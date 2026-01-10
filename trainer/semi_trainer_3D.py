from logging import raiseExceptions
import os
import sys
import time
from cv2 import threshold
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
import random
import wandb
from tqdm import tqdm
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join

from utils import losses,ramps,cac_loss,feature_memory,correlation
from dataset.BCVData import BCVDataset, BCVDatasetCAC,DatasetSR
from dataset.dataset import DatasetSemi
from dataset.sampler import BatchSampler, ClassRandomSampler
from networks.net_factory_3d import net_factory_3d
from dataset.dataset import TwoStreamBatchSampler
from inference.val_3D import test_all_case
from unet3d.losses import DiceLoss #test loss







class SemiSupervisedTrainer3D:
    def __init__(self, config, output_folder, logging,
                 continue_training: bool = False) -> None:
        self.config = config
        self.device = torch.device(f"cuda:{config['gpu']}")
        self.output_folder = output_folder
        self.logging = logging
        self.exp = config['exp']
        self.weight_decay = config['weight_decay']
        self.lr_scheduler_eps = config['lr_scheduler_eps']
        self.lr_scheduler_patience = config['lr_scheduler_patience']
        self.seed = config['seed']
        self.initial_lr = config['initial_lr']
        self.initial2_lr = config['initial2_lr']
        self.model1_thresh = config['model1_thresh'] #threshold for model1 pred
        self.model2_thresh = config['model2_thresh'] #threshold for model2 pred
        self.optimizer_type = config['optimizer_type']
        self.optimizer2_type = config['optimizer2_type']
        self.backbone = config['backbone']
        self.backbone2 = config['backbone2']
        self.max_iterations = config['max_iterations']
        self.began_semi_iter = config['began_semi_iter']
        self.ema_decay = config['ema_decay']
        self.began_condition_iter = config['began_condition_iter']
        self.began_eval_iter = config['began_eval_iter']
        self.show_img_freq = config['show_img_freq']
        self.save_checkpoint_freq = config['save_checkpoint_freq']
        self.val_freq = config['val_freq']

        # config for training from checkpoint
        self.continue_wandb = config['continue_wandb']
        self.continue_training = continue_training
        self.wandb_id = config['wandb_id']
        self.network_checkpoint = config['model_checkpoint']
        self.network2_checkpoint = config['model2_checkpoint'] # for CPS based methods

        # config for semi-supervised
        self.consistency_rampup = config['consistency_rampup']
        self.consistency = config['consistency']
        
        # config for dataset
        dataset = config['DATASET']
        self.patch_size = dataset['patch_size']
        self.labeled_num = dataset['labeled_num']
        self.labeled_num_pl = dataset['labeled_num_pl']

        self.batch_size = dataset['batch_size']
        self.labeled_bs = dataset['labeled_bs']
        self.cutout = dataset['cutout']
        self.rotate_trans = dataset['rotate_trans']
        self.scale_trans = dataset['scale_trans']
        self.random_rotflip = dataset['random_rotflip']
        self.edge_prob = dataset['edge_prob']
        self.normalization = dataset['normalization']
        self.dataset_name = config['dataset_name']
        
        dataset_config = dataset[self.dataset_name]
        self.num_classes = dataset_config['num_classes']
        self.class_name_list = dataset_config['class_name_list']
        self.training_data_num = dataset_config['training_data_num']
        self.testing_data_num = dataset_config['testing_data_num']
        self.train_list = dataset_config['train_list']
        #train_list_pl: train list for partial labeled data
        self.train_list_pl = dataset_config['train_list_pl']
        self.test_list = dataset_config['test_list']
        self.cut_upper = dataset_config['cut_upper']
        self.cut_lower = dataset_config['cut_lower']
        self.weights = dataset_config['weights']

        # config for method
        self.method_name = config['method']
        self.method_config = config['METHOD'][self.method_name]
        self.use_CAC = config['use_CAC']
        self.use_PL = config['use_PL']

        self.experiment_name = None
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = losses.DiceLoss(self.num_classes)
        self.dice_loss_con = losses.DiceLoss(2)
        self.dice_loss2 = DiceLoss(normalization='softmax')
        self.cvcl_loss = None
        self.best_performance = 0.0
        self.best_performance2 = 0.0 # for CPS based methods
        self.current_iter = 0
        self.model = None
        self.ema_model = None # for MT based methods
        self.model2 = None # for CPS based methods
        self.scaler = None 
        self.scaler2 = None

        self.dataset = None
        self.dataset_pl = None # dataset for partial labeled data
        self.dataloader = None
        self.dataloader_pl = None # dataloader for partial label data
        self.wandb_logger = None
        self.tensorboard_writer = None
        self.labeled_idxs = None 
        self.unlabeled_idxs = None

        # generate by training process
        self.current_lr = self.initial_lr
        self.current2_lr = self.initial2_lr
        self.loss = None
        self.loss_ce = None 
        self.loss_dice = None
        self.loss_dice_con = None # for conditional network
        self.consistency_loss = None
        self.consistency_weight = None
        self.grad_scaler1 = GradScaler()
        self.grad_scaler2 = GradScaler()

    
    def initialize(self):
        self.experiment_name = f"{self.dataset_name}_{self.method_name}_"\
                               f"labeled{self.labeled_num}_"\
                               f"{self.optimizer_type}_{self.optimizer2_type}"\
                               f"_{self.exp}"

        self.wandb_logger = wandb.init( name=self.experiment_name,
                                        project="semi-supervised-segmentation",
                                        config = self.config)
    
        wandb.tensorboard.patch(root_logdir=self.output_folder + '/log')
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')
        self.load_dataset()
        self.initialize_optimizer_and_scheduler()
    
    def initialize_network(self):
        self.model = net_factory_3d(
            net_type=self.backbone,in_chns=1, class_num=self.num_classes,
            model_config=self.config['model'], device=self.device
        )
        if self.method_name in ['MT','UAMT']:
            self.ema_model = net_factory_3d(
                net_type=self.backbone,in_chns=1, class_num=self.num_classes,
                model_config=self.config['model'], device=self.device
            )
            for param in self.ema_model.parameters():
                param.detach_()
        elif self.method_name == 'CPS':
            self.model2 = net_factory_3d(
                net_type=self.backbone,in_chns=1, class_num=self.num_classes,
                model_config=self.config['model'], device=self.device
            )
            self._kaiming_normal_init_weight()
        elif self.method_name in ['C3PS','ConNet']:
            self.model2 = net_factory_3d(
                self.backbone2, in_chns=1, class_num=2,device=self.device
            )
            self._kaiming_normal_init_weight()
        elif self.method_name == 'CSSR':
            self.model2 = net_factory_3d(
                self.backbone2, in_chns=1, class_num=self.num_classes,
                device=self.device,
                large_patch_size=self.method_config['patch_size_large']
            )
            self._kaiming_normal_init_weight()
        elif self.method_name == 'URPC':
            print("URPC")
            self.model = net_factory_3d(
                net_type='URPC', class_num=self.num_classes
            )
        elif self.method_name == 'McNet':
            self.model = net_factory_3d(
                net_type='McNet',in_chns=1, class_num=self.num_classes, 
                device=self.device
            )
    
    def load_checkpoint(self, fname="latest"):
        checkpoint  = torch.load(join(self.output_folder,
                                      "model1_"+fname+".pth"))
        network_weights = checkpoint['network_weights']
        self.model.load_state_dict(network_weights)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if self.grad_scaler1 is not None:
            self.grad_scaler1.load_state_dict(checkpoint['grad_scaler_state'])
        self.current_iter = checkpoint['current_iter']
        # self.wandb_logger = wandb.init(name=self.experiment_name,
        #                                     project="semi-supervised-segmentation",
        #                                     config = self.config,
        #                                     id=checkpoint['wandb_id'],
        #                                     resume='must')
        # wandb.tensorboard.patch(root_logdir=self.output_folder + '/log')
        print(f"=====> Load checkpoint from {join(self.output_folder, 'model1_'+fname+'.pth')} for model1 Successfully")
        
        # load  checkpoint for model2 
        if self.model2 is not None:
            checkpoint2  = torch.load(join(self.output_folder,
                                      "model2_"+fname+".pth"))
            network_weights2 = checkpoint2['network_weights']
            self.model2.load_state_dict(network_weights2)
            self.optimizer2.load_state_dict(checkpoint2['optimizer_state'])
            if self.grad_scaler2 is not None:
                self.grad_scaler2.load_state_dict(checkpoint2['grad_scaler_state'])
            print(f"=====> Load checkpoint from {join(self.output_folder, 'model2_'+fname+'.pth')} for model2 Successfully")
        

    def load_dataset(self):
        train_supervised = False
        if self.method_name == 'Baseline':
            train_supervised = True

        self.dataset = DatasetSemi(img_list_file=self.train_list, 
                                        cutout=self.cutout,
                                        rotate_trans=self.rotate_trans, 
                                        scale_trans=self.scale_trans,
                                        random_rotflip=self.random_rotflip,
                                        patch_size=self.patch_size, 
                                        num_class=self.num_classes, 
                                        edge_prob=self.edge_prob,
                                        upper=self.cut_upper,
                                        lower=self.cut_lower,
                                        labeled_num=self.labeled_num,
                                        train_supervised=train_supervised,
                                        normalization=self.normalization)
        if self.method_name in ['C3PS','ConNet','CVCL','CVCL_partial']:
            self.dataset = BCVDatasetCAC(
                img_list_file=self.train_list,
                patch_size=self.patch_size,
                num_class=self.num_classes,
                stride=self.method_config['stride'],
                iou_bound=[self.method_config['iou_bound_low'],
                           self.method_config['iou_bound_high']],
                labeled_num=self.labeled_num,
                cutout=self.cutout,
                rotate_trans=self.rotate_trans,
                scale_trans=self.scale_trans,
                random_rotflip=self.random_rotflip,
                upper=self.cut_upper,
                lower=self.cut_lower,
                con_list=self.method_config['con_list'],
                addi_con_list=self.method_config['addition_con_list'],
                weights=self.weights
            )
            self.dataset_pl = DatasetSemi(
                img_list_file=self.train_list_pl, 
                cutout=self.cutout,
                rotate_trans=self.rotate_trans, 
                random_rotflip=self.random_rotflip,
                patch_size=self.patch_size, 
                num_class=2, 
                edge_prob=self.edge_prob,
                upper=self.cut_upper,
                lower=self.cut_lower,
                labeled_num=self.labeled_num_pl, #TODO labeled num for partial label
                train_supervised=True,
                normalization=self.normalization
            )
        if self.method_name == 'CSSR':
            self.dataset = DatasetSR(
                img_list_file=self.train_list,
                patch_size_small=self.patch_size,
                patch_size_large=self.method_config['patch_size_large'],
                num_class=self.num_classes,
                stride=self.method_config['stride'],
                iou_bound=[self.method_config['iou_bound_low'],
                           self.method_config['iou_bound_high']],
                labeled_num=self.labeled_num,
                cutout=self.cutout,
                rotate_trans=self.rotate_trans,
                scale_trans=self.scale_trans,
                random_rotflip=self.random_rotflip,
                upper=self.cut_upper,
                lower=self.cut_lower,
                weights=self.weights
            )

    def get_dataloader(self):
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=2, 
                                     pin_memory=True)
        self.dataloader_pl = DataLoader(self.dataset_pl, batch_size=self.batch_size, 
                                     shuffle=True, num_workers=2, 
                                     pin_memory=True)

    def initialize_optimizer_and_scheduler(self):
        assert self.model is not None, "self.initialize_network must be called first"
        self.scaler = GradScaler()
        if self.optimizer_type == "Adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          self.initial_lr, 
                                          weight_decay=self.weight_decay,
                                          amsgrad=True)
        
        # self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3, 
        #                                   weight_decay=self.weight_decay)
        elif self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                            lr=self.initial_lr, momentum=0.9, 
                                            weight_decay=self.weight_decay)
        else:
            print("unrecognized optimizer, use Adam instead")
            self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                          self.initial_lr, 
                                          weight_decay=self.weight_decay,
                                          amsgrad=True)
        
        if self.method_name in ['CPS', 'C3PS', 'ConNet', 'CSSR']:
            self.scaler2 = GradScaler()
            if self.optimizer2_type == 'Adam':
                self.optimizer2 = torch.optim.Adam(
                    self.model2.parameters(), 
                    self.initial2_lr, 
                    weight_decay=self.weight_decay,
                    amsgrad=True)
            elif self.optimizer2_type == 'SGD':
                self.optimizer2 = torch.optim.SGD(self.model2.parameters(), 
                    lr=self.initial2_lr, momentum=0.9, 
                    weight_decay=self.weight_decay)
            else:
                print("unrecognized optimizer type, use adam instead!")
                self.optimizer = torch.optim.Adam(
                    self.model2.parameters(), 
                    self.initial_lr, 
                    weight_decay=self.weight_decay,
                    amsgrad=True
                )
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', factor=0.2,
            patience=self.lr_scheduler_patience,
            verbose=True, threshold=self.lr_scheduler_eps,
            threshold_mode="abs")
    
    def train(self):
        self.labeled_idxs = list(range(0, self.labeled_num))
        self.unlabeled_idxs = list(range(self.labeled_num, self.training_data_num))
        if self.method_name == "Baseline":
            self.dataloader = DataLoader(self.dataset, 
                                         batch_size=self.batch_size,
                                         shuffle=True, num_workers=2,
                                         pin_memory=False)
            self.max_epoch = self.max_iterations // len(self.dataloader) + 1
            print(f"max epochs:{self.max_epoch}, max iterations:{self.max_iterations}")
            print(f"len dataloader:{len(self.dataloader)}")
            self._train_baseline()
        elif self.method_name == 'CVCL_partial':
            self.dataloader = DataLoader(self.dataset, 
                                         batch_sampler = BatchSampler(
                                            ClassRandomSampler(self.dataset), 
                                            self.batch_size, True), 
                                         num_workers=2, pin_memory=True)
            self.max_epoch = self.max_iterations // len(self.dataloader) + 1
            self.cvcl_loss = cac_loss.CAC(
                self.num_classes, 
                stride=self.method_config['stride'], 
                selected_num=400, 
                b = 500, 
                step_save=self.method_config['step_save'], 
                temp=0.1, proj_final_dim=64, 
                pos_thresh_value=self.method_config['threshold'], 
                weight=0.1
            )
            self._train_CVCL_partial()      
        else:           
            batch_sampler = TwoStreamBatchSampler(self.labeled_idxs, 
                                            self.unlabeled_idxs,
                                            self.batch_size, 
                                            self.batch_size-self.labeled_bs)
            self.dataloader = DataLoader(self.dataset, batch_sampler=batch_sampler,
                            num_workers=2, pin_memory=True)
            self.max_epoch = self.max_iterations // len(self.dataloader) + 1
            if self.method_name == 'UAMT':
                self._train_UAMT()
            elif self.method_name == 'MT':
                self._train_MT()
            elif self.method_name == 'CPS':
                self._train_CPS()
            elif self.method_name == 'C3PS':
                self.dataloader_pl = DataLoader(self.dataset_pl, 
                                batch_size=4, 
                                shuffle=True, num_workers=2, 
                                pin_memory=True)
                self._train_C3PS()
            elif self.method_name == 'DAN':
                self._train_DAN()
            elif self.method_name == 'URPC':
                self._train_URPC()
            elif self.method_name == 'EM':
                self._train_EM()
            elif self.method_name == 'ConNet':
                self._train_ConditionNet()
            elif self.method_name == 'McNet':
                self._train_McNet()
            elif self.method_name == 'CVCL':
                self.cvcl_loss = cac_loss.CAC(
                    self.num_classes, 
                    stride=self.method_config['stride'], 
                    selected_num=400, 
                    b = 500, 
                    step_save=self.method_config['step_save'], 
                    temp=0.1, proj_final_dim=64, 
                    pos_thresh_value=self.method_config['threshold'], 
                    weight=0.1
                )
                self._train_CVCL()      
            elif self.method_name == 'CSSR':
                self._train_CSSR()
            elif self.method_name == "CAML":
                self._train_CAML()
            else:
                print(f"no such method {self.method_name}"+"!"*10)
                sys.exit(0)
    
    def evaluation(self, model, do_condition=False, do_SR=False, model_name="model"):
        """
        do_SR: whether do super resolution model
        """
        print("began evaluation!")
        model.eval()
        class_id_list = range(1, self.num_classes)
        if do_condition:
            best_performance = self.best_performance2
            model_name = "model2"
            con_list = self.method_config['con_list'] + self.method_config['addition_con_list']
            class_id_list = con_list
        else:
            best_performance = self.best_performance
            con_list = None
        test_num = self.testing_data_num // 1.5
        if self.current_iter % 1000==0 or (self.method_name=='ConNet' and self.current_iter % 400==0):
            test_num = self.testing_data_num

        avg_metric = test_all_case(model,test_list=self.test_list,
                                       num_classes=self.num_classes,
                                       patch_size=self.method_config['patch_size_large'] if do_SR else self.patch_size,
                                       batch_size=2,
                                       stride_xy=64, stride_z=64,
                                       overlap=0.2,
                                       cut_upper=self.cut_upper,
                                       cut_lower=self.cut_lower,
                                       do_condition=do_condition,
                                       do_SR=do_SR,
                                       test_num=test_num,
                                       method=self.method_name.lower(),
                                       con_list=con_list,
                                       normalization=self.normalization)
        print("avg metric shape:",avg_metric.shape)
        if avg_metric[:, 0].mean() > best_performance:
            best_performance = avg_metric[:, 0].mean()
            save_name = f'iter_{self.current_iter}_dice_{round(best_performance,4)}'
            self._save_checkpoint(save_name)

        self.tensorboard_writer.add_scalar(f'info/{model_name}_val_dice_score',
                        avg_metric[:, 0].mean(), self.current_iter)
        self.tensorboard_writer.add_scalar(f'info/{model_name}val_hd95',
                        avg_metric[:, 1].mean(), self.current_iter)
        self.logging.info(
            'iteration %d : %s_dice_score : %f %s_hd95 : %f' % (
                self.current_iter, 
                model_name,
                avg_metric[:, 0].mean(), 
                model_name,
                avg_metric[:, 1].mean()))
        # print metric of each class
        for i,class_id in enumerate(class_id_list):
            class_name = self.class_name_list[class_id-1]
            self.tensorboard_writer.add_scalar(
                f'DSC_each_class/{model_name}_{class_name}',
                        avg_metric[i, 0], self.current_iter
            )
            self.tensorboard_writer.add_scalar(
                f'HD_each_class/{model_name}_{class_name}',
                            avg_metric[i, 1], self.current_iter
            )

        return avg_metric


    def _train_baseline(self):
        print("================> Training Baseline <===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.model.train()
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'].long())
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                self.optimizer.zero_grad()
                with autocast():
                    outputs = self.model(volume_batch)
                    outputs_soft = torch.softmax(outputs, dim=1)

                    label_batch = torch.argmax(label_batch, dim=1)
                    self.loss_ce = self.ce_loss(outputs, label_batch.long())
                    self.loss_dice = self.dice_loss(outputs_soft, 
                                                    label_batch.unsqueeze(1))
                    self.loss = 0.5 * (self.loss_dice + self.loss_ce)
                self.grad_scaler1.scale(self.loss).backward()
                self.grad_scaler1.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.grad_scaler1.step(self.optimizer)
                self.grad_scaler1.update()

                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()

                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                      self.current_iter)

                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                    grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                        0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                    grid_image, self.current_iter)

                if (self.current_iter > self.began_eval_iter and 
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    self.evaluation(model=self.model)

                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint(filename="latest")

                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter>= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)

    def _train_DAN(self):
        print("================> Training DAN <===============")
        self.model2 = net_factory_3d(
            net_type="DAN", class_num=self.num_classes, device=self.device
        )
        self.optimizer2 = torch.optim.Adam(
            self.model2.parameters(), lr=0.0001, betas=(0.9, 0.99)
        )
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = (
                    volume_batch.to(self.device), label_batch.to(self.device)
                )
                DAN_target = torch.tensor([1, 0]).to(self.device)
                self.model.train()
                self.model2.eval()

                outputs = self.model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                label_batch = torch.argmax(label_batch, dim=1)
                self.loss_ce = self.ce_loss(
                    outputs[:self.labeled_bs],label_batch[:self.labeled_bs]
                )
                self.loss_dice = self.dice_loss(
                    outputs_soft[:self.labeled_bs],
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                supervised_loss = 0.5 * (self.loss_dice + self.loss_ce)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter // 6
                )
                DAN_outputs = self.model2(
                    outputs_soft[self.labeled_bs:], 
                    volume_batch[self.labeled_bs:]
                )
                if self.current_iter > self.began_semi_iter:
                    self.consistency_loss = self.ce_loss(
                        DAN_outputs, (DAN_target[:self.labeled_bs]).long()
                    )
                else:
                    self.consistency_loss = torch.FloatTensor([0.0]).to(self.device)
                self.loss = supervised_loss + self.consistency_weight * \
                       self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self.model.eval()
                self.model2.train()
                with torch.no_grad():
                    outputs = self.model(volume_batch)
                    outputs_soft = torch.softmax(outputs, dim=1)
                
                DAN_outputs = self.model2(outputs_soft, volume_batch)
                DAN_loss = self.ce_loss(DAN_outputs, DAN_target.long())
                self.optimizer2.zero_grad()
                DAN_loss.backward()
                self.optimizer2.step()

                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()

                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image(
                        'train/Image', grid_image, self.current_iter
                    )

                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                    grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                        0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                    grid_image, self.current_iter)
                if (
                    self.current_iter > self.began_eval_iter and 
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    self.evaluation(model=self.model)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break 
        self.tensorboard_writer.close()
        print("Training done!")

    def _train_URPC(self):
        print("================> Training URPC <===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        kl_distance = nn.KLDivLoss(reduction='none')
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
                label_batch = torch.argmax(label_batch, dim=1)
                outputs_aux1, outputs_aux2, outputs_aux3, outputs_aux4 = (
                    self.model(volume_batch)
                )
                outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
                outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
                outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)
                outputs_aux4_soft = torch.softmax(outputs_aux4, dim=1)

                loss_ce_aux1 = self.ce_loss(
                    outputs_aux1[:self.labeled_bs], label_batch[:self.labeled_bs]
                )
                loss_ce_aux2 = self.ce_loss(
                    outputs_aux2[:self.labeled_bs], label_batch[:self.labeled_bs]
                )
                loss_ce_aux3 = self.ce_loss(
                    outputs_aux3[:self.labeled_bs], label_batch[:self.labeled_bs]
                )
                loss_ce_aux4 = self.ce_loss(
                    outputs_aux4[:self.labeled_bs], label_batch[:self.labeled_bs]
                )

                loss_dice_aux1 = self.dice_loss(
                    outputs_aux1_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                loss_dice_aux2 = self.dice_loss(
                    outputs_aux2_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                loss_dice_aux3 = self.dice_loss(
                    outputs_aux3_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                loss_dice_aux4 = self.dice_loss(
                    outputs_aux4_soft[:self.labeled_bs], 
                    label_batch[:self.labeled_bs].unsqueeze(1)
                )
                
                self.loss_ce = (
                    loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4
                ) / 4.
                self.loss_dice = (
                    loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4
                ) / 4.
                supervised_loss = (
                    loss_ce_aux1+loss_ce_aux2+loss_ce_aux3+loss_ce_aux4+
                    loss_dice_aux1+loss_dice_aux2+loss_dice_aux3+loss_dice_aux4
                ) / 8.

                preds = (
                    outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft +
                    outputs_aux4_soft
                ) / 4.

                variance_aux1 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux1_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux1 = torch.exp(-variance_aux1)

                variance_aux2 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux2_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux2 = torch.exp(-variance_aux2)

                variance_aux3 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux3_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux3 = torch.exp(-variance_aux3)

                variance_aux4 = torch.sum(
                    kl_distance(
                        torch.log(outputs_aux4_soft[self.labeled_bs:]),
                        preds[self.labeled_bs:]
                    ),
                    dim=1, keepdim=True
                )
                exp_variance_aux4 = torch.exp(-variance_aux4)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter // 150
                )
                consistency_dist_aux1 = (
                    preds[self.labeled_bs:] - outputs_aux1_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux1 = torch.mean(
                    consistency_dist_aux1 * exp_variance_aux1
                ) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(variance_aux1)

                consistency_dist_aux2 = (
                    preds[self.labeled_bs:] - outputs_aux2_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux2 = torch.mean(
                    consistency_dist_aux2 * exp_variance_aux2
                ) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(variance_aux2)

                consistency_dist_aux3 = (
                    preds[self.labeled_bs:] - outputs_aux3_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux3 = torch.mean(
                    consistency_dist_aux3 * exp_variance_aux3
                ) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(variance_aux3)

                consistency_dist_aux4 = (
                    preds[self.labeled_bs:] - outputs_aux4_soft[self.labeled_bs:]
                ) ** 2
                consistency_loss_aux4 = torch.mean(
                    consistency_dist_aux4 * exp_variance_aux4
                ) / (torch.mean(exp_variance_aux4) + 1e-8) + torch.mean(variance_aux4)
                self.consistency_loss = (
                    consistency_loss_aux1 + consistency_loss_aux2 + 
                    consistency_loss_aux3 + consistency_loss_aux4
                ) / 4.

                if self.current_iter<self.began_semi_iter:
                    self.consistency_weight = 0.0
                self.loss = supervised_loss + self.consistency_weight * self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()

                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image(
                        'train/Image', grid_image, self.current_iter
                    )

                    image = torch.argmax(outputs_aux1_soft, dim=1, keepdim=True)[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Predicted_label',grid_image, self.current_iter
                    )

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                        0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1) * 100
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Groundtruth_label',grid_image, self.current_iter
                    )
                if (
                        self.current_iter > self.began_eval_iter and 
                        self.current_iter % self.val_freq ==0
                ) or self.current_iter == 20:
                    self.evaluation(model=self.model)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break 
        self.tensorboard_writer.close()
        print('Training Finished')
    
    
    def _train_McNet(self):
        print("================> Training McNet <===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'])
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                label_batch = torch.argmax(label_batch,dim=1)
                outputs = self.model(volume_batch)
                num_outputs = len(outputs)
                y_ori = torch.zeros((num_outputs,) + outputs[0].shape)
                y_pseudo_label = torch.zeros((num_outputs,) + outputs[0].shape)
                self.loss_ce = 0
                self.loss_dice = 0 
                for idx in range(num_outputs):
                    y = outputs[idx][:self.labeled_bs,...]
                    y_prob = torch.softmax(y, dim=1)
                    self.loss_ce += self.ce_loss(
                        y[:self.labeled_bs],label_batch[:self.labeled_bs]
                    )
                    self.loss_dice += self.dice_loss(
                        y_prob, label_batch[:self.labeled_bs,...].unsqueeze(1)
                    )

                    y_all = outputs[idx]
                    y_prob_all = torch.softmax(y_all, dim=1)
                    y_ori[idx] = y_prob_all
                    y_pseudo_label[idx] = self._sharpening(y_prob_all)
                
                self.consistency_loss = 0
                if self.current_iter > self.began_semi_iter:
                    for i in range(num_outputs):
                        for j in range(num_outputs):
                            if i != j:
                                self.consistency_loss += losses.mse_loss(
                                    y_ori[i], y_pseudo_label[j]
                                )


                consistency_weight = self._get_current_consistency_weight(
                    self.current_iter//150
                )
                
                self.loss = 0.5 * (self.loss_dice) + consistency_weight * self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if (
                    self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    self.evaluation(model=self.model)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
                
    
    def _train_MT(self):
        print("================> Training MT <===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'])
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                unlabeled_volume_batch = volume_batch[self.labeled_bs:]
                noise = torch.clamp(torch.randn_like(
                    unlabeled_volume_batch)*0.1, -0.2, 0.2)
                ema_inputs = unlabeled_volume_batch + noise
                ema_inputs = ema_inputs.to(self.device)

                outputs = self.model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output = self.ema_model(ema_inputs)
                    ema_output_soft = torch.softmax(ema_output, dim=1)
                label_batch = torch.argmax(label_batch,dim=1)
                self.loss_ce = self.ce_loss(outputs[:self.labeled_bs],
                                            label_batch[:self.labeled_bs][:])
                self.loss_dice = self.dice_loss(outputs_soft[:self.labeled_bs],
                    label_batch[:self.labeled_bs].unsqueeze(1)
                    )
                supervised_loss = 0.5 * (self.loss_dice + self.loss_ce)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter // 4
                )
                if self.current_iter > self.began_semi_iter:
                    self.consistency_loss = torch.mean(
                        (outputs_soft[self.labeled_bs:] - ema_output_soft)**2
                    )
                else:
                    self.consistency_loss = torch.FloatTensor([0]).to(self.device)
                self.loss = supervised_loss + self.consistency_weight * \
                            self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self._update_ema_variables()
                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image,
                                                      self.current_iter)
                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(2, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                                      grid_image,
                                                      self.current_iter)
                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image,
                                                      self.current_iter)
                if (
                    self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    self.evaluation(model=self.model)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint("latest")
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)

    def _train_CPS(self):
        print("================> Training CPS <===============")
        self.model2.train()
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = (
                    volume_batch.to(self.device), label_batch.to(self.device)
                )
                noise1 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                )
                label_batch = torch.argmax(label_batch, dim=1)
                outputs1 = self.model(volume_batch + noise1)
                outputs_soft1 = torch.softmax(outputs1, dim=1)
                noise2 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                )
                outputs2 = self.model2(volume_batch + noise2)
                outputs_soft2 = torch.softmax(outputs2, dim=1)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter//150
                )
                loss1 = 0.5 * (self.ce_loss(outputs1[:self.labeled_bs],
                                   label_batch[:][:self.labeled_bs].long()) + 
                               self.dice_loss(outputs_soft1[:self.labeled_bs], 
                                             label_batch[:self.labeled_bs].\
                                                unsqueeze(1)))
                loss2 = 0.5 * (self.ce_loss(outputs2[:self.labeled_bs],
                                   label_batch[:][:self.labeled_bs].long()) + 
                               self.dice_loss(outputs_soft2[:self.labeled_bs], 
                                             label_batch[:self.labeled_bs].\
                                                unsqueeze(1)))
                pseudo_outputs1 = torch.argmax(
                    outputs_soft1[self.labeled_bs:].detach(),
                    dim=1, keepdim=False
                )
                pseudo_outputs2 = torch.argmax(
                    outputs_soft2[self.labeled_bs:].detach(),
                    dim=1, keepdim=False
                )
                if self.current_iter < self.began_semi_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                    pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                else:
                    pseudo_supervision1 = self.ce_loss(
                        outputs1[self.labeled_bs:],
                        pseudo_outputs2
                    )
                    pseudo_supervision2 = self.ce_loss(
                        outputs2[self.labeled_bs:],
                        pseudo_outputs1
                    )
                model1_loss = loss1 + self.consistency_weight *  \
                                      pseudo_supervision1
                model2_loss = loss2 + self.consistency_weight * \
                                      pseudo_supervision2
                loss = model1_loss + model2_loss 
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()

                loss.backward()
                self.optimizer.step()
                self.optimizer2.step()
                
                self._adjust_learning_rate()
                self.current_iter += 1

                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar(
                'consistency_weight/consistency_weight',self.consistency_weight, 
                self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model1_loss', model1_loss, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', model2_loss, 
                                               self.current_iter)
                self.logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (
                    self.current_iter,model1_loss.item(), 
                    model2_loss.item()))
            
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                  self.current_iter)

                    image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                    'train/Model1_Predicted_label',
                     grid_image, self.current_iter)

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Model2_Predicted_label',
                                 grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                 grid_image, self.current_iter)

                if (
                    self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter==20:
                    self.evaluation(model=self.model)
                    self.evaluation(model=self.model2,model_name='model2')
                    self.model.train()
                    self.model2.train()
            
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_mode_path = os.path.join(
                    self.output_folder, 'model1_iter_' + \
                        str(self.current_iter) + '.pth')
                    torch.save(self.model.state_dict(), save_mode_path)
                    self.logging.info("save model1 to {}".format(save_mode_path))

                    save_mode_path = os.path.join(
                        self.output_folder, 
                        'model2_iter_' + str(self.current_iter) + '.pth')
                    torch.save(self.model2.state_dict(), save_mode_path)
                    self.logging.info("save model2 to {}".format(save_mode_path))
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        
            
    def _get_condition(self, pred_con_list):
        """
        get conditon number for conditional network
        """
        if 0 in pred_con_list:
            pred_con_list.remove(0)
        inter_label_list = list(
            set(pred_con_list) & set(self.method_config['con_list'])
        )
        # use num_class as con label
        inter_label_list+=self.method_config['addition_con_list']
        if len(inter_label_list) == 0:
            inter_label_list = self.method_config['con_list']
        con = np.random.choice(inter_label_list)
        return con

    def _train_CVCL_partial(self):
        print("===========> Training CVCL for partially labeled data<========")   
        iterator = tqdm(range(20000), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self._adjust_learning_rate()  
                self.model.train() 
                volume_batch, label_batch = (
                    sampled_batch['image'],sampled_batch['label']
                )
                task_id = sampled_batch['task_id']
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                if self.current_iter<10000 and torch.sum(task_id)>0:
                    print("==========>break for partial label data!")
                    break
                if torch.sum(task_id) != task_id[0]*len(task_id):
                    continue
                volume_batch = torch.cat(
                    [volume_batch[:,0,...],volume_batch[:,1,...]],
                    dim=0
                )
                label_batch = torch.cat(
                    [label_batch[:,0,...],label_batch[:,1,...]],
                    dim=0
                )
                noise1 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                ).to(self.device)
                outputs,CL_outputs = self.model(volume_batch + noise1)
                print("task_id:",task_id)
                outputs_soft1 = torch.softmax(outputs, dim=1)
                # gt_onehot = torch.zeros((self.batch_size, self.num_class, self.patch_size[0], self.patch_size[1],self.patch_size[2]))
                # gt_onehot.scatter_(0, label_batch[0].long(), 1)
                # mask_array = gt_onehot     
                if torch.sum(task_id)>0:
                    #conbine partially labeled dataset
                    # bg_class = [0,1,2,3,4,5]
                    # bg_class.remove(task_id[0])
                    # merge_mask= mask[:,[0,task_id[0]],:,:,:]
                    # if task_id[0] == 2:
                    #     bg_class.remove(task_id[0]+1)
                    #     merge_mask= mask[:,[0,task_id[0], task_id[0]+1],:,:,:]
                    # merge_output = torch.zeros_like(merge_mask)
                    # merge_output[:,0,:,:,:] = torch.sum(output[:,bg_class,:,:,:],dim=1)
                    # merge_output[:,1,:,:,:] = output[:,task_id[0],:,:,:]
                    # if task_id[0] == 2:
                    #     merge_output[:,2,:,:,:] = output[:,task_id[0]+1,:,:,:]
                    # output = merge_output 
                    # mask = merge_mask
                    # label_batch[label_batch==0] = 255 
                    # loss_ce = self.ce_loss(
                    #         outputs,
                    #         label_batch.long()
                    #     )
                    
                    #label_batch[label_batch==255] = 0
                    loss_dice = self.dice_loss(
                            outputs_soft1,
                            label_batch.unsqueeze(1),
                            skip_id=0
                        ) 
                    loss_sup = loss_dice
                else:
                    loss_sup =  (
                        self.ce_loss(
                            outputs,
                            label_batch.long()
                        ) +
                        self.dice_loss(
                            outputs_soft1,
                            label_batch.unsqueeze(1)
                        )
                    )
                print(f"current iter:{self.current_iter}")
                print(f"loss sup:{loss_sup.item()}")
                loss_cl = torch.FloatTensor([0.0]).to(self.device)
                if self.current_iter> self.began_semi_iter:
                    loss_cl = self.cvcl_loss(
                        output_ul1=CL_outputs[1:2], 
                        output_ul2=CL_outputs[3:4], 
                        logits1=outputs[1:2], 
                        logits2=outputs[3:4], 
                        ul1=[x[1] for x in sampled_batch['ul1']], 
                        br1=[x[1] for x in sampled_batch['br1']], 
                        ul2=[x[1] for x in sampled_batch['ul2']], 
                        br2=[x[1] for x in sampled_batch['br2']]
                    )
                
                loss = loss_sup + loss_cl
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.current_iter += 1
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/loss', 
                                                   loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/loss_sup', 
                                                   loss_sup, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/loss_cl', 
                                    loss_cl, 
                                    self.current_iter)
                # self.tensorboard_writer.add_scalar(
                #     'loss/pseudo_supervision1',
                #     pseudo_supervision1, self.current_iter
                # )
                self.logging.info(
                    'iteration %d:'
                    ' loss: %f' 
                    ' supvised loss: %f'
                    ' CVCL loss: %f' % (
                        self.current_iter, loss.item(), 
                        loss_sup.item(), 
                        loss_cl.item()
                    )
                )
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    with torch.no_grad():
                        self.evaluation(model=self.model)
                    self.model.train()
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_model_path = os.path.join(
                        self.output_folder,
                        'model_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.model.state_dict(), save_model_path)
                    self.logging.info(f"save model to {save_model_path}")
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break    
            
    def _train_CVCL(self):
        print("================> Training CVCL<===============")   
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self._adjust_learning_rate()  
                self.model.train() 
                volume_batch, label_batch = (
                    sampled_batch['image'],sampled_batch['label']
                )
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                labeled_idxs_batch = torch.arange(0, self.labeled_bs)
                unlabeled_idx_batch = torch.arange(self.labeled_bs, 
                                                   self.batch_size)
                volume_batch = torch.cat(
                    [volume_batch[:,0,...],volume_batch[:,1,...]],
                    dim=0
                )
                label_batch = torch.cat(
                    [label_batch[:,0,...],label_batch[:,1,...]],
                    dim=0
                )
                labeled_idxs2_batch = torch.arange(
                    self.batch_size,
                    self.batch_size+self.labeled_bs
                )
                labeled_idxs1_batch = torch.arange(0,self.labeled_bs)
                labeled_idxs_batch = torch.cat(
                    [labeled_idxs1_batch,labeled_idxs2_batch]
                )
                unlabeled_idxs1_batch = torch.arange(self.labeled_bs,
                                                        self.batch_size)
                unlabeled_idxs2_batch = torch.arange(
                    self.batch_size+self.labeled_bs, 
                    2 * self.batch_size
                )
                unlabeled_idxs_batch = torch.cat(
                    [unlabeled_idxs1_batch,unlabeled_idxs2_batch]
                ) 
                noise1 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                ).to(self.device)
                outputs,CL_outputs = self.model(volume_batch + noise1)
                outputs_soft1 = torch.softmax(outputs, dim=1) 
                loss_sup =  (
                    self.ce_loss(
                        outputs[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].long()
                    ) +
                    self.dice_loss(
                        outputs_soft1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].unsqueeze(1)
                    )
                )
                print(f"current iter:{self.current_iter}")
                print(f"loss sup:{loss_sup.item()}")
                loss_cl = torch.FloatTensor([0.0]).to(self.device)
                if self.current_iter> self.began_semi_iter:
                    loss_cl = self.cvcl_loss(
                        output_ul1=CL_outputs[unlabeled_idxs1_batch], 
                        output_ul2=CL_outputs[unlabeled_idxs2_batch], 
                        logits1=outputs[unlabeled_idxs1_batch], 
                        logits2=outputs[unlabeled_idxs2_batch], 
                        ul1=[x[1] for x in sampled_batch['ul1']], 
                        br1=[x[1] for x in sampled_batch['br1']], 
                        ul2=[x[1] for x in sampled_batch['ul2']], 
                        br2=[x[1] for x in sampled_batch['br2']]
                    )
                loss = loss_sup + loss_cl
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.current_iter += 1
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                # self.tensorboard_writer.add_scalar(
                #     'consistency_weight/consistency_weight', 
                #     self.consistency_weight, 
                #     self.current_iter
                # )
                self.tensorboard_writer.add_scalar('loss/loss', 
                                                   loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/loss_sup', 
                                                   loss_sup, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/loss_cl', 
                                    loss_cl, 
                                    self.current_iter)
                # self.tensorboard_writer.add_scalar(
                #     'loss/pseudo_supervision1',
                #     pseudo_supervision1, self.current_iter
                # )
                self.logging.info(
                    'iteration %d:'
                    ' loss: %f' 
                    ' supvised loss: %f'
                    ' CVCL loss: %f' % (
                        self.current_iter, loss.item(), 
                        loss_sup.item(), 
                        loss_cl.item()
                    )
                )
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    with torch.no_grad():
                        self.evaluation(model=self.model)
                    self.model.train()
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_model_path = os.path.join(
                        self.output_folder,
                        'model_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.model.state_dict(), save_model_path)
                    self.logging.info(f"save model to {save_model_path}")
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break     
                            
    def _train_C3PS(self):
        print("================> Training C3PS<===============")   
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self._adjust_learning_rate()  
                self.model.train()
                self.model2.train()
                volume_batch, label_batch = (
                    sampled_batch['image'],sampled_batch['label']
                )
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                #prepare input for condition net
                condition_batch = sampled_batch['condition']
                condition_batch = torch.cat([
                        condition_batch[:,0],
                        condition_batch[:,1]
                    ],dim=0).unsqueeze(1).to(self.device)
                ul1, br1, ul2, br2 = [], [], [], []
                labeled_idxs_batch = torch.arange(0, self.labeled_bs)
                unlabeled_idx_batch = torch.arange(self.labeled_bs, 
                                                   self.batch_size)
                if self.use_CAC:
                    ul1,br1 = sampled_batch['ul1'],sampled_batch['br1']
                    ul2,br2 = sampled_batch['ul2'],sampled_batch['br2']
                    volume_batch = torch.cat(
                        [volume_batch[:,0,...],volume_batch[:,1,...]],
                        dim=0
                    )
                    label_batch = torch.cat(
                        [label_batch[:,0,...],label_batch[:,1,...]],
                        dim=0
                    )
                    labeled_idxs2_batch = torch.arange(
                        self.batch_size,
                        self.batch_size+self.labeled_bs
                    )
                    labeled_idxs1_batch = torch.arange(0,self.labeled_bs)
                    labeled_idxs_batch = torch.cat(
                        [labeled_idxs1_batch,labeled_idxs2_batch]
                    )
                    unlabeled_idxs1_batch = torch.arange(self.labeled_bs,
                                                         self.batch_size)
                    unlabeled_idxs2_batch = torch.arange(
                        self.batch_size+self.labeled_bs, 
                        2 * self.batch_size
                    )
                    unlabeled_idxs_batch = torch.cat(
                        [unlabeled_idxs1_batch,unlabeled_idxs2_batch]
                    )
                noise1 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                ).to(self.device)
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                with autocast():
                    outputs1 = self.model(volume_batch + noise1)
                    outputs_soft1 = torch.softmax(outputs1, dim=1)
                    
                    #get condition list:
                    if self.use_CAC and (
                        self.current_iter >= min(
                            self.began_semi_iter,self.began_condition_iter
                        )
                    ):
                        overlap_soft1_list = []
                        overlap_outputs1_list = []
                        overlap_filter1_list = []
                        for unlabeled_idx1, unlabeled_idx2 in zip(
                            unlabeled_idxs1_batch,
                            unlabeled_idxs2_batch
                        ):
                            overlap1_soft1 = outputs_soft1[
                                unlabeled_idx1,
                                :,
                                ul1[0][1]:br1[0][1],
                                ul1[1][1]:br1[1][1],
                                ul1[2][1]:br1[2][1]
                            ]
                            overlap2_soft1 = outputs_soft1[
                                unlabeled_idx2,
                                :,
                                ul2[0][1]:br2[0][1],
                                ul2[1][1]:br2[1][1],
                                ul2[2][1]:br2[2][1]
                            ]
                            assert overlap1_soft1.shape == overlap2_soft1.shape,(
                                "overlap region size must equal"
                            )

                            # overlap region pred by model1
                            overlap1_outputs1 = outputs1[
                                unlabeled_idx1,
                                :,
                                ul1[0][1]:br1[0][1],
                                ul1[1][1]:br1[1][1],
                                ul1[2][1]:br1[2][1]
                            ] # overlap batch1
                            overlap2_outputs1 = outputs1[
                                unlabeled_idx2,
                                :,
                                ul2[0][1]:br2[0][1],
                                ul2[1][1]:br2[1][1],
                                ul2[2][1]:br2[2][1]
                            ] # overlap batch2
                            assert overlap1_outputs1.shape == overlap2_outputs1.shape,(
                                "overlap region size must equal"
                            )
                            overlap_outputs1_list.append(overlap1_outputs1.unsqueeze(0))
                            overlap_outputs1_list.append(overlap2_outputs1.unsqueeze(0))
                            
                            overlap_soft1_tmp = (overlap1_soft1 + overlap2_soft1) / 2.
                            max1,pseudo_mask1 = torch.max(overlap_soft1_tmp, dim=0)
                            pred_con_list = pseudo_mask1.unique().tolist()
                            con = self._get_condition(pred_con_list)
                            if self.num_classes==2:  #如果类别数为2则可以利用到背景像素
                                overlap_filter1_tmp = (
                                    (((max1>self.model1_thresh)&(pseudo_mask1!=con))|
                                    ((max1>0.8)&(pseudo_mask1==con))
                                    )
                                ).type(torch.int16)
                            else:
                                if con< self.num_classes:
                                    overlap_filter1_tmp = (
                                        (((max1>0.99)&(pseudo_mask1==0))|
                                        ((max1>0.9)&(pseudo_mask1!=con)&(pseudo_mask1!=0))|
                                        ((max1>0.9)&(pseudo_mask1==con))
                                        )
                                    ).type(torch.int16)
                                else:
                                    overlap_filter1_tmp = (
                                        (((max1>0.99)&(pseudo_mask1==0))|
                                        ((max1>0.9)&(pseudo_mask1!=0)))
                                        ).type(torch.int16)
                                # overlap_filter1_tmp = (
                                #     (max1>0.9)&(pseudo_mask1==con)
                                # ).type(torch.int16)
                            
                            overlap_soft1_list.append(overlap_soft1_tmp.unsqueeze(0))
                            overlap_filter1_list.append(overlap_filter1_tmp.unsqueeze(0))
                        overlap_soft1 = torch.cat(overlap_soft1_list, 0)
                        overlap_outputs1 = torch.cat(overlap_outputs1_list, 0)
                        overlap_filter1 = torch.cat(overlap_filter1_list, 0)

                    
                    
                        #get condition list pred by model1
                        condition_batch[unlabeled_idxs_batch] = con
                    
                    # random noise add to input of model2
                    noise2 = torch.clamp(
                        torch.randn_like(volume_batch) * 0.1, 
                        -0.2, 
                        0.2
                    ).to(self.device)

                    outputs2 = self.model2(volume_batch+noise2, condition_batch)
                    outputs_soft2 = torch.softmax(outputs2, dim=1)
                    # label_batch_con = (
                    #     label_batch==condition_batch.unsqueeze(-1).unsqueeze(-1)
                    # ).long()
                    label_batch_con = self._get_label_batch_for_conditional_net(
                        label_batch, condition_batch
                    )

                    self.consistency_weight = self._get_current_consistency_weight(
                        self.current_iter//150
                    )
                    loss1 = 0.5 * (
                        self.ce_loss(
                            outputs1[labeled_idxs_batch],
                            label_batch[labeled_idxs_batch].long()
                        ) +
                        self.dice_loss(
                            outputs_soft1[labeled_idxs_batch],
                            label_batch[labeled_idxs_batch].unsqueeze(1)
                        )
                    )
                    loss2 = 0.5 * (
                        self.ce_loss(
                            outputs2[labeled_idxs_batch],
                            label_batch_con[labeled_idxs_batch].long()
                        ) + 
                        self.dice_loss_con(
                            outputs_soft2[labeled_idxs_batch],
                            label_batch_con[labeled_idxs_batch].unsqueeze(1)
                        )
                    )

                    if self.use_CAC and (
                        self.current_iter >= min(
                            self.began_semi_iter,self.began_condition_iter
                        )
                    ):
                        overlap_soft2_list = []
                        overlap_outputs2_list = []
                        overlap_filter2_list = []
                        for unlabeled_idx1, unlabeled_idx2 in zip(
                            unlabeled_idxs1_batch,
                            unlabeled_idxs2_batch
                        ):
                            # overlap region pred by model2
                            overlap1_soft2 = outputs_soft2[
                                unlabeled_idx1,
                                :,
                                ul1[0][1]:br1[0][1],
                                ul1[1][1]:br1[1][1],
                                ul1[2][1]:br1[2][1]
                            ]
                            overlap2_soft2 = outputs_soft2[
                                unlabeled_idx2,
                                :,
                                ul2[0][1]:br2[0][1],
                                ul2[1][1]:br2[1][1],
                                ul2[2][1]:br2[2][1]
                            ]
                            assert overlap1_soft2.shape == overlap2_soft2.shape,(
                                "overlap region size must equal"
                            )
                            
                            # overlap region outputs pred by model2
                            overlap1_outputs2 = outputs2[
                                unlabeled_idx1,
                                :,
                                ul1[0][1]:br1[0][1],
                                ul1[1][1]:br1[1][1],
                                ul1[2][1]:br1[2][1]
                            ]
                            overlap2_outputs2 = outputs2[
                                unlabeled_idx2,
                                :,
                                ul2[0][1]:br2[0][1],
                                ul2[1][1]:br2[1][1],
                                ul2[2][1]:br2[2][1]
                            ]
                            assert overlap1_outputs2.shape == overlap2_outputs2.shape,(
                                "overlap region size must equal"
                            )
                            overlap_outputs2_list.append(overlap1_outputs2.unsqueeze(0))
                            overlap_outputs2_list.append(overlap2_outputs2.unsqueeze(0))

                            
                            overlap_soft2_tmp = (overlap1_soft2 + overlap2_soft2) / 2.
                            max2,pseudo_mask2 = torch.max(overlap_soft2_tmp, dim=0)
                            if self.num_classes==2:
                                overlap_filter2_tmp = ((
                                    max2>self.model2_thresh
                                )).type(torch.int16)
                            else:
                                if con < self.num_classes:
                                    overlap_filter2_tmp = ((
                                        max2>self.model2_thresh
                                    ) & (
                                        pseudo_mask2>0
                                    )).type(torch.int16)
                                else:
                                    overlap_filter2_tmp = ((
                                        max2>self.model2_thresh
                                    ) & (
                                        pseudo_mask2==0
                                    )).type(torch.int16)
                            overlap_soft2_list.append(overlap_soft2_tmp.unsqueeze(0))
                            overlap_filter2_list.append(overlap_filter2_tmp.unsqueeze(0))
                        overlap_soft2 = torch.cat(overlap_soft2_list, 0)
                        overlap_outputs2 = torch.cat(overlap_outputs2_list, 0)
                        overlap_filter2 = torch.cat(overlap_filter2_list, 0)
                    if self.current_iter < self.began_condition_iter:
                        pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                    else:
                        if self.use_CAC:
                            overlap_pseudo_outputs2 = torch.argmax(
                                overlap_soft2.detach(),
                                dim=1,
                                keepdim=False
                            )
                            if overlap_pseudo_outputs2.sum() == 0 or overlap_filter2.sum()==0:
                                pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                            else:
                                overlap_pseudo_outputs2 = torch.cat(
                                    [overlap_pseudo_outputs2, overlap_pseudo_outputs2]
                                )
                                overlap_pseudo_filter2 = torch.cat(
                                    [overlap_filter2, overlap_filter2]
                                )
                                ce_pseudo_supervision1 = self._cross_entropy_loss_con(
                                    overlap_outputs1,
                                    overlap_pseudo_outputs2,
                                    condition_batch[unlabeled_idx_batch],
                                    overlap_pseudo_filter2
                                )
                                B,C,D,W,H = overlap_outputs1.shape
                                # overlap_soft1_con = torch.softmax(overlap_outputs1,dim=1)
                                # overlap_outputs_con = torch.zeros(B,2,D,W,H)
                                # overlap_outputs_con[:,1,:,:,:] = overlap_soft1_con[
                                #     :,condition_batch[unlabeled_idx_batch].item(),:,:,:
                                # ]
                                # overlap_outputs_con[:,0,:,:,:] = 1 - overlap_outputs_con[:,1,:,:,:]
                                # overlap_outputs_con[:,1,:,:,:][overlap_pseudo_filter2==0]=0 # filer output
                                # dice_pseudo_supervision1 = self.dice_loss_con(
                                #     overlap_outputs_con.to(self.device),
                                #     (overlap_pseudo_outputs2*overlap_pseudo_filter2).unsqueeze(1),
                                #     skip_id=0
                                # )
                                pseudo_supervision1 = ce_pseudo_supervision1 
                        else:
                            pseudo_outputs2 = torch.argmax(
                                outputs_soft2[self.labeled_bs:].detach(),
                                dim=1,
                                keepdim=False
                            )
                            pseudo_supervision1 = self._cross_entropy_loss_con(
                                outputs1[self.labeled_bs:],
                                pseudo_outputs2,
                                condition_batch[self.labeled_bs:]
                            )
                    if self.current_iter < self.began_semi_iter or overlap_filter1.sum()==0:
                        pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                    else:
                        if self.use_CAC:
                            overlap_pseudo_outputs1 = torch.argmax(
                                overlap_soft1.detach(), 
                                dim=1, 
                                keepdim=False
                            )
                            overlap_pseudo_outputs1 = torch.cat(
                                [overlap_pseudo_outputs1, overlap_pseudo_outputs1]
                            )
                            overlap_pseudo_filter1 = torch.cat(
                                [overlap_filter1, overlap_filter1]
                            )
                            target_ce_con = self._get_label_batch_for_conditional_net(
                                overlap_pseudo_outputs1,condition_batch[unlabeled_idxs_batch]
                            )
                            target_ce_con[overlap_pseudo_filter1==0] = 255
                            ce_pseudo_supervision2 = self.ce_loss(
                                overlap_outputs2, 
                                target_ce_con
                            )
                            dice_pseudo_supervision2 = self.dice_loss_con(
                                torch.softmax(overlap_outputs2,dim=1)*overlap_pseudo_filter1,
                                ((
                                    overlap_pseudo_outputs1==condition_batch[unlabeled_idxs_batch].unsqueeze(-1).unsqueeze(-1)
                                ).long()*overlap_pseudo_filter1).unsqueeze(1),
                                skip_id=0
                            )
                            pseudo_supervision2 = ce_pseudo_supervision2 + dice_pseudo_supervision2
                        else:
                            pseudo_outputs1 = torch.argmax(
                                outputs_soft1[self.labeled_bs:].detach(), 
                                dim=1, 
                                keepdim=False
                            )
                            pseudo_supervision2 = self.ce_loss(
                                outputs2[self.labeled_bs:], 
                                (pseudo_outputs1==condition_batch[self.labeled_bs:].\
                                    unsqueeze(-1).unsqueeze(-1)).long()
                            )
                    

                    model1_loss = loss1 + self.consistency_weight * pseudo_supervision1
                    model2_loss = loss2 + self.consistency_weight * pseudo_supervision2

                    loss = model1_loss + model2_loss
                self.grad_scaler1.scale(model1_loss).backward()
                self.grad_scaler1.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.grad_scaler1.step(self.optimizer)
                self.grad_scaler1.update()
                self.grad_scaler2.scale(model2_loss).backward()
                self.grad_scaler2.unscale_(self.optimizer2)
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 12)
                self.grad_scaler2.step(self.optimizer2)
                self.grad_scaler2.update()

                self.current_iter += 1
                for param_group in self.optimizer.param_groups:
                    self.current_lr = param_group['lr']
                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'consistency_weight/consistency_weight', 
                    self.consistency_weight, 
                    self.current_iter
                )
                self.tensorboard_writer.add_scalar('loss/model1_loss', 
                                                   model1_loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', 
                                                   model2_loss, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision1',
                    pseudo_supervision1, self.current_iter
                )
                self.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision2',
                    pseudo_supervision2, 
                    self.current_iter
                )
                self.logging.info(
                    'iteration %d :'
                    'model1 loss : %f' 
                    'model2 loss : %f' 
                    'pseudo_supervision1 : %f'
                    'pseudo_supervision2 : %f' % (
                        self.current_iter, model1_loss.item(), 
                        model2_loss.item(), 
                        pseudo_supervision1.item(), 
                        pseudo_supervision2.item()
                    )
                )
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                      self.current_iter)

                    image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model1_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model2_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image, 
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    with torch.no_grad():
                        self.evaluation(model=self.model)
                        self.evaluation(model=self.model2, do_condition=True)
                    self.model.train()
                    self.model2.train()
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_model_path = os.path.join(
                        self.output_folder,
                        'model_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.model.state_dict(), save_model_path)
                    self.logging.info(f"save model to {save_model_path}")

                    save_model_path = os.path.join(
                        self.output_folder,
                        'model2_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.model2.state_dict(), save_model_path)
                    self.logging.info(f'save model2 to {save_model_path}')
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break

            # train network2 for partial dataset
            if self.use_PL:
                full_volume_batch = volume_batch[labeled_idxs_batch]
                full_label_batch = label_batch[labeled_idxs_batch]
                full_condition_batch = condition_batch[labeled_idxs_batch]
                con_list1 = full_label_batch[0].unique().tolist()
                con_list2 = full_label_batch[1].unique().tolist()
                if 0 in con_list1:
                    con_list1.remove(0)
                inter_label_list = list(
                    set(con_list1) & set(self.method_config['con_list'])
                )
                if len(inter_label_list) == 0:
                    inter_label_list = self.method_config['con_list']
                con1 = np.random.choice(inter_label_list)

                if 0 in con_list2:
                    con_list2.remove(0)
                inter_label_list = list(
                    set(con_list2) & set(self.method_config['con_list'])
                )
                if len(inter_label_list) == 0:
                    inter_label_list = self.method_config['con_list']
                con2 = np.random.choice(inter_label_list)
                full_condition_batch[0] = con1 
                full_condition_batch[1] = con2
                for i_batch, sampled_batch in enumerate(self.dataloader_pl):
                    self.model2.train()
                    volume_batch, label_batch = (
                        sampled_batch['image'],sampled_batch['label']
                    )
                    label_batch = torch.argmax(label_batch, dim=1)
                    volume_batch, label_batch = (volume_batch.to(self.device), 
                                    label_batch.to(self.device))
                    condition_batch = torch.cat([
                            torch.Tensor([5]),
                            torch.Tensor([5]),
                            torch.Tensor([5]),
                            torch.Tensor([5]),
                        ],dim=0).unsqueeze(1).to(self.device)
              
                    noise = torch.clamp(
                        torch.randn_like(volume_batch) * 0.1, 
                        -0.2, 
                        0.2
                    ).to(self.device)

                    outputs2 = self.model2(volume_batch+noise, condition_batch)
                    outputs_soft2 = torch.softmax(outputs2, dim=1)
                    
                    # get condition label of 0 vs 1
                    # condition_batch[2:] = 1 # since partial label is binary mask
                    # label_batch_con = (
                    #     label_batch==condition_batch.unsqueeze(-1).unsqueeze(-1)
                    # ).long()     
                    loss2 = 0.1 * (
                        self.ce_loss(
                            outputs2,
                            label_batch
                        ) + 
                        self.dice_loss_con(
                            outputs_soft2,
                            label_batch.unsqueeze(1)
                        )
                    )
                    model2_loss = loss2 
                    self.optimizer2.zero_grad() 
                    model2_loss.backward()
                    self.optimizer2.step()

                    self.tensorboard_writer.add_scalar('loss/model2_loss_pl', 
                                                    model2_loss, 
                                                    self.current_iter)

                    self.logging.info(
                        'iteration %d :'
                        'model2 loss pl : %f' % (
                            self.current_iter,
                            model2_loss.item()
                        )
                    )
                    break
        self.tensorboard_writer.close()
    
    def _train_GFEL(self):
        "General Feature Enhanced Learning"
        print("================> Training GFEL<===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                volume_batch, label_batch = (
                    volume_batch.to(self.device), label_batch.to(self.device)
                )
                noise1 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                )
                label_batch = torch.argmax(label_batch, dim=1)
                outputs1 = self.model(volume_batch + noise1)
                outputs_soft1 = torch.softmax(outputs1, dim=1)
                noise2 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                )
                outputs2 = self.model2(volume_batch + noise2)
                outputs_soft2 = torch.softmax(outputs2, dim=1)

                self.consistency_weight = self._get_current_consistency_weight(
                    self.current_iter//150
                )
                loss1 = 0.5 * (self.ce_loss(outputs1[:self.labeled_bs],
                                   label_batch[:][:self.labeled_bs].long()) + 
                               self.dice_loss(outputs_soft1[:self.labeled_bs], 
                                             label_batch[:self.labeled_bs].\
                                                unsqueeze(1)))
                loss2 = 0.5 * (self.ce_loss(outputs2[:self.labeled_bs],
                                   label_batch[:][:self.labeled_bs].long()) + 
                               self.dice_loss(outputs_soft2[:self.labeled_bs], 
                                             label_batch[:self.labeled_bs].\
                                                unsqueeze(1)))
                pseudo_outputs1 = torch.argmax(
                    outputs_soft1[self.labeled_bs:].detach(),
                    dim=1, keepdim=False
                )
                pseudo_outputs2 = torch.argmax(
                    outputs_soft2[self.labeled_bs:].detach(),
                    dim=1, keepdim=False
                )
                if self.current_iter < self.began_semi_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                    pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                else:
                    pseudo_supervision1 = self.ce_loss(
                        outputs1[self.labeled_bs:],
                        pseudo_outputs2
                    )
                    pseudo_supervision2 = self.ce_loss(
                        outputs2[self.labeled_bs:],
                        pseudo_outputs1
                    )
                model1_loss = loss1 + self.consistency_weight *  \
                                      pseudo_supervision1
                model2_loss = loss2 + self.consistency_weight * \
                                      pseudo_supervision2
                loss = model1_loss + model2_loss 
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()

                loss.backward()
                self.optimizer.step()
                self.optimizer2.step()
                
                self._adjust_learning_rate()
                self.current_iter += 1

                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar(
                'consistency_weight/consistency_weight',self.consistency_weight, 
                self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model1_loss', model1_loss, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', model2_loss, 
                                               self.current_iter)
                self.logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (
                    self.current_iter,model1_loss.item(), 
                    model2_loss.item()))
            
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                  self.current_iter)

                    image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                    'train/Model1_Predicted_label',
                     grid_image, self.current_iter)

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Model2_Predicted_label',
                                 grid_image, self.current_iter)

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                 grid_image, self.current_iter)

                if (
                    self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter==20:
                    self.evaluation(model=self.model)
                    self.evaluation(model=self.model2,model_name='model2')
                    self.model.train()
                    self.model2.train()
            
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_mode_path = os.path.join(
                    self.output_folder, 'model1_iter_' + \
                        str(self.current_iter) + '.pth')
                    torch.save(self.model.state_dict(), save_mode_path)
                    self.logging.info("save model1 to {}".format(save_mode_path))

                    save_mode_path = os.path.join(
                        self.output_folder, 
                        'model2_iter_' + str(self.current_iter) + '.pth')
                    torch.save(self.model2.state_dict(), save_mode_path)
                    self.logging.info("save model2 to {}".format(save_mode_path))
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()

    def _train_CAML(self):
        "Correlation-Aware Mutual Learning"
        print("================> Training CAML<===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        consistency_criterion = losses.mse_loss
        memory_num = 256
        num_filtered = 12800
        lambda_s = 0.5
        memory_bank = feature_memory.MemoryBank(num_labeled_samples=self.labeled_num, num_cls=self.num_classes)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch, idx = sampled_batch['image'], sampled_batch['label'], sampled_batch['idx']
                volume_batch, label_batch, idx = volume_batch.cuda(), label_batch.cuda(), idx.cuda()
                label_batch = torch.argmax(label_batch, dim=1)
                self.model.train()
                outputs_v, outputs_a, embedding_v, embedding_a = self.model(volume_batch)
                outputs_list = [outputs_v, outputs_a]
                num_outputs = len(outputs_list)

                y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape)
                y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape)

                loss_s = 0
                for i in range(num_outputs):
                    y = outputs_list[i][:self.labeled_bs, ...]
                    y_prob = F.softmax(y, dim=1)
                    loss_s += self.dice_loss(y_prob[:, ...], label_batch[:self.labeled_bs].unsqueeze(1))

                    y_all = outputs_list[i]
                    y_prob_all = F.softmax(y_all, dim=1)
                    y_ori[i] = y_prob_all
                    y_pseudo_label[i] = self._sharpening(y_prob_all)

                loss_c = 0
                for i in range(num_outputs):
                    for j in range(num_outputs):
                        if i != j:
                            loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

                outputs_v_soft = F.softmax(outputs_v, dim=1)  # [batch, num_class, h, w, d]
                outputs_a_soft = F.softmax(outputs_a, dim=1)  # soft prediction of fa
                labeled_features_v = embedding_v[:self.labeled_bs, ...]
                labeled_features_a = embedding_a[:self.labeled_bs, ...]

                # unlabeled embeddings to calculate correlation matrix with embeddings sampled from the memory bank
                unlabeled_features_v = embedding_v[self.labeled_bs:, ...]
                unlabeled_features_a = embedding_a[self.labeled_bs:, ...]

                y_v = outputs_v_soft[:self.labeled_bs]
                y_a = outputs_a_soft[:self.labeled_bs]
                true_labels = label_batch[:self.labeled_bs]

                _, prediction_label_v = torch.max(y_v, dim=1)
                _, prediction_label_a = torch.max(y_a, dim=1)
                predicted_unlabel_prob_v, predicted_unlabel_v = torch.max(outputs_v_soft[self.labeled_bs:],
                                                                        dim=1)  # v_unlabeled_mask
                predicted_unlabel_prob_a, predicted_unlabel_a = torch.max(outputs_a_soft[self.labeled_bs:],
                                                                        dim=1)  # a_unlabeled_mask

                # Select the correct predictions including the foreground class and the background class
                mask_prediction_correctly = (
                        ((prediction_label_a == true_labels).float() + (prediction_label_v == true_labels).float()) == 2)

                labeled_features_v = labeled_features_v.permute(0, 2, 3, 4, 1).contiguous()
                b, h, w, d, labeled_features_dim = labeled_features_v.shape

                # get projected features
                self.model.eval()
                proj_labeled_features_v = self.model.projection_head1(labeled_features_v.view(-1, labeled_features_dim))
                proj_labeled_features_v = proj_labeled_features_v.view(b, h, w, d, -1)

                proj_labeled_features_a = self.model.projection_head2(labeled_features_a.view(-1, labeled_features_dim))
                proj_labeled_features_a = proj_labeled_features_a.view(b, h, w, d, -1)
                self.model.train()

                labels_correct_list = []
                labeled_features_correct_list = []
                labeled_index_list = []
                for i in range(self.labeled_bs):
                    labels_correct_list.append(true_labels[i][mask_prediction_correctly[i]])
                    labeled_features_correct_list.append((proj_labeled_features_v[i][mask_prediction_correctly[i]] +
                                                        proj_labeled_features_a[i][mask_prediction_correctly[i]]) / 2)
                    labeled_index_list.append(idx[i])

                # updated memory bank
                labeled_index = idx[:self.labeled_bs]
                memory_bank.update_labeled_features(labeled_features_correct_list, labels_correct_list,
                                                    labeled_index_list)

                # sample memory bank size labeled features from memory bank
                memory = memory_bank.sample_labeled_features(memory_num)

                # get the mask with the same prediction between fv and fa on unlabeled data
                mask_consist_unlabeled = predicted_unlabel_v == predicted_unlabel_a  # [b, h, w, d]
                # use model V's predicted label and prob to filter unlabeled feature online
                consist_unlabel = predicted_unlabel_v[mask_consist_unlabeled]  # [num_consist]
                consist_unlabel_prob = predicted_unlabel_prob_v[mask_consist_unlabeled]  # [num_consist]

                unlabeled_features_v = unlabeled_features_v.permute(0, 2, 3, 4, 1)
                unlabeled_features_a = unlabeled_features_a.permute(0, 2, 3, 4, 1)
                unlabeled_features_v = unlabeled_features_v[mask_consist_unlabeled, :]  # [num_consist, feat_dim]
                unlabeled_features_a = unlabeled_features_a[mask_consist_unlabeled, :]

                # get fv's correlation matrix
                projected_feature_v = self.model.projection_head1(unlabeled_features_v)
                predicted_feature_v = self.model.prediction_head1(projected_feature_v)
                corr_v, corr_v_available = correlation.cal_correlation_matrix(predicted_feature_v,
                                                                            consist_unlabel_prob,
                                                                            consist_unlabel,
                                                                            memory,
                                                                            self.num_classes,
                                                                            num_filtered=num_filtered)

                # get fa's correlation matrix
                projected_feature_a = self.model.projection_head2(unlabeled_features_a)
                predicted_feature_a = self.model.prediction_head2(projected_feature_a)
                corr_a, corr_a_available = correlation.cal_correlation_matrix(predicted_feature_a,
                                                                            consist_unlabel_prob,
                                                                            consist_unlabel,
                                                                            memory,
                                                                            self.num_classes,
                                                                            num_filtered=num_filtered)

                # calculate omni-correlation consistency loss
                if corr_v_available and corr_a_available:
                    num_samples = corr_a.shape[0]
                    loss_o = torch.sum(torch.sum(-corr_a * torch.log(corr_v + 1e-8), dim=1)) / num_samples
                else:
                    loss_o = 0

                lambda_c = self._get_lambda_c(self.current_iter // 150)
                lambda_o = self._get_lambda_o(self.current_iter // 150)

                loss = lambda_s * loss_s  #+ lambda_c * loss_c #+ lambda_o * loss_o

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self._adjust_learning_rate()
                self.current_iter += 1
                self.logging.info('iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_o: %03f' % (
                    self.current_iter, loss, loss_s, loss_c, loss_o))

                self.tensorboard_writer.add_scalar('Labeled_loss/loss_s', loss_s, self.current_iter)
                self.tensorboard_writer.add_scalar('Co_loss/loss_c', loss_c, self.current_iter)
                self.tensorboard_writer.add_scalar('Co_loss/loss_o', loss_o, self.current_iter)
                if (
                    self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter==20:
                    self.evaluation(model=self.model)
                    #self.evaluation(model=self.model2,do_SR=True,model_name='model2')
                    self.model.train()
            
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint("latest")
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
    def _train_CSSR(self):
        "cross supervision use high resolution"
        print("================> Training CSSR<===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        self.model = self.model.float().to(self.device)
        self.model2 = self.model2.float().to(self.device)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.model.train()
                self.model2.train()
                volume_large, label_large = (
                    sampled_batch['image_large'].float().to(self.device),
                    sampled_batch['label_large'].float().to(self.device)
                )
                volume_small, label_small = (
                    sampled_batch['image_small'].float().to(self.device),
                    sampled_batch['label_small'].float().to(self.device)
                )

                ul_large,br_large = sampled_batch['ul1'],sampled_batch['br1']
                ul_small,br_small = sampled_batch['ul2'],sampled_batch['br2']
                ul_large_u = [x[self.labeled_bs:] for x in ul_large ]
                br_large_u = [x[self.labeled_bs:] for x in br_large ]
                ul_small_u = [x[self.labeled_bs:] for x in ul_small ]
                br_small_u = [x[self.labeled_bs:] for x in br_small ]
                noise1 = torch.clamp(
                    torch.randn_like(volume_small) * 0.1, 
                    -0.2, 
                    0.2
                ).to(self.device)
                noise2 = torch.clamp(
                    torch.randn_like(volume_large) * 0.1, 
                        -0.2, 
                        0.2
                    ).to(self.device)
                self.optimizer.zero_grad()
                self.optimizer2.zero_grad()
                with autocast():
                    outputs1 = self.model(volume_small+noise1)
                    outputs_soft1 = torch.softmax(outputs1, dim=1)
                    outputs2 = self.model2(volume_large + noise2)
                    outputs_soft2 = torch.softmax(outputs2, dim=1)

                    self.consistency_weight = self._get_current_consistency_weight(
                        self.current_iter//150
                    )
                    loss1 = 0.5 * (self.ce_loss(outputs1[:self.labeled_bs],
                                    label_small[:self.labeled_bs].long()) + 
                                self.dice_loss(outputs_soft1[:self.labeled_bs], 
                                                label_small[:self.labeled_bs].\
                                                    unsqueeze(1)))
                    loss2 = 0.5 * (self.ce_loss(outputs2[:self.labeled_bs],
                                    label_large[:self.labeled_bs].long()) + 
                                self.dice_loss(outputs_soft2[:self.labeled_bs], 
                                                label_large[:self.labeled_bs].\
                                                    unsqueeze(1)))
                    max_prob1,pseudo_outputs1 = torch.max(
                        outputs_soft1[self.labeled_bs:].detach(), dim=1
                    )
                    #assert (pseudo_outputs1_old!=pseudo_outputs1).sum()==0,'error of pseudo mask1'
                    # filter the pseudo mask by max prob
                    filter1 = (
                        ((max_prob1>0.99)&(pseudo_outputs1==0))|
                        ((max_prob1>0.95)&(pseudo_outputs1!=0))
                    )
                    

                    max_prob2,pseudo_outputs2 = torch.max(
                        outputs_soft2[self.labeled_bs:].detach(), dim=1
                    )
                    #assert (pseudo_outputs2_old!=pseudo_outputs2).sum()==0,'error of pseudo mask2'
                    # filter the pseudo mask by max prob
                    filter2 = (
                        ((max_prob2>0.99)&(pseudo_outputs2==0))|
                        ((max_prob2>0.95)&(pseudo_outputs2!=0))
                    )
                    
                    if self.current_iter < self.began_semi_iter:
                        pseudo_supervision1 = torch.FloatTensor([0]).to(self.device)
                        pseudo_supervision2 = torch.FloatTensor([0]).to(self.device)
                    else:
                        pseudo_outputs2[filter2==0] = 255
                        pseudo_supervision1 = self.ce_loss(
                            outputs1[self.labeled_bs:,:,ul_small_u[0]:br_small_u[0],ul_small_u[1]:br_small_u[1],ul_small_u[2]:br_small_u[2]],
                            pseudo_outputs2[:,ul_large_u[0]:br_large_u[0],ul_large_u[1]:br_large_u[1],ul_large_u[2]:br_large_u[2]]
                        ) 
                        # + self.dice_loss(
                        #     outputs1[self.labeled_bs:,:,ul_small_u[0]:br_small_u[0],ul_small_u[1]:br_small_u[1],ul_small_u[2]:br_small_u[2]],
                        #     pseudo_outputs2[:,ul_large_u[0]:br_large_u[0],ul_large_u[1]:br_large_u[1],ul_large_u[2]:br_large_u[2]].unsqueeze(1)
                        # )
                        pseudo_outputs1[filter1==0] = 255
                        pseudo_supervision2 = self.ce_loss(
                            outputs2[self.labeled_bs:,:,ul_large_u[0]:br_large_u[0],ul_large_u[1]:br_large_u[1],ul_large_u[2]:br_large_u[2]],
                            pseudo_outputs1[:,ul_small_u[0]:br_small_u[0],ul_small_u[1]:br_small_u[1],ul_small_u[2]:br_small_u[2]]
                        ) 
                        # + self.dice_loss(
                        #     outputs2[self.labeled_bs:,:,ul_large_u[0]:br_large_u[0],ul_large_u[1]:br_large_u[1],ul_large_u[2]:br_large_u[2]],
                        #     pseudo_outputs1[:,ul_small_u[0]:br_small_u[0],ul_small_u[1]:br_small_u[1],ul_small_u[2]:br_small_u[2]].unsqueeze(1)
                        # )
                    model1_loss = loss1 + self.consistency_weight *  \
                                        pseudo_supervision1
                    model2_loss = loss2 + self.consistency_weight * \
                                        pseudo_supervision2
                    loss = model1_loss + model2_loss 

                self.grad_scaler1.scale(model1_loss).backward()
                self.grad_scaler1.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
                self.grad_scaler1.step(self.optimizer)
                self.grad_scaler1.update()
                self.grad_scaler2.scale(model2_loss).backward()
                self.grad_scaler2.unscale_(self.optimizer2)
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), 12)
                self.grad_scaler2.step(self.optimizer2)
                self.grad_scaler2.update()
                #loss.backward()
                # self.optimizer.step()
                # self.optimizer2.step()
                
                self._adjust_learning_rate()
                self.current_iter += 1

                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar(
                'consistency_weight/consistency_weight',self.consistency_weight, 
                self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model1_loss', model1_loss, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', model2_loss, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar('loss/pseudo1_loss', pseudo_supervision1, 
                                               self.current_iter)
                self.tensorboard_writer.add_scalar('loss/pseudo2_loss', pseudo_supervision2, 
                                               self.current_iter)
                
                self.logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (
                    self.current_iter,model1_loss.item(), 
                    model2_loss.item()))

                if (
                    self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter==20:
                    self.evaluation(model=self.model)
                    #self.evaluation(model=self.model2,do_SR=True,model_name='model2')
                    self.model.train()
                    self.model2.train()
            
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint("latest")
                if self.current_iter >= self.max_iterations:
                    break 
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
    
    def _train_EMSSL(self):
        """
        code for "Bayesian Pseudo Labels: Expectation Maximization for Robust
        and Efficient Semi-supervised Segmentation"
        """
        print("================> Training EMSSL<===============")
    
    def _train_ConditionNet(self):
        print("================> Training ConditionNet<===============")   
        iterator = tqdm(range(self.max_epoch), ncols=70)
        iter_each_epoch = len(self.dataloader)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                self.model2.train()
                volume_batch, label_batch = (
                    sampled_batch['image'],sampled_batch['label']
                )
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                
                                    #prepare input for condition net
                condition_batch = sampled_batch['condition']
                condition_batch = torch.cat([
                        condition_batch[0],
                        condition_batch[1]
                    ],dim=0).unsqueeze(1).to(self.device)
                labeled_idxs_batch = torch.arange(0, self.labeled_bs)
                if self.use_CAC:
                    volume_batch = torch.cat(
                        [volume_batch[:,0,...],volume_batch[:,1,...]],
                        dim=0
                    )
                    label_batch = torch.cat(
                        [label_batch[:,0,...],label_batch[:,1,...]],
                        dim=0
                    )
                    labeled_idxs2_batch = torch.arange(
                        self.batch_size,
                        self.batch_size+self.labeled_bs
                    )
                    labeled_idxs1_batch = torch.arange(0,self.labeled_bs)
                    labeled_idxs_batch = torch.cat(
                        [labeled_idxs1_batch,labeled_idxs2_batch]
                    )
                noise = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, 
                    -0.2, 
                    0.2
                ).to(self.device)

                outputs2 = self.model2(volume_batch+noise, condition_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                label_batch_con = self._get_label_batch_for_conditional_net(
                    label_batch,condition_batch
                )
                loss2 =  (
                    self.ce_loss(
                        outputs2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].long()
                    ) + 
                    self.dice_loss_con(
                        outputs_soft2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].unsqueeze(1)
                    )
                )


                model2_loss = loss2 

                self.optimizer2.zero_grad() 
                model2_loss.backward()
                self.optimizer2.step()

                self.current_iter += 1
                self._adjust_learning_rate()  
                for param_group in self.optimizer2.param_groups:
                    self.current_lr = param_group['lr']
                self.tensorboard_writer.add_scalar('lr', self.current_lr, 
                                                   self.current_iter)
                self.tensorboard_writer.add_scalar('loss/model2_loss', 
                                                   model2_loss, 
                                                   self.current_iter)

                self.logging.info(
                    'iteration %d :'
                    'model2 loss : %f' % (
                        self.current_iter,
                        model2_loss.item()
                    )
                )
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image, 
                                                      self.current_iter)

                    image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image(
                        'train/Model2_Predicted_label',
                        grid_image, 
                        self.current_iter
                    )

                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image, 
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter == 20:
                    with torch.no_grad():
                        self.evaluation(model=self.model2, do_condition=True)
                    self.model2.train()
                if self.current_iter % self.save_checkpoint_freq == 0:
                    save_model_path = os.path.join(
                        self.output_folder,
                        'model2_iter_' + str(self.current_iter) + '.pth'
                    )
                    torch.save(self.model2.state_dict(), save_model_path)
                    self.logging.info(f'save model2 to {save_model_path}')
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
    
    def _train_EM(self):
        pass

    def _train_UAMT(self):
        print("================> Training UAMT<===============")
        iterator = tqdm(range(self.max_epoch), ncols=70)
        for epoch_num in iterator:
            for i_batch, sampled_batch in enumerate(self.dataloader):
                volume_batch, label_batch = (sampled_batch['image'], 
                                             sampled_batch['label'])
                volume_batch, label_batch = (volume_batch.to(self.device), 
                                             label_batch.to(self.device))
                unlabeled_volume_batch = volume_batch[self.labeled_bs:]
              
                label_batch = torch.argmax(label_batch, dim=1)
                noise = torch.clamp(torch.randn_like(
                    unlabeled_volume_batch)*0.1, -0.2, 0.2
                )
                ema_inputs = unlabeled_volume_batch + noise 
                ema_inputs = ema_inputs.to(self.device)

                outputs = self.model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                with torch.no_grad():
                    ema_output = self.ema_model(ema_inputs)
                T = 8
                _, _, d, w, h = unlabeled_volume_batch.shape
                volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
                stride = volume_batch_r.shape[0] // 2
                preds = torch.zeros([stride*T, 
                                     self.num_classes, d, w, h]).to(self.device)
                for i in range(T//2):
                    ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(
                        volume_batch_r)*0.1, -0.2, 0.2)
                    with torch.no_grad():
                        preds[2*stride*i:2*stride*(i+1)] = self.ema_model(
                            ema_inputs)
                preds = torch.softmax(preds, dim=1)
                preds = preds.reshape(T, stride, self.num_classes, d, w, h)
                preds = torch.mean(preds, dim=0)
                uncertainty = -1.0 * torch.sum(preds*torch.log(preds+1e-6),
                                               dim=1, keepdim=True)
                self.loss_ce = self.ce_loss(outputs[:self.labeled_bs],
                                       label_batch[:self.labeled_bs])
                self.loss_dice = self.dice_loss(outputs_soft[:self.labeled_bs],
                                           label_batch[:self.labeled_bs].unsqueeze(1))
                supervised_loss = 0.5 * (self.loss_dice + self.loss_ce)
                self.consistency_weight = (
                    self._get_current_consistency_weight(self.current_iter//150)
                )
                consistency_dist = losses.softmax_dice_loss(
                    outputs[self.labeled_bs:], ema_output)
                threshold = (0.75+0.25*ramps.sigmoid_rampup(self.current_iter,
                self.max_iterations))*np.log(2)
                #if self.current_iter > self.began_semi_iter:
                mask = (uncertainty < threshold).float()
                self.consistency_loss = torch.sum(mask*consistency_dist)/(
                    2*torch.sum(mask)+1e-16
                )
                # else:
                #     self.consistency_loss = torch.FloatTensor([0.0]).to(self.device)
                self.loss = supervised_loss + \
                    self.consistency_weight * self.consistency_loss
                self.optimizer.zero_grad()
                self.loss.backward()
                self.optimizer.step()
                self._update_ema_variables()
                self._adjust_learning_rate()
                self.current_iter += 1
                self._add_information_to_writer()
                if self.current_iter % self.show_img_freq == 0:
                    image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=True)
                    self.tensorboard_writer.add_image('train/Image', grid_image,
                                                      self.current_iter)
                    image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                        3, 0, 1, 2).repeat(2, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Predicted_label',
                                                      grid_image,
                                                      self.current_iter)
                    image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                        3, 0, 1, 2).repeat(1, 3, 1, 1)
                    grid_image = make_grid(image, 5, normalize=False)
                    self.tensorboard_writer.add_image('train/Groundtruth_label',
                                                      grid_image,
                                                      self.current_iter)
                if (self.current_iter > self.began_eval_iter and
                    self.current_iter % self.val_freq == 0
                ) or self.current_iter==20:
                    self.evaluation(model=self.model)
                if self.current_iter % self.save_checkpoint_freq == 0:
                    self._save_checkpoint()
                if self.current_iter >= self.max_iterations:
                    break
            if self.current_iter >= self.max_iterations:
                iterator.close()
                break
        self.tensorboard_writer.close()
        print("*"*10,"training done!","*"*10)

    
    def _get_current_consistency_weight(self, epoch):
        return self.consistency * ramps.sigmoid_rampup(epoch, 
                                                       self.consistency_rampup)
    def _get_lambda_c(self,epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return 1.0 * ramps.sigmoid_rampup(epoch, self.consistency_rampup)


    def _get_lambda_o(self,epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return 0.05 * ramps.sigmoid_rampup(epoch, self.consistency_rampup)
 
    def _update_ema_variables(self):
        # use the true average until the exponential average is more correct
        alpha = min(1-1/(self.current_iter + 1), self.ema_decay)
        for ema_param, param in zip(self.ema_model.parameters(),
                                    self.model.parameters()):
            ema_param.data.mul_(alpha).add_(1-alpha, param.data)
    
    def _worker_init_fn(self, worker_id):
        random.seed(self.seed + worker_id)     
        
    def _save_checkpoint(self, filename: str) -> None:
        checkpoint1 = {
                    'network_weights': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'grad_scaler_state': self.grad_scaler1.state_dict() if self.grad_scaler1 is not None else None,
                    'current_iter': self.current_iter + 1,
                    'wandb_id': self.wandb_logger.id
        }
        torch.save(checkpoint1, join(self.output_folder, "model1_" + filename + ".pth"))
        if self.model2 is not None:
            checkpoint2 = {
                        'network_weights': self.model2.state_dict(),
                        'optimizer_state': self.optimizer2.state_dict(),
                        'grad_scaler_state': self.grad_scaler2.state_dict() if self.grad_scaler2 is not None else None,
                        'current_iter': self.current_iter + 1,
                        'wandb_id': self.wandb_logger.id
            }
            torch.save(checkpoint2, join(self.output_folder, "model2_" + filename + ".pth"))
        self.logging.info(f'save model to {join(self.output_folder, filename)}')
    
    
    def _get_label_batch_for_conditional_net(self, label_batch, condition_batch):
        """
        convert label batch to condition label batch
        """
        if condition_batch.max() < self.num_classes:
            return (
                        label_batch==condition_batch.unsqueeze(-1).unsqueeze(-1)
            ).long()
        else:
            label_batch_con = torch.zeros_like(label_batch)
            for i,con in enumerate(condition_batch):
                if con == self.num_classes:
                    label_batch_con[i][label_batch[i]>0] = 1
                else:
                    label_batch_con[i][label_batch[i]!=con] = 0
                    label_batch_con[i][label_batch[i]==con] = 1
            return label_batch_con
                    
    def _kaiming_normal_init_weight(self):
        for m in self.model2.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _xavier_normal_init_weight(self):
        for m in self.model.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.xavier_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def _cross_entropy_loss_con(self, output, target, condition, filter):
        """
        cross entropy loss for conditional network
        """
        softmax = torch.softmax(output,dim=1)
        B,C,D,H,W = softmax.shape
        softmax_con = torch.zeros(B,2,D,H,W).to(self.device)
        if condition[0] < self.num_classes:
            softmax_con[:,1,...] = softmax[np.arange(B),condition.squeeze().long(),...] 
            softmax_con[:,0,...] = 1.0 - softmax_con[:,1,...]
            log = -torch.log(softmax_con.gather(1, target.unsqueeze(1)) + 1e-7)
            #loss = log.mean()
            loss = (log*filter.unsqueeze(1)).sum()/filter.sum() # with filter
        else:
            softmax_con[:,0,...] = softmax[np.arange(B),0,...] 
            softmax_con[:,1,...] = 1.0 - softmax_con[:,0,...]
            log = -torch.log(softmax_con.gather(1, target.unsqueeze(1)) + 1e-7)
            #loss = log.mean()
            loss = (log*filter.unsqueeze(1)).sum()/filter.sum() # with filter
        return loss

    def _adjust_learning_rate(self):
        if self.optimizer_type == 'SGD': # no need to adjust learning rate for adam optimizer   
            print("current learning rate: ",self.current_lr)
            self.current_lr = self.initial_lr * (
                1.0 - self.current_iter / self.max_iterations
            ) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
        
        if (
            self.method_name in ['CPS','C3PS','ConNet'] and
            self.optimizer2_type == 'SGD'
        ):
                print("current learning rate model2: ",self.current_lr)
                self.current2_lr = self.initial2_lr * (
                1.0 - self.current_iter / self.max_iterations
                ) ** 0.9
                for param_group in self.optimizer2.param_groups:
                    param_group['lr'] = self.current2_lr
    
    def _add_information_to_writer(self):
        for param_group in self.optimizer.param_groups:
            self.current_lr = param_group['lr']
        self.tensorboard_writer.add_scalar('info/lr', self.current_lr, 
                                            self.current_iter)
        self.tensorboard_writer.add_scalar('info/total_loss', self.loss, 
                                            self.current_iter)
        self.tensorboard_writer.add_scalar('info/loss_ce', self.loss_ce, 
                                            self.current_iter)
        self.tensorboard_writer.add_scalar('info/loss_dice', self.loss_dice, 
                                            self.current_iter)
        self.logging.info(
            'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
            (self.current_iter, self.loss.item(), self.loss_ce.item(), 
                self.loss_dice.item()))
        self.tensorboard_writer.add_scalar('loss/loss', self.loss, 
                                            self.current_iter)
        if self.consistency_loss:
            self.tensorboard_writer.add_scalar('info/consistency_loss',
                                        self.consistency_loss, 
                                        self.current_iter)
        if self.consistency_weight:
            self.tensorboard_writer.add_scalar('info/consistency_weight',
                                                self.consistency_weight,
                                                self.current_iter)
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _sharpening(self, P):
        T = 1/0.1
        P_sharpen = P ** T / (P ** T + (1-P) ** T)
        return P_sharpen

    

if __name__ == "__main__":
    # test semiTrainer
    pass