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
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
import random
from glob import glob
import wandb
from tqdm import tqdm
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join

from utils import losses,ramps,cac_loss,feature_memory,correlation
from dataset.BCVData import BCVDataset, BCVDatasetCAC,DatasetSR
from dataset.dataset import DatasetSemi
from dataset.sampler import BatchSampler, ClassRandomSampler
from networks.net_factory_3d import (
    net_factory_3d,
    resolve_feature_module_3d,
)
from dataset.dataset import TwoStreamBatchSampler
from inference.val_3D import test_all_case
from unet3d.losses import DiceLoss #test loss
from trainer.train_utils import normalize_slice_to_01, sharpening
from trainer.sam_pipeline import SAMPseudoLabelPipeline
from trainer.checkpoint_manager import CheckpointManager
from trainer.training_logger import TrainingLogger
from trainer.methods.baseline import train_baseline
from trainer.methods.dan import train_DAN
from trainer.methods.mcnet import train_McNet
from trainer.methods.mt import train_MT
from trainer.methods.uamt import train_UAMT
from trainer.methods.cps import train_CPS
from trainer.methods.cvcl_partial import train_CVCL_partial
from trainer.methods.cvcl import train_CVCL
from trainer.methods.c3ps import train_C3PS
from trainer.methods.gfel import train_GFEL
from trainer.methods.caml import train_CAML
from trainer.methods.cssr import train_CSSR
from trainer.methods.condition_net import train_ConditionNet







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
        self.iters_per_epoch = config.get('iters_per_epoch')
        self.num_epochs = config.get('num_epochs')
        self.began_semi_iter = config['began_semi_iter']
        self.ema_decay = config['ema_decay']
        self.began_condition_iter = config['began_condition_iter']
        self.began_eval_iter = config['began_eval_iter']
        self.show_img_freq = config['show_img_freq']
        self.save_checkpoint_freq = config['save_checkpoint_freq']
        self.val_freq = config['val_freq']
        self.val_sample_num = config.get('val_sample_num')
        self.condition_eval_batch_size = config.get('condition_eval_batch_size', 1)
        self.train_log_interval = max(1, int(config.get('train_log_interval', 20)))
        self.dataloader_num_workers = int(config.get('dataloader_num_workers', 2))
        self.dataloader_prefetch_factor = int(config.get('dataloader_prefetch_factor', 2))
        self.dataloader_persistent_workers = bool(config.get('dataloader_persistent_workers', True))
        self.dataloader_pin_memory = bool(config.get('dataloader_pin_memory', True))

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
        
        # SAM pipeline (encapsulates all SAM/FN-recovery config and state)
        sam_cfg = config.get('sam', {})
        sam_cfg['show_img_freq'] = self.show_img_freq  # pass for visualize_interval default
        self.sam_pipeline = SAMPseudoLabelPipeline(
            sam_cfg=sam_cfg,
            global_cfg=config,
            device=self.device,
            num_classes=self.num_classes,
            output_folder=output_folder,
            backbone=self.backbone,
            logging=logging,
        )

        # Checkpoint manager (encapsulates save/load logic)
        self.ckpt_manager = CheckpointManager(output_folder, logging)

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
        self._cached_val_cases_iter = None
        self._cached_val_cases = None

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
        self.logger = TrainingLogger(self.tensorboard_writer, self.logging)
        self.load_dataset()
        self.initialize_optimizer_and_scheduler()
        self.sam_pipeline.initialize()
    
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

    def _forward_logits(self, model, inputs):
        outputs = model(inputs)
        if isinstance(outputs, (tuple, list)):
            for item in outputs:
                if torch.is_tensor(item):
                    return item
            raise ValueError("model outputs does not contain tensor logits")
        return outputs

    def load_checkpoint(self, fname="latest"):
        self.current_iter = self.ckpt_manager.load(
            fname=fname,
            model=self.model,
            optimizer=self.optimizer,
            grad_scaler=self.grad_scaler1,
            model2=self.model2,
            optimizer2=getattr(self, 'optimizer2', None),
            grad_scaler2=self.grad_scaler2,
        )
        

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

        # Inform the SAM pipeline which dataset to use for prototype computation
        self.sam_pipeline.set_dataset(
            dataset=self.dataset,
            labeled_num=self.labeled_num,
            labeled_bs=self.labeled_bs,
            batch_size=self.batch_size,
        )

    def get_dataloader(self):
        dataloader_kwargs = self._build_dataloader_kwargs()
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            **dataloader_kwargs,
        )
        self.dataloader_pl = DataLoader(
            self.dataset_pl,
            batch_size=self.batch_size,
            shuffle=True,
            **dataloader_kwargs,
        )

    def _build_dataloader_kwargs(self):
        kwargs = {
            "num_workers": self.dataloader_num_workers,
            "pin_memory": self.dataloader_pin_memory,
        }
        if self.dataloader_num_workers > 0:
            kwargs["persistent_workers"] = self.dataloader_persistent_workers
            kwargs["prefetch_factor"] = self.dataloader_prefetch_factor
        return kwargs

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
                self.optimizer2 = torch.optim.Adam(
                    self.model2.parameters(), 
                    self.initial2_lr, 
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
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                shuffle=True,
                **self._build_dataloader_kwargs(),
            )
            self.max_epoch = self.max_iterations // len(self.dataloader) + 1
            print(f"max epochs:{self.max_epoch}, max iterations:{self.max_iterations}")
            print(f"len dataloader:{len(self.dataloader)}")
            self._train_baseline()
        elif self.method_name == 'CVCL_partial':
            self.dataloader = DataLoader(
                self.dataset,
                batch_sampler=BatchSampler(
                    ClassRandomSampler(self.dataset),
                    self.batch_size,
                    True,
                ),
                **self._build_dataloader_kwargs(),
            )
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
                            **self._build_dataloader_kwargs())
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
                                shuffle=True, **self._build_dataloader_kwargs())
                if self.iters_per_epoch is not None and self.num_epochs is not None:
                    self.iters_per_epoch = int(self.iters_per_epoch)
                    self.num_epochs = int(self.num_epochs)
                    if self.iters_per_epoch <= 0 or self.num_epochs <= 0:
                        raise ValueError("iters_per_epoch and num_epochs must be positive integers")
                    self.max_epoch = self.num_epochs
                    self.max_iterations = self.iters_per_epoch * self.num_epochs
                    self.logging.info(
                        "C3PS fixed-epoch mode: %d iters/epoch x %d epochs => max_iterations=%d",
                        self.iters_per_epoch,
                        self.num_epochs,
                        self.max_iterations,
                    )
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
        selected_cases = self._get_validation_cases()
        if selected_cases is None:
            test_num = self.testing_data_num // 1.5
            if self.current_iter % 1000==0 or (self.method_name=='ConNet' and self.current_iter % 400==0):
                test_num = self.testing_data_num
        else:
            test_num = len(selected_cases)

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
                                       normalization=self.normalization,
                                       selected_cases=selected_cases,
                                       condition_batch_size=self.condition_eval_batch_size)
        print("avg metric shape:",avg_metric.shape)
        if avg_metric[:, 0].mean() > best_performance:
            best_performance = avg_metric[:, 0].mean()
            if do_condition:
                self.best_performance2 = best_performance
            else:
                self.best_performance = best_performance
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

    def _load_validation_case_list(self):
        if os.path.isdir(self.test_list):
            return sorted(glob(os.path.join(self.test_list, "*.nii.gz")))
        with open(self.test_list, 'r') as f:
            return [img.strip() for img in f.readlines() if img.strip()]

    def _get_validation_cases(self):
        if self.val_sample_num in (None, 0):
            return None
        if self.val_sample_num < 0:
            raise ValueError("val_sample_num must be >= 0 or null")
        if self._cached_val_cases_iter == self.current_iter:
            return self._cached_val_cases

        image_list = self._load_validation_case_list()
        sample_num = min(int(self.val_sample_num), len(image_list))
        selected_cases = random.sample(image_list, sample_num)
        self._cached_val_cases_iter = self.current_iter
        self._cached_val_cases = selected_cases
        self.logging.info(
            "iteration %d: validation sampled %d/%d cases",
            self.current_iter,
            sample_num,
            len(image_list),
        )
        return selected_cases


    def _train_baseline(self):
        train_baseline(self)

    def _train_DAN(self):
        train_DAN(self)

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
        self.logger.close()
        print('Training Finished')
    
    
    def _train_McNet(self):
        train_McNet(self)
                
    
    def _train_MT(self):
        train_MT(self)

    def _train_CPS(self):
        train_CPS(self)

        
            
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
        train_CVCL_partial(self)

    def _train_CVCL(self):
        train_CVCL(self)
    def _train_C3PS(self):
        train_C3PS(self)
    def _train_GFEL(self):
        train_GFEL(self)
    def _train_CAML(self):
        train_CAML(self)
    def _train_CSSR(self):
        train_CSSR(self)
    def _train_EMSSL(self):
        """
        code for "Bayesian Pseudo Labels: Expectation Maximization for Robust
        and Efficient Semi-supervised Segmentation"
        """
        print("================> Training EMSSL<===============")
    
    def _train_ConditionNet(self):
        train_ConditionNet(self)
    def _train_EM(self):
        pass

    def _train_UAMT(self):
        train_UAMT(self)

    
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
        
    def _save_checkpoint(self, filename: str = "latest") -> None:
        self.ckpt_manager.save(
            filename=filename,
            model=self.model,
            optimizer=self.optimizer,
            grad_scaler=self.grad_scaler1,
            current_iter=self.current_iter,
            wandb_id=self.wandb_logger.id,
            model2=self.model2,
            optimizer2=getattr(self, 'optimizer2', None),
            grad_scaler2=self.grad_scaler2,
        )
    
    
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
            self.current_lr = self.initial_lr * (
                1.0 - self.current_iter / self.max_iterations
            ) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.current_lr
        
        if (
            self.method_name in ['CPS','C3PS','ConNet'] and
            self.optimizer2_type == 'SGD'
        ):
                self.current2_lr = self.initial2_lr * (
                1.0 - self.current_iter / self.max_iterations
                ) ** 0.9
                for param_group in self.optimizer2.param_groups:
                    param_group['lr'] = self.current2_lr
    
    def _add_information_to_writer(self):
        self.current_lr = self.logger.log_iter_scalars(
            current_iter=self.current_iter,
            optimizer=self.optimizer,
            loss=self.loss,
            loss_ce=self.loss_ce,
            loss_dice=self.loss_dice,
            consistency_loss=self.consistency_loss,
            consistency_weight=self.consistency_weight,
        )
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.num_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _sharpening(self, P):
        return sharpening(P)

    

if __name__ == "__main__":
    # test semiTrainer
    pass
