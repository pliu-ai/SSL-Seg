import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.cuda.amp as amp
from torch.cuda.amp import GradScaler
from tensorboardX import SummaryWriter
import random
import wandb
from tqdm import tqdm
import numpy as np
from scipy.ndimage import zoom

from dataset.dataset_old import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from dataset.dataset_2d_cac import ACDCDatasetCAC, ACDCDatasetVal
from networks.net_factory import net_factory
from utils import losses, ramps
from trainer.training_logger import TrainingLogger
from trainer.methods.c3ps_2d import train_C3PS_2D


PATIENTS_TO_SLICES = {
    "ACDC": {"3": 68, "7": 136, "14": 256, "21": 396,
             "28": 512, "35": 664, "140": 1312},
    "Prostate": {"2": 27, "4": 53, "8": 120, "12": 179,
                 "16": 256, "21": 312, "42": 623},
}


class SemiSupervisedTrainer2D:
    def __init__(self, config, output_folder=None, logging=None,
                 continue_training=False) -> None:
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
        self.model1_thresh = config['model1_thresh']
        self.model2_thresh = config['model2_thresh']
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

        self.continue_wandb = config['continue_wandb']
        self.continue_training = continue_training
        self.wandb_id = config['wandb_id']
        self.network_checkpoint = config['model_checkpoint']
        self.network2_checkpoint = config['model2_checkpoint']

        self.consistency_rampup = config['consistency_rampup']
        self.consistency = config['consistency']

        dataset = config['DATASET']
        self.patch_size = dataset['patch_size']
        self.labeled_num = dataset['labeled_num']
        self.labeled_num_pl = dataset.get('labeled_num_pl', 0)
        self.batch_size = dataset['batch_size']
        self.labeled_bs = dataset['labeled_bs']
        self.dataset_name = config['dataset_name']

        dataset_config = dataset[self.dataset_name]
        self.num_classes = dataset_config['num_classes']
        self.class_name_list = dataset_config.get('class_name_list', [])
        self.training_data_num = dataset_config['training_data_num']
        self.testing_data_num = dataset_config['testing_data_num']
        self.root_path = dataset_config['root_path']

        self.method_name = config['method']
        self.method_config = config['METHOD'][self.method_name]
        self.use_CAC = config['use_CAC']
        self.use_PL = config.get('use_PL', False)

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255)
        self.dice_loss = losses.DiceLoss(self.num_classes)
        self.dice_loss_con = losses.DiceLoss(2)
        self.best_performance = 0.0
        self.best_performance2 = 0.0
        self.current_iter = 0
        self.model = None
        self.model2 = None
        self.optimizer = None
        self.optimizer2 = None
        self.dataset = None
        self.dataset_val = None
        self.dataset_pl = None
        self.dataloader = None
        self.dataloader_pl = None
        self.dataloader_val = None
        self.tensorboard_writer = None
        self.wandb_logger = None
        self.logger = None
        self.max_epoch = 0
        self.current_lr = self.initial_lr
        self.current2_lr = self.initial2_lr
        self.consistency_weight = 0.0
        self.grad_scaler1 = GradScaler()
        self.grad_scaler2 = GradScaler()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def initialize_network(self):
        self.model = net_factory(
            net_type=self.backbone, in_chns=1,
            class_num=self.num_classes, device=self.device,
        )
        if self.method_name in ('C3PS', 'ConNet'):
            self.model2 = net_factory(
                net_type=self.backbone2, in_chns=1,
                class_num=2, device=self.device,
            )
            self._kaiming_normal_init_weight()

    def initialize(self):
        self.experiment_name = (
            f"{self.dataset_name}_{self.method_name}_"
            f"labeled{self.labeled_num}_{self.optimizer_type}_"
            f"{self.optimizer2_type}_{self.exp}"
        )
        self.wandb_logger = wandb.init(
            name=self.experiment_name,
            project="semi-supervised-segmentation",
            config=self.config,
        )
        wandb.tensorboard.patch(root_logdir=self.output_folder + '/log')
        self.tensorboard_writer = SummaryWriter(self.output_folder + '/log')
        self.logger = TrainingLogger(self.tensorboard_writer, self.logging)
        self.load_dataset()
        self.get_dataloader()
        self.initialize_optimizer_and_scheduler()

    def load_dataset(self):
        if self.method_name in ('C3PS', 'ConNet'):
            self.dataset = ACDCDatasetCAC(
                base_dir=self.root_path,
                patch_size=self.patch_size,
                num_class=self.num_classes,
                stride=self.method_config['stride'],
                iou_bound=(self.method_config['iou_bound_low'],
                           self.method_config['iou_bound_high']),
                labeled_num=self._patients_to_slices(),
                con_list=self.method_config['con_list'],
                addi_con_list=self.method_config['addition_con_list'],
            )
        else:
            self.dataset = BaseDataSets(
                base_dir=self.root_path, split='train', num=None,
                transform=transforms.Compose([
                    RandomGenerator(self.patch_size),
                ]),
            )
        self.dataset_val = ACDCDatasetVal(base_dir=self.root_path)

    def get_dataloader(self):
        labeled_slice = self._patients_to_slices()
        total_slices = len(self.dataset)
        print(f"Total slices: {total_slices}, labeled: {labeled_slice}")
        labeled_idxs = list(range(0, labeled_slice))
        unlabeled_idxs = list(range(labeled_slice, total_slices))
        batch_sampler = TwoStreamBatchSampler(
            labeled_idxs, unlabeled_idxs,
            self.batch_size, self.batch_size - self.labeled_bs,
        )
        self.dataloader = DataLoader(
            self.dataset, batch_sampler=batch_sampler,
            num_workers=4, pin_memory=True,
        )
        self.dataloader_val = DataLoader(
            self.dataset_val, batch_size=1, shuffle=False, num_workers=1,
        )
        self.max_epoch = self.max_iterations // len(self.dataloader) + 1

    def _patients_to_slices(self):
        ref = PATIENTS_TO_SLICES.get(self.dataset_name, {})
        return ref.get(str(self.labeled_num), self.labeled_num)

    def initialize_optimizer_and_scheduler(self):
        assert self.model is not None
        if self.optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), lr=self.initial_lr,
                momentum=0.9, weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), self.initial_lr,
                weight_decay=self.weight_decay, amsgrad=True,
            )
        if self.method_name in ('C3PS', 'ConNet', 'CPS', 'CTCT'):
            if self.optimizer2_type == 'SGD':
                self.optimizer2 = torch.optim.SGD(
                    self.model2.parameters(), lr=self.initial2_lr,
                    momentum=0.9, weight_decay=self.weight_decay,
                )
            else:
                self.optimizer2 = torch.optim.Adam(
                    self.model2.parameters(), self.initial2_lr,
                    weight_decay=self.weight_decay, amsgrad=True,
                )
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.2,
            patience=self.lr_scheduler_patience, verbose=True,
            threshold=self.lr_scheduler_eps, threshold_mode='abs',
        )

    # ------------------------------------------------------------------
    # Training dispatch
    # ------------------------------------------------------------------
    def train(self):
        if self.method_name == 'C3PS':
            train_C3PS_2D(self)
        else:
            raise NotImplementedError(
                f"Method {self.method_name} not yet supported in 2D trainer"
            )

    # ------------------------------------------------------------------
    # Evaluation (slice-by-slice on full volumes)
    # ------------------------------------------------------------------
    def evaluation(self, model, do_condition=False, model_name="model"):
        from medpy import metric as medpy_metric
        model.eval()
        metric_list = 0.0
        for _, sampled_batch in enumerate(self.dataloader_val):
            volume = sampled_batch['image']
            label_vol = sampled_batch['label']
            metric_i = self._test_single_volume(
                model, volume, label_vol,
                classes=self.num_classes,
                patch_size=self.patch_size,
                do_condition=do_condition,
            )
            metric_list += np.array(metric_i)
        metric_list = metric_list / len(self.dataset_val)

        for ci in range(self.num_classes - 1):
            self.tensorboard_writer.add_scalar(
                f'info/{model_name}_val_{ci + 1}_dice',
                metric_list[ci, 0], self.current_iter,
            )
            self.tensorboard_writer.add_scalar(
                f'info/{model_name}_val_{ci + 1}_hd95',
                metric_list[ci, 1], self.current_iter,
            )
        perf = np.mean(metric_list, axis=0)[0]
        hd95 = np.mean(metric_list, axis=0)[1]
        self.tensorboard_writer.add_scalar(
            f'mean/{model_name}_val_mean_dice', perf, self.current_iter,
        )
        self.tensorboard_writer.add_scalar(
            f'mean/{model_name}_val_mean_hd', hd95, self.current_iter,
        )
        best = self.best_performance2 if do_condition else self.best_performance
        if perf > best:
            if do_condition:
                self.best_performance2 = perf
            else:
                self.best_performance = perf
            path = os.path.join(
                self.output_folder,
                f'{model_name}_iter_{self.current_iter}_dice_{perf:.4f}.pth',
            )
            torch.save(model.state_dict(), path)
            self.logging.info(f'save best {model_name} to {path}')
        self.logging.info(
            f'iter {self.current_iter}: {model_name}_dice={perf:.4f}, '
            f'{model_name}_hd95={hd95:.4f}'
        )
        model.train()
        return metric_list

    def _test_single_volume(self, net, image, label, classes,
                            patch_size, do_condition=False):
        from medpy import metric as medpy_metric
        image = image.squeeze(0).cpu().numpy()
        label = label.squeeze(0).cpu().numpy()
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            sl = image[ind]
            x, y = sl.shape
            sl = zoom(sl, (patch_size[0] / x, patch_size[1] / y), order=0)
            inp = torch.from_numpy(sl).unsqueeze(0).unsqueeze(0).float().to(
                self.device)
            net.eval()
            with torch.no_grad():
                if do_condition:
                    con_preds = []
                    for c in self.method_config['con_list']:
                        cond = torch.FloatTensor([[c]]).to(self.device)
                        out_c = torch.softmax(net(inp, cond), dim=1)
                        con_preds.append(out_c[:, 1:2])
                    stacked = torch.cat(con_preds, dim=1)
                    bg = 1.0 - stacked.sum(dim=1, keepdim=True).clamp(0, 1)
                    full = torch.cat([bg, stacked], dim=1)
                    out = torch.argmax(full, dim=1).squeeze(0)
                else:
                    out = torch.argmax(
                        torch.softmax(net(inp), dim=1), dim=1
                    ).squeeze(0)
                out = out.cpu().numpy()
                pred = zoom(out, (x / patch_size[0], y / patch_size[1]),
                            order=0)
                prediction[ind] = pred
        result = []
        for i in range(1, classes):
            p = (prediction == i)
            g = (label == i)
            if p.sum() == 0 and g.sum() == 0:
                result.append([1.0, 0.0])
            elif p.sum() == 0 or g.sum() == 0:
                result.append([0.0, 100.0])
            else:
                result.append([
                    medpy_metric.dc(p, g),
                    medpy_metric.hd95(p, g),
                ])
        return result

    # ------------------------------------------------------------------
    # Helper methods used by C3PS training loop
    # ------------------------------------------------------------------
    def _get_condition(self, pred_con_list):
        if 0 in pred_con_list:
            pred_con_list.remove(0)
        inter = list(set(pred_con_list) & set(self.method_config['con_list']))
        inter += self.method_config['addition_con_list']
        if not inter:
            inter = self.method_config['con_list']
        return int(np.random.choice(inter))

    def _get_current_consistency_weight(self, epoch):
        return self.consistency * ramps.sigmoid_rampup(
            epoch, self.consistency_rampup)

    def _adjust_learning_rate(self):
        if self.optimizer_type == 'SGD':
            self.current_lr = self.initial_lr * (
                1.0 - self.current_iter / self.max_iterations) ** 0.9
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.current_lr
        if (self.method_name in ('C3PS', 'ConNet', 'CPS')
                and self.optimizer2_type == 'SGD'):
            self.current2_lr = self.initial2_lr * (
                1.0 - self.current_iter / self.max_iterations) ** 0.9
            for pg in self.optimizer2.param_groups:
                pg['lr'] = self.current2_lr

    def _kaiming_normal_init_weight(self):
        for m in self.model2.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _worker_init_fn(self, worker_id):
        random.seed(self.seed + worker_id)

    def load_checkpoint(self, fname="latest"):
        pass
