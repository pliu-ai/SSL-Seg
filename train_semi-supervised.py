import sys
import os
import shutil
import argparse
import logging
import time
import yaml
import random 
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from trainer.semi_trainer_3D import SemiSupervisedTrainer3D
#from trainer.semi_trainer_2D import SemiSupervisedTrainer2D
from utils.util import save_config


# from torch.utils.data import dataloader
# from torch.multiprocessing import reductions
# from multiprocessing.reduction import ForkingPickler

# default_collate_func = dataloader.default_collate


# def default_collate_override(batch):
#   dataloader._use_shared_memory = False
#   return default_collate_func(batch)

# setattr(dataloader, 'default_collate', default_collate_override)

# for t in torch._storage_classes:
#   if sys.version_info[0] == 2:
#     if t in ForkingPickler.dispatch:
#         del ForkingPickler.dispatch[t]
#   else:
#     if t in ForkingPickler._extra_reducers:
#         del ForkingPickler._extra_reducers[t]

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str,
                    default='configs/train_config_3d.yaml', help='training configuration')
parser.add_argument('--c', action='store_true',required=False,
                    help="[OPTIONAL] Continue training from latest checkpoint")


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    #os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
    save_config(config) # save config to yaml file with timestamp
    if not config['deterministic']:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])

    snapshot_path = "../model/{}_{}_{}_{}_{}_{}/{}".format(
        config['dataset_name'], 
        config['DATASET']['labeled_num'], 
        config['method'], 
        config['exp'],
        config['optimizer_type'],
        config['optimizer2_type'],
        config['backbone']   
    )
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    # move config to snapshot_path
    shutil.copyfile(args.config, snapshot_path+"/"+
                                 time.strftime("%Y-%m-%d=%H-%M-%S", 
                                               time.localtime())+
                                               "_train_config.yaml")
    logging.basicConfig(
        filename=snapshot_path+"/log.txt", 
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s', 
        datefmt='%H-%M-%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(config))
    if config['train_3D']:
        trainer = SemiSupervisedTrainer3D(config=config, 
                                          output_folder=snapshot_path,
                                          logging=logging,
                                          continue_training=args.c
                                          )
    # else:
    #     trainer = SemiSupervisedTrainer2D(config=config, 
    #                                       output_folder=snapshot_path,
    #                                       logging=logging,
    #                                       root_path=config['root_path'])
    trainer.initialize_network()
    trainer.initialize()
    if args.c: # continue training
        trainer.load_checkpoint()
    trainer.train()

