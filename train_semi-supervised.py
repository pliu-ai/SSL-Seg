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
from urllib.parse import urlparse

from trainer.semi_trainer_3D import SemiSupervisedTrainer3D
from trainer.semi_trainer_2D import SemiSupervisedTrainer2D
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

SAM_CHECKPOINT_URLS = {
    'vit_h': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth',
    'vit_l': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
    'vit_b': 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth',
}

MEDSAM_CHECKPOINT_URLS = {
    'vit_b': 'https://zenodo.org/records/10689643/files/medsam_vit_b.pth?download=1',
}


def maybe_download_sam_checkpoint(config):
    sam_cfg = config.get('sam', {}) or {}
    sam_backend = str(sam_cfg.get('sam_backend', 'sam')).lower()
    use_sam = bool(
        sam_cfg.get('use_sam_refinement', False) or
        sam_cfg.get('use_fn_recovery', False)
    )
    if not use_sam:
        return
    if sam_backend not in ('sam', 'medsam'):
        raise ValueError(
            f"Unsupported sam.sam_backend={sam_backend}. Expected 'sam' or 'medsam'."
        )

    sam_checkpoint = str(sam_cfg.get('sam_checkpoint', '')).strip()
    if len(sam_checkpoint) == 0:
        raise ValueError("SAM is enabled but sam.sam_checkpoint is empty.")
    if os.path.isfile(sam_checkpoint):
        return

    auto_download = bool(sam_cfg.get('auto_download', False))
    if not auto_download:
        raise FileNotFoundError(
            f"SAM checkpoint not found: {sam_checkpoint}. "
            "Set sam.auto_download=true to download automatically."
        )

    download_url = str(sam_cfg.get('download_url', '')).strip()
    if len(download_url) == 0:
        model_type = str(sam_cfg.get('sam_model_type', 'vit_h'))
        if sam_backend == 'medsam':
            if model_type not in MEDSAM_CHECKPOINT_URLS:
                raise ValueError(
                    f"Unsupported MedSAM sam_model_type={model_type}. "
                    "Use 'vit_b' for MedSAM or set sam.download_url explicitly."
                )
            download_url = MEDSAM_CHECKPOINT_URLS[model_type]
        else:
            if model_type not in SAM_CHECKPOINT_URLS:
                raise ValueError(
                    f"Unsupported sam_model_type={model_type}. "
                    "Set sam.download_url explicitly."
                )
            download_url = SAM_CHECKPOINT_URLS[model_type]

    ckpt_dir = os.path.dirname(sam_checkpoint)
    if len(ckpt_dir) > 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    parsed = urlparse(download_url)
    if parsed.scheme not in ('http', 'https'):
        raise ValueError(f"Invalid SAM download URL: {download_url}")

    print(f"[SAM] backend={sam_backend}, checkpoint missing, downloading from {download_url}")
    tmp_path = sam_checkpoint + ".tmp"
    try:
        torch.hub.download_url_to_file(download_url, tmp_path, progress=True)
        os.replace(tmp_path, sam_checkpoint)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    print(f"[SAM] checkpoint ready at {sam_checkpoint}")


def sanitize_gpu_config(config):
    requested_gpu_raw = str(config.get('gpu', '0'))
    try:
        requested_gpu = int(requested_gpu_raw)
    except Exception:
        requested_gpu = 0
        print(f"[GPU] invalid gpu='{requested_gpu_raw}', fallback to 0")

    try:
        cuda_available = torch.cuda.is_available()
        visible_gpus = torch.cuda.device_count() if cuda_available else 0
    except Exception as e:
        raise RuntimeError(
            f"[GPU] failed to query CUDA devices: {e}\n"
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}"
        )

    if visible_gpus <= 0:
        raise RuntimeError(
            "[GPU] no visible CUDA device. "
            "Please check CUDA driver/runtime and CUDA_VISIBLE_DEVICES."
        )

    if requested_gpu < 0 or requested_gpu >= visible_gpus:
        print(
            f"[GPU] requested gpu={requested_gpu} but visible_gpus={visible_gpus}, "
            "fallback to gpu=0"
        )
        requested_gpu = 0

    config['gpu'] = str(requested_gpu)


if __name__ == "__main__":
    args = parser.parse_args()
    config = yaml.safe_load(open(args.config, 'r'))
    maybe_download_sam_checkpoint(config)
    sanitize_gpu_config(config)
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
    else:
        trainer = SemiSupervisedTrainer2D(config=config, 
                                          output_folder=snapshot_path,
                                          logging=logging,
                                          continue_training=args.c)
    trainer.initialize_network()
    trainer.initialize()
    if args.c: # continue training
        trainer.load_checkpoint()
    trainer.train()
