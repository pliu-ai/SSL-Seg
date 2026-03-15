"""
SAMPseudoLabelPipeline: SAM-guided pseudo label refinement and FN recovery.

Extracted from SemiSupervisedTrainer3D to isolate SAM-specific logic.
The trainer creates one instance and calls run_pipeline() in training loops.
"""
import os
import time
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from networks.net_factory_3d import resolve_feature_module_3d
from utils.sam_refiner import MultiPlanarSAMRefiner, SAM_REFINER_DEFAULTS
from utils.confidence_estimator import DualSourceConfidence
from utils.fn_recovery import SAMFalseNegativeRecovery, FN_RECOVERY_DEFAULTS
from trainer.train_utils import normalize_slice_to_01


class SAMPseudoLabelPipeline:
    """Encapsulates SAM-guided pseudo label refinement and false-negative recovery.

    Usage in trainer::

        self.sam_pipeline = SAMPseudoLabelPipeline(sam_cfg, config, ...)
        self.sam_pipeline.initialize()                 # in trainer.initialize()
        self.sam_pipeline.set_dataset(dataset, ...)   # after load_dataset()

        # in training loop:
        if self.sam_pipeline.is_active:
            refined, reliability, iou = self.sam_pipeline.run_pipeline(
                unlabeled_volume_batch, teacher_logits, pseudo_label, epoch_num,
                current_iter=self.current_iter,
                tensorboard_writer=self.tensorboard_writer,
                ema_model=self.ema_model,
            )
    """

    def __init__(
        self,
        sam_cfg: dict,
        global_cfg: dict,
        device,
        num_classes: int,
        output_folder: str,
        backbone: str,
        logging,
    ):
        self._logging = logging
        self._device = device
        self._num_classes = num_classes
        self._backbone = backbone

        # --- refinement config ---
        self.use_sam_refinement: bool = sam_cfg.get(
            'use_sam_refinement',
            global_cfg.get('use_sam_refinement', False),
        )
        self._sam_checkpoint: str = sam_cfg.get(
            'sam_checkpoint', global_cfg.get('sam_checkpoint', '')
        )
        self._auto_download: bool = bool(sam_cfg.get('auto_download', False))
        self._download_url: str = sam_cfg.get('download_url', '')
        self._sam_model_type: str = sam_cfg.get(
            'sam_model_type', global_cfg.get('sam_model_type', 'vit_b')
        )
        self._sam_backend: str = str(
            sam_cfg.get('sam_backend', global_cfg.get('sam_backend', 'medsam'))
        ).lower()
        self._sam_refine_interval: int = int(
            sam_cfg.get('sam_refine_interval', global_cfg.get('sam_refine_interval', 20))
        )
        self._sam_min_area: int = int(
            sam_cfg.get('min_component_area', sam_cfg.get('min_area', SAM_REFINER_DEFAULTS['min_area']))
        )
        self._sam_bbox_expand_ratio: float = float(
            sam_cfg.get('bbox_expand_ratio', sam_cfg.get('expand_ratio', SAM_REFINER_DEFAULTS['expand_ratio']))
        )
        self._sam_max_expansion_ratio: float = float(
            sam_cfg.get('max_expansion_ratio', SAM_REFINER_DEFAULTS['max_expansion_ratio'])
        )
        self._sam_use_multiplanar: bool = bool(
            sam_cfg.get('use_multiplanar', SAM_REFINER_DEFAULTS['use_multiplanar'])
        )
        self._sam_use_propagation: bool = bool(
            sam_cfg.get('use_propagation', SAM_REFINER_DEFAULTS['use_propagation'])
        )
        self._confidence_alpha: float = float(
            sam_cfg.get('confidence_alpha', global_cfg.get('confidence_alpha', 0.5))
        )
        self._confidence_temperature: float = float(
            sam_cfg.get('confidence_temperature', global_cfg.get('confidence_temperature', 1.0))
        )
        self._spatial_smooth_sigma: float = float(
            sam_cfg.get('spatial_smooth_sigma', global_cfg.get('spatial_smooth_sigma', 2.0))
        )

        # --- FN recovery config ---
        self.use_fn_recovery: bool = bool(
            sam_cfg.get('use_fn_recovery', global_cfg.get('use_fn_recovery', False))
        )
        self._fn_recovery_start_epoch: int = int(
            sam_cfg.get('fn_recovery_start_epoch', global_cfg.get('fn_recovery_start_epoch', 50))
        )
        self._fn_recovery_interval: int = int(
            sam_cfg.get('fn_recovery_interval', global_cfg.get('fn_recovery_interval', 100))
        )
        self._fn_similarity_threshold: float = float(
            sam_cfg.get('fn_similarity_threshold', FN_RECOVERY_DEFAULTS['similarity_threshold'])
        )
        self._fn_consistency_slices: int = int(
            sam_cfg.get('fn_consistency_slices', FN_RECOVERY_DEFAULTS['consistency_slices'])
        )
        self._prototype_update_interval: int = int(
            sam_cfg.get('prototype_update_interval', global_cfg.get('prototype_update_interval', 5))
        )
        self._sam_feature_layer: str = sam_cfg.get('feature_layer', '')
        self._sam_points_per_side: int = int(
            sam_cfg.get('sam_points_per_side', FN_RECOVERY_DEFAULTS['sam_points_per_side'])
        )
        self._sam_pred_iou_thresh: float = float(
            sam_cfg.get('sam_pred_iou_thresh', FN_RECOVERY_DEFAULTS['sam_pred_iou_thresh'])
        )
        self._fn_iou_threshold = sam_cfg.get('fn_iou_threshold', FN_RECOVERY_DEFAULTS['iou_threshold'])

        # --- visualization config ---
        self._visualize_refine: bool = bool(sam_cfg.get('visualize_refine', False))
        self._visualize_interval: int = int(
            sam_cfg.get('visualize_interval', max(sam_cfg.get('show_img_freq', 1), 1))
        )
        self._visualize_case_index: int = int(sam_cfg.get('visualize_case_index', 0))
        self._visualize_slice_strategy: str = str(sam_cfg.get('visualize_slice_strategy', 'max_change'))
        self._save_compare_npz: bool = bool(sam_cfg.get('save_compare_npz', True))
        self._vis_output_dir = os.path.join(output_folder, "sam_refine_vis")

        # --- runtime state ---
        self._sam_refiner = None
        self._confidence_estimator = None
        self._fn_recoverer = None
        self._class_prototypes = None
        self._cached_refined_label = None
        self._cached_iou_scores = None
        self._teacher_feature_module = None
        self._teacher_feature_layer_name = ""
        self._last_visualize_iter = -1

        # dataset state (set via set_dataset)
        self._dataset = None
        self._labeled_num = 0
        self._labeled_bs = 1
        self._batch_size = 2
        self._labeled_loader_cache = None

    @property
    def is_active(self) -> bool:
        """True if any SAM module is enabled."""
        return self.use_sam_refinement or self.use_fn_recovery

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _maybe_download_checkpoint(self) -> None:
        """Download the checkpoint file if auto_download is enabled and the file is missing."""
        ckpt = self._sam_checkpoint
        if os.path.isfile(ckpt):
            return
        if not self._auto_download:
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt}. "
                "Set auto_download: true and download_url in the sam config, "
                "or download it manually."
            )
        url = self._download_url
        if not url:
            raise ValueError(
                f"Checkpoint not found: {ckpt} and download_url is empty. "
                "For MedSAM (vit_b), download medsam_vit_b.pth from "
                "https://github.com/bowang-lab/MedSAM and set its path in sam_checkpoint."
            )
        os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
        self._logging.info("Downloading checkpoint from %s → %s", url, ckpt)
        try:
            urllib.request.urlretrieve(url, ckpt)
        except Exception as e:
            raise RuntimeError(
                f"Failed to download checkpoint from {url}: {e}. "
                "Please download medsam_vit_b.pth manually."
            ) from e
        self._logging.info("Checkpoint saved to %s", ckpt)

    def initialize(self) -> None:
        """Create SAM/FN-recovery objects. Call once before training starts."""
        if not self.is_active:
            return
        if not self._sam_checkpoint:
            raise ValueError("SAM is enabled but sam_checkpoint is empty in config.")

        self._maybe_download_checkpoint()

        if self.use_fn_recovery and self._sam_backend == 'medsam':
            raise ValueError(
                "use_fn_recovery=True is not compatible with sam_backend='medsam'. "
                "MedSAM uses box prompts only; the automatic mask generator used by FN "
                "recovery requires point-grid prompts. Set use_fn_recovery: false in config."
            )

        if self.use_sam_refinement:
            self._sam_refiner = MultiPlanarSAMRefiner(
                sam_checkpoint=self._sam_checkpoint,
                model_type=self._sam_model_type,
                sam_backend=self._sam_backend,
                device=str(self._device),
                min_area=self._sam_min_area,
                expand_ratio=self._sam_bbox_expand_ratio,
                max_expansion_ratio=self._sam_max_expansion_ratio,
                use_multiplanar=self._sam_use_multiplanar,
                use_propagation=self._sam_use_propagation,
            )
            self._confidence_estimator = DualSourceConfidence(
                alpha=self._confidence_alpha,
                temperature=self._confidence_temperature,
                spatial_smooth_sigma=self._spatial_smooth_sigma,
            )

        if self.use_fn_recovery:
            self._fn_recoverer = SAMFalseNegativeRecovery(
                sam_checkpoint=self._sam_checkpoint,
                model_type=self._sam_model_type,
                sam_backend=self._sam_backend,
                device=str(self._device),
                iou_threshold=self._fn_iou_threshold,
                similarity_threshold=self._fn_similarity_threshold,
                min_area=self._sam_min_area,
                consistency_slices=self._fn_consistency_slices,
                sam_points_per_side=self._sam_points_per_side,
                sam_pred_iou_thresh=self._sam_pred_iou_thresh,
            )

        self._logging.info(
            "SAM module init: use_refine=%s, use_fn_recovery=%s, backend=%s, model=%s",
            self.use_sam_refinement,
            self.use_fn_recovery,
            self._sam_backend,
            self._sam_model_type,
        )

    def set_dataset(self, dataset, labeled_num: int, labeled_bs: int, batch_size: int) -> None:
        """Provide the training dataset for prototype computation (FN recovery).
        Call after trainer.load_dataset().
        """
        self._dataset = dataset
        self._labeled_num = labeled_num
        self._labeled_bs = labeled_bs
        self._batch_size = batch_size
        self._labeled_loader_cache = None  # reset on dataset change

    def run_pipeline(
        self,
        unlabeled_volume_batch,
        teacher_logits,
        pseudo_label,
        epoch_num: int,
        current_iter: int,
        tensorboard_writer,
        ema_model,
        ground_truth_label=None,
    ):
        """Run SAM refinement + FN recovery on a batch of unlabeled volumes.

        Returns:
            refined_pseudo_label: refined label tensor (same shape as pseudo_label)
            reliability_map: per-voxel reliability weights
            sam_iou_scores: per-slice SAM IoU scores (or None)
            ground_truth_label: optional GT tensor for visualization (same shape as pseudo_label)
        """
        refined_pseudo_label = pseudo_label.clone()
        sam_iou_scores = None
        reliability_map = torch.ones_like(
            pseudo_label, dtype=torch.float32, device=self._device
        )

        # --- SAM refinement ---
        if self.use_sam_refinement and self._sam_refiner is not None:
            refine_start = time.time()
            run_refine = (
                self._cached_refined_label is None
                or current_iter % max(self._sam_refine_interval, 1) == 0
            )
            if run_refine:
                refined_labels, iou_scores_list = [], []
                for b in range(unlabeled_volume_batch.shape[0]):
                    vol_np = unlabeled_volume_batch[b, 0].detach().cpu().numpy()
                    pl_np = pseudo_label[b].detach().cpu().numpy()
                    refined_np, iou_scores_np = self._sam_refiner.refine_volume(
                        vol_np, pl_np, self._num_classes - 1
                    )
                    refined_labels.append(refined_np.astype(np.int16))
                    iou_scores_list.append(iou_scores_np.astype(np.float32))
                refined_pseudo_label = (
                    torch.from_numpy(np.stack(refined_labels)).long().to(self._device)
                )
                sam_iou_scores = (
                    torch.from_numpy(np.stack(iou_scores_list)).float().to(self._device)
                )
                self._cached_refined_label = refined_pseudo_label.detach().clone()
                self._cached_iou_scores = sam_iou_scores.detach().clone()
            else:
                if (
                    self._cached_refined_label.shape == pseudo_label.shape
                    and self._cached_refined_label.device == pseudo_label.device
                ):
                    refined_pseudo_label = self._cached_refined_label.detach().clone()
                    sam_iou_scores = (
                        self._cached_iou_scores.detach().clone()
                        if self._cached_iou_scores is not None
                        else None
                    )
                else:
                    refined_pseudo_label = pseudo_label.clone()
                    sam_iou_scores = None

            reliability_map = self._confidence_estimator.compute_reliability(
                teacher_logits=teacher_logits,
                pseudo_label=pseudo_label,
                sam_refined_label=refined_pseudo_label,
                sam_iou_scores=sam_iou_scores,
            )
            agreement = (pseudo_label == refined_pseudo_label).float().mean().item()
            refine_cost = time.time() - refine_start
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar('sam/refinement_time', refine_cost, current_iter)
                tensorboard_writer.add_scalar('sam/agreement_rate', agreement, current_iter)
                if sam_iou_scores is not None:
                    tensorboard_writer.add_scalar(
                        'sam/avg_iou_score', sam_iou_scores.mean().item(), current_iter
                    )
            if run_refine:
                self._maybe_visualize(
                    unlabeled_volume_batch=unlabeled_volume_batch,
                    pseudo_label=pseudo_label,
                    refined_pseudo_label=refined_pseudo_label,
                    ground_truth_label=ground_truth_label,
                    reliability_map=reliability_map,
                    sam_iou_scores=sam_iou_scores,
                    current_iter=current_iter,
                    tensorboard_writer=tensorboard_writer,
                )

        # --- FN recovery ---
        if (
            self.use_fn_recovery
            and self._fn_recoverer is not None
            and epoch_num >= self._fn_recovery_start_epoch
            and current_iter % max(self._fn_recovery_interval, 1) == 0
        ):
            if (
                self._class_prototypes is None
                or epoch_num % max(self._prototype_update_interval, 1) == 0
            ):
                labeled_loader = self._get_labeled_loader()
                feature_layer = self._sam_feature_layer or self._teacher_feature_layer_name
                self._class_prototypes = self._fn_recoverer.compute_prototypes(
                    labeled_loader=labeled_loader,
                    teacher_model=ema_model,
                    feature_layer=feature_layer,
                )

            recovered_voxels = 0
            if self._class_prototypes:
                for b in range(unlabeled_volume_batch.shape[0]):
                    teacher_feat = self._extract_teacher_features(
                        unlabeled_volume_batch[b:b + 1], ema_model
                    )
                    if teacher_feat is None:
                        continue
                    recovered_np = self._fn_recoverer.recover(
                        volume=unlabeled_volume_batch[b, 0].detach().cpu().numpy(),
                        pseudo_label=refined_pseudo_label[b].detach().cpu().numpy(),
                        teacher_features=teacher_feat,
                        class_prototypes=self._class_prototypes,
                    )
                    recovered_tensor = torch.from_numpy(recovered_np).long().to(self._device)
                    recovered_voxels += torch.logical_and(
                        recovered_tensor != refined_pseudo_label[b],
                        refined_pseudo_label[b] == 0,
                    ).sum().item()
                    refined_pseudo_label[b] = recovered_tensor
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    'sam/fn_recovered_voxels', recovered_voxels, current_iter
                )

        return refined_pseudo_label, reliability_map, sam_iou_scores

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_labeled_loader(self):
        if self._labeled_loader_cache is not None:
            return self._labeled_loader_cache
        if self._dataset is None:
            return None
        labeled_indices = list(range(min(self._labeled_num, len(self._dataset))))
        subset = Subset(self._dataset, labeled_indices)
        self._labeled_loader_cache = DataLoader(
            subset,
            batch_size=max(1, min(self._labeled_bs, self._batch_size)),
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        return self._labeled_loader_cache

    def _resolve_feature_module(self, ema_model):
        if self._teacher_feature_module is not None:
            return self._teacher_feature_module
        if ema_model is None:
            return None

        named = dict(ema_model.named_modules())
        if self._sam_feature_layer and self._sam_feature_layer in named:
            self._teacher_feature_module = named[self._sam_feature_layer]
            self._teacher_feature_layer_name = self._sam_feature_layer
            return self._teacher_feature_module

        layer_name, module = resolve_feature_module_3d(ema_model, self._backbone)
        self._teacher_feature_module = module
        self._teacher_feature_layer_name = layer_name
        return self._teacher_feature_module

    def _extract_teacher_features(self, x, ema_model):
        if ema_model is None:
            return None
        feature_module = self._resolve_feature_module(ema_model)
        if feature_module is None:
            with torch.no_grad():
                outputs = ema_model(x)
            if isinstance(outputs, (tuple, list)):
                for item in outputs[1:]:
                    if torch.is_tensor(item):
                        return item.unsqueeze(0) if item.dim() == 4 else item
            return None

        feature_dict = {}

        def hook_fn(_module, _inputs, output):
            if isinstance(output, (tuple, list)):
                for item in output:
                    if torch.is_tensor(item):
                        feature_dict['feat'] = item
                        return
            elif torch.is_tensor(output):
                feature_dict['feat'] = output

        handle = feature_module.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = ema_model(x)
        handle.remove()
        feat = feature_dict.get('feat', None)
        if feat is not None and feat.dim() == 4:
            feat = feat.unsqueeze(0)
        return feat

    def _maybe_visualize(
        self,
        unlabeled_volume_batch,
        pseudo_label,
        refined_pseudo_label,
        ground_truth_label,
        reliability_map,
        sam_iou_scores,
        current_iter: int,
        tensorboard_writer,
    ):
        if not self._visualize_refine:
            return
        if tensorboard_writer is None:
            return
        if self._visualize_interval <= 0:
            return
        if current_iter % self._visualize_interval != 0:
            return
        if self._last_visualize_iter == current_iter:
            return
        if pseudo_label.shape != refined_pseudo_label.shape:
            return

        case_idx = min(
            max(self._visualize_case_index, 0),
            int(unlabeled_volume_batch.shape[0]) - 1,
        )
        diff_mask = (pseudo_label[case_idx] != refined_pseudo_label[case_idx]).float()
        changed_per_slice = diff_mask.sum(dim=(1, 2))
        if (
            self._visualize_slice_strategy == 'middle'
            or torch.sum(changed_per_slice).item() == 0
        ):
            slice_idx = pseudo_label.shape[1] // 2
        else:
            slice_idx = int(torch.argmax(changed_per_slice).item())

        image_slice = normalize_slice_to_01(unlabeled_volume_batch[case_idx, 0, slice_idx])
        class_denom = float(max(self._num_classes - 1, 1))
        pseudo_slice = pseudo_label[case_idx, slice_idx].detach().float().cpu() / class_denom
        refined_slice = refined_pseudo_label[case_idx, slice_idx].detach().float().cpu() / class_denom
        diff_slice = diff_mask[slice_idx].detach().float().cpu()
        reliability_slice = reliability_map[case_idx, slice_idx].detach().float().cpu()
        has_gt = (
            ground_truth_label is not None
            and ground_truth_label.shape == pseudo_label.shape
        )
        if has_gt:
            gt_slice = (
                ground_truth_label[case_idx, slice_idx].detach().float().cpu() / class_denom
            )
            panel_items = [
                image_slice, pseudo_slice, refined_slice, gt_slice, diff_slice, reliability_slice
            ]
            grid_nrow = 6
            image_tag = 'sam_refine/axial_compare_input_pseudo_refined_gt_diff_reliability'
        else:
            panel_items = [image_slice, pseudo_slice, refined_slice, diff_slice, reliability_slice]
            grid_nrow = 5
            image_tag = 'sam_refine/axial_compare_input_pseudo_refined_diff_reliability'

        panel = torch.stack(panel_items, dim=0).unsqueeze(1).repeat(1, 3, 1, 1)
        grid = make_grid(panel, nrow=grid_nrow, normalize=False)
        tensorboard_writer.add_image(
            image_tag,
            grid,
            current_iter,
        )

        changed_ratio = (pseudo_label != refined_pseudo_label).float().mean().item()
        changed_case = int(diff_mask.sum().item())
        tensorboard_writer.add_scalar('sam_refine/changed_voxel_ratio', changed_ratio, current_iter)
        tensorboard_writer.add_scalar('sam_refine/changed_voxels_case', changed_case, current_iter)
        tensorboard_writer.add_scalar('sam_refine/selected_slice_index', slice_idx, current_iter)
        if sam_iou_scores is not None and sam_iou_scores.numel() > 0:
            tensorboard_writer.add_scalar(
                'sam_refine/selected_slice_iou_score',
                sam_iou_scores[case_idx, slice_idx].item(),
                current_iter,
            )

        if self._save_compare_npz:
            os.makedirs(self._vis_output_dir, exist_ok=True)
            save_path = os.path.join(
                self._vis_output_dir,
                f"iter_{current_iter:06d}_case_{case_idx:02d}_slice_{slice_idx:03d}.npz",
            )
            save_dict = {
                'image': unlabeled_volume_batch[case_idx, 0].detach().cpu().numpy().astype(np.float32),
                'pseudo_before': pseudo_label[case_idx].detach().cpu().numpy().astype(np.int16),
                'pseudo_after': refined_pseudo_label[case_idx].detach().cpu().numpy().astype(np.int16),
                'reliability': reliability_map[case_idx].detach().cpu().numpy().astype(np.float32),
                'sam_iou_scores': (
                    sam_iou_scores[case_idx].detach().cpu().numpy().astype(np.float32)
                    if sam_iou_scores is not None
                    else np.array([])
                ),
                'slice_index': np.array([slice_idx], dtype=np.int32),
                'changed_ratio': np.array([changed_ratio], dtype=np.float32),
            }
            if has_gt:
                save_dict['ground_truth'] = (
                    ground_truth_label[case_idx].detach().cpu().numpy().astype(np.int16)
                )
            np.savez_compressed(save_path, **save_dict)

        self._last_visualize_iter = current_iter
