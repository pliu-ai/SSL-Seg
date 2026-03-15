"""
SAM-Guided False Negative Recovery.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
except Exception:
    sam_model_registry = None
    SamAutomaticMaskGenerator = None


FN_RECOVERY_DEFAULTS = {
    "iou_threshold": 0.1,
    "similarity_threshold": 0.7,
    "min_area": 100,
    "consistency_slices": 3,
    "sam_points_per_side": 32,
    "sam_pred_iou_thresh": 0.86,
}


class SAMFalseNegativeRecovery:
    def __init__(
        self,
        sam_checkpoint,
        model_type="vit_h",
        sam_backend="sam",
        device="cuda",
        iou_threshold=0.1,
        similarity_threshold=0.7,
        min_area=100,
        consistency_slices=3,
        sam_points_per_side=32,
        sam_pred_iou_thresh=0.86,
        sam_stability_score_thresh=0.92,
    ):
        if SamAutomaticMaskGenerator is None or sam_model_registry is None:
            raise RuntimeError(
                "segment-anything is not available. Install it with: pip install segment-anything"
            )
        self.sam_backend = str(sam_backend).lower()
        if self.sam_backend not in ("sam", "medsam"):
            raise ValueError(
                f"Unsupported sam_backend={sam_backend}. Expected one of: sam, medsam."
            )
        if self.sam_backend == "medsam":
            raise NotImplementedError(
                "FN recovery with MedSAM is not supported. "
                "MedSAM was fine-tuned with box prompts only; "
                "SamAutomaticMaskGenerator uses point-grid prompts and gives unreliable results. "
                "Set use_fn_recovery: false in your config when using sam_backend: medsam."
            )
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=int(sam_points_per_side),
            pred_iou_thresh=float(sam_pred_iou_thresh),
            stability_score_thresh=float(sam_stability_score_thresh),
            min_mask_region_area=int(min_area),
        )
        self.iou_threshold = float(iou_threshold)
        self.similarity_threshold = float(similarity_threshold)
        self.min_area = int(min_area)
        self.consistency_slices = int(consistency_slices)

    def recover(self, volume, pseudo_label, teacher_features, class_prototypes):
        volume = np.asarray(volume)
        pseudo_label = np.asarray(pseudo_label).astype(np.int16)
        if not class_prototypes:
            return pseudo_label

        feat = self._format_teacher_features(teacher_features, depth=pseudo_label.shape[0])
        if feat is None:
            return pseudo_label

        recovered_label = pseudo_label.copy()
        candidates = np.zeros_like(pseudo_label, dtype=np.int16)
        depth = pseudo_label.shape[0]

        for d in range(depth):
            slice_image = self._prepare_slice_image(volume[d])
            try:
                proposals = self.mask_generator.generate(slice_image)
            except Exception:
                proposals = []
            if len(proposals) == 0:
                continue
            feature_slice = feat[d]  # [C, Hf, Wf]

            for proposal in proposals:
                mask = proposal.get("segmentation", None)
                if mask is None:
                    continue
                mask = mask.astype(bool)
                area = int(mask.sum())
                if area < self.min_area:
                    continue

                pseudo_fg = pseudo_label[d] > 0
                inter = np.logical_and(mask, pseudo_fg).sum()
                union = np.logical_or(mask, pseudo_fg).sum()
                iou = float(inter) / float(union + 1e-6)
                if iou >= self.iou_threshold:
                    continue

                cls_id, similarity = self._match_prototype(mask, feature_slice, class_prototypes)
                if cls_id <= 0 or similarity < self.similarity_threshold:
                    continue
                candidates[d][mask] = cls_id

        candidates = self._apply_slice_consistency(candidates, class_prototypes.keys())
        write_mask = np.logical_and(candidates > 0, recovered_label == 0)
        recovered_label[write_mask] = candidates[write_mask]
        return recovered_label

    def compute_prototypes(self, labeled_loader, teacher_model, feature_layer=""):
        if labeled_loader is None:
            return {}
        teacher_model.eval()
        feature_cache = {}
        module = self._resolve_feature_module(teacher_model, feature_layer=feature_layer)
        if module is None:
            return {}

        def hook_fn(_module, _inputs, output):
            if isinstance(output, (list, tuple)):
                for item in output:
                    if torch.is_tensor(item):
                        feature_cache["feat"] = item
                        return
            elif torch.is_tensor(output):
                feature_cache["feat"] = output

        handle = module.register_forward_hook(hook_fn)
        proto_bank = {}

        with torch.no_grad():
            for sampled_batch in labeled_loader:
                if not isinstance(sampled_batch, dict):
                    continue
                image = sampled_batch["image"].to(next(teacher_model.parameters()).device)
                label = sampled_batch["label"].to(image.device)
                if label.dim() == 5 and label.shape[1] > 1:
                    label = torch.argmax(label, dim=1)
                elif label.dim() == 5:
                    label = label[:, 0].long()
                else:
                    label = label.long()

                feature_cache.clear()
                _ = teacher_model(image)
                feat = feature_cache.get("feat", None)
                if feat is None or feat.dim() != 5:
                    continue
                # [B, D, H, W] -> [B, 1, Df, Hf, Wf]
                label_resized = F.interpolate(
                    label.unsqueeze(1).float(),
                    size=feat.shape[2:],
                    mode="nearest",
                ).squeeze(1).long()

                cls_max = int(label.max().item())
                for cls_id in range(1, cls_max + 1):
                    class_mask = (label_resized == cls_id).float()
                    denom = class_mask.sum()
                    if denom <= 0:
                        continue
                    pooled = (feat * class_mask.unsqueeze(1)).sum(dim=(0, 2, 3, 4)) / (denom + 1e-6)
                    proto_bank.setdefault(cls_id, []).append(pooled.detach().cpu())

        handle.remove()

        prototypes = {}
        for cls_id, vecs in proto_bank.items():
            if len(vecs) == 0:
                continue
            mean_vec = torch.stack(vecs, dim=0).mean(dim=0)
            prototypes[cls_id] = F.normalize(mean_vec, dim=0)
        return prototypes

    def _match_prototype(self, proposal_mask, feature_slice, class_prototypes):
        # feature_slice: [C, Hf, Wf]
        mask_tensor = torch.from_numpy(proposal_mask.astype(np.float32))[None, None]  # [1,1,H,W]
        feat_h, feat_w = feature_slice.shape[-2], feature_slice.shape[-1]
        mask_resized = F.interpolate(mask_tensor, size=(feat_h, feat_w), mode="nearest")[0, 0]
        mask_bool = mask_resized > 0.5
        if mask_bool.sum() == 0:
            return 0, 0.0

        pooled = (feature_slice * mask_bool.unsqueeze(0).float()).sum(dim=(1, 2))
        pooled = pooled / (mask_bool.sum().float() + 1e-6)
        pooled = F.normalize(pooled, dim=0)

        best_cls = 0
        best_sim = -1.0
        for cls_id, proto in class_prototypes.items():
            proto_t = proto.to(pooled.device)
            proto_t = F.normalize(proto_t, dim=0)
            sim = float(F.cosine_similarity(pooled, proto_t, dim=0).item())
            if sim > best_sim:
                best_sim = sim
                best_cls = int(cls_id)
        return best_cls, best_sim

    def _apply_slice_consistency(self, candidate_labels, class_ids):
        depth = candidate_labels.shape[0]
        if self.consistency_slices <= 1:
            return candidate_labels

        for cls_id in class_ids:
            cls_id = int(cls_id)
            if cls_id <= 0:
                continue
            presence = np.array(
                [np.any(candidate_labels[d] == cls_id) for d in range(depth)],
                dtype=bool,
            )
            keep = np.zeros_like(presence, dtype=bool)
            idx = 0
            while idx < depth:
                if not presence[idx]:
                    idx += 1
                    continue
                start = idx
                while idx < depth and presence[idx]:
                    idx += 1
                end = idx
                if (end - start) >= self.consistency_slices:
                    keep[start:end] = True
            for d in range(depth):
                if not keep[d]:
                    candidate_labels[d][candidate_labels[d] == cls_id] = 0
        return candidate_labels

    def _resolve_feature_module(self, model, feature_layer=""):
        named = dict(model.named_modules())
        if feature_layer and feature_layer in named:
            return named[feature_layer]
        preferred = [
            "center",
            "block_five",
            "encoder",
            "conv_blocks_context.4",
            "down4",
        ]
        for name in preferred:
            if name in named:
                return named[name]
        for _, module in reversed(list(model.named_modules())):
            if isinstance(module, nn.Conv3d):
                return module
        return None

    def _prepare_slice_image(self, slice_2d):
        arr = np.asarray(slice_2d).astype(np.float32)
        lo = np.percentile(arr, 0.5)
        hi = np.percentile(arr, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip(arr, lo, hi)
        arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=2)

    def _format_teacher_features(self, teacher_features, depth):
        if teacher_features is None:
            return None
        if not torch.is_tensor(teacher_features):
            teacher_features = torch.as_tensor(teacher_features)
        feat = teacher_features.detach().float().cpu()
        # [1, C, D, H, W] -> [D, C, H, W]
        if feat.dim() == 5 and feat.shape[0] == 1:
            feat = feat[0].permute(1, 0, 2, 3)
        elif feat.dim() == 5 and feat.shape[1] >= 1:
            # [B, C, D, H, W] use first sample
            feat = feat[0].permute(1, 0, 2, 3)
        elif feat.dim() == 4 and feat.shape[0] == depth:
            # [D, C, H, W]
            pass
        elif feat.dim() == 4 and feat.shape[1] == depth:
            # [C, D, H, W] -> [D, C, H, W]
            feat = feat.permute(1, 0, 2, 3)
        else:
            return None
        return feat
