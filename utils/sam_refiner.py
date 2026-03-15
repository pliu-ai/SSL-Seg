"""
Multi-Planar SAM Refinement for 3D pseudo labels.
"""

from __future__ import annotations

import numpy as np
import cv2
import torch
import torch.nn.functional as F

try:
    from segment_anything import sam_model_registry, SamPredictor
except Exception:
    sam_model_registry = None
    SamPredictor = None


SAM_REFINER_DEFAULTS = {
    "min_area": 100,
    "expand_ratio": 0.05,
    "max_expansion_ratio": 1.5,
    "use_multiplanar": True,
    "use_propagation": False,
}


class MultiPlanarSAMRefiner:
    def __init__(
        self,
        sam_checkpoint,
        model_type="vit_h",
        sam_backend="sam",
        device="cuda",
        min_area=100,
        expand_ratio=0.05,
        max_expansion_ratio=1.5,
        use_multiplanar=True,
        use_propagation=False,
    ):
        if SamPredictor is None or sam_model_registry is None:
            raise RuntimeError(
                "segment-anything is not available. Install it with: pip install segment-anything"
            )
        self.sam_backend = str(sam_backend).lower()
        if self.sam_backend not in ("sam", "medsam"):
            raise ValueError(
                f"Unsupported sam_backend={sam_backend}. Expected one of: sam, medsam."
            )
        if self.sam_backend == "medsam" and model_type != "vit_b":
            raise ValueError(
                f"MedSAM currently supports only model_type='vit_b', got '{model_type}'."
            )
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device)
        self.predictor = (
            _MedSAMBoxPredictor(sam, device=device)
            if self.sam_backend == "medsam"
            else SamPredictor(sam)
        )
        self.min_area = int(min_area)
        self.expand_ratio = float(expand_ratio)
        self.max_expansion_ratio = float(max_expansion_ratio)
        self.use_multiplanar = bool(use_multiplanar)
        self.use_propagation = bool(use_propagation)

    def refine_volume(self, volume, pseudo_label, num_classes):
        volume = np.asarray(volume)
        pseudo_label = np.asarray(pseudo_label).astype(np.int16)
        assert volume.ndim == 3, f"Expected [D,H,W], got {volume.shape}"
        assert pseudo_label.shape == volume.shape

        num_classes = int(num_classes)
        if num_classes <= 0:
            return pseudo_label.copy(), np.zeros(volume.shape[0], dtype=np.float32)

        axes = [0]
        if self.use_multiplanar:
            axes.extend([1, 2])

        refined_by_axis = []
        axial_iou_sum = np.zeros(volume.shape[0], dtype=np.float32)
        axial_iou_cnt = np.zeros(volume.shape[0], dtype=np.float32)

        for axis in axes:
            refined_axis, iou_sum_axis, iou_cnt_axis = self._refine_along_axis(
                volume, pseudo_label, num_classes, axis
            )
            refined_by_axis.append(refined_axis)
            if axis == 0:
                axial_iou_sum += iou_sum_axis
                axial_iou_cnt += iou_cnt_axis

        if len(refined_by_axis) == 1:
            refined_label = refined_by_axis[0]
        else:
            refined_label = pseudo_label.copy()
            vote_stack = np.stack(refined_by_axis, axis=0)
            max_votes = np.zeros_like(refined_label, dtype=np.int16)
            voted_class = np.zeros_like(refined_label, dtype=np.int16)
            for cls in range(1, num_classes + 1):
                votes = np.sum(vote_stack == cls, axis=0).astype(np.int16)
                update = votes > max_votes
                max_votes[update] = votes[update]
                voted_class[update] = cls
            # Majority voting across 3 planes.
            keep_mask = max_votes >= 2
            refined_label[keep_mask] = voted_class[keep_mask]

        iou_scores = np.zeros(volume.shape[0], dtype=np.float32)
        valid = axial_iou_cnt > 0
        iou_scores[valid] = axial_iou_sum[valid] / axial_iou_cnt[valid]
        return refined_label.astype(np.int16), iou_scores

    def _prepare_slice_image(self, slice_2d):
        arr = np.asarray(slice_2d).astype(np.float32)
        if arr.size == 0:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        lo = np.percentile(arr, 0.5)
        hi = np.percentile(arr, 99.5)
        if hi <= lo:
            hi = lo + 1.0
        arr = np.clip(arr, lo, hi)
        arr = ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)
        return np.repeat(arr[..., None], 3, axis=2)

    def _area_constrained_refine(self, original_mask, sam_mask):
        """Guard against SAM over-segmentation by checking the area expansion ratio.

        If the SAM mask is more than ``self.max_expansion_ratio`` times larger
        than the original pseudo mask, SAM has expanded too aggressively.
        In that case fall back to the intersection of the two masks so we keep
        only the region both sources agree on.
        """
        original_area = int(original_mask.sum())
        if original_area == 0:
            return original_mask.astype(np.uint8)
        sam_area = int(sam_mask.sum())
        if sam_area > original_area * self.max_expansion_ratio:
            return np.logical_and(original_mask, sam_mask).astype(np.uint8)
        return sam_mask.astype(np.uint8)

    def _build_mask_input(self, component_mask):
        """Convert a binary component mask to low-res logits for SAM mask prompt.

        Args:
            component_mask: [H, W] bool/uint8 array for one connected component.

        Returns:
            np.ndarray of shape [1, 256, 256] with logit values:
            foreground ≈ +4.0, background ≈ −4.0.  This matches the
            format expected by both SamPredictor.predict(mask_input=...)
            and _MedSAMBoxPredictor.predict(mask_input=...).
        """
        mask_f32 = component_mask.astype(np.float32)
        # Resize to SAM's low-res mask size (256×256).
        mask_resized = cv2.resize(mask_f32, (256, 256), interpolation=cv2.INTER_LINEAR)
        # Map [0,1] → logits: foreground≈+4, background≈−4.
        mask_logits = (mask_resized * 8.0) - 4.0
        return mask_logits[None]  # [1, 256, 256]

    def _refine_single_slice(self, slice_image, slice_mask, prev_mask=None):
        binary = (slice_mask > 0).astype(np.uint8)
        if binary.sum() == 0:
            return np.zeros_like(binary, dtype=np.uint8), 0.0

        self.predictor.set_image(slice_image)
        num_cc, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        refined = np.zeros_like(binary, dtype=np.uint8)
        iou_list = []

        for cc_idx in range(1, num_cc):
            area = int(stats[cc_idx, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            x = int(stats[cc_idx, cv2.CC_STAT_LEFT])
            y = int(stats[cc_idx, cv2.CC_STAT_TOP])
            w = int(stats[cc_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[cc_idx, cv2.CC_STAT_HEIGHT])
            expand_x = int(self.expand_ratio * w)
            expand_y = int(self.expand_ratio * h)
            x0 = max(0, x - expand_x)
            y0 = max(0, y - expand_y)
            x1 = min(binary.shape[1] - 1, x + w + expand_x)
            y1 = min(binary.shape[0] - 1, y + h + expand_y)
            bbox = np.array([x0, y0, x1, y1], dtype=np.float32)

            # Mask prompt: encode the coarse pseudo mask as logits so SAM
            # refines the boundary rather than searching the whole bbox.
            mask_input = self._build_mask_input(labels == cc_idx)

            point_coords = None
            point_labels = None
            if self.use_propagation and prev_mask is not None:
                point_coords, point_labels = self._prompt_propagation(prev_mask, labels == cc_idx)

            try:
                masks, scores, _ = self.predictor.predict(
                    point_coords=point_coords,
                    point_labels=point_labels,
                    box=bbox,
                    mask_input=mask_input,
                    multimask_output=True,
                )
            except Exception:
                masks, scores = None, None

            if masks is None or len(masks) == 0:
                continue

            best_idx = int(np.argmax(scores))
            best_mask = self._area_constrained_refine(
                original_mask=(labels == cc_idx),
                sam_mask=masks[best_idx] > 0,
            )
            refined = np.logical_or(refined > 0, best_mask > 0).astype(np.uint8)
            iou_list.append(float(scores[best_idx]))

        if refined.sum() == 0:
            # Fallback to original coarse pseudo mask.
            refined = binary
        avg_iou = float(np.mean(iou_list)) if len(iou_list) > 0 else 0.0
        return refined, avg_iou

    def _prompt_propagation(self, prev_mask, current_mask):
        prev = prev_mask.astype(bool)
        curr = current_mask.astype(bool)
        inter = np.logical_and(prev, curr)
        source = inter if inter.sum() > 0 else prev
        ys, xs = np.where(source)
        if len(xs) == 0:
            return None, None
        cx = float(xs.mean())
        cy = float(ys.mean())
        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        return point_coords, point_labels

    def _refine_along_axis(self, volume, pseudo_label, num_classes, axis):
        volume_axis = np.moveaxis(volume, axis, 0)
        pseudo_axis = np.moveaxis(pseudo_label, axis, 0)

        refined_axis = pseudo_axis.copy()
        best_score = np.zeros_like(pseudo_axis, dtype=np.float32)

        # Always report IoU statistics in axial index space.
        axial_len = volume.shape[0]
        iou_sum = np.zeros(axial_len, dtype=np.float32)
        iou_cnt = np.zeros(axial_len, dtype=np.float32)

        for cls_id in range(1, num_classes + 1):
            prev_refined = None
            for sidx in range(volume_axis.shape[0]):
                mask_2d = (pseudo_axis[sidx] == cls_id).astype(np.uint8)
                if mask_2d.sum() == 0:
                    prev_refined = None
                    continue
                slice_image = self._prepare_slice_image(volume_axis[sidx])
                refined_2d, avg_iou = self._refine_single_slice(
                    slice_image=slice_image,
                    slice_mask=mask_2d,
                    prev_mask=prev_refined,
                )
                prev_refined = refined_2d

                refined_bool = refined_2d > 0
                if not refined_bool.any():
                    continue
                score = max(avg_iou, 1e-4)
                score_map = np.zeros_like(mask_2d, dtype=np.float32)
                score_map[refined_bool] = score
                update = score_map > best_score[sidx]
                refined_axis[sidx][update] = cls_id
                best_score[sidx][update] = score_map[update]

                if axis == 0:
                    iou_sum[sidx] += avg_iou
                    iou_cnt[sidx] += 1.0

        return np.moveaxis(refined_axis, 0, axis), iou_sum, iou_cnt


class _MedSAMBoxPredictor:
    """Minimal MedSAM box predictor compatible with SamPredictor-style calls."""

    def __init__(self, sam_model, device="cuda"):
        self.model = sam_model
        self.device = torch.device(device)
        self.model.eval()
        self._img_embed = None
        self._orig_hw = None
        self._input_size = 1024

    def set_image(self, image):
        image_np = np.asarray(image)
        if image_np.ndim != 3 or image_np.shape[2] != 3:
            raise ValueError(f"Expected image [H,W,3], got {image_np.shape}")

        h, w = image_np.shape[:2]
        self._orig_hw = (h, w)
        image_1024 = cv2.resize(
            image_np, (self._input_size, self._input_size), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)
        image_1024 = image_1024 / 255.0
        img_tensor = (
            torch.from_numpy(image_1024)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device, dtype=torch.float32)
        )
        with torch.no_grad():
            self._img_embed = self.model.image_encoder(img_tensor)

    def predict(
        self,
        point_coords=None,
        point_labels=None,
        box=None,
        mask_input=None,
        multimask_output=True,
    ):
        if self._img_embed is None or self._orig_hw is None:
            raise RuntimeError("Call set_image() before predict().")
        if box is None:
            raise ValueError("MedSAM predictor currently requires a box prompt.")

        h, w = self._orig_hw
        box_np = np.asarray(box, dtype=np.float32).reshape(1, 4)
        scale = np.array(
            [self._input_size / max(w, 1), self._input_size / max(h, 1)] * 2,
            dtype=np.float32,
        )
        box_1024 = box_np * scale[None, :]
        box_t = torch.from_numpy(box_1024).to(self.device, dtype=torch.float32)[:, None, :]

        points = None
        if point_coords is not None and point_labels is not None:
            pt_np = np.asarray(point_coords, dtype=np.float32).reshape(-1, 2)
            lb_np = np.asarray(point_labels, dtype=np.int64).reshape(-1)
            pt_scale = np.array(
                [self._input_size / max(w, 1), self._input_size / max(h, 1)],
                dtype=np.float32,
            )
            pt_1024 = pt_np * pt_scale[None, :]
            pt_t = torch.from_numpy(pt_1024).to(self.device, dtype=torch.float32)[None, ...]
            lb_t = torch.from_numpy(lb_np).to(self.device, dtype=torch.int64)[None, ...]
            points = (pt_t, lb_t)

        # mask_input: numpy [1,256,256] logits → tensor [1,1,256,256]
        masks_t = None
        if mask_input is not None:
            m = np.asarray(mask_input, dtype=np.float32)
            if m.ndim == 2:
                m = m[None, None]  # [H,W] → [1,1,H,W]
            elif m.ndim == 3:
                m = m[None]        # [1,H,W] → [1,1,H,W]
            masks_t = torch.from_numpy(m).to(self.device, dtype=torch.float32)

        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=points,
                boxes=box_t,
                masks=masks_t,
            )
            low_res_logits, iou_pred = self.model.mask_decoder(
                image_embeddings=self._img_embed,
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=bool(multimask_output),
            )
            logits = F.interpolate(
                low_res_logits,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            probs = torch.sigmoid(logits)

        masks_np = (probs[0] > 0.5).detach().cpu().numpy().astype(np.uint8)
        scores_np = iou_pred[0].detach().cpu().numpy().astype(np.float32)
        logits_np = logits[0].detach().cpu().numpy().astype(np.float32)
        return masks_np, scores_np, logits_np
