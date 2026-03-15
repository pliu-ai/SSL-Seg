"""
Dual-Source Confidence Estimation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


class DualSourceConfidence:
    def __init__(self, alpha=0.5, temperature=1.0, spatial_smooth_sigma=2.0):
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.sigma = float(spatial_smooth_sigma)

    def compute_reliability(
        self,
        teacher_logits,
        pseudo_label,
        sam_refined_label,
        sam_iou_scores=None,
    ):
        probs = F.softmax(teacher_logits / max(self.temperature, 1e-6), dim=1)
        c_int = probs.max(dim=1)[0]

        agreement = (pseudo_label == sam_refined_label).float()
        if sam_iou_scores is not None:
            iou_weight = sam_iou_scores[:, :, None, None].expand_as(agreement)
            c_ext = agreement * iou_weight
        else:
            c_ext = agreement

        reliability = self.alpha * c_int + (1.0 - self.alpha) * c_ext
        reliability = reliability.clamp(0.0, 1.0)

        if self.sigma > 0:
            smoothed = []
            for b in range(reliability.shape[0]):
                rel_np = reliability[b].detach().cpu().numpy()
                rel_np = gaussian_filter(rel_np, sigma=self.sigma)
                smoothed.append(torch.from_numpy(rel_np).float())
            reliability = torch.stack(smoothed, dim=0).to(teacher_logits.device)
            reliability = reliability.clamp(0.0, 1.0)

        return reliability
