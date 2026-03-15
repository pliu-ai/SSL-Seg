"""
TrainingLogger: centralised scalar and image logging for semi-supervised training.

Extracted from SemiSupervisedTrainer3D to isolate TensorBoard / Python-logging
calls that were scattered across the trainer and repeated verbatim in every
_train_* method.
"""
from typing import Optional

import torch
from torchvision.utils import make_grid


class TrainingLogger:
    """Wraps TensorBoard SummaryWriter + Python logging for training loops.

    Create one instance in ``trainer.initialize()`` after the SummaryWriter is
    ready, then call the methods inside training loops:

    Usage::

        # in initialize():
        self.logger = TrainingLogger(self.tensorboard_writer, self.logging)

        # every iteration:
        self.current_lr = self.logger.log_iter_scalars(
            current_iter=self.current_iter,
            optimizer=self.optimizer,
            loss=self.loss, loss_ce=self.loss_ce, loss_dice=self.loss_dice,
            consistency_loss=self.consistency_loss,
            consistency_weight=self.consistency_weight,
        )

        # every show_img_freq iterations:
        if self.current_iter % self.show_img_freq == 0:
            self.logger.log_train_images_3d(
                self.current_iter, volume_batch, outputs_soft, label_batch
            )
    """

    def __init__(self, writer, logging):
        """
        Args:
            writer: TensorBoard SummaryWriter instance.
            logging: Python logging module / logger instance.
        """
        self._writer = writer
        self._logging = logging

    # ------------------------------------------------------------------
    # Scalar logging
    # ------------------------------------------------------------------

    def log_iter_scalars(
        self,
        current_iter: int,
        optimizer,
        loss,
        loss_ce,
        loss_dice,
        consistency_loss=None,
        consistency_weight=None,
    ) -> float:
        """Log per-iteration scalars and return the current learning rate.

        Replaces ``_add_information_to_writer`` in the trainer.

        Returns:
            Current learning rate read from the first param group.
        """
        current_lr: float = optimizer.param_groups[0]['lr']

        self._writer.add_scalar('info/lr', current_lr, current_iter)
        self._writer.add_scalar('info/total_loss', loss, current_iter)
        self._writer.add_scalar('info/loss_ce', loss_ce, current_iter)
        self._writer.add_scalar('info/loss_dice', loss_dice, current_iter)
        self._writer.add_scalar('loss/loss', loss, current_iter)

        self._logging.info(
            'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f',
            current_iter, loss.item(), loss_ce.item(), loss_dice.item(),
        )

        if consistency_loss:
            self._writer.add_scalar(
                'info/consistency_loss', consistency_loss, current_iter
            )
        if consistency_weight:
            self._writer.add_scalar(
                'info/consistency_weight', consistency_weight, current_iter
            )

        return current_lr

    # ------------------------------------------------------------------
    # Image logging
    # ------------------------------------------------------------------

    def log_train_images_3d(
        self,
        current_iter: int,
        volume_batch: torch.Tensor,
        outputs_soft: torch.Tensor,
        label_batch: torch.Tensor,
        pred_channel: int = 1,
        pred_repeat: int = 1,
    ) -> None:
        """Log axial-slice image grids for input / prediction / ground-truth.

        Slices the depth dimension at indices 20:61:10 (5 evenly-spaced slices).
        This is the standard pattern repeated verbatim in most _train_* methods.

        Args:
            current_iter:  Global training iteration (used as step for TensorBoard).
            volume_batch:  Input volume, shape (B, C, D, H, W).
            outputs_soft:  Softmax predictions, shape (B, num_classes, D, H, W).
            label_batch:   Integer label map, shape (B, D, H, W).
            pred_channel:  Which channel of outputs_soft to display (default 1).
            pred_repeat:   Repeat factor for the prediction grid rows (default 1;
                           use 2 for UAMT which originally used .repeat(2, 3, 1, 1)).
        """
        # Input image (sample 0, channel 0)
        img = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        self._writer.add_image(
            'train/Image', make_grid(img, 5, normalize=True), current_iter
        )

        # Prediction (sample 0, specified channel)
        pred = (
            outputs_soft[0, pred_channel:pred_channel + 1, :, :, 20:61:10]
            .permute(3, 0, 1, 2)
            .repeat(pred_repeat, 3, 1, 1)
        )
        self._writer.add_image(
            'train/Predicted_label', make_grid(pred, 5, normalize=False), current_iter
        )

        # Ground truth (sample 0)
        gt = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
        self._writer.add_image(
            'train/Groundtruth_label', make_grid(gt, 5, normalize=False), current_iter
        )

    # ------------------------------------------------------------------
    # Misc
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying TensorBoard writer."""
        self._writer.close()
