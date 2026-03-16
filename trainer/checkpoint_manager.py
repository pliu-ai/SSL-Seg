"""
CheckpointManager: save and load full training checkpoints.

Extracted from SemiSupervisedTrainer3D to isolate checkpoint I/O logic.
"""
import os
import torch
from batchgenerators.utilities.file_and_folder_operations import join


class CheckpointManager:
    """Handles save/load of model1 (and optionally model2) checkpoints.

    Usage in trainer::

        self.ckpt_manager = CheckpointManager(output_folder, logging)

        # save
        self.ckpt_manager.save(
            filename="latest",
            model=self.model, optimizer=self.optimizer,
            grad_scaler=self.grad_scaler1, current_iter=self.current_iter,
            wandb_id=self.wandb_logger.id,
            model2=self.model2, optimizer2=self.optimizer2,
            grad_scaler2=self.grad_scaler2,
        )

        # load — returns current_iter stored in checkpoint
        current_iter = self.ckpt_manager.load(
            fname="latest",
            model=self.model, optimizer=self.optimizer,
            grad_scaler=self.grad_scaler1,
            model2=self.model2, optimizer2=self.optimizer2,
            grad_scaler2=self.grad_scaler2,
        )
    """

    def __init__(self, output_folder: str, logging):
        self._output_folder = output_folder
        self._logging = logging

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        filename: str,
        model,
        optimizer,
        grad_scaler,
        current_iter: int,
        wandb_id,
        model2=None,
        optimizer2=None,
        grad_scaler2=None,
        only: str = None,
    ) -> None:
        """Save full checkpoint for model1 and/or model2.

        Args:
            only: If ``'model1'`` or ``'model2'``, save only that model.
                  If *None* (default), save both.
        """
        if only is None or only == 'model1':
            checkpoint1 = {
                'network_weights': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'grad_scaler_state': grad_scaler.state_dict() if grad_scaler is not None else None,
                'current_iter': current_iter + 1,
                'wandb_id': wandb_id,
            }
            path1 = join(self._output_folder, f"model1_{filename}.pth")
            torch.save(checkpoint1, path1)

        if model2 is not None and (only is None or only == 'model2'):
            checkpoint2 = {
                'network_weights': model2.state_dict(),
                'optimizer_state': optimizer2.state_dict(),
                'grad_scaler_state': grad_scaler2.state_dict() if grad_scaler2 is not None else None,
                'current_iter': current_iter + 1,
                'wandb_id': wandb_id,
            }
            path2 = join(self._output_folder, f"model2_{filename}.pth")
            torch.save(checkpoint2, path2)

        self._logging.info(f'save model to {join(self._output_folder, filename)}')

    def load(
        self,
        fname: str,
        model,
        optimizer,
        grad_scaler,
        model2=None,
        optimizer2=None,
        grad_scaler2=None,
    ) -> int:
        """Load checkpoint for model1 (and model2 if provided).

        Returns:
            current_iter stored in the checkpoint.
        """
        path1 = join(self._output_folder, f"model1_{fname}.pth")
        checkpoint = torch.load(path1, weights_only=False)
        model.load_state_dict(checkpoint['network_weights'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        if grad_scaler is not None and checkpoint.get('grad_scaler_state') is not None:
            grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])
        current_iter = checkpoint['current_iter']
        print(f"=====> Load checkpoint from {path1} for model1 Successfully")

        if model2 is not None:
            path2 = join(self._output_folder, f"model2_{fname}.pth")
            checkpoint2 = torch.load(path2, weights_only=False)
            model2.load_state_dict(checkpoint2['network_weights'])
            optimizer2.load_state_dict(checkpoint2['optimizer_state'])
            if grad_scaler2 is not None and checkpoint2.get('grad_scaler_state') is not None:
                grad_scaler2.load_state_dict(checkpoint2['grad_scaler_state'])
            print(f"=====> Load checkpoint from {path2} for model2 Successfully")

        return current_iter
