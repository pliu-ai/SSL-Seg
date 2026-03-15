"""Fully-supervised baseline training loop."""
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def train_baseline(trainer) -> None:
    print("================> Training Baseline <===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            trainer.model.train()
            volume_batch, label_batch = (sampled_batch['image'],
                                         sampled_batch['label'].long())
            volume_batch, label_batch = (volume_batch.to(trainer.device),
                                         label_batch.to(trainer.device))
            trainer.optimizer.zero_grad()
            with autocast():
                outputs = trainer.model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                label_batch = torch.argmax(label_batch, dim=1)
                trainer.loss_ce = trainer.ce_loss(outputs, label_batch.long())
                trainer.loss_dice = trainer.dice_loss(outputs_soft,
                                                      label_batch.unsqueeze(1))
                trainer.loss = 0.5 * (trainer.loss_dice + trainer.loss_ce)
            trainer.grad_scaler1.scale(trainer.loss).backward()
            trainer.grad_scaler1.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 12)
            trainer.grad_scaler1.step(trainer.optimizer)
            trainer.grad_scaler1.update()

            trainer._adjust_learning_rate()
            trainer.current_iter += 1
            trainer._add_information_to_writer()

            if trainer.current_iter % trainer.show_img_freq == 0:
                trainer.logger.log_train_images_3d(
                    trainer.current_iter, volume_batch, outputs_soft, label_batch
                )

            if (trainer.current_iter > trainer.began_eval_iter and
                    trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                trainer.evaluation(model=trainer.model)

            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                trainer._save_checkpoint(filename="latest")

            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
    print("*" * 10, "training done!", "*" * 10)
