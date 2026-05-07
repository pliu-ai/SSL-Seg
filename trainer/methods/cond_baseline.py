"""Fully-supervised Conditional Network baseline.

Uses only the conditional network (model2 from C3PS) trained with full supervision.
All samples are labeled; no pseudo-labels or cross-teaching.
Reuses BCVDatasetCAC without modification.

At each iteration the conditional net receives an image patch and a condition
(single- or multi-class, depending on condition_group_mode) and predicts a
binary foreground/background mask.  At evaluation time, predictions are
assembled over all conditions to recover the full multi-class segmentation
(same as C3PS model2 evaluation).
"""
import os
import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from tqdm import tqdm


def train_CondBaseline(trainer) -> None:
    print("================> Training CondBaseline (fully-supervised conditional net) <===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    log_interval = trainer.train_log_interval
    fixed_iters = (
        int(trainer.iters_per_epoch)
        if getattr(trainer, "iters_per_epoch", None) is not None
        else len(trainer.dataloader)
    )
    data_iter = iter(trainer.dataloader)

    for epoch_num in iterator:
        for i_batch in range(fixed_iters):
            try:
                sampled_batch = next(data_iter)
            except StopIteration:
                data_iter = iter(trainer.dataloader)
                sampled_batch = next(data_iter)

            trainer._adjust_learning_rate()
            trainer.model2.train()

            volume_batch = sampled_batch['image'].to(trainer.device, non_blocking=True)
            label_batch = sampled_batch['label'].to(trainer.device, non_blocking=True)

            if trainer.use_CAC:
                # BCVDatasetCAC returns (B, 2, 1, D, W, H) and (B, 2, D, W, H)
                volume_batch = torch.cat(
                    [volume_batch[:, 0, ...], volume_batch[:, 1, ...]], dim=0
                )
                label_batch = torch.cat(
                    [label_batch[:, 0, ...], label_batch[:, 1, ...]], dim=0
                )
            N = volume_batch.shape[0]

            # Build condition for each patch from ground-truth labels
            condition_batch = []
            for i in range(N):
                label_list = label_batch[i].unique().tolist()
                if 0 in label_list:
                    label_list.remove(0)
                if len(label_list) == 0:
                    label_list = list(range(1, trainer.num_classes))
                con = trainer._get_condition(label_list)
                condition_batch.append(con)
            condition_batch = torch.stack(condition_batch).to(trainer.device)

            label_batch_con = trainer._get_label_batch_for_conditional_net(
                label_batch, condition_batch
            )

            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            trainer.optimizer2.zero_grad(set_to_none=True)

            with autocast():
                outputs = trainer.model2(volume_batch + noise, condition_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

                loss_ce = trainer.ce_loss(outputs, label_batch_con.long())
                loss_dice = trainer.dice_loss_con(
                    outputs_soft, label_batch_con.unsqueeze(1)
                )
                loss = 0.5 * (loss_ce + loss_dice)

            trainer.grad_scaler2.scale(loss).backward()
            trainer.grad_scaler2.unscale_(trainer.optimizer2)
            torch.nn.utils.clip_grad_norm_(trainer.model2.parameters(), 12)
            trainer.grad_scaler2.step(trainer.optimizer2)
            trainer.grad_scaler2.update()

            trainer.current_iter += 1
            trainer.current_lr = trainer.optimizer2.param_groups[0]['lr']

            if trainer.current_iter % log_interval == 0 or trainer.current_iter == 1:
                trainer.tensorboard_writer.add_scalar(
                    'lr', trainer.current_lr, trainer.current_iter
                )
                trainer.tensorboard_writer.add_scalar(
                    'loss/total_loss', loss.item(), trainer.current_iter
                )
                trainer.tensorboard_writer.add_scalar(
                    'loss/loss_ce', loss_ce.item(), trainer.current_iter
                )
                trainer.tensorboard_writer.add_scalar(
                    'loss/loss_dice', loss_dice.item(), trainer.current_iter
                )
                trainer.logging.info(
                    'iteration %d : loss : %f  loss_ce : %f  loss_dice : %f' % (
                        trainer.current_iter, loss.item(),
                        loss_ce.item(), loss_dice.item()
                    )
                )

            if trainer.current_iter % trainer.show_img_freq == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Image', make_grid(image, 5, normalize=True),
                    trainer.current_iter
                )
                image = outputs_soft[0, 1:2, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Predicted_fg_prob',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )
                image = label_batch_con[0, :, :, 20:61:10].unsqueeze(0).permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1).float()
                trainer.tensorboard_writer.add_image(
                    'train/Groundtruth_binary',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )

            if (trainer.current_iter > trainer.began_eval_iter and
                    trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                with torch.no_grad():
                    trainer.evaluation(model=trainer.model2, do_condition=True)
                trainer.model2.train()

            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                save_path = os.path.join(
                    trainer.output_folder,
                    'model2_iter_' + str(trainer.current_iter) + '.pth'
                )
                torch.save(trainer.model2.state_dict(), save_path)
                trainer.logging.info(f"save model to {save_path}")

            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
    print("*" * 10, "CondBaseline training done!", "*" * 10)
