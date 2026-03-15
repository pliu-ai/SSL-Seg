"""ConditionNet standalone training loop."""
import os
import torch
from torchvision.utils import make_grid
from tqdm import tqdm


def train_ConditionNet(trainer) -> None:
    print("================> Training ConditionNet<===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            trainer.model2.train()
            volume_batch, label_batch = (
                sampled_batch['image'], sampled_batch['label']
            )
            volume_batch, label_batch = (
                volume_batch.to(trainer.device), label_batch.to(trainer.device)
            )
            condition_batch = sampled_batch['condition']
            condition_batch = torch.cat([
                condition_batch[0], condition_batch[1]
            ], dim=0).unsqueeze(1).to(trainer.device)
            labeled_idxs_batch = torch.arange(0, trainer.labeled_bs)
            if trainer.use_CAC:
                volume_batch = torch.cat(
                    [volume_batch[:, 0, ...], volume_batch[:, 1, ...]], dim=0
                )
                label_batch = torch.cat(
                    [label_batch[:, 0, ...], label_batch[:, 1, ...]], dim=0
                )
                labeled_idxs2_batch = torch.arange(
                    trainer.batch_size, trainer.batch_size + trainer.labeled_bs
                )
                labeled_idxs1_batch = torch.arange(0, trainer.labeled_bs)
                labeled_idxs_batch = torch.cat(
                    [labeled_idxs1_batch, labeled_idxs2_batch]
                )
            noise = torch.clamp(
                torch.randn_like(volume_batch) * 0.1, -0.2, 0.2
            ).to(trainer.device)
            outputs2 = trainer.model2(volume_batch + noise, condition_batch)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
            label_batch_con = trainer._get_label_batch_for_conditional_net(
                label_batch, condition_batch
            )
            loss2 = (
                trainer.ce_loss(
                    outputs2[labeled_idxs_batch],
                    label_batch_con[labeled_idxs_batch].long()
                ) +
                trainer.dice_loss_con(
                    outputs_soft2[labeled_idxs_batch],
                    label_batch_con[labeled_idxs_batch].unsqueeze(1)
                )
            )
            model2_loss = loss2
            trainer.optimizer2.zero_grad()
            model2_loss.backward()
            trainer.optimizer2.step()

            trainer.current_iter += 1
            trainer._adjust_learning_rate()
            for param_group in trainer.optimizer2.param_groups:
                trainer.current_lr = param_group['lr']
            trainer.tensorboard_writer.add_scalar(
                'lr', trainer.current_lr, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/model2_loss', model2_loss, trainer.current_iter
            )
            trainer.logging.info(
                'iteration %d :model2 loss : %f' % (
                    trainer.current_iter, model2_loss.item()
                )
            )
            if trainer.current_iter % trainer.show_img_freq == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Image', make_grid(image, 5, normalize=True),
                    trainer.current_iter
                )
                image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Model2_Predicted_label',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )
                image = label_batch[0, :, :, 20:61:10].unsqueeze(0).permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Groundtruth_label',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )
            if (trainer.current_iter > trainer.began_eval_iter and
                    trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                with torch.no_grad():
                    trainer.evaluation(model=trainer.model2, do_condition=True)
                trainer.model2.train()
            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                save_model_path = os.path.join(
                    trainer.output_folder,
                    'model2_iter_' + str(trainer.current_iter) + '.pth'
                )
                torch.save(trainer.model2.state_dict(), save_model_path)
                trainer.logging.info(f'save model2 to {save_model_path}')
            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
