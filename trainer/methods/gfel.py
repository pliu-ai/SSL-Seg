"""GFEL (General Feature Enhanced Learning) semi-supervised training loop."""
import os
import torch
from torchvision.utils import make_grid
from tqdm import tqdm


def train_GFEL(trainer) -> None:
    "General Feature Enhanced Learning"
    print("================> Training GFEL<===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            volume_batch, label_batch = (
                sampled_batch['image'], sampled_batch['label']
            )
            volume_batch, label_batch = (
                volume_batch.to(trainer.device), label_batch.to(trainer.device)
            )
            noise1 = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            label_batch = torch.argmax(label_batch, dim=1)
            outputs1 = trainer.model(volume_batch + noise1)
            outputs_soft1 = torch.softmax(outputs1, dim=1)
            noise2 = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            outputs2 = trainer.model2(volume_batch + noise2)
            outputs_soft2 = torch.softmax(outputs2, dim=1)

            trainer.consistency_weight = trainer._get_current_consistency_weight(
                trainer.current_iter // 150
            )
            loss1 = 0.5 * (
                trainer.ce_loss(outputs1[:trainer.labeled_bs],
                                label_batch[:][:trainer.labeled_bs].long()) +
                trainer.dice_loss(outputs_soft1[:trainer.labeled_bs],
                                  label_batch[:trainer.labeled_bs].unsqueeze(1))
            )
            loss2 = 0.5 * (
                trainer.ce_loss(outputs2[:trainer.labeled_bs],
                                label_batch[:][:trainer.labeled_bs].long()) +
                trainer.dice_loss(outputs_soft2[:trainer.labeled_bs],
                                  label_batch[:trainer.labeled_bs].unsqueeze(1))
            )
            pseudo_outputs1 = torch.argmax(
                outputs_soft1[trainer.labeled_bs:].detach(), dim=1, keepdim=False
            )
            pseudo_outputs2 = torch.argmax(
                outputs_soft2[trainer.labeled_bs:].detach(), dim=1, keepdim=False
            )
            if trainer.current_iter < trainer.began_semi_iter:
                pseudo_supervision1 = torch.FloatTensor([0]).to(trainer.device)
                pseudo_supervision2 = torch.FloatTensor([0]).to(trainer.device)
            else:
                pseudo_supervision1 = trainer.ce_loss(
                    outputs1[trainer.labeled_bs:], pseudo_outputs2
                )
                pseudo_supervision2 = trainer.ce_loss(
                    outputs2[trainer.labeled_bs:], pseudo_outputs1
                )
            model1_loss = loss1 + trainer.consistency_weight * pseudo_supervision1
            model2_loss = loss2 + trainer.consistency_weight * pseudo_supervision2
            loss = model1_loss + model2_loss
            trainer.optimizer.zero_grad()
            trainer.optimizer2.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer.optimizer2.step()

            trainer._adjust_learning_rate()
            trainer.current_iter += 1

            trainer.tensorboard_writer.add_scalar(
                'lr', trainer.current_lr, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'consistency_weight/consistency_weight',
                trainer.consistency_weight, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/model1_loss', model1_loss, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/model2_loss', model2_loss, trainer.current_iter
            )
            trainer.logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (
                    trainer.current_iter, model1_loss.item(), model2_loss.item()
                )
            )

            if trainer.current_iter % trainer.show_img_freq == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Image', make_grid(image, 5, normalize=True),
                    trainer.current_iter
                )
                image = outputs_soft1[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Model1_Predicted_label',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )
                image = outputs_soft2[0, 0:1, :, :, 20:61:10].permute(
                    3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Model2_Predicted_label',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )
                image = label_batch[0, :, :, 20:61:10].unsqueeze(
                    0).permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Groundtruth_label',
                    make_grid(image, 5, normalize=False), trainer.current_iter
                )

            if (
                trainer.current_iter > trainer.began_eval_iter and
                trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                trainer.evaluation(model=trainer.model)
                trainer.evaluation(model=trainer.model2, model_name='model2')
                trainer.model.train()
                trainer.model2.train()

            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                save_mode_path = os.path.join(
                    trainer.output_folder,
                    'model1_iter_' + str(trainer.current_iter) + '.pth'
                )
                torch.save(trainer.model.state_dict(), save_mode_path)
                trainer.logging.info("save model1 to {}".format(save_mode_path))
                save_mode_path = os.path.join(
                    trainer.output_folder,
                    'model2_iter_' + str(trainer.current_iter) + '.pth'
                )
                torch.save(trainer.model2.state_dict(), save_mode_path)
                trainer.logging.info("save model2 to {}".format(save_mode_path))
            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
