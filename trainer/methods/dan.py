"""DAN (Domain Adversarial Network) semi-supervised training loop."""
import torch
from tqdm import tqdm
from networks.net_factory_3d import net_factory_3d


def train_DAN(trainer) -> None:
    print("================> Training DAN <===============")
    trainer.model2 = net_factory_3d(
        net_type="DAN", class_num=trainer.num_classes, device=trainer.device
    )
    trainer.optimizer2 = torch.optim.Adam(
        trainer.model2.parameters(), lr=0.0001, betas=(0.9, 0.99)
    )
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            volume_batch, label_batch = (
                sampled_batch['image'], sampled_batch['label']
            )
            volume_batch, label_batch = (
                volume_batch.to(trainer.device), label_batch.to(trainer.device)
            )
            DAN_target = torch.tensor([1, 0]).to(trainer.device)
            trainer.model.train()
            trainer.model2.eval()

            outputs = trainer.model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            label_batch = torch.argmax(label_batch, dim=1)
            trainer.loss_ce = trainer.ce_loss(
                outputs[:trainer.labeled_bs], label_batch[:trainer.labeled_bs]
            )
            trainer.loss_dice = trainer.dice_loss(
                outputs_soft[:trainer.labeled_bs],
                label_batch[:trainer.labeled_bs].unsqueeze(1)
            )
            supervised_loss = 0.5 * (trainer.loss_dice + trainer.loss_ce)

            trainer.consistency_weight = trainer._get_current_consistency_weight(
                trainer.current_iter // 6
            )
            DAN_outputs = trainer.model2(
                outputs_soft[trainer.labeled_bs:],
                volume_batch[trainer.labeled_bs:]
            )
            if trainer.current_iter > trainer.began_semi_iter:
                trainer.consistency_loss = trainer.ce_loss(
                    DAN_outputs, (DAN_target[:trainer.labeled_bs]).long()
                )
            else:
                trainer.consistency_loss = torch.FloatTensor([0.0]).to(trainer.device)
            trainer.loss = (supervised_loss
                            + trainer.consistency_weight * trainer.consistency_loss)
            trainer.optimizer.zero_grad()
            trainer.loss.backward()
            trainer.optimizer.step()

            trainer.model.eval()
            trainer.model2.train()
            with torch.no_grad():
                outputs = trainer.model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)

            DAN_outputs = trainer.model2(outputs_soft, volume_batch)
            DAN_loss = trainer.ce_loss(DAN_outputs, DAN_target.long())
            trainer.optimizer2.zero_grad()
            DAN_loss.backward()
            trainer.optimizer2.step()

            trainer._adjust_learning_rate()
            trainer.current_iter += 1
            trainer._add_information_to_writer()

            if trainer.current_iter % trainer.show_img_freq == 0:
                trainer.logger.log_train_images_3d(
                    trainer.current_iter, volume_batch, outputs_soft, label_batch
                )
            if (
                trainer.current_iter > trainer.began_eval_iter and
                trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                trainer.evaluation(model=trainer.model)
            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                trainer._save_checkpoint()
            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
    print("Training done!")
