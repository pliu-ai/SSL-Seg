"""McNet multi-head consistency semi-supervised training loop."""
import torch
from tqdm import tqdm
from utils import losses
from trainer.train_utils import sharpening


def train_McNet(trainer) -> None:
    print("================> Training McNet <===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            volume_batch, label_batch = (sampled_batch['image'],
                                         sampled_batch['label'])
            volume_batch, label_batch = (volume_batch.to(trainer.device),
                                         label_batch.to(trainer.device))
            label_batch = torch.argmax(label_batch, dim=1)
            outputs = trainer.model(volume_batch)
            num_outputs = len(outputs)
            y_ori = outputs[0].new_zeros((num_outputs,) + outputs[0].shape)
            y_pseudo_label = outputs[0].new_zeros((num_outputs,) + outputs[0].shape)
            trainer.loss_ce = 0
            trainer.loss_dice = 0
            for idx in range(num_outputs):
                y = outputs[idx][:trainer.labeled_bs, ...]
                y_prob = torch.softmax(y, dim=1)
                trainer.loss_ce += trainer.ce_loss(
                    y[:trainer.labeled_bs], label_batch[:trainer.labeled_bs]
                )
                trainer.loss_dice += trainer.dice_loss(
                    y_prob, label_batch[:trainer.labeled_bs, ...].unsqueeze(1)
                )

                y_all = outputs[idx]
                y_prob_all = torch.softmax(y_all, dim=1)
                y_ori[idx] = y_prob_all
                y_pseudo_label[idx] = sharpening(y_prob_all)

            trainer.consistency_loss = 0
            if trainer.current_iter > trainer.began_semi_iter:
                for i in range(num_outputs):
                    for j in range(num_outputs):
                        if i != j:
                            trainer.consistency_loss += losses.mse_loss(
                                y_ori[i], y_pseudo_label[j]
                            )

            consistency_weight = trainer._get_current_consistency_weight(
                trainer.current_iter // 150
            )

            trainer.loss = (0.5 * trainer.loss_dice
                            + consistency_weight * trainer.consistency_loss)
            trainer.optimizer.zero_grad()
            trainer.loss.backward()
            trainer.optimizer.step()
            trainer._adjust_learning_rate()
            trainer.current_iter += 1
            trainer._add_information_to_writer()
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
