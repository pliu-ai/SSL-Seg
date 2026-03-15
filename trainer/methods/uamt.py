"""UAMT (Uncertainty-aware Mean Teacher) semi-supervised training loop."""
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import losses, ramps


def train_UAMT(trainer) -> None:
    print("================> Training UAMT<===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            volume_batch, label_batch = (sampled_batch['image'],
                                         sampled_batch['label'])
            volume_batch, label_batch = (volume_batch.to(trainer.device),
                                         label_batch.to(trainer.device))
            unlabeled_volume_batch = volume_batch[trainer.labeled_bs:]

            label_batch = torch.argmax(label_batch, dim=1)
            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = (unlabeled_volume_batch + noise).to(trainer.device)

            outputs = trainer._forward_logits(trainer.model, volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output = trainer._forward_logits(trainer.ema_model, ema_inputs)
            T = 8
            _, _, d, w, h = unlabeled_volume_batch.shape
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros(
                [stride * T, trainer.num_classes, d, w, h]
            ).to(trainer.device)
            for i in range(T // 2):
                ema_inputs = volume_batch_r + torch.clamp(
                    torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2
                )
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = trainer._forward_logits(
                        trainer.ema_model, ema_inputs
                    )
            preds = torch.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, trainer.num_classes, d, w, h)
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(
                preds * torch.log(preds + 1e-6), dim=1, keepdim=True
            )
            trainer.loss_ce = trainer.ce_loss(
                outputs[:trainer.labeled_bs], label_batch[:trainer.labeled_bs]
            )
            trainer.loss_dice = trainer.dice_loss(
                outputs_soft[:trainer.labeled_bs],
                label_batch[:trainer.labeled_bs].unsqueeze(1)
            )
            supervised_loss = 0.5 * (trainer.loss_dice + trainer.loss_ce)
            trainer.consistency_weight = trainer._get_current_consistency_weight(
                trainer.current_iter // 150
            )
            threshold = (
                0.75 + 0.25 * ramps.sigmoid_rampup(trainer.current_iter, trainer.max_iterations)
            ) * np.log(2)
            mask = (uncertainty < threshold).float()
            if trainer.sam_pipeline.is_active:
                with torch.no_grad():
                    ema_output_soft = torch.softmax(ema_output, dim=1)
                    pseudo_label = torch.argmax(ema_output_soft, dim=1)
                refined_pseudo_label, reliability_map, _ = trainer.sam_pipeline.run_pipeline(
                    unlabeled_volume_batch=unlabeled_volume_batch,
                    teacher_logits=ema_output,
                    pseudo_label=pseudo_label,
                    ground_truth_label=label_batch[trainer.labeled_bs:],
                    epoch_num=epoch_num,
                    current_iter=trainer.current_iter,
                    tensorboard_writer=trainer.tensorboard_writer,
                    ema_model=trainer.ema_model,
                )
                per_voxel_loss = F.cross_entropy(
                    outputs[trainer.labeled_bs:],
                    refined_pseudo_label.long(),
                    reduction='none',
                )
                uncertainty_mask = mask.squeeze(1)
                trainer.consistency_loss = torch.sum(
                    uncertainty_mask * reliability_map * per_voxel_loss
                ) / (torch.sum(uncertainty_mask) + 1e-16)
            else:
                consistency_dist = losses.softmax_dice_loss(
                    outputs[trainer.labeled_bs:], ema_output
                )
                trainer.consistency_loss = torch.sum(mask * consistency_dist) / (
                    2 * torch.sum(mask) + 1e-16
                )
            trainer.loss = (supervised_loss
                            + trainer.consistency_weight * trainer.consistency_loss)
            trainer.optimizer.zero_grad()
            trainer.loss.backward()
            trainer.optimizer.step()
            trainer._update_ema_variables()
            trainer._adjust_learning_rate()
            trainer.current_iter += 1
            trainer._add_information_to_writer()
            if trainer.current_iter % trainer.show_img_freq == 0:
                trainer.logger.log_train_images_3d(
                    trainer.current_iter, volume_batch, outputs_soft, label_batch,
                    pred_repeat=2,
                )
            if (trainer.current_iter > trainer.began_eval_iter and
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
    print("*" * 10, "training done!", "*" * 10)
