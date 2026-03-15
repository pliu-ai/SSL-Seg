"""C3PS-2D: 2D version of C3PS (Conditional Cross-pseudo-label with Patch-level Supervision)."""
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from tqdm import tqdm


def _get_label_batch_for_conditional_net_2d(label_batch, condition_batch, num_classes):
    """
    Convert label batch to binary condition label batch for 2D spatial tensors.
    label_batch: (B, H, W), condition_batch: (B, 1)
    """
    if condition_batch.max() < num_classes:
        # (B, 1) -> (B, 1, 1) to broadcast with (B, H, W)
        return (label_batch == condition_batch.unsqueeze(-1)).long()
    else:
        label_batch_con = torch.zeros_like(label_batch)
        for i, con in enumerate(condition_batch):
            if con == num_classes:
                label_batch_con[i][label_batch[i] > 0] = 1
            else:
                label_batch_con[i][label_batch[i] != con] = 0
                label_batch_con[i][label_batch[i] == con] = 1
        return label_batch_con


def _cross_entropy_loss_con_2d(output, target, condition, filter_mask,
                                num_classes, device):
    """
    Cross entropy loss for conditional network with 2D spatial tensors.
    output: (B, C, H, W), target: (B, H, W), condition: (B, 1), filter_mask: (B, H, W)
    """
    softmax = torch.softmax(output, dim=1)
    B, C, H, W = softmax.shape
    softmax_con = torch.zeros(B, 2, H, W).to(device)
    if condition[0] < num_classes:
        softmax_con[:, 1, ...] = softmax[
            np.arange(B), condition.squeeze().long(), ...
        ]
        softmax_con[:, 0, ...] = 1.0 - softmax_con[:, 1, ...]
    else:
        softmax_con[:, 0, ...] = softmax[np.arange(B), 0, ...]
        softmax_con[:, 1, ...] = 1.0 - softmax_con[:, 0, ...]
    log = -torch.log(softmax_con.gather(1, target.unsqueeze(1)) + 1e-7)
    loss = (log * filter_mask.unsqueeze(1)).sum() / filter_mask.sum()
    return loss


def train_C3PS_2D(trainer) -> None:
    """2D version of the C3PS training loop."""
    print("================> Training C3PS-2D <===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            trainer._adjust_learning_rate()
            trainer.model.train()
            trainer.model2.train()
            volume_batch, label_batch = (
                sampled_batch['image'], sampled_batch['label']
            )
            volume_batch, label_batch = (
                volume_batch.to(trainer.device), label_batch.to(trainer.device)
            )
            condition_batch = sampled_batch['condition']
            condition_batch = torch.cat([
                condition_batch[:, 0], condition_batch[:, 1]
            ], dim=0).unsqueeze(1).to(trainer.device)
            ul1, br1, ul2, br2 = [], [], [], []
            labeled_idxs_batch = torch.arange(0, trainer.labeled_bs)
            unlabeled_idx_batch = torch.arange(
                trainer.labeled_bs, trainer.batch_size
            )
            if trainer.use_CAC:
                ul1, br1 = sampled_batch['ul1'], sampled_batch['br1']
                ul2, br2 = sampled_batch['ul2'], sampled_batch['br2']
                volume_batch = torch.cat(
                    [volume_batch[:, 0, ...], volume_batch[:, 1, ...]], dim=0
                )
                label_batch = torch.cat(
                    [label_batch[:, 0, ...], label_batch[:, 1, ...]], dim=0
                )
                labeled_idxs2_batch = torch.arange(
                    trainer.batch_size,
                    trainer.batch_size + trainer.labeled_bs
                )
                labeled_idxs1_batch = torch.arange(0, trainer.labeled_bs)
                labeled_idxs_batch = torch.cat(
                    [labeled_idxs1_batch, labeled_idxs2_batch]
                )
                unlabeled_idxs1_batch = torch.arange(
                    trainer.labeled_bs, trainer.batch_size
                )
                unlabeled_idxs2_batch = torch.arange(
                    trainer.batch_size + trainer.labeled_bs,
                    2 * trainer.batch_size
                )
                unlabeled_idxs_batch = torch.cat(
                    [unlabeled_idxs1_batch, unlabeled_idxs2_batch]
                )
            noise1 = torch.clamp(
                torch.randn_like(volume_batch) * 0.1, -0.2, 0.2
            ).to(trainer.device)
            trainer.optimizer.zero_grad()
            trainer.optimizer2.zero_grad()
            with autocast():
                outputs1 = trainer.model(volume_batch + noise1)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                if trainer.use_CAC and (
                    trainer.current_iter >= min(
                        trainer.began_semi_iter,
                        trainer.began_condition_iter
                    )
                ):
                    overlap_soft1_list = []
                    overlap_outputs1_list = []
                    overlap_filter1_list = []
                    for unlabeled_idx1, unlabeled_idx2 in zip(
                        unlabeled_idxs1_batch, unlabeled_idxs2_batch
                    ):
                        # 2D overlap: index with 2 spatial dims (H, W)
                        overlap1_soft1 = outputs_soft1[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1]
                        ]
                        overlap2_soft1 = outputs_soft1[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1]
                        ]
                        assert overlap1_soft1.shape == overlap2_soft1.shape, \
                            "overlap region size must equal"
                        overlap1_outputs1 = outputs1[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1]
                        ]
                        overlap2_outputs1 = outputs1[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1]
                        ]
                        assert overlap1_outputs1.shape == overlap2_outputs1.shape, \
                            "overlap region size must equal"
                        overlap_outputs1_list.append(
                            overlap1_outputs1.unsqueeze(0)
                        )
                        overlap_outputs1_list.append(
                            overlap2_outputs1.unsqueeze(0)
                        )
                        overlap_soft1_tmp = (
                            overlap1_soft1 + overlap2_soft1
                        ) / 2.0
                        max1, pseudo_mask1 = torch.max(
                            overlap_soft1_tmp, dim=0
                        )
                        pred_con_list = pseudo_mask1.unique().tolist()
                        con = trainer._get_condition(pred_con_list)
                        if trainer.num_classes == 2:
                            overlap_filter1_tmp = (
                                (((max1 > trainer.model1_thresh)
                                  & (pseudo_mask1 != con))
                                 | ((max1 > 0.8) & (pseudo_mask1 == con)))
                            ).type(torch.int16)
                        else:
                            if con < trainer.num_classes:
                                overlap_filter1_tmp = (
                                    (((max1 > 0.99) & (pseudo_mask1 == 0))
                                     | ((max1 > 0.9) & (pseudo_mask1 != con)
                                        & (pseudo_mask1 != 0))
                                     | ((max1 > 0.9) & (pseudo_mask1 == con)))
                                ).type(torch.int16)
                            else:
                                overlap_filter1_tmp = (
                                    (((max1 > 0.99) & (pseudo_mask1 == 0))
                                     | ((max1 > 0.9) & (pseudo_mask1 != 0)))
                                ).type(torch.int16)
                        overlap_soft1_list.append(
                            overlap_soft1_tmp.unsqueeze(0)
                        )
                        overlap_filter1_list.append(
                            overlap_filter1_tmp.unsqueeze(0)
                        )
                    overlap_soft1 = torch.cat(overlap_soft1_list, 0)
                    overlap_outputs1 = torch.cat(overlap_outputs1_list, 0)
                    overlap_filter1 = torch.cat(overlap_filter1_list, 0)
                    condition_batch[unlabeled_idxs_batch] = con

                noise2 = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, -0.2, 0.2
                ).to(trainer.device)
                outputs2 = trainer.model2(volume_batch + noise2, condition_batch)
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                label_batch_con = _get_label_batch_for_conditional_net_2d(
                    label_batch, condition_batch, trainer.num_classes
                )
                trainer.consistency_weight = (
                    trainer._get_current_consistency_weight(
                        trainer.current_iter // 150
                    )
                )
                loss1 = 0.5 * (
                    trainer.ce_loss(
                        outputs1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].long()
                    )
                    + trainer.dice_loss(
                        outputs_soft1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].unsqueeze(1)
                    )
                )
                loss2 = 0.5 * (
                    trainer.ce_loss(
                        outputs2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].long()
                    )
                    + trainer.dice_loss_con(
                        outputs_soft2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].unsqueeze(1)
                    )
                )

                if trainer.use_CAC and (
                    trainer.current_iter >= min(
                        trainer.began_semi_iter,
                        trainer.began_condition_iter
                    )
                ):
                    overlap_soft2_list = []
                    overlap_outputs2_list = []
                    overlap_filter2_list = []
                    for unlabeled_idx1, unlabeled_idx2 in zip(
                        unlabeled_idxs1_batch, unlabeled_idxs2_batch
                    ):
                        # 2D overlap: index with 2 spatial dims (H, W)
                        overlap1_soft2 = outputs_soft2[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1]
                        ]
                        overlap2_soft2 = outputs_soft2[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1]
                        ]
                        assert overlap1_soft2.shape == overlap2_soft2.shape, \
                            "overlap region size must equal"
                        overlap1_outputs2 = outputs2[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1]
                        ]
                        overlap2_outputs2 = outputs2[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1]
                        ]
                        assert overlap1_outputs2.shape == overlap2_outputs2.shape, \
                            "overlap region size must equal"
                        overlap_outputs2_list.append(
                            overlap1_outputs2.unsqueeze(0)
                        )
                        overlap_outputs2_list.append(
                            overlap2_outputs2.unsqueeze(0)
                        )
                        overlap_soft2_tmp = (
                            overlap1_soft2 + overlap2_soft2
                        ) / 2.0
                        max2, pseudo_mask2 = torch.max(
                            overlap_soft2_tmp, dim=0
                        )
                        if trainer.num_classes == 2:
                            overlap_filter2_tmp = (
                                max2 > trainer.model2_thresh
                            ).type(torch.int16)
                        else:
                            if con < trainer.num_classes:
                                overlap_filter2_tmp = (
                                    (max2 > trainer.model2_thresh)
                                    & (pseudo_mask2 > 0)
                                ).type(torch.int16)
                            else:
                                overlap_filter2_tmp = (
                                    (max2 > trainer.model2_thresh)
                                    & (pseudo_mask2 == 0)
                                ).type(torch.int16)
                        overlap_soft2_list.append(
                            overlap_soft2_tmp.unsqueeze(0)
                        )
                        overlap_filter2_list.append(
                            overlap_filter2_tmp.unsqueeze(0)
                        )
                    overlap_soft2 = torch.cat(overlap_soft2_list, 0)
                    overlap_outputs2 = torch.cat(overlap_outputs2_list, 0)
                    overlap_filter2 = torch.cat(overlap_filter2_list, 0)

                # --- pseudo supervision 1: model2 -> model1 ---
                if trainer.current_iter < trainer.began_condition_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(
                        trainer.device
                    )
                else:
                    if trainer.use_CAC:
                        overlap_pseudo_outputs2 = torch.argmax(
                            overlap_soft2.detach(), dim=1, keepdim=False
                        )
                        if (overlap_pseudo_outputs2.sum() == 0
                                or overlap_filter2.sum() == 0):
                            pseudo_supervision1 = torch.FloatTensor([0]).to(
                                trainer.device
                            )
                        else:
                            overlap_pseudo_outputs2 = torch.cat([
                                overlap_pseudo_outputs2,
                                overlap_pseudo_outputs2
                            ])
                            overlap_pseudo_filter2 = torch.cat([
                                overlap_filter2, overlap_filter2
                            ])
                            ce_pseudo_supervision1 = (
                                _cross_entropy_loss_con_2d(
                                    overlap_outputs1,
                                    overlap_pseudo_outputs2,
                                    condition_batch[unlabeled_idx_batch],
                                    overlap_pseudo_filter2,
                                    trainer.num_classes,
                                    trainer.device,
                                )
                            )
                            pseudo_supervision1 = ce_pseudo_supervision1
                    else:
                        pseudo_outputs2 = torch.argmax(
                            outputs_soft2[trainer.labeled_bs:].detach(),
                            dim=1, keepdim=False
                        )
                        pseudo_supervision1 = _cross_entropy_loss_con_2d(
                            outputs1[trainer.labeled_bs:],
                            pseudo_outputs2,
                            condition_batch[trainer.labeled_bs:],
                            torch.ones_like(pseudo_outputs2).type(
                                torch.int16
                            ),
                            trainer.num_classes,
                            trainer.device,
                        )

                # --- pseudo supervision 2: model1 -> model2 ---
                if (trainer.current_iter < trainer.began_semi_iter
                        or overlap_filter1.sum() == 0):
                    pseudo_supervision2 = torch.FloatTensor([0]).to(
                        trainer.device
                    )
                else:
                    if trainer.use_CAC:
                        overlap_pseudo_outputs1 = torch.argmax(
                            overlap_soft1.detach(), dim=1, keepdim=False
                        )
                        overlap_pseudo_outputs1 = torch.cat([
                            overlap_pseudo_outputs1, overlap_pseudo_outputs1
                        ])
                        overlap_pseudo_filter1 = torch.cat([
                            overlap_filter1, overlap_filter1
                        ])
                        target_ce_con = (
                            _get_label_batch_for_conditional_net_2d(
                                overlap_pseudo_outputs1,
                                condition_batch[unlabeled_idxs_batch],
                                trainer.num_classes,
                            )
                        )
                        target_ce_con[overlap_pseudo_filter1 == 0] = 255
                        ce_pseudo_supervision2 = trainer.ce_loss(
                            overlap_outputs2, target_ce_con
                        )
                        # 2D: condition broadcast uses single unsqueeze
                        dice_pseudo_supervision2 = trainer.dice_loss_con(
                            torch.softmax(overlap_outputs2, dim=1)
                            * overlap_pseudo_filter1,
                            ((overlap_pseudo_outputs1
                              == condition_batch[
                                  unlabeled_idxs_batch
                              ].unsqueeze(-1)
                              ).long() * overlap_pseudo_filter1).unsqueeze(1),
                            skip_id=0,
                        )
                        pseudo_supervision2 = (
                            ce_pseudo_supervision2 + dice_pseudo_supervision2
                        )
                    else:
                        pseudo_outputs1 = torch.argmax(
                            outputs_soft1[trainer.labeled_bs:].detach(),
                            dim=1, keepdim=False
                        )
                        # 2D: condition broadcast uses single unsqueeze
                        pseudo_supervision2 = trainer.ce_loss(
                            outputs2[trainer.labeled_bs:],
                            (pseudo_outputs1
                             == condition_batch[
                                 trainer.labeled_bs:
                             ].unsqueeze(-1)).long()
                        )

                model1_loss = (
                    loss1 + trainer.consistency_weight * pseudo_supervision1
                )
                model2_loss = (
                    loss2 + trainer.consistency_weight * pseudo_supervision2
                )
            trainer.grad_scaler1.scale(model1_loss).backward()
            trainer.grad_scaler1.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 12)
            trainer.grad_scaler1.step(trainer.optimizer)
            trainer.grad_scaler1.update()
            trainer.grad_scaler2.scale(model2_loss).backward()
            trainer.grad_scaler2.unscale_(trainer.optimizer2)
            torch.nn.utils.clip_grad_norm_(trainer.model2.parameters(), 12)
            trainer.grad_scaler2.step(trainer.optimizer2)
            trainer.grad_scaler2.update()

            trainer.current_iter += 1
            for param_group in trainer.optimizer.param_groups:
                trainer.current_lr = param_group['lr']
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
            trainer.tensorboard_writer.add_scalar(
                'loss/pseudo_supervision1',
                pseudo_supervision1, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/pseudo_supervision2',
                pseudo_supervision2, trainer.current_iter
            )
            trainer.logging.info(
                'iteration %d : model1 loss : %f  model2 loss : %f  '
                'pseudo_supervision1 : %f  pseudo_supervision2 : %f' % (
                    trainer.current_iter, model1_loss.item(),
                    model2_loss.item(),
                    pseudo_supervision1.item(),
                    pseudo_supervision2.item(),
                )
            )
            # 2D tensorboard visualization
            if trainer.current_iter % trainer.show_img_freq == 0:
                image = volume_batch[0:1, :, :, :].repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Image',
                    make_grid(image, 1, normalize=True),
                    trainer.current_iter,
                )
                pred1 = outputs_soft1[0:1, 0:1, :, :].repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Model1_Predicted_label',
                    make_grid(pred1, 1, normalize=False),
                    trainer.current_iter,
                )
                pred2 = outputs_soft2[0:1, 0:1, :, :].repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Model2_Predicted_label',
                    make_grid(pred2, 1, normalize=False),
                    trainer.current_iter,
                )
                gt = label_batch[0:1, :, :].unsqueeze(1).repeat(1, 3, 1, 1)
                trainer.tensorboard_writer.add_image(
                    'train/Groundtruth_label',
                    make_grid(gt.float(), 1, normalize=False),
                    trainer.current_iter,
                )
            if (
                (trainer.current_iter > trainer.began_eval_iter
                 and trainer.current_iter % trainer.val_freq == 0)
                or trainer.current_iter == 20
            ):
                with torch.no_grad():
                    trainer.evaluation(model=trainer.model)
                    trainer.evaluation(model=trainer.model2, do_condition=True)
                trainer.model.train()
                trainer.model2.train()
            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                save_model_path = os.path.join(
                    trainer.output_folder,
                    'model_iter_' + str(trainer.current_iter) + '.pth'
                )
                torch.save(trainer.model.state_dict(), save_model_path)
                trainer.logging.info(f"save model to {save_model_path}")
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

        if trainer.use_PL:
            full_volume_batch = volume_batch[labeled_idxs_batch]
            full_label_batch = label_batch[labeled_idxs_batch]
            full_condition_batch = condition_batch[labeled_idxs_batch]
            con_list1 = full_label_batch[0].unique().tolist()
            con_list2 = full_label_batch[1].unique().tolist()
            if 0 in con_list1:
                con_list1.remove(0)
            inter_label_list = list(
                set(con_list1) & set(trainer.method_config['con_list'])
            )
            if len(inter_label_list) == 0:
                inter_label_list = trainer.method_config['con_list']
            con1 = np.random.choice(inter_label_list)
            if 0 in con_list2:
                con_list2.remove(0)
            inter_label_list = list(
                set(con_list2) & set(trainer.method_config['con_list'])
            )
            if len(inter_label_list) == 0:
                inter_label_list = trainer.method_config['con_list']
            con2 = np.random.choice(inter_label_list)
            full_condition_batch[0] = con1
            full_condition_batch[1] = con2
            for i_batch, sampled_batch in enumerate(trainer.dataloader_pl):
                trainer.model2.train()
                volume_batch, label_batch = (
                    sampled_batch['image'], sampled_batch['label']
                )
                label_batch = torch.argmax(label_batch, dim=1)
                volume_batch, label_batch = (
                    volume_batch.to(trainer.device),
                    label_batch.to(trainer.device),
                )
                condition_batch = torch.cat([
                    torch.Tensor([5]), torch.Tensor([5]),
                    torch.Tensor([5]), torch.Tensor([5]),
                ], dim=0).unsqueeze(1).to(trainer.device)
                noise = torch.clamp(
                    torch.randn_like(volume_batch) * 0.1, -0.2, 0.2
                ).to(trainer.device)
                outputs2 = trainer.model2(
                    volume_batch + noise, condition_batch
                )
                outputs_soft2 = torch.softmax(outputs2, dim=1)
                loss2 = 0.1 * (
                    trainer.ce_loss(outputs2, label_batch)
                    + trainer.dice_loss_con(
                        outputs_soft2, label_batch.unsqueeze(1)
                    )
                )
                model2_loss = loss2
                trainer.optimizer2.zero_grad()
                model2_loss.backward()
                trainer.optimizer2.step()
                trainer.tensorboard_writer.add_scalar(
                    'loss/model2_loss_pl',
                    model2_loss, trainer.current_iter
                )
                trainer.logging.info(
                    'iteration %d : model2 loss pl : %f' % (
                        trainer.current_iter, model2_loss.item()
                    )
                )
                break
    trainer.logger.close()
