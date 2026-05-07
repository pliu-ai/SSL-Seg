"""SharedC3PS: C3PS with shared encoder between multi-class and condition decoders."""
import os
import numpy as np
import torch
from torch.cuda.amp import autocast
from torchvision.utils import make_grid
from tqdm import tqdm


def train_SharedC3PS(trainer) -> None:
    print("================> Training SharedC3PS <===============")
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
            trainer.model.train()
            volume_batch, label_batch = (
                sampled_batch['image'], sampled_batch['label']
            )
            volume_batch, label_batch = (
                volume_batch.to(trainer.device, non_blocking=True),
                label_batch.to(trainer.device, non_blocking=True),
            )
            # Convert per-sample int conditions to (2B, num_classes) multi-hot
            condition_raw = sampled_batch['condition']  # (B, 2) long
            condition_raw = torch.cat([
                condition_raw[:, 0], condition_raw[:, 1]
            ], dim=0)  # (2B,)
            condition_batch = torch.stack([
                trainer._con_group_to_multihot([int(c.item())])
                for c in condition_raw
            ]).to(trainer.device, non_blocking=True)
            ul1, br1, ul2, br2 = [], [], [], []
            labeled_idxs_batch = torch.arange(0, trainer.labeled_bs)
            unlabeled_idx_batch = torch.arange(trainer.labeled_bs, trainer.batch_size)
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
                    trainer.batch_size, trainer.batch_size + trainer.labeled_bs
                )
                labeled_idxs1_batch = torch.arange(0, trainer.labeled_bs)
                labeled_idxs_batch = torch.cat([labeled_idxs1_batch, labeled_idxs2_batch])
                unlabeled_idxs1_batch = torch.arange(trainer.labeled_bs, trainer.batch_size)
                unlabeled_idxs2_batch = torch.arange(
                    trainer.batch_size + trainer.labeled_bs, 2 * trainer.batch_size
                )
                unlabeled_idxs_batch = torch.cat(
                    [unlabeled_idxs1_batch, unlabeled_idxs2_batch]
                )
            noise = torch.clamp(torch.randn_like(volume_batch) * 0.1, -0.2, 0.2)
            trainer.optimizer.zero_grad(set_to_none=True)
            with autocast():
                # Step 1: Shared encode + decode1
                noisy_input = volume_batch + noise
                enc_feats = trainer.model.encode(noisy_input)
                # enc_feats = (conv1, conv2, conv3, conv4, center)
                outputs1 = trainer.model.decode1(*enc_feats)
                outputs_soft1 = torch.softmax(outputs1, dim=1)

                # Step 2: If CAC, use outputs1 to compute overlap and update condition
                if trainer.use_CAC and (
                    trainer.current_iter >= min(
                        trainer.began_semi_iter, trainer.began_condition_iter
                    )
                ):
                    overlap_soft1_list = []
                    overlap_outputs1_list = []
                    overlap_filter1_list = []
                    for unlabeled_idx1, unlabeled_idx2 in zip(
                        unlabeled_idxs1_batch, unlabeled_idxs2_batch
                    ):
                        overlap1_soft1 = outputs_soft1[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1], ul1[2][1]:br1[2][1]
                        ]
                        overlap2_soft1 = outputs_soft1[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1], ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_soft1.shape == overlap2_soft1.shape, \
                            "overlap region size must equal"
                        overlap1_outputs1 = outputs1[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1], ul1[2][1]:br1[2][1]
                        ]
                        overlap2_outputs1 = outputs1[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1], ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_outputs1.shape == overlap2_outputs1.shape, \
                            "overlap region size must equal"
                        overlap_outputs1_list.append(overlap1_outputs1.unsqueeze(0))
                        overlap_outputs1_list.append(overlap2_outputs1.unsqueeze(0))
                        overlap_soft1_tmp = (overlap1_soft1 + overlap2_soft1) / 2.
                        max1, pseudo_mask1 = torch.max(overlap_soft1_tmp, dim=0)
                        pred_con_list = pseudo_mask1.unique().tolist()
                        con = trainer._get_condition(pred_con_list)
                        con_dev = con.to(trainer.device)
                        is_cond = con_dev[pseudo_mask1.long()].bool()
                        is_bg = (pseudo_mask1 == 0)
                        if trainer.num_classes == 2:
                            thresh_map = torch.where(
                                is_cond,
                                torch.tensor(0.8, device=max1.device),
                                torch.tensor(float(trainer.model1_thresh), device=max1.device)
                            )
                        else:
                            thresh_map = torch.where(
                                is_bg,
                                torch.tensor(0.99, device=max1.device),
                                torch.tensor(0.9, device=max1.device)
                            )
                        overlap_filter1_tmp = trainer._compute_filter(max1, thresh_map)
                        overlap_soft1_list.append(overlap_soft1_tmp.unsqueeze(0))
                        overlap_filter1_list.append(overlap_filter1_tmp.unsqueeze(0))
                    overlap_soft1 = torch.cat(overlap_soft1_list, 0)
                    overlap_outputs1 = torch.cat(overlap_outputs1_list, 0)
                    overlap_filter1 = torch.cat(overlap_filter1_list, 0)
                    condition_batch[unlabeled_idxs_batch] = con_dev

                # Step 3: decode2 with (possibly updated) condition — reuse encoder features
                from networks.unet_3D_condition import _prepare_condition
                cond_vec = _prepare_condition(
                    condition_batch, noisy_input.shape[0],
                    trainer.model.num_conditions, trainer.device
                )
                cond_emb = (trainer.model.cond_enc(cond_vec)
                            if trainer.model.condition_mode == 'film' else None)
                outputs2 = trainer.model.decode2(*enc_feats, cond_vec, cond_emb)
                outputs_soft2 = torch.softmax(outputs2, dim=1)

                label_batch_con = trainer._get_label_batch_for_conditional_net(
                    label_batch, condition_batch
                )
                trainer.consistency_weight = trainer._get_current_consistency_weight(
                    trainer.current_iter // 150
                )
                loss1 = 0.5 * (
                    trainer.ce_loss(
                        outputs1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].long()
                    ) +
                    trainer.dice_loss(
                        outputs_soft1[labeled_idxs_batch],
                        label_batch[labeled_idxs_batch].unsqueeze(1)
                    )
                )
                loss2 = 0.5 * (
                    trainer.ce_loss(
                        outputs2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].long()
                    ) +
                    trainer.dice_loss_con(
                        outputs_soft2[labeled_idxs_batch],
                        label_batch_con[labeled_idxs_batch].unsqueeze(1)
                    )
                )

                if trainer.use_CAC and (
                    trainer.current_iter >= min(
                        trainer.began_semi_iter, trainer.began_condition_iter
                    )
                ):
                    overlap_soft2_list = []
                    overlap_outputs2_list = []
                    overlap_filter2_list = []
                    for unlabeled_idx1, unlabeled_idx2 in zip(
                        unlabeled_idxs1_batch, unlabeled_idxs2_batch
                    ):
                        overlap1_soft2 = outputs_soft2[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1], ul1[2][1]:br1[2][1]
                        ]
                        overlap2_soft2 = outputs_soft2[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1], ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_soft2.shape == overlap2_soft2.shape, \
                            "overlap region size must equal"
                        overlap1_outputs2 = outputs2[
                            unlabeled_idx1, :,
                            ul1[0][1]:br1[0][1], ul1[1][1]:br1[1][1], ul1[2][1]:br1[2][1]
                        ]
                        overlap2_outputs2 = outputs2[
                            unlabeled_idx2, :,
                            ul2[0][1]:br2[0][1], ul2[1][1]:br2[1][1], ul2[2][1]:br2[2][1]
                        ]
                        assert overlap1_outputs2.shape == overlap2_outputs2.shape, \
                            "overlap region size must equal"
                        overlap_outputs2_list.append(overlap1_outputs2.unsqueeze(0))
                        overlap_outputs2_list.append(overlap2_outputs2.unsqueeze(0))
                        overlap_soft2_tmp = (overlap1_soft2 + overlap2_soft2) / 2.
                        max2, pseudo_mask2 = torch.max(overlap_soft2_tmp, dim=0)
                        is_all_fg = (con_dev[1:].sum() >= trainer.num_classes - 1)
                        if trainer.num_classes == 2:
                            overlap_filter2_tmp = trainer._compute_filter(
                                max2, trainer.model2_thresh
                            )
                        else:
                            if not is_all_fg:
                                region_mask = (pseudo_mask2 > 0).float()
                            else:
                                region_mask = (pseudo_mask2 == 0).float()
                            overlap_filter2_tmp = (
                                trainer._compute_filter(max2, trainer.model2_thresh)
                                * region_mask
                            )
                        overlap_soft2_list.append(overlap_soft2_tmp.unsqueeze(0))
                        overlap_filter2_list.append(overlap_filter2_tmp.unsqueeze(0))
                    overlap_soft2 = torch.cat(overlap_soft2_list, 0)
                    overlap_outputs2 = torch.cat(overlap_outputs2_list, 0)
                    overlap_filter2 = torch.cat(overlap_filter2_list, 0)

                # pseudo_supervision1: model2 -> model1
                if trainer.current_iter < trainer.began_condition_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(trainer.device)
                else:
                    if trainer.use_CAC:
                        overlap_pseudo_outputs2 = torch.argmax(
                            overlap_soft2.detach(), dim=1, keepdim=False
                        )
                        if overlap_pseudo_outputs2.sum() == 0 or overlap_filter2.sum() == 0:
                            pseudo_supervision1 = torch.FloatTensor([0]).to(trainer.device)
                        else:
                            overlap_pseudo_outputs2 = torch.cat(
                                [overlap_pseudo_outputs2, overlap_pseudo_outputs2]
                            )
                            overlap_pseudo_filter2 = torch.cat(
                                [overlap_filter2, overlap_filter2]
                            )
                            ce_pseudo_supervision1 = trainer._cross_entropy_loss_con(
                                overlap_outputs1,
                                overlap_pseudo_outputs2,
                                condition_batch[unlabeled_idx_batch],
                                overlap_pseudo_filter2
                            )
                            pseudo_supervision1 = ce_pseudo_supervision1
                    else:
                        pseudo_outputs2 = torch.argmax(
                            outputs_soft2[trainer.labeled_bs:].detach(),
                            dim=1, keepdim=False
                        )
                        pseudo_supervision1 = trainer._cross_entropy_loss_con(
                            outputs1[trainer.labeled_bs:],
                            pseudo_outputs2,
                            condition_batch[trainer.labeled_bs:]
                        )

                # pseudo_supervision2: model1 -> model2
                if trainer.current_iter < trainer.began_semi_iter or (
                    trainer.use_CAC and overlap_filter1.sum() == 0
                ):
                    pseudo_supervision2 = torch.FloatTensor([0]).to(trainer.device)
                else:
                    if trainer.use_CAC:
                        overlap_pseudo_outputs1 = torch.argmax(
                            overlap_soft1.detach(), dim=1, keepdim=False
                        )
                        overlap_pseudo_outputs1 = torch.cat(
                            [overlap_pseudo_outputs1, overlap_pseudo_outputs1]
                        )
                        overlap_pseudo_filter1 = torch.cat(
                            [overlap_filter1, overlap_filter1]
                        )
                        target_ce_con = trainer._get_label_batch_for_conditional_net(
                            overlap_pseudo_outputs1,
                            condition_batch[unlabeled_idxs_batch]
                        )
                        ce_pseudo_supervision2 = trainer._weighted_ce_loss(
                            overlap_outputs2, target_ce_con, overlap_pseudo_filter1
                        )
                        dice_pseudo_supervision2 = trainer.dice_loss_con(
                            torch.softmax(overlap_outputs2, dim=1) * overlap_pseudo_filter1.unsqueeze(1),
                            (target_ce_con * overlap_pseudo_filter1).unsqueeze(1),
                            skip_id=0
                        )
                        pseudo_supervision2 = ce_pseudo_supervision2 + dice_pseudo_supervision2
                    else:
                        pseudo_outputs1 = torch.argmax(
                            outputs_soft1[trainer.labeled_bs:].detach(),
                            dim=1, keepdim=False
                        )
                        target_ce_con = trainer._get_label_batch_for_conditional_net(
                            pseudo_outputs1,
                            condition_batch[trainer.labeled_bs:]
                        )
                        pseudo_supervision2 = trainer.ce_loss(
                            outputs2[trainer.labeled_bs:],
                            target_ce_con
                        )

                model1_loss = loss1 + trainer.consistency_weight * pseudo_supervision1
                model2_loss = loss2 + trainer.consistency_weight * pseudo_supervision2
                total_loss = model1_loss + model2_loss

            # Single backward for shared encoder
            trainer.grad_scaler1.scale(total_loss).backward()
            trainer.grad_scaler1.unscale_(trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 12)
            trainer.grad_scaler1.step(trainer.optimizer)
            trainer.grad_scaler1.update()

            trainer.current_iter += 1
            trainer.current_lr = trainer.optimizer.param_groups[0]['lr']
            if trainer.current_iter % log_interval == 0 or trainer.current_iter == 1:
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
                    'loss/pseudo_supervision1', pseudo_supervision1, trainer.current_iter
                )
                trainer.tensorboard_writer.add_scalar(
                    'loss/pseudo_supervision2', pseudo_supervision2, trainer.current_iter
                )
                trainer.logging.info(
                    'iteration %d :model1 loss : %fmodel2 loss : %f'
                    'pseudo_supervision1 : %fpseudo_supervision2 : %f' % (
                        trainer.current_iter, model1_loss.item(),
                        model2_loss.item(),
                        pseudo_supervision1.item(),
                        pseudo_supervision2.item()
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
                    # model(x) -> out1 (multi-class eval)
                    trainer.evaluation(model=trainer.model)
                    # model(x, condition) -> out2 in eval mode (condition eval)
                    trainer.evaluation(model=trainer.model, do_condition=True)
                trainer.model.train()
            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                save_model_path = os.path.join(
                    trainer.output_folder,
                    'model_iter_' + str(trainer.current_iter) + '.pth'
                )
                torch.save(trainer.model.state_dict(), save_model_path)
                trainer.logging.info(f"save model to {save_model_path}")
            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
