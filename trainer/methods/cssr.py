"""CSSR (Cross-Scale Semi-supervised with high Resolution) training loop."""
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm


def train_CSSR(trainer) -> None:
    "cross supervision use high resolution"
    print("================> Training CSSR<===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    trainer.model = trainer.model.float().to(trainer.device)
    trainer.model2 = trainer.model2.float().to(trainer.device)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            trainer.model.train()
            trainer.model2.train()
            volume_large = sampled_batch['image_large'].float().to(trainer.device)
            label_large = sampled_batch['label_large'].float().to(trainer.device)
            volume_small = sampled_batch['image_small'].float().to(trainer.device)
            label_small = sampled_batch['label_small'].float().to(trainer.device)

            ul_large, br_large = sampled_batch['ul1'], sampled_batch['br1']
            ul_small, br_small = sampled_batch['ul2'], sampled_batch['br2']
            ul_large_u = [x[trainer.labeled_bs:] for x in ul_large]
            br_large_u = [x[trainer.labeled_bs:] for x in br_large]
            ul_small_u = [x[trainer.labeled_bs:] for x in ul_small]
            br_small_u = [x[trainer.labeled_bs:] for x in br_small]
            noise1 = torch.clamp(
                torch.randn_like(volume_small) * 0.1, -0.2, 0.2
            ).to(trainer.device)
            noise2 = torch.clamp(
                torch.randn_like(volume_large) * 0.1, -0.2, 0.2
            ).to(trainer.device)
            trainer.optimizer.zero_grad()
            trainer.optimizer2.zero_grad()
            with autocast():
                outputs1 = trainer.model(volume_small + noise1)
                outputs_soft1 = torch.softmax(outputs1, dim=1)
                outputs2 = trainer.model2(volume_large + noise2)
                outputs_soft2 = torch.softmax(outputs2, dim=1)

                trainer.consistency_weight = trainer._get_current_consistency_weight(
                    trainer.current_iter // 150
                )
                loss1 = 0.5 * (
                    trainer.ce_loss(outputs1[:trainer.labeled_bs],
                                    label_small[:trainer.labeled_bs].long()) +
                    trainer.dice_loss(outputs_soft1[:trainer.labeled_bs],
                                      label_small[:trainer.labeled_bs].unsqueeze(1))
                )
                loss2 = 0.5 * (
                    trainer.ce_loss(outputs2[:trainer.labeled_bs],
                                    label_large[:trainer.labeled_bs].long()) +
                    trainer.dice_loss(outputs_soft2[:trainer.labeled_bs],
                                      label_large[:trainer.labeled_bs].unsqueeze(1))
                )
                max_prob1, pseudo_outputs1 = torch.max(
                    outputs_soft1[trainer.labeled_bs:].detach(), dim=1
                )
                filter1 = (
                    ((max_prob1 > 0.99) & (pseudo_outputs1 == 0)) |
                    ((max_prob1 > 0.95) & (pseudo_outputs1 != 0))
                )
                max_prob2, pseudo_outputs2 = torch.max(
                    outputs_soft2[trainer.labeled_bs:].detach(), dim=1
                )
                filter2 = (
                    ((max_prob2 > 0.99) & (pseudo_outputs2 == 0)) |
                    ((max_prob2 > 0.95) & (pseudo_outputs2 != 0))
                )
                if trainer.current_iter < trainer.began_semi_iter:
                    pseudo_supervision1 = torch.FloatTensor([0]).to(trainer.device)
                    pseudo_supervision2 = torch.FloatTensor([0]).to(trainer.device)
                else:
                    pseudo_outputs2[filter2 == 0] = 255
                    pseudo_supervision1 = trainer.ce_loss(
                        outputs1[trainer.labeled_bs:, :,
                                 ul_small_u[0]:br_small_u[0],
                                 ul_small_u[1]:br_small_u[1],
                                 ul_small_u[2]:br_small_u[2]],
                        pseudo_outputs2[:, ul_large_u[0]:br_large_u[0],
                                        ul_large_u[1]:br_large_u[1],
                                        ul_large_u[2]:br_large_u[2]]
                    )
                    pseudo_outputs1[filter1 == 0] = 255
                    pseudo_supervision2 = trainer.ce_loss(
                        outputs2[trainer.labeled_bs:, :,
                                 ul_large_u[0]:br_large_u[0],
                                 ul_large_u[1]:br_large_u[1],
                                 ul_large_u[2]:br_large_u[2]],
                        pseudo_outputs1[:, ul_small_u[0]:br_small_u[0],
                                        ul_small_u[1]:br_small_u[1],
                                        ul_small_u[2]:br_small_u[2]]
                    )
                model1_loss = loss1 + trainer.consistency_weight * pseudo_supervision1
                model2_loss = loss2 + trainer.consistency_weight * pseudo_supervision2

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
            trainer.tensorboard_writer.add_scalar(
                'loss/pseudo1_loss', pseudo_supervision1, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/pseudo2_loss', pseudo_supervision2, trainer.current_iter
            )
            trainer.logging.info(
                'iteration %d : model1 loss : %f model2 loss : %f' % (
                    trainer.current_iter, model1_loss.item(), model2_loss.item()
                )
            )
            if (
                trainer.current_iter > trainer.began_eval_iter and
                trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                trainer.evaluation(model=trainer.model)
                trainer.model.train()
                trainer.model2.train()

            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                trainer._save_checkpoint("latest")
            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
