"""CVCL (Cross-View Consistency Learning) semi-supervised training loop."""
import os
import torch
from tqdm import tqdm


def train_CVCL(trainer) -> None:
    print("================> Training CVCL<===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            trainer._adjust_learning_rate()
            trainer.model.train()
            volume_batch, label_batch = (
                sampled_batch['image'], sampled_batch['label']
            )
            volume_batch, label_batch = (
                volume_batch.to(trainer.device), label_batch.to(trainer.device)
            )
            labeled_idxs_batch = torch.arange(0, trainer.labeled_bs)
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
            unlabeled_idxs2_batch = torch.arange(
                trainer.batch_size + trainer.labeled_bs, 2 * trainer.batch_size
            )
            noise1 = torch.clamp(
                torch.randn_like(volume_batch) * 0.1, -0.2, 0.2
            ).to(trainer.device)
            outputs, CL_outputs = trainer.model(volume_batch + noise1)
            outputs_soft1 = torch.softmax(outputs, dim=1)
            loss_sup = (
                trainer.ce_loss(
                    outputs[labeled_idxs_batch],
                    label_batch[labeled_idxs_batch].long()
                ) +
                trainer.dice_loss(
                    outputs_soft1[labeled_idxs_batch],
                    label_batch[labeled_idxs_batch].unsqueeze(1)
                )
            )
            print(f"current iter:{trainer.current_iter}")
            print(f"loss sup:{loss_sup.item()}")
            loss_cl = torch.FloatTensor([0.0]).to(trainer.device)
            if trainer.current_iter > trainer.began_semi_iter:
                loss_cl = trainer.cvcl_loss(
                    output_ul1=CL_outputs[unlabeled_idxs1_batch],
                    output_ul2=CL_outputs[unlabeled_idxs2_batch],
                    logits1=outputs[unlabeled_idxs1_batch],
                    logits2=outputs[unlabeled_idxs2_batch],
                    ul1=[x[1] for x in sampled_batch['ul1']],
                    br1=[x[1] for x in sampled_batch['br1']],
                    ul2=[x[1] for x in sampled_batch['ul2']],
                    br2=[x[1] for x in sampled_batch['br2']]
                )
            loss = loss_sup + loss_cl
            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer.current_iter += 1
            for param_group in trainer.optimizer.param_groups:
                trainer.current_lr = param_group['lr']
            trainer.tensorboard_writer.add_scalar(
                'lr', trainer.current_lr, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/loss', loss, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/loss_sup', loss_sup, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'loss/loss_cl', loss_cl, trainer.current_iter
            )
            trainer.logging.info(
                'iteration %d: loss: %f supvised loss: %f CVCL loss: %f' % (
                    trainer.current_iter, loss.item(),
                    loss_sup.item(), loss_cl.item()
                )
            )
            if (trainer.current_iter > trainer.began_eval_iter and
                    trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                with torch.no_grad():
                    trainer.evaluation(model=trainer.model)
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
