"""CAML (Correlation-Aware Mutual Learning) semi-supervised training loop."""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils import losses, feature_memory, correlation
from trainer.train_utils import sharpening


def train_CAML(trainer) -> None:
    "Correlation-Aware Mutual Learning"
    print("================> Training CAML<===============")
    iterator = tqdm(range(trainer.max_epoch), ncols=70)
    consistency_criterion = losses.mse_loss
    memory_num = 256
    num_filtered = 12800
    lambda_s = 0.5
    memory_bank = feature_memory.MemoryBank(
        num_labeled_samples=trainer.labeled_num, num_cls=trainer.num_classes
    )
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainer.dataloader):
            volume_batch = sampled_batch['image']
            label_batch = sampled_batch['label']
            idx = sampled_batch['idx']
            volume_batch, label_batch, idx = (
                volume_batch.cuda(), label_batch.cuda(), idx.cuda()
            )
            label_batch = torch.argmax(label_batch, dim=1)
            trainer.model.train()
            outputs_v, outputs_a, embedding_v, embedding_a = trainer.model(volume_batch)
            outputs_list = [outputs_v, outputs_a]
            num_outputs = len(outputs_list)

            y_ori = torch.zeros((num_outputs,) + outputs_list[0].shape)
            y_pseudo_label = torch.zeros((num_outputs,) + outputs_list[0].shape)

            loss_s = 0
            for i in range(num_outputs):
                y = outputs_list[i][:trainer.labeled_bs, ...]
                y_prob = F.softmax(y, dim=1)
                loss_s += trainer.dice_loss(
                    y_prob[:, ...], label_batch[:trainer.labeled_bs].unsqueeze(1)
                )
                y_all = outputs_list[i]
                y_prob_all = F.softmax(y_all, dim=1)
                y_ori[i] = y_prob_all
                y_pseudo_label[i] = sharpening(y_prob_all)

            loss_c = 0
            for i in range(num_outputs):
                for j in range(num_outputs):
                    if i != j:
                        loss_c += consistency_criterion(y_ori[i], y_pseudo_label[j])

            outputs_v_soft = F.softmax(outputs_v, dim=1)
            outputs_a_soft = F.softmax(outputs_a, dim=1)
            labeled_features_v = embedding_v[:trainer.labeled_bs, ...]
            labeled_features_a = embedding_a[:trainer.labeled_bs, ...]
            unlabeled_features_v = embedding_v[trainer.labeled_bs:, ...]
            unlabeled_features_a = embedding_a[trainer.labeled_bs:, ...]

            y_v = outputs_v_soft[:trainer.labeled_bs]
            y_a = outputs_a_soft[:trainer.labeled_bs]
            true_labels = label_batch[:trainer.labeled_bs]

            _, prediction_label_v = torch.max(y_v, dim=1)
            _, prediction_label_a = torch.max(y_a, dim=1)
            predicted_unlabel_prob_v, predicted_unlabel_v = torch.max(
                outputs_v_soft[trainer.labeled_bs:], dim=1
            )
            predicted_unlabel_prob_a, predicted_unlabel_a = torch.max(
                outputs_a_soft[trainer.labeled_bs:], dim=1
            )

            mask_prediction_correctly = (
                ((prediction_label_a == true_labels).float() +
                 (prediction_label_v == true_labels).float()) == 2
            )

            labeled_features_v = labeled_features_v.permute(0, 2, 3, 4, 1).contiguous()
            b, h, w, d, labeled_features_dim = labeled_features_v.shape

            trainer.model.eval()
            proj_labeled_features_v = trainer.model.projection_head1(
                labeled_features_v.view(-1, labeled_features_dim)
            )
            proj_labeled_features_v = proj_labeled_features_v.view(b, h, w, d, -1)
            proj_labeled_features_a = trainer.model.projection_head2(
                labeled_features_a.view(-1, labeled_features_dim)
            )
            proj_labeled_features_a = proj_labeled_features_a.view(b, h, w, d, -1)
            trainer.model.train()

            labels_correct_list = []
            labeled_features_correct_list = []
            labeled_index_list = []
            for i in range(trainer.labeled_bs):
                labels_correct_list.append(
                    true_labels[i][mask_prediction_correctly[i]]
                )
                labeled_features_correct_list.append(
                    (proj_labeled_features_v[i][mask_prediction_correctly[i]] +
                     proj_labeled_features_a[i][mask_prediction_correctly[i]]) / 2
                )
                labeled_index_list.append(idx[i])

            labeled_index = idx[:trainer.labeled_bs]
            memory_bank.update_labeled_features(
                labeled_features_correct_list, labels_correct_list, labeled_index_list
            )
            memory = memory_bank.sample_labeled_features(memory_num)

            mask_consist_unlabeled = predicted_unlabel_v == predicted_unlabel_a
            consist_unlabel = predicted_unlabel_v[mask_consist_unlabeled]
            consist_unlabel_prob = predicted_unlabel_prob_v[mask_consist_unlabeled]

            unlabeled_features_v = unlabeled_features_v.permute(0, 2, 3, 4, 1)
            unlabeled_features_a = unlabeled_features_a.permute(0, 2, 3, 4, 1)
            unlabeled_features_v = unlabeled_features_v[mask_consist_unlabeled, :]
            unlabeled_features_a = unlabeled_features_a[mask_consist_unlabeled, :]

            projected_feature_v = trainer.model.projection_head1(unlabeled_features_v)
            predicted_feature_v = trainer.model.prediction_head1(projected_feature_v)
            corr_v, corr_v_available = correlation.cal_correlation_matrix(
                predicted_feature_v, consist_unlabel_prob, consist_unlabel,
                memory, trainer.num_classes, num_filtered=num_filtered
            )

            projected_feature_a = trainer.model.projection_head2(unlabeled_features_a)
            predicted_feature_a = trainer.model.prediction_head2(projected_feature_a)
            corr_a, corr_a_available = correlation.cal_correlation_matrix(
                predicted_feature_a, consist_unlabel_prob, consist_unlabel,
                memory, trainer.num_classes, num_filtered=num_filtered
            )

            if corr_v_available and corr_a_available:
                num_samples = corr_a.shape[0]
                loss_o = torch.sum(
                    torch.sum(-corr_a * torch.log(corr_v + 1e-8), dim=1)
                ) / num_samples
            else:
                loss_o = 0

            lambda_c = trainer._get_lambda_c(trainer.current_iter // 150)
            lambda_o = trainer._get_lambda_o(trainer.current_iter // 150)

            loss = lambda_s * loss_s

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()
            trainer._adjust_learning_rate()
            trainer.current_iter += 1
            trainer.logging.info(
                'iteration %d : loss : %03f, loss_s: %03f, loss_c: %03f, loss_o: %03f' % (
                    trainer.current_iter, loss, loss_s, loss_c, loss_o
                )
            )
            trainer.tensorboard_writer.add_scalar(
                'Labeled_loss/loss_s', loss_s, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'Co_loss/loss_c', loss_c, trainer.current_iter
            )
            trainer.tensorboard_writer.add_scalar(
                'Co_loss/loss_o', loss_o, trainer.current_iter
            )
            if (
                trainer.current_iter > trainer.began_eval_iter and
                trainer.current_iter % trainer.val_freq == 0
            ) or trainer.current_iter == 20:
                trainer.evaluation(model=trainer.model)
                trainer.model.train()

            if trainer.current_iter % trainer.save_checkpoint_freq == 0:
                trainer._save_checkpoint("latest")
            if trainer.current_iter >= trainer.max_iterations:
                break
        if trainer.current_iter >= trainer.max_iterations:
            iterator.close()
            break
    trainer.logger.close()
