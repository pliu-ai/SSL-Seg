# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.networks_other import init_weights
from networks.utils import UnetConv3, UnetUp3, UnetUp3_CT


def _prepare_condition_index(condition, batch_size, device):
    """Convert condition input to long index tensor of shape (B,)."""
    if not torch.is_tensor(condition):
        condition = torch.as_tensor(condition, device=device)
    else:
        condition = condition.to(device=device)
    condition = condition.reshape(-1).long()
    if condition.numel() == 1 and batch_size > 1:
        condition = condition.expand(batch_size)
    elif condition.numel() != batch_size:
        raise ValueError(
            f"condition batch size mismatch: got {condition.numel()} values for batch {batch_size}"
        )
    return condition


def _condition_to_vector(condition, num_conditions):
    """Convert condition index to one-hot/multi-hot vector.

    Args:
        condition: (B,) long tensor.
        num_conditions: int, number of classes (including background).
    Returns:
        (B, num_conditions) float tensor.
    """
    B = condition.shape[0]
    cond_vec = torch.zeros(B, num_conditions, device=condition.device)

    normal_mask = condition < num_conditions
    if normal_mask.any():
        cond_vec[normal_mask] = cond_vec[normal_mask].scatter_(
            1, condition[normal_mask].unsqueeze(1), 1.0
        )

    all_fg_mask = condition >= num_conditions
    if all_fg_mask.any():
        cond_vec[all_fg_mask, 1:] = 1.0

    return cond_vec


class ScalarCondition(nn.Module):
    """Scalar mode (original): int -> fill spatial tensor with scalar value -> concat.

    No learnable parameters.
    """

    def __init__(self, embed_dim=2):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, condition, spatial_shape):
        B = condition.shape[0]
        cond = condition.float().view(B, 1, 1, 1, 1)
        cond = cond.expand(B, self.embed_dim, *spatial_shape)
        return cond


class ConditionEmbedding(nn.Module):
    """Concat mode: int -> one-hot/multi-hot -> Linear -> spatial expand -> concat."""

    def __init__(self, num_conditions, embed_dim):
        super().__init__()
        self.num_conditions = num_conditions
        self.embed_dim = embed_dim
        self.proj = nn.Linear(num_conditions, embed_dim)

    def forward(self, condition, spatial_shape):
        """
        Args:
            condition: (B,) long tensor of class indices.
            spatial_shape: tuple (D, W, H).
        Returns:
            (B, embed_dim, D, W, H) condition feature map.
        """
        cond_vec = _condition_to_vector(condition, self.num_conditions)
        cond_emb = self.proj(cond_vec)  # (B, embed_dim)
        cond_emb = cond_emb.view(cond_emb.shape[0], self.embed_dim, 1, 1, 1)
        cond_emb = cond_emb.expand(-1, -1, *spatial_shape)
        return cond_emb


class ConditionEncoder(nn.Module):
    """Shared condition encoder: int -> one-hot/multi-hot -> MLP -> cond_emb."""

    def __init__(self, num_conditions, cond_dim=32):
        super().__init__()
        self.num_conditions = num_conditions
        self.mlp = nn.Sequential(
            nn.Linear(num_conditions, cond_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, condition):
        """
        Args:
            condition: (B,) long tensor.
        Returns:
            (B, cond_dim) condition embedding.
        """
        cond_vec = _condition_to_vector(condition, self.num_conditions)
        return self.mlp(cond_vec)


class FiLMLayer(nn.Module):
    """Generate per-channel scale (gamma) & shift (beta) from condition embedding."""

    def __init__(self, cond_dim, num_channels):
        super().__init__()
        self.fc = nn.Linear(cond_dim, num_channels * 2)
        # Identity init: gamma=1, beta=0 so FiLM is a no-op at start
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        self.fc.bias.data[:num_channels] = 1.0

    def forward(self, feature, cond_emb):
        """
        Args:
            feature:  (B, C, D, W, H)
            cond_emb: (B, cond_dim)
        Returns:
            (B, C, D, W, H) modulated feature.
        """
        gamma_beta = self.fc(cond_emb)                    # (B, C*2)
        gamma, beta = gamma_beta.chunk(2, dim=1)           # each (B, C)
        gamma = gamma.view(gamma.shape[0], -1, 1, 1, 1)   # (B, C, 1, 1, 1)
        beta = beta.view(beta.shape[0], -1, 1, 1, 1)
        return gamma * feature + beta


class unet_3D_Condition(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True,
                 in_channels=3, is_batchnorm=True,
                 num_conditions=None, embed_dim=8,
                 condition_mode='concat', cond_dim=32):
        super(unet_3D_Condition, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.condition_mode = condition_mode

        if num_conditions is None:
            num_conditions = 16

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        if condition_mode == 'scalar':
            self.embed_dim = embed_dim
            self.cond_embed = ScalarCondition(embed_dim)
            ed = embed_dim
        elif condition_mode == 'concat':
            self.embed_dim = embed_dim
            self.cond_embed = ConditionEmbedding(num_conditions, embed_dim)
            ed = embed_dim
        elif condition_mode == 'film':
            ed = 0
            self.cond_enc = ConditionEncoder(num_conditions, cond_dim)
            self.film_mp1 = FiLMLayer(cond_dim, filters[0])
            self.film_mp2 = FiLMLayer(cond_dim, filters[1])
            self.film_mp3 = FiLMLayer(cond_dim, filters[2])
            self.film_mp4 = FiLMLayer(cond_dim, filters[3])
            self.film_center = FiLMLayer(cond_dim, filters[4])
            self.film_up4 = FiLMLayer(cond_dim, filters[3])
            self.film_up3 = FiLMLayer(cond_dim, filters[2])
            self.film_up2 = FiLMLayer(cond_dim, filters[1])
            self.film_up1 = FiLMLayer(cond_dim, filters[0])
        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0] + ed, filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1] + ed, filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2] + ed, filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3] + ed, filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4] + ed, filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3] + ed, filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2] + ed, filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1] + ed, filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0] + ed, n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def _inject(self, feature, condition, cond_emb, film_layer):
        if self.condition_mode in ('scalar', 'concat'):
            c = self.cond_embed(condition, feature.shape[2:])
            return torch.cat((c, feature), dim=1)
        else:
            return film_layer(feature, cond_emb)

    def forward(self, inputs, condition=1):
        condition = _prepare_condition_index(condition, inputs.shape[0], inputs.device)
        cond_emb = self.cond_enc(condition) if self.condition_mode == 'film' else None

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        maxpool1 = self._inject(maxpool1, condition, cond_emb,
                                getattr(self, 'film_mp1', None))

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        maxpool2 = self._inject(maxpool2, condition, cond_emb,
                                getattr(self, 'film_mp2', None))

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        maxpool3 = self._inject(maxpool3, condition, cond_emb,
                                getattr(self, 'film_mp3', None))

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        maxpool4 = self._inject(maxpool4, condition, cond_emb,
                                getattr(self, 'film_mp4', None))

        center = self.center(maxpool4)
        center = self.dropout1(center)
        center = self._inject(center, condition, cond_emb,
                              getattr(self, 'film_center', None))

        up4 = self.up_concat4(conv4, center)
        up4 = self._inject(up4, condition, cond_emb,
                           getattr(self, 'film_up4', None))

        up3 = self.up_concat3(conv3, up4)
        up3 = self._inject(up3, condition, cond_emb,
                           getattr(self, 'film_up3', None))

        up2 = self.up_concat2(conv2, up3)
        up2 = self._inject(up2, condition, cond_emb,
                           getattr(self, 'film_up2', None))

        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)
        up1 = self._inject(up1, condition, cond_emb,
                           getattr(self, 'film_up1', None))

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p



class Unet3DConditionDecoder(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True,
                 in_channels=3, is_batchnorm=True,
                 num_conditions=None, embed_dim=8,
                 condition_mode='concat', cond_dim=32):
        super(Unet3DConditionDecoder, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.condition_mode = condition_mode

        if num_conditions is None:
            num_conditions = 16

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        if condition_mode == 'scalar':
            self.embed_dim = embed_dim
            self.cond_embed = ScalarCondition(embed_dim)
            ed = embed_dim
        elif condition_mode == 'concat':
            self.embed_dim = embed_dim
            self.cond_embed = ConditionEmbedding(num_conditions, embed_dim)
            ed = embed_dim
        elif condition_mode == 'film':
            ed = 0
            self.cond_enc = ConditionEncoder(num_conditions, cond_dim)
            self.film_mp4 = FiLMLayer(cond_dim, filters[3])
            self.film_center = FiLMLayer(cond_dim, filters[4])
            self.film_up4 = FiLMLayer(cond_dim, filters[3])
            self.film_up3 = FiLMLayer(cond_dim, filters[2])
            self.film_up2 = FiLMLayer(cond_dim, filters[1])
            self.film_up1 = FiLMLayer(cond_dim, filters[0])
        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        # downsampling (encoder: no condition injection)
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3] + ed, filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling (decoder: with condition injection)
        self.up_concat4 = UnetUp3_CT(filters[4] + ed, filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3] + ed, filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2] + ed, filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1] + ed, filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0] + ed, n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def _inject(self, feature, condition, cond_emb, film_layer):
        if self.condition_mode in ('scalar', 'concat'):
            c = self.cond_embed(condition, feature.shape[2:])
            return torch.cat((c, feature), dim=1)
        else:
            return film_layer(feature, cond_emb)

    def forward(self, inputs, condition=1):
        condition = _prepare_condition_index(condition, inputs.shape[0], inputs.device)
        cond_emb = self.cond_enc(condition) if self.condition_mode == 'film' else None

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        maxpool4 = self._inject(maxpool4, condition, cond_emb,
                                getattr(self, 'film_mp4', None))

        center = self.center(maxpool4)
        center = self.dropout1(center)

        center = self._inject(center, condition, cond_emb,
                              getattr(self, 'film_center', None))

        up4 = self.up_concat4(conv4, center)
        up4 = self._inject(up4, condition, cond_emb,
                           getattr(self, 'film_up4', None))

        up3 = self.up_concat3(conv3, up4)
        up3 = self._inject(up3, condition, cond_emb,
                           getattr(self, 'film_up3', None))

        up2 = self.up_concat2(conv2, up3)
        up2 = self._inject(up2, condition, cond_emb,
                           getattr(self, 'film_up2', None))

        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)
        up1 = self._inject(up1, condition, cond_emb,
                           getattr(self, 'film_up1', None))

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class Unet3DConditionBottom(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True,
                 in_channels=3, is_batchnorm=True,
                 num_conditions=None, embed_dim=8,
                 condition_mode='concat', cond_dim=32):
        super(Unet3DConditionBottom, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.condition_mode = condition_mode

        if num_conditions is None:
            num_conditions = 16

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        if condition_mode == 'scalar':
            self.embed_dim = embed_dim
            self.cond_embed = ScalarCondition(embed_dim)
            ed = embed_dim
        elif condition_mode == 'concat':
            self.embed_dim = embed_dim
            self.cond_embed = ConditionEmbedding(num_conditions, embed_dim)
            ed = embed_dim
        elif condition_mode == 'film':
            ed = 0
            self.cond_enc = ConditionEncoder(num_conditions, cond_dim)
            self.film_center = FiLMLayer(cond_dim, filters[4])
        else:
            raise ValueError(f"Unknown condition_mode: {condition_mode}")

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling (only bottleneck gets condition)
        self.up_concat4 = UnetUp3_CT(filters[4] + ed, filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, condition=1):
        condition = _prepare_condition_index(condition, inputs.shape[0], inputs.device)

        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)
        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)
        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        center = self.dropout1(center)

        if self.condition_mode in ('scalar', 'concat'):
            c = self.cond_embed(condition, center.shape[2:])
            center = torch.cat((c, center), dim=1)
        else:
            cond_emb = self.cond_enc(condition)
            center = self.film_center(center, cond_emb)

        up4 = self.up_concat4(conv4, center)

        up3 = self.up_concat3(conv3, up4)

        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
