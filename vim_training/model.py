"""Taken from Hendrycks et al. and added vim scoring"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm, pinv

from energy_ood.CIFAR.models.wrn import BasicBlock, NetworkBlock
from pytorch_pca.pca import PCA
from util.get_ood_score import to_np


class WideResVIMNet(nn.Module):
    def __init__(
        self,
        depth,
        num_classes,
        loader,
        widen_factor=1,
        dropRate=0.0,
        threshold=0.982,
        is_using_vim: bool = True,
    ):
        super(WideResVIMNet, self).__init__()

        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        # self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        training_data, _ = next(iter(loader))
        if next(self.parameters()).is_cuda:
            training_data = training_data.cuda()
        with torch.no_grad():
            penultimate = self.forward_skip_vim(training_data)
        self.vim = ViMBlock(num_classes, nChannels[3], penultimate, threshold, is_using_vim)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        vlogits, out = self.vim(out)

        return vlogits, out

    def forward_skip_vim(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out

    def update_vim_parameters(self, loader):
        # Getting the first batch of the training data to calculate principal_space_perp and alpha
        training_data, _ = next(iter(loader))
        if next(self.parameters()).is_cuda:
            training_data = training_data.cuda()
        with torch.no_grad():
            penultimate = self.forward_skip_vim(training_data)
        self.vim.update_vim_parameters(penultimate)

    def compute_anomaly_score(self, output):
        probs = torch.nn.functional.softmax(output, dim=-1)
        return to_np(probs[:, -1])


class ViMBlock(nn.Module):
    def __init__(
        self,
        num_classes,
        n_channels,
        penultimate,
        threshold: float = 0.982,
        is_using_vim: bool = True,
    ):
        super().__init__()
        self.threshold = threshold
        self.is_using_vim = is_using_vim
        self.fc = nn.Linear(n_channels, num_classes)
        self.update_vim_parameters(penultimate)

    def forward(self, x):
        out = self.fc(x)
        vlogit = torch.mul(
            norm(
                torch.matmul(
                    x - self.u,
                    self.orthogonal_space,
                ),
                axis=-1,
            ),
            self.alpha,
        )
        vlogits = torch.cat([out, torch.unsqueeze(vlogit, dim=1)], dim=1)
        return vlogits, out

    def update_vim_parameters(self, penultimate):
        # Extraction fully connected layer's weights and biases
        w, b = self.fc.weight, self.fc.bias
        # Origin of a new coordinate system of feature space to remove bias
        u = -torch.matmul(pinv(w), b).detach()

        logit_id_train = self.fc(penultimate)
        feature_id_train = penultimate

        centered = feature_id_train - u
        orthogonal_space = compute_orthogonal_space(centered.float(), self.threshold)

        max_logit, _ = torch.max(logit_id_train, dim=-1)
        vlogit_id_training = norm(torch.matmul(centered, orthogonal_space), axis=-1)
        alpha = torch.sum(max_logit) / torch.sum(vlogit_id_training).detach()
        if not self.is_using_vim:
            alpha -= torch.absolute(alpha) * 1e3  # mask alpha to approach - inf

        self.u = torch.nn.Parameter(u, requires_grad=False)
        self.orthogonal_space = torch.nn.Parameter(orthogonal_space, requires_grad=False)
        self.alpha = torch.nn.Parameter(alpha, requires_grad=False)

    def cuda(self, device=None):
        self.u, self.orthogonal_space, self.alpha = (
            self.u.cuda(),
            self.orthogonal_space.cuda(),
            self.alpha.cuda(),
        )
        return self._apply(lambda t: t.cuda(device))


def compute_orthogonal_space(matrix, threshold):
    pca = PCA(n_components=None).fit(matrix)

    cumulative_explained_variance = torch.cumsum(pca.explained_variance_ratio_, dim=-1)

    eigen_vectors = pca.components_.detach().clone()

    eigen_vectors[cumulative_explained_variance <= threshold, :] = 0

    return eigen_vectors
