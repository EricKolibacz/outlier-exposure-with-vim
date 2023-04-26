"""Taken from Hendrycks et al. and added vim scoring"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import norm, pinv

from energy_ood.CIFAR.models.wrn import BasicBlock, NetworkBlock


class WideResVIMNet(nn.Module):
    def __init__(self, depth, num_classes, loader, widen_factor=1, dropRate=0.0, threshold=0.982):
        super(WideResVIMNet, self).__init__()

        self.threshold = threshold
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
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.update_vim_parameters(loader)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        vlogit = torch.mul(
            norm(
                torch.matmul(
                    out - self.u,
                    self.principal_space_perp,
                ),
                axis=-1,
            ),
            self.alpha,
        )
        out = self.fc(out)
        vlogits = torch.hstack([out, torch.unsqueeze(vlogit, dim=1)])

        return vlogits, out

    def forward_skip_vim(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out

    def update_vim_parameters(self, data_loader):
        # Extraction fully connected layer's weights and biases
        w, b = self.fc.weight, self.fc.bias
        # Origin of a new coordinate system of feature space to remove bias
        self.u = -torch.matmul(pinv(w), b)

        # Getting the first batch of the training data to calculate principal_space_perp and alpha
        training_data, _ = next(iter(data_loader))
        if next(self.parameters()).is_cuda:
            training_data = training_data.cuda()
        with torch.no_grad():
            logit, penultimate = self.forward_skip_vim(training_data)

        logit_id_train = logit
        feature_id_train = penultimate

        centered = feature_id_train - self.u
        covariance_matrix = torch.cov(centered.T)
        eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)
        variance_explained = eigen_values.real / torch.sum(
            eigen_values.real, dim=-1
        )  # TODO just taking the real values, ok?
        cumulative_variance_explained = torch.cumsum(variance_explained, dim=-1)

        self.principal_space_perp = (
            eigen_vectors.real.T[:][cumulative_variance_explained > self.threshold]
        ).T  # TODO just taking the real values, ok?

        max_logit, _ = torch.max(logit_id_train, dim=-1)
        vlogit_id_training = norm(torch.matmul(centered, self.principal_space_perp), axis=-1)
        self.alpha = torch.sum(max_logit) / torch.sum(vlogit_id_training)

    def cuda(self, device=None):
        self.u, self.principal_space_perp, self.alpha = (
            self.u.cuda(),
            self.principal_space_perp.cuda(),
            self.alpha.cuda(),
        )
        return self._apply(lambda t: t.cuda(device))


def compute_orthogonal_space(matrix, threshold):
    pca = PCA(n_components=None).fit(matrix)

    cumulative_explained_variance = torch.cumsum(pca.explained_variance_ratio_, dim=-1)

    eigen_vectors = pca.components_.detach().clone()

    eigen_vectors[cumulative_explained_variance <= threshold, :] = 0

    return eigen_vectors
