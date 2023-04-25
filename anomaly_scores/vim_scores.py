"""Module which includes the virtual logit matching (vim) idea. 
Copied and adapted from the jupyter notebook submission on anomaly detection 
of the Intro to ML Safety course from the Center of AI Safety"""


import torch
from scipy.special import logsumexp
from torch.linalg import norm, pinv

from util.get_ood_score import to_np

THRESHOLD = 0.982  # how much variance should be in the principale space spanned by the pca
# (the rest will be in orthogonal space)


class VIM:
    """
    Calculates and returns the space orthogonal to the principal space (principal_space_perp)
    and alpha values given the training data the model used.


    training_data_loader: A DataLoader that contains the loaded data of a
                        training dataset.
    model: The classifier model. Needs to return last two (!) layer outputs before the softmax (!)
    """

    def __init__(self, training_data_loader, model) -> None:
        # Extraction fully connected layer's weights and biases
        w, b = model.fc.weight, model.fc.bias
        # Origin of a new coordinate system of feature space to remove bias
        self.u = -torch.matmul(pinv(w), b).detach()

        self.principal_space_perp, self.alpha = self._get_parameters(training_data_loader, model)

    def _get_parameters(self, training_data_loader, model):
        # Getting the first batch of the training data to calculate principal_space_perp and alpha
        training_data, _ = next(iter(training_data_loader))
        if torch.cuda.is_available():
            training_data = training_data.cuda()

        result = model(training_data)
        logit = result[0]  # Logits (values before softmax)
        penultimate = result[1]  # Features/Penultimate (values before fully connected layer)

        logit_id_train = logit
        feature_id_train = penultimate

        centered = feature_id_train - self.u
        covariance_matrix = torch.cov(centered.T)
        eigen_values, eigen_vectors = torch.linalg.eig(covariance_matrix)
        variance_explained = eigen_values.real / torch.sum(
            eigen_values.real, dim=-1
        )  # TODO just taking the real values, ok?
        cumulative_variance_explained = torch.cumsum(variance_explained, dim=-1)

        principal_space_perp = (
            eigen_vectors.real.T[:][cumulative_variance_explained > THRESHOLD]
        ).T  # TODO just taking the real values, ok?

        max_logit, _ = torch.max(logit_id_train, dim=-1)
        vlogit_id_training = norm(torch.matmul(centered, principal_space_perp), axis=-1)
        alpha = torch.sum(max_logit) / torch.sum(vlogit_id_training)

        return principal_space_perp.detach(), alpha.detach()

    def compute_anomaly_score(self, output, penultimate):
        _, vprobs = self.compute_vlogits(output, penultimate)
        return to_np(vprobs[:, -1])

    def compute_vlogits(self, output, penultimate):
        vlogit = self.alpha * norm(
            torch.matmul(
                penultimate - self.u,
                self.principal_space_perp,
            ),
            axis=-1,
        )
        vlogits = torch.hstack([output, torch.unsqueeze(vlogit, dim=1)])
        vprobs = torch.nn.functional.softmax(vlogits, dim=-1)
        return vlogits, vprobs
