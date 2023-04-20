"""Module which includes the virtual logit matching (vim) idea. 
Copied and adapted from the jupyter notebook submission on anomaly detection 
of the Intro to ML Safety course from the Center of AI Safety"""


import numpy as np
import torch
from numpy.linalg import norm, pinv
from scipy.special import logsumexp


class VIM:
    """
    Calculates and returns the space orthogonal to the principal space (principal_space_perp)
    and alpha values given the training data the model used.


    training_data_loader: A DataLoader that contains the loaded data of a
                        training dataset.
    model: The classifier model. Needs to return last two (!) layer outputs before the softmax (!)
    """

    def __init__(self, training_data_loader, model) -> None:
        result = []

        # Extraction fully connected layer's weights and biases
        w, b = model.fc.weight.cpu().detach().numpy(), model.fc.bias.cpu().detach().numpy()
        # Origin of a new coordinate system of feature space to remove bias
        self.u = -np.matmul(pinv(w), b)

        self.principal_space_perp, self.alpha = self._get_parameters(training_data_loader, model)

    def _get_parameters(self, training_data_loader, model):
        # Getting the first batch of the training data to calculate principal_space_perp and alpha
        training_data, _ = next(iter(training_data_loader))
        if torch.cuda.is_available():
            training_data = training_data.cuda()

        result = model(training_data)
        logit = result[0]  # Logits (values before softmax)
        penultimate = result[1]  # Features/Penultimate (values before fully connected layer)

        logit_id_train = logit.cpu().detach().numpy().squeeze()
        feature_id_train = penultimate.cpu().detach().numpy().squeeze()

        centered = feature_id_train - self.u
        covariance_matrix = np.cov(centered.T)
        _, eigen_vectors = np.linalg.eig(covariance_matrix)
        principal_space_perp = (eigen_vectors.T[:][12:]).T

        vlogit_id_training = norm(np.matmul(centered, principal_space_perp), axis=-1)
        alpha = np.sum(logit_id_train.max(axis=-1)) / np.sum(vlogit_id_training)

        return principal_space_perp, alpha

    def compute_anomaly_score(self, output, penultimate):
        logit_id_val = output.cpu().detach().numpy().squeeze()
        feature_id_val = penultimate.cpu().detach().numpy().squeeze()

        vlogit = self.alpha * norm(np.matmul(feature_id_val - self.u, self.principal_space_perp), axis=-1)
        energy = logsumexp(logit_id_val, axis=-1)

        score_id = vlogit - energy

        return score_id
