"""Score by the maximum logit"""
import torch

from anomaly_scores.max_logit import max_logit_anomaly_score


def max_softmax_anomaly_score(output):
    """Computes the anomly score by taking the max probability from softmax
    (see paper by Hendrycks)"""
    probabilities = torch.nn.functional.softmax(output, dim=-1)
    return max_logit_anomaly_score(probabilities)
