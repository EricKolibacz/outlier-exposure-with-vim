"""Score by the maximum logit"""
import torch

from util.get_ood_score import to_np


def max_logit_anomaly_score(output):
    """Computes the anomly score by taking the max logit (similar to the paper by Hendrycks)"""
    maxima, _ = torch.max(output, -1)
    
    return -1 * to_np(maxima)
