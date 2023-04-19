"""Score by the energy"""
import torch

from util.get_ood_score import to_np


def energy_anomaly_score(output):
    """Computes the anomly score by taking the denominator of the softmax output (energy based)"""
    energy = -torch.logsumexp(output, -1)
    return to_np(energy)
