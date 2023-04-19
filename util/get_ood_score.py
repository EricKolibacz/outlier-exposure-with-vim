"""Code taken from the notebook to ood submission notebook of the Intro to ML Safety Course"""
import numpy as np
import torch


def concat(x):
    """Concatenates a vector"""
    return np.concatenate(x, axis=0)


def to_np(x):
    """Transforms a torch tensor to a numpy tensor"""
    return x.data.cpu().numpy()


def get_ood_scores(loader, anomaly_score_calculator, model_net, use_penultimate=False):
    """
    Calculates the anomaly scores for a portion of the given dataset.
    If a GPU is not available, will run on a smaller fraction of the
    dataset, so that calculations will be faster.

    loader: A DataLoader that contains the loaded data of a dataset
    anomaly_score_calculator: A function that takes in the output
                            logit of a batch of data and/or the
                            penultimate.
    model_net: The classifier model.
    usePenultimate: True if anomaly_score_calculator needs the
                    penultimate as a parameter. False otherwise.
    """
    _score = []

    ood_num_examples = len(loader.dataset) // 5
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if torch.cuda.is_available():
                fraction = 200
            else:
                fraction = 1000

            if batch_idx >= ood_num_examples // fraction:
                break

            if torch.cuda.is_available():
                data = data.cuda()

            output = model_net(data)
            probs = torch.softmax(output, dim=-1)

            if use_penultimate:
                score = anomaly_score_calculator(output)
            else:
                score = anomaly_score_calculator(probs)
            _score.append(score)

    return concat(_score).copy()
