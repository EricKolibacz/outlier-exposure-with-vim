"""Code taken from the notebook to ood submission notebook of the Intro to ML Safety Course"""
import numpy as np
import torch

from energy_ood.utils.display_results import get_measures


def concat(x):
    """Concatenates a vector"""
    return np.concatenate(x, axis=0)


def to_np(x):
    """Transforms a torch tensor to a numpy tensor"""
    return x.data.cpu().numpy()


def get_ood_scores(loader, model, anomaly_score_calculator, ood_num_examples, in_dist=False, is_using="last"):
    """
    Calculates the anomaly scores for a portion of the given dataset.
    If a GPU is not available, will run on a smaller fraction of the
    dataset, so that calculations will be faster.

    loader: A DataLoader that contains the loaded data of a dataset
    model: The classifier model.
    anomaly_score_calculator: A function that takes in the output
                            logit of a batch of data and/or the
                            penultimate.
    is_using: "last" if anomaly score calculator needs last output
              "last_penultimate" if needs last and penultimate
    """
    _score = []

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if batch_idx >= ood_num_examples // loader.batch_size and in_dist is False:
                break

            if torch.cuda.is_available():
                data = data.cuda()

            output, penultimate_output = model(data)

            if is_using == "last":
                score = anomaly_score_calculator(output)
            elif is_using == "last_penultimate":
                score = anomaly_score_calculator(output, penultimate_output)
            else:
                raise ValueError(f"Did not recognize is_using option: {is_using}")
            _score.append(score)

    return concat(_score)[:ood_num_examples].copy()


def get_ood_score_for_multiple_datasets(
    loaders,
    model,
    anomaly_score_calculator,
    is_using="last",
    verbose=False,
    runs=1,
):
    ood_num_examples = len(loaders[0][1].dataset) // 5

    if verbose:
        print(f"In distribution dataset: {loaders[0][0]}")

    in_score = get_ood_scores(
        loaders[0][1],
        model,
        anomaly_score_calculator,
        ood_num_examples,
        in_dist=True,
        is_using=is_using,
    )
    results = []

    for i in range(1, len(loaders)):
        if verbose:
            print(f"OOD dataset: {loaders[i][0]}")
        aurocs, auprs, fprs = [], [], []
        for _ in range(runs):
            out_score = get_ood_scores(
                loaders[i][1],
                model,
                anomaly_score_calculator,
                ood_num_examples,
                is_using=is_using,
            )
            auroc, aupr, fpr = get_measures(out_score[:], in_score[:])
            aurocs.append([auroc])
            auprs.append([aupr])
            fprs.append([fpr])
        results.append([aurocs, auprs, fprs])

    return results
