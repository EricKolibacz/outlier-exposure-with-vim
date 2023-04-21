"""Module to restore a pytorch model"""
import os

import torch


def restore_model(model, file_prefix: str, parent_folder: str):
    """Recursively look for a model trained on dataset in a folder"""
    model_found = False
    for epoch in range(1000 - 1, -1, -1):
        model_name = os.path.join(
            parent_folder, f"{file_prefix}_epoch_{str(epoch)}.pt"
        )
        if os.path.isfile(model_name):
            model.load_state_dict(torch.load(model_name))
            print("Model restored! Epoch:", epoch)
            model_found = True
            break
    if not model_found:
        assert False, "could not find model to restore"

    return model
