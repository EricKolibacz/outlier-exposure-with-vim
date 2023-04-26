"""A module taken from the energy-ood paper (http://arxiv.org/abs/2010.03759),
with improved re-usability and adapted to use vim for training."""
import os
import time

import numpy as np
import torch
from torchvision import datasets

from energy_ood.CIFAR.models.wrn import WideResNet
from energy_ood.utils.svhn_loader import SVHN
from energy_ood.utils.tinyimages_80mn_loader import TinyImages
from energy_ood.utils.validation_dataset import validation_split
from util import TEST_TRANSFORM, TINY_TRANSFORM, TRAIN_TRANSFORM
from vim_training.model import WideResVIMNet
from vim_training.regimes import ENERGY, PRETRAIN_VIM, PRETRAINING, VANILLA_FT, VIM
from vim_training.restore_model import restore_model
from vim_training.testing import test
from vim_training.train import cosine_annealing, pretrain, train, train_with_energy, train_with_vim

REGIME = PRETRAIN_VIM
print(REGIME)
OOD_DATA = "tiny"
print(OOD_DATA)
SEED = 1
MODEL_NAME = "WRN"
DATASET_NAME = "CIFAR10"
SNAPSHOT_FOLDER = "snapshots/"
DATA_FOLDER = "data/"

FILE_PREFIX = f"{DATASET_NAME}_{MODEL_NAME}"

EPOCHS = REGIME["epochs"]
BATCH_SIZE = 128
BATCH_SIZE_TEST = 200
OOD_BATCH_SIZE = 256
LEARNING_RATE = REGIME["learning_rate"]
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

NUM_CLASSES = 10

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)

train_data_in = datasets.CIFAR10(f"{DATA_FOLDER}/cifar10", train=True, transform=TRAIN_TRANSFORM)
test_data = datasets.CIFAR10(f"{DATA_FOLDER}/cifar10", train=False, transform=TEST_TRANSFORM)
if REGIME["calibration"]:
    train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
if OOD_DATA == "svhn":
    ood_data = SVHN(root=f"{DATA_FOLDER}/svhn/", split="test", transform=TEST_TRANSFORM, download=False)
elif OOD_DATA == "tiny":
    ood_data = TinyImages(transform=TINY_TRANSFORM, folder="data/TinyImages")
else:
    raise ValueError(f"The dataset {OOD_DATA} is not known.")

train_loader_in = torch.utils.data.DataLoader(
    train_data_in, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
)
train_loader_out = torch.utils.data.DataLoader(
    ood_data, batch_size=OOD_BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE_TEST, shuffle=False, num_workers=4, pin_memory=True
)

# Create model
if "vim" in REGIME["name"]:
    print("Using WideResVIMNet")
    is_using_vim = "pretrain" not in REGIME["name"]
    print(f"Is using VIM? {is_using_vim}")
    model = WideResVIMNet(40, NUM_CLASSES, train_loader_in, 2, 0.3, is_using_vim=is_using_vim)
else:
    model = WideResNet(40, NUM_CLASSES, 2, 0.3)
if REGIME["loading"] != "":
    model.load_state_dict(torch.load(REGIME["loading"]))
model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        EPOCHS * len(train_loader_in),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / LEARNING_RATE,
    ),
)
model.eval()  # enter test mode
test_loss, test_accuracy = test(model, test_loader)
print(f"Starting with test accuracy of {test_accuracy*100:.2f}%")
print(f"and vanilla loss of {test_loss:.2f}")

print("Beginning Training\n")
print(f" {'Epoch':<5s} |  Time | Train Loss | Test Loss | Test Accuracy")
# Main loop
for epoch in range(0, EPOCHS):
    begin_epoch = time.time()
    model.train()  # enter train mode
    if REGIME["name"] == "default" or REGIME["name"] == "pretrain_vim":
        train_loss = pretrain(model, train_loader_in, scheduler, optimizer)
    elif REGIME["name"] == "energy":
        train_loss = train_with_energy(
            model,
            train_loader_in,
            train_loader_out,
            scheduler,
            optimizer,
            REGIME["m_in"],
            REGIME["m_out"],
        )
    elif REGIME["name"] == "vim":
        train_loss = train_with_vim(
            model,
            train_loader_in,
            train_loader_out,
            scheduler,
            optimizer,
            REGIME["m_in"],
            REGIME["m_out"],
        )
    else:
        train_loss = train(model, train_loader_in, train_loader_out, scheduler, optimizer, REGIME)
    model.eval()  # enter test mode
    test_loss, test_accuracy = test(model, test_loader)

    # Save model
    if (epoch + 1) % (EPOCHS / 5) == 0:  # save 5 models
        torch.save(
            model.state_dict(),
            os.path.join(
                SNAPSHOT_FOLDER,
                os.path.join(
                    REGIME["name"],
                    f"{FILE_PREFIX}_epoch_{str(epoch)}.pt",
                ),
            ),
        )
    end_time = int(time.time() - begin_epoch)
    print(f" {epoch + 1:5d} | {end_time:5d} | {train_loss:10.4f} | {test_loss:9.3f} | {test_accuracy:10.2f}")
