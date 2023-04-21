"""A module taken from the energy-ood paper (http://arxiv.org/abs/2010.03759),
with improved re-usability and adapted to use vim for training."""
import os
import time

import numpy as np
import torch
from torchvision import datasets

from energy_ood.CIFAR.models.wrn import WideResNet
from energy_ood.utils.svhn_loader import SVHN
from energy_ood.utils.validation_dataset import validation_split
from util import TEST_TRANSFORM, TRAIN_TRANSFORM
from vim_training.regimes import ENERGY, PRETRAINING
from vim_training.restore_model import restore_model
from vim_training.test import test
from vim_training.train import cosine_annealing, train

IS_RESTORING = False
REGIME = PRETRAINING
SEED = 1
MODEL_NAME = "WRN"
DATASET_NAME = "CIFAR10"
SNAPSHOT_FOLDER = "snapshots/"
DATA_FOLDER = "data/"

FILE_PREFIX = f"{DATASET_NAME}_{MODEL_NAME}"

EPOCHS = REGIME["epochs"]
BATCH_SIZE = 128
OOD_BATCH_SIZE = 256
LEARNING_RATE = REGIME["learning_rate"]
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005

NUM_CLASSES = 10

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

train_data_in = datasets.CIFAR10(f"{DATA_FOLDER}/cifar10", train=True, transform=TRAIN_TRANSFORM)
test_data = datasets.CIFAR10(f"{DATA_FOLDER}/cifar10", train=False, transform=TEST_TRANSFORM)
train_data_in, val_data = validation_split(train_data_in, val_share=0.1)
ood_data = SVHN(root=f"{DATA_FOLDER}/svhn/", split="test", transform=TEST_TRANSFORM, download=False)

train_loader_in = torch.utils.data.DataLoader(
    train_data_in, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
train_loader_out = torch.utils.data.DataLoader(
    ood_data, batch_size=OOD_BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
)

# Create model
model = WideResNet(40, NUM_CLASSES, 2, 0.3)
if IS_RESTORING:
    model = restore_model(model, MODEL_NAME, DATASET_NAME, SNAPSHOT_FOLDER)
model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
    nesterov=REGIME["is_using_nestrov"],
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

print("Beginning Training\n")
print(f" {'Epoch':<5s} |  Time | Train Loss | Test Loss | Test Accuracy")
# Main loop
for epoch in range(0, EPOCHS):
    begin_epoch = time.time()

    model, train_loss = train(model, train_loader_in, train_loader_out, scheduler, optimizer, REGIME)
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