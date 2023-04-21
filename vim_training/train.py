"""A module taken from the energy-ood paper (http://arxiv.org/abs/2010.03759),
with improved re-usability and adapted to use vim for training."""


import numpy as np
import torch
import torch.nn.functional as F


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def train(model, train_loader_in, train_loader_out, scheduler, optimizer, loss_method: dict):
    model.train()  # enter train mode
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        input_in, label_in = in_set
        input_out, _ = out_set

        input_in, label_in, input_out = input_in.cuda(), label_in.cuda(), input_out.cuda()

        # forward
        output_in, penultimate_in = model(input_in)

        loss = F.cross_entropy(output_in, label_in)
        # cross-entropy from softmax distribution to uniform distribution
        if loss_method["name"] == "energy":
            output_out, penultimate_out = model(input_out)
            Ec_in = -torch.logsumexp(output_in, dim=1)
            Ec_out = -torch.logsumexp(output_out, dim=1)
            loss += 0.1 * (
                torch.pow(F.relu(Ec_in - loss_method["m_in"]), 2).mean()
                + torch.pow(F.relu(loss_method["m_out"] - Ec_out), 2).mean()
            )
        elif loss_method["name"] == "OE":
            output_out, penultimate_out = model(input_out)
            loss += 0.5 * -(output_out.mean(1) - torch.logsumexp(output_in, dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return model, loss_avg
