"""A module taken from the energy-ood paper (http://arxiv.org/abs/2010.03759),
with improved re-usability and adapted to use vim for training."""
import numpy as np
import torch
import torch.nn.functional as F

from anomaly_scores.vim_scores import VIM


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def pretrain(model, loader, scheduler, optimizer):
    loss_avg = 0.0
    for data, label in loader:
        data, label = data.cuda(), label.cuda()

        # forward
        output, _ = model(data)

        loss = F.cross_entropy(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return loss_avg


def train_with_energy(
    model,
    train_loader_in,
    train_loader_out,
    scheduler,
    optimizer,
    m_in: float,
    m_out: float,
):
    loss_avg = 0.0

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x, _ = model(data)

        loss = F.cross_entropy(x[: len(in_set[0])], target)

        Ec_out = -torch.logsumexp(x[len(in_set[0]) :], dim=1)
        Ec_in = -torch.logsumexp(x[: len(in_set[0])], dim=1)
        loss += 0.1 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return loss_avg


def train_with_vim(
    model,
    train_loader_in,
    train_loader_out,
    scheduler,
    optimizer,
    m_in: float,
    m_out: float,
):
    loss_avg = 0.0
    model.update_vim_parameters(train_loader_in)
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x, _ = model(data)

        loss = F.cross_entropy(x[: len(in_set[0]), :-1], target)

        vprobs = torch.nn.functional.softmax(x, dim=-1)

        loss += 10 * (
            torch.pow(F.relu(vprobs[: len(in_set[0]), -1] - m_in), 2).mean()
            + torch.pow(F.relu(m_out - vprobs[len(in_set[0]) :, -1]), 2).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return loss_avg


def train_with_energy_and_vim(
    model,
    train_loader_in,
    train_loader_out,
    scheduler,
    optimizer,
    m_ins: float,
    m_outs: float,
):
    loss_avg = 0.0
    model.update_vim_parameters(train_loader_in)
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        data = torch.cat((in_set[0], out_set[0]), 0)
        target = in_set[1]

        data, target = data.cuda(), target.cuda()

        # forward
        x, _ = model(data)
        loss = F.cross_entropy(x[: len(in_set[0]), :-1], target)

        Ec_out = -torch.logsumexp(x[len(in_set[0]) :, :10], dim=1)
        Ec_in = -torch.logsumexp(x[: len(in_set[0]), :10], dim=1)
        loss += 0.1 * (torch.pow(F.relu(Ec_in - m_ins[0]), 2).mean() + torch.pow(F.relu(m_outs[0] - Ec_out), 2).mean())

        vprobs = torch.nn.functional.softmax(x, dim=-1)

        loss += 10 * (
            torch.pow(F.relu(vprobs[: len(in_set[0]), -1] - m_ins[1]), 2).mean()
            + torch.pow(F.relu(m_outs[1] - vprobs[len(in_set[0]) :, -1]), 2).mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return loss_avg


def train(model, train_loader_in, train_loader_out, scheduler, optimizer, loss_method: dict):
    raise ValueError
    loss_avg = 0.0

    if loss_method["name"] == "vim":
        model.update_vim_parameters(train_loader_in)

    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    train_loader_out.dataset.offset = np.random.randint(len(train_loader_out.dataset))
    for in_set, out_set in zip(train_loader_in, train_loader_out):
        input_in, label_in = in_set
        input_out, _ = out_set

        input_in, label_in, input_out = input_in.cuda(), label_in.cuda(), input_out.cuda()

        # forward
        output_in, _ = model(input_in)

        scheduler.step()
        optimizer.zero_grad()

        if loss_method["name"] != "vim":
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
            # elif loss_method["name"] == "OE":
            #     output_out, penultimate_out = model(input_out)
            #     loss += 0.5 * -(output_out.mean(1) - torch.logsumexp(output_in, dim=1)).mean()
        else:
            loss = F.cross_entropy(output_in[:, :-1], label_in)

            vprobs_in = torch.nn.functional.softmax(output_in, dim=-1)
            output_out, _ = model(input_out)
            vprobs_out = torch.nn.functional.softmax(output_out, dim=-1)

            loss += 0.00 * (
                torch.pow(F.relu(vprobs_in[:, -1] - loss_method["m_in"]), 2).mean()
                + torch.pow(F.relu(loss_method["m_out"] - vprobs_out[:, -1]), 2).mean()
            )

        loss.backward(retain_graph=True)
        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    return loss_avg
