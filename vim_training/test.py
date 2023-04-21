"""A module taken from the energy-ood paper (http://arxiv.org/abs/2010.03759)
everything related to testing a model"""
import torch
import torch.nn.functional as F


# test function
def test(model, loader):
    model.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, label in loader:
            data, label = data.cuda(), label.cuda()

            # forward
            output, _ = model(data)

            loss = F.cross_entropy(output, label)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(label.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    test_loss = loss_avg / len(loader)
    test_accuracy = correct / len(loader.dataset)

    return test_loss, test_accuracy
