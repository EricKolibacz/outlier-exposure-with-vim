PRETRAINING = {
    "name": "default",
    "loading": "",
    "epochs": 100,
    "learning_rate": 0.1,
}

ENERGY = {
    "name": "energy",
    "loading": "energy_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
    "epochs": 10,
    "learning_rate": 0.001,
    "m_in": -15,  # as higher as less effect
    "m_out": -5,  # as lower as less effect
}

VIM = {
    "name": "vim",
    "loading": "energy_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
    "epochs": 10,
    "learning_rate": 0.001,
    "m_in": 0.5,
    "m_out": 0.5,
}
