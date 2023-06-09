PRETRAINING = {
    "name": "default",
    "loading": "",
    "epochs": 100,
    "learning_rate": 0.1,
    "calibration": False,
}

ENERGY = {
    "name": "energy",
    "loading": "energy_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
    "epochs": 10,
    "learning_rate": 0.001,
    "m_in": -23,  # as higher as less effect
    "m_out": -5,  # as lower as less effect
    "calibration": False,
}

PRETRAIN_VIM = {
    "name": "pretrain_vim",
    "loading": "",  # "energy_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
    "epochs": 100,
    "learning_rate": 0.1,
    "calibration": False,
}

TRAIN_WITH_VIM = {
    "name": "train_with_vim",
    "loading": "",  # "energy_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
    "epochs": 100,
    "learning_rate": 0.1,
    "calibration": False,
}


VIM_FT = {
    "name": "vim_ft",
    "loading": "snapshots/pretrain_vim/CIFAR10_WRN_epoch_99.pt",
    "epochs": 10,
    "learning_rate": 0.001,
    "m_in": 0.10,
    "m_out": 0.90,
    "calibration": False,
}


ENERGY_VIM_FT = {
    "name": "energy_vim_ft",
    "loading": "snapshots/pretrain_vim/CIFAR10_WRN_epoch_99.pt",
    "epochs": 10,
    "learning_rate": 0.001,
    "m_ins": [-23, 0.10],
    "m_outs": [-5, 0.90],
    "calibration": False,
}

VANILLA_FT = {
    "name": "vanilla_ft",
    "loading": "energy_ood/CIFAR/snapshots/pretrained/cifar10_wrn_pretrained_epoch_99.pt",
    "epochs": 10,
    "learning_rate": 0.001,
    "calibration": False,
}
