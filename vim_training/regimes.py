PRETRAINING = {
    "name": "default",
    "loading": "",
    "epochs": 100,
    "learning_rate": 0.1,
    "is_using_nestrov": False,
}

ENERGY = {
    "name": "energy",
    "loading": "default",
    "epochs": 10,
    "learning_rate": 0.001,
    "is_using_nestrov": True,
    "m_in": -23,
    "m_out": -5,
}
