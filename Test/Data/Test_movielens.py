import numpy as np
from Data import MovielensLoader


def test_movielens():
    dataloader = MovielensLoader("m-100k", split=0.8, min_sample_len=5)
    xs, ys, us = dataloader.get_train_batch(3)
    print("=======\nTrain batch: min_sample_len = 5, split = 0.8")
    for i in range(3):
        print("xs:", xs[i, :])
        print("ys:", ys[i, :])
        print("us:", us[i])
    dataloader = MovielensLoader("m-100k", split=0.8, min_sample_len=5, label_mode="all")
    xs, ys, us = dataloader.get_train_batch(3)
    print("=======\nTrain batch: min_sample_len = 5, split = 0.8, label_mode = 'all'")
    for i in range(3):
        print("xs:", xs[i, :])
        print("ys:", ys[i, :])
        print("us:", us[i])

    dataloader = MovielensLoader("m-100k", split=0, min_sample_len=5, label_mode="all")
    xs, ys, us = dataloader.get_train_batch(3)
    print("=======\nTrain batch: min_sample_len = 5, split = 0.0, label_mode='all'")
    for i in range(3):
        print("xs:", xs[i, :])
        print("ys:", ys[i, :])
        print("us:", us[i])
    xs, ys, us = dataloader.get_test_batch(3)
    print("=======\nTest batch: min_sample_len = 5, split = 0.0, label_mode='all'")
    for i in range(3):
        print("xs:", xs[i, :])
        print("ys:", ys[i, :])
        print("us:", us[i])


test_movielens()