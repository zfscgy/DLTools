import numpy as np
import matplotlib.pyplot as plt
from Data import Cifar10Loader

def test_cifar10():
    dataloader = Cifar10Loader()
    xs, ys = dataloader.get_train_batch(3)
    for i in range(3):
        plt.imshow(xs[i])
        plt.title("Train batch label {}".format(ys[i]))
        plt.show()
    xs, ys = dataloader.get_test_batch(3)
    for i in range(3):
        plt.imshow(xs[i])
        plt.title("Test batch label {}".format(ys[i]))
        plt.show()

test_cifar10()