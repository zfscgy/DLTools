import numpy as np
import matplotlib.pyplot as plt
from Data import LFWLoader


def test_lfw():
    dataloader = LFWLoader()
    x0s, x1s, ys = dataloader.get_train_batch(2, 3)
    x0s = x0s.reshape([-1, 250, 3])
    x1s = x1s.reshape([-1, 250, 3])
    xs = np.concatenate([x0s, x1s], axis=1)
    plt.title("Train batch, no crop")
    plt.imshow(xs)
    plt.show()
    dataloader = LFWLoader(crop=[224, 224])
    x0s, x1s, ys = dataloader.get_train_batch(2, 3)
    x0s = x0s.reshape([-1, 224, 3])
    x1s = x1s.reshape([-1, 224, 3])
    xs = np.concatenate([x0s, x1s], axis=1)
    plt.title("Train batch, crop=[224,224]")
    plt.imshow(xs)
    plt.show()


test_lfw()