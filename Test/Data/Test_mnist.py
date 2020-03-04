import matplotlib.pyplot as plt
from Data import MnistLoader


def test_mnist():
    data_loader = MnistLoader()
    xs, ys = data_loader.get_train_batch(2)
    for i in range(xs.shape[0]):
        pic = xs[i, :].reshape([28, 28])
        label = ys[i]
        plt.title("Train batch: index: {} label: {}".format(i, label))
        plt.imshow(pic)
        plt.show()
    xs, ys = data_loader.get_test_batch(2)
    for i in range(xs.shape[0]):
        pic = xs[i, :].reshape([28, 28])
        label = ys[i]
        plt.title("Test batch: index: {} label: {}".format(i, label))
        plt.imshow(pic)
        plt.show()

test_mnist()