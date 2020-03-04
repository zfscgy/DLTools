import numpy as np
from Evaluation.Metrics import binary_accuracy, binary_cross_entropy,\
    top1_accuracy, hit_ratio, cross_entropy, discounted_cumulative_gain


def test_binary_accuracy():
    print("=======\nTest binary_accuracy...")
    ys = np.array([1, 0, 0, 1])
    pred_mid = np.array([0.5, 0.5, 0.5, 0.5])
    pred_close = np.array([0.8, 0.2, 0.3, 0.9])
    pred_far = np.array([0.1, 0.8, 0.9, 0.3])
    print("Mid:", binary_accuracy(ys, pred_mid))
    print("Close:", binary_accuracy(ys, pred_close))
    print("Far", binary_accuracy(ys, pred_far))


def test_binary_cross_entropy():
    print("=======\nTest binary_cross_entropy")
    ys = np.array([1, 0, 0, 1])
    pred_mid = np.array([0.5, 0.5, 0.5, 0.5])
    pred_close = np.array([0.8, 0.2, 0.3, 0.9])
    pred_far = np.array([0.1, 0.8, 0.9, 0.3])
    print("Mid:", binary_cross_entropy(ys, pred_mid))
    print("Close:", binary_cross_entropy(ys, pred_close))
    print("Far", binary_cross_entropy(ys, pred_far))


def test_top1_accuracy():
    print("=======\nTest top1_accuracy...")
    ys = np.array([0, 1, 2, 3])
    ys_pred = np.array([
        [0.1, 0.6, 0.2, 0.1],
        [0.1, 0.8, 0, 0.1],
        [0.4, 0, 0.5, 0.1],
        [0.1, 0.1, 0.1, 0.7]])
    print("Top1 accuracy:", top1_accuracy(ys, ys_pred))


def test_hit_ratio():
    print("=======\nTest hit_ratio")
    ys = np.array([1, 2, 1, 4])
    ys_pred = np.array([
        [1, 2],
        [3, 4],
        [2, 3],
        [1, 4],
    ])
    print("Hit ratio:", hit_ratio(ys, ys_pred))


def test_cross_entropy():
    print("======\nTest cross_entropy...")
    ys = np.array([[0, 0, 0, 1]])
    ys_close = np.array([[0.1, 0.1, 0.1, 0.7]])
    ys_mid = np.array([[0.2, 0.2, 0.2, 0.3]])
    ys_far = np.array([[0.5, 0.3, 0.1, 0.1]])
    print("Close:", cross_entropy(ys, ys_close))
    print("Mid: ", cross_entropy(ys, ys_mid))
    print("Far: ", cross_entropy(ys, ys_far))


def test_dcg():
    print("======\nTest discounted_cumulative_gain")
    ys = np.array([0])
    pred_ys = np.array([[0, 1, 3, 2]])
    print("DCG at 1:", discounted_cumulative_gain(ys, pred_ys))
    pred_ys = np.array([[1, 0, 3, 2]])
    print("DCG at 2:", discounted_cumulative_gain(ys, pred_ys))
    pred_ys = np.array([[1, 2, 0, 3]])
    print("DCG at 3:", discounted_cumulative_gain(ys, pred_ys))


test_binary_accuracy()
test_binary_cross_entropy()
test_top1_accuracy()
test_hit_ratio()
test_cross_entropy()
test_dcg()
