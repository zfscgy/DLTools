import numpy as np
from sklearn.metrics import roc_auc_score as auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score


"""
    One prediction, one label
"""


def binary_cross_entropy(targets, predictions, epsilon=1e-9):
    """

    :param targets:  [batch_size], range [0, 1]
    :param predictions:  [batch_size], range [0, 1]
    :param epsilon: In order to handle extreme predictions close to 0 or 1
    :return:
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    ce = - np.mean(np.log(predictions) * targets + np.log(1 - predictions) * (1 - targets))
    return ce


def binary_accuracy(y_true, y_pred):
    """
    :param y_true: [batch_size], range [0, 1]
    :param y_pred: [batch_size], range [0, 1]
    :return:
    """
    matches = np.sum(np.abs(y_true - y_pred) < 0.5) + np.sum(np.abs(y_true - y_pred) == 0.5) / 2
    return matches / len(y_true)


"""
    Multiple predictions, one label
"""

"""
    Prediction is distribution
"""


def top1_accuracy(y_true: np.ndarray, y_pred: np.ndarray):
    """
    :param y_pred: shape [batch_size, pred_len] range [0, 1]
    :param y_true: shape [batch_size] range integer in [0, pred_len)
    :return:
    """
    batch_size = y_pred.shape[0]
    y_pred_class = np.argmax(y_pred, axis=1)
    hit_array = np.equal(y_pred_class, y_true).astype(np.float)
    return np.sum(hit_array) / batch_size


"""
    Prediction is sequence
"""


def hit_ratio(y_true: np.ndarray, y_pred: np.ndarray):
    """
    The ratio label is in the prediction array
    :param y_pred: shape [batch, pred_len] range integer in [0, n_items)
    :param y_true: shape [batch] range integer in [0, n_items)
    :return:
    """
    batch_size = y_pred.shape[0]
    pred_len = y_pred.shape[1]
    y_true = np.broadcast_to(y_true, [pred_len, batch_size]).transpose()
    hit_array = np.equal(y_pred, y_true).astype(np.float)
    return np.sum(hit_array) / batch_size


"""
    Multiple predictions, multiple labels
"""


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon=1e-9):
    """
    :param y_true: shape [batch_size, n_classes] range in [0, 1] and sum to 1 (should be outputs of softmax)
    :param y_pred: shape [batch_size, n_classes] range in [0, 1]
    :param epsilon:
    :return:
    """
    clipped_yp = np.clip(y_pred, epsilon, 1. - epsilon)
    return - np.mean(np.log(clipped_yp) * y_true)


def discounted_cumulative_gain(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Formula: sum rel_i / log2(i + 1)
    :param y_true: shape [batch_size, n_items]
    :param y_pred: shape [batch_size]
    :return:
    """
    batch_size = y_pred.shape[0]
    pred_len = y_pred.shape[1]
    y_true = np.broadcast_to(y_true, [pred_len, batch_size]).transpose()
    hit_array = np.equal(y_pred, y_true).astype(np.float)  # [batch, pred_len]
    scores = hit_array / np.log2(np.arange(2, pred_len + 2))
    return np.mean(np.sum(scores, axis=1))


def oneshot_accuracy(y_pred: np.ndarray, support_size: int=2):
    """
    :param y_pred: shape [batch_size * support_size, 1]
    Assume in each batch, the 1 label is in the first place
    i.e. y_true must be [1 0 0 ...(0 for support_size - 1) 1 0 0 ... 1 0 0 ... ...]
    :return:
    """
    len = y_pred.shape[0]
    ys = y_pred.reshape([int(len/support_size), support_size])
    max_idx = np.argmax(ys[:, ::-1], axis=1)
    return np.mean(max_idx == support_size - 1)


def cmc_curve(y_pred, support_size:int=2):
    """

    :param y_pred:
    :param support_size:
    :return:
    """
    len = y_pred.shape[0]
    ys = y_pred.reshape([int(len/support_size), support_size])
    ys_idx = np.argsort(-ys, axis=1)
    cumulative_match = [0]
    for i in range(support_size):
        cumulative_match.append(np.mean(ys_idx[:, i] == 0) + cumulative_match[i])
    return cumulative_match[1:]
