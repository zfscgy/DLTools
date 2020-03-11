import tensorflow as tf
k = tf.keras
L = k.layers
A = k.activations
K = k.backend
I = k.initializers
import numpy as np


class SampleOutputLayer(L.Layer):
    """
    This layer is used for negative sampling
    For example, there're 1000 items, and 1 item is our label.
    We want to get the gradients on some of the losses for SGD, so we want to get the model output on,
    let's say, 10 items, since calculating 1000 items' gradients is difficult
    And the 10 items must contain our target item since we want to maximize the model's output on this one

    For example, there are 10 items, and the target item(label) is 1, and our model output is
    [0.1, 0.3, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1], we cam choose some of those:
      x    x    x         x     x
    -> [0.3, 0.1, 0.1, 0.05, 0.05] and the label becomes one-hot vector
    -> [1,   0,   0,   0,    0   ]
    Notice for convenience, we move the model's output on the 'right' item to the first place.
    So the label vector is always leading by 1, with all 0's behind
    """
    def __init__(self, n_negative_samples, batch_size):
        self.n_negative_samples = n_negative_samples
        self.batch_size = batch_size
        super(SampleOutputLayer, self).__init__()
        
    def build(self, input_shape):
        model_out_shape, label_shape = input_shape
        self.n_items = model_out_shape[-1]
        self.leading_shape = model_out_shape[1:-1]
        y_true = np.array([1] + [0] * self.n_negative_samples)
        # always a [1, 0, 0, 0, ..., 0] tensor
        self.targets = self.add_weight(name="target", shape=(1 + self.n_negative_samples,),
                                       initializer=I.Constant(value=y_true), trainable=False)
        super(SampleOutputLayer, self).build(input_shape)

    def call(self, inputs):
        y_preds, labels = inputs
        # labels: [batch, ..]
        # probs: [batch, .., item_size]
        labels = K.expand_dims(labels, -1)
        samples_indices = K.random_uniform([self.batch_size] + self.leading_shape + [self.n_negative_samples],
                                           0, self.n_items, dtype='int32')
        indices = K.concatenate([labels, samples_indices], axis=-1)  # [batch, .., 1 + n_samples]
        indices = K.expand_dims(indices, -1)  # [batch, .., 1 + n_samples, 1]
        sampled_outputs = tf.gather_nd(y_preds, indices, batch_dims=1 + len(self.leading_shape))
        #  gathered_probs: [batch, .., 1 + n_samples]
        targets = self.targets + K.zeros_like(sampled_outputs)
        #  [batch, .., 1 + n_samples]
        return targets, sampled_outputs
