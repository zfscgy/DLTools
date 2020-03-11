import tensorflow as tf
import numpy as np
k = tf.keras

from Modules.Keras.SampleLayers import SampleOutputLayer


def test_sampleOutput():
    print("======\nTest sample output layer, n_samples=3, batch_size=3")
    sample_output_layer = SampleOutputLayer(3, 3)
    model_output = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]])
    model_labels = np.array([1, 2, 3])
    sampled_labels, sampled_output = sample_output_layer([model_output, model_labels])
    print("labels\n", sampled_labels)
    print("ouptuts\n", sampled_output)
    print("======\nTest sample output layer, n_samples=3, batch_size=2, (leading_shape=[2])")
    sample_output_layer = SampleOutputLayer(3, 2)
    model_output = np.array([[[0.1, 0.2, 0.3, 0.4, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5]],
                             [[0.6, 0.7, 0.8, 0.9, 1.0], [0.6, 0.7, 0.8, 0.9, 1.0]]])
    model_labels = np.array([[1, 2], [3, 4]])
    sampled_labels, sampled_output = sample_output_layer([model_output, model_labels])
    print("labels\n", sampled_labels)
    print("ouptuts\n", sampled_output)



test_sampleOutput()
