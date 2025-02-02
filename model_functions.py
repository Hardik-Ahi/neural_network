import numpy as np

leak_ = 0
epsilon_ = 1e-7

@np.vectorize  # decorator, allows this func to execute on all elements of an ndarray
def relu(x):
    return x if x > 0 else x*leak_

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def round(x):
    return 1 if x >= 0.5 else 0

def binary_loss(labels, predictions, epsilon = epsilon_):
    total = predictions.shape[0]
    summation = 0

    for label, prediction in np.nditer([labels, predictions]):
        summation -= label * np.log(prediction + epsilon) + (1 - label) * np.log(1 - prediction + epsilon)
    return (1 / total) * summation

def bce_single_sample(label, prediction, epsilon = epsilon_):
    return -(label * np.log(prediction + epsilon) + (1 - label) * np.log(1 - prediction + epsilon))

def der_binary_cross_entropy(label, output, epsilon = epsilon_):  # output belongs to range [0, 1]
    return ((1 - label)/((1 - output) + epsilon)) - (label / (output + epsilon))

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def der_sigmoid(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

@np.vectorize
def der_relu(x):
    return 1 if x > 0 else leak_