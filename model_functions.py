import numpy as np

leak_ = 0.4
epsilon_ = 1e-7

@np.vectorize
def relu(x):
    return x if x > 0 else x*leak_

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def round(x):
    return 1 if x >= 0.5 else 0

def binary_loss(labels, predictions):
    total = predictions.size
    labels = labels.reshape(predictions.shape)
    summation = 0

    predictions = np.clip(predictions, epsilon_, 1 - epsilon_)  # avoiding values of 0 ( => inf loss) and 1 ( => 0 loss)
    iterator = zip(labels, predictions)
    for label, prediction in iterator:
        summation += label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
    return (-1 / total) * summation

def der_binary_cross_entropy(label, output, epsilon = epsilon_):  # output belongs to range [0, 1]
    output = np.clip(output, epsilon_, 1 - epsilon_)
    return ((1 - label)/(1 - output)) - (label / output)

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def der_sigmoid(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

@np.vectorize
def der_relu(x):
    return 1 if x > 0 else leak_