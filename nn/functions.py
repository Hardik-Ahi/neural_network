import numpy as np

@np.vectorize
def relu(x):
    return x if x > 0 else 0

def leaky_relu(leak = 0.1, alpha = 1):
    @np.vectorize
    def func(x):
        return alpha * x if x > 0 else x*leak  # leak is accessed through a 'closure'
    return func

def der_leaky_relu(leak = 0.1, alpha = 1):
    @np.vectorize
    def func(x):
        return alpha if x > 0 else leak
    return func

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def round_off(x):
    return 1 if x >= 0.5 else 0

@np.vectorize
def mirror(x):
    return x

@np.vectorize
def der_mirror(x):
    return 1

def binary_loss(labels, predictions, epsilon = 1e-7):
    total = predictions.size
    labels = labels.reshape(predictions.shape)
    summation = 0

    predictions = np.clip(predictions, epsilon, 1 - epsilon)  # avoiding values of 0 ( => inf loss) and 1 ( => 0 loss)
    iterator = zip(labels, predictions)
    for label, prediction in iterator:
        summation += label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
    return (-1 / total) * summation

def der_binary_cross_entropy(label, output, epsilon = 1e-7):  # output belongs to range [0, 1]
    output = np.clip(output, epsilon, 1 - epsilon)
    return ((1 - label)/(1 - output)) - (label / output)

def mean_square_error(labels, predictions):
    labels = labels.reshape(predictions.shape)
    return (1 / labels.size) * np.sum((labels - predictions)**2)

def der_mean_square_error(label, output):
    return 2 * (output - label)

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def der_sigmoid(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

@np.vectorize
def der_relu(x):
    return 1 if x > 0 else 0