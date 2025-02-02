import numpy as np
from numpy.random import default_rng
from math import sqrt

from dataset_utils import get_minibatch
from model_functions import binary_loss

# internal dependencies (functions used within the classes only)
def init_weights(destination_neurons, source_neurons, seed):  # destination_neurons = fan_in to this layer = fan_out of previous layer
    std = sqrt(2 / source_neurons)  # standard deviation for 'He' initialization (use it if you use ReLU functions)
    generator = default_rng(seed)

    weights = generator.standard_normal((destination_neurons, source_neurons)) * std  # z = x-mu / sigma, therefore x = z . sigma + mu; z = standard normal
    # here we need mu (mean) = 0, but standard deviation as per we calculated using fan_in.
    return weights

# is this external or internal dependency?
def accuracy(labels, predictions):  # both must be of same shape
    confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
    for label, prediction in np.nditer([labels, predictions]):
        if label == 1:
            if prediction == 1:
                confusion_matrix['tp'] += 1
            else:
                confusion_matrix['fn'] += 1
        else:
            if prediction == 1:
                confusion_matrix['fp'] += 1
            else:
                confusion_matrix['tn'] += 1
    total = 0
    for i in confusion_matrix.values():
        total += i
    return (confusion_matrix['tp'] + confusion_matrix['tn']) / total

class Layer:

    def __init__(self, n_neurons, activation = None, der_activation = None):
        self.n_neurons = n_neurons
        self.activation = activation
        self.der_activation = der_activation
        self.z_ = np.zeros((n_neurons, 1))
        self.del_ = np.zeros((n_neurons, 1))
        self.a_ = np.zeros((n_neurons, 1))
        self.b_ = np.zeros((n_neurons, 1))

class Weights:

    def __init__(self, layer_1, layer_2, seed = 1000):
        self.rows = layer_2.n_neurons
        self.cols = layer_1.n_neurons
        self.seed = seed
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.matrix = init_weights(self.rows, self.cols, self.seed)
        self.gradients = np.zeros((self.rows, self.cols))

class Model:

    def __init__(self, loss_function, der_loss_function):
        self.layers = list()
        self.weights = list()
        self.loss_function = loss_function  # should be applicable to single instances and batches
        self.der_loss_function = der_loss_function
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    # part of model compiler
    def init_weights(self):
        generator = default_rng(seed = 100)
        seeds = generator.integers(0, len(self.layers)*100, (len(self.layers)-1,))

        for i in range(len(self.layers)-1):
            self.weights.append(Weights(self.layers[i], self.layers[i+1], seed = seeds[i]))
    
    def show_weights(self):
        for i in range(len(self.weights)):
            print(i, self.weights[i].matrix, sep = "\n")
    
    def show_biases(self):
        for i in range(1, len(self.layers)):
            print(i, self.layers[i].b_, sep = "\n")