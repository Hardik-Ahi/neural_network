import numpy as np
from numpy.random import default_rng
from math import sqrt
import os

class Layer:

    def __init__(self, n_neurons, activation = None, der_activation = None):
        self.n_neurons = n_neurons
        self.activation = activation
        self.der_activation = der_activation
        self.z_ = np.zeros((n_neurons, 1))
        self.del_ = np.zeros((n_neurons, 1))
        self.b_gradients = np.zeros((n_neurons, 1))
        self.a_ = np.zeros((n_neurons, 1))
        self.b_ = np.zeros((n_neurons, 1))
    
    def init_biases(self):
        self.b_ = np.zeros((self.n_neurons, 1))
    
    def activate(self):
        self.a_ = self.activation(self.z_)
    
    def der_activate(self):
        return self.der_activation(self.z_)

class PolyLayer:

    def __init__(self, *layers):
        self.layers = layers  # list as it is
        self.n_neurons = sum(list(map(lambda layer: layer.n_neurons, self.layers)))
        self.z_ = np.zeros((self.n_neurons, 1))
        self.del_ = np.zeros((self.n_neurons, 1))
        self.b_gradients = np.zeros((self.n_neurons, 1))
        self.a_ = np.zeros((self.n_neurons, 1))
        self.b_ = np.zeros((self.n_neurons, 1))
    
    def init_biases(self):
        start = 0
        for i in range(len(self.layers)):
            self.layers[i].init_biases()
            self.b_[start : start+self.layers[i].n_neurons] = self.layers[i].b_
            start += self.layers[i].n_neurons
    
    def activate(self):
        # prerequisite: layer.z_
        start = 0
        for i in range(len(self.layers)):
            self.layers[i].z_ = self.z_[start : start+self.layers[i].n_neurons]
            start += self.layers[i].n_neurons

        start = 0
        for i in range(len(self.layers)):
            self.layers[i].activate()
            self.a_[start : start+self.layers[i].n_neurons] = self.layers[i].a_
            start += self.layers[i].n_neurons

    def der_activate(self):
        result = []
        for i in range(len(self.layers)):
            result.append(self.layers[i].der_activate())
        return np.asarray(result).reshape((self.n_neurons, 1))

class Weights:

    def __init__(self, layer_1, layer_2, seed = 1000):
        self.rows = layer_2.n_neurons
        self.cols = layer_1.n_neurons
        self.seed = seed
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.matrix = self.init_weights(self.rows, self.cols, self.seed)
        self.gradients = np.zeros((self.rows, self.cols))
    
    def init_weights(self, destination_neurons, source_neurons, seed):  # source_neurons = fan_in to this layer
        std = sqrt(2 / source_neurons)  # standard deviation for 'He' initialization (use it if you use ReLU functions)
        generator = default_rng(seed)

        weights = generator.standard_normal((destination_neurons, source_neurons)) * std  # z = x-mu / sigma, therefore x = z * sigma + mu; z = standard normal
        # here we need mu (mean) = 0, but standard deviation as per we calculated using fan_in.
        return weights

class Model:

    def __init__(self, loss, seed = 1000):
        self.layers = list()
        self.weights = list()
        self.loss = loss  # 'class' for the loss function
        self.seed = seed
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    # part of model compiler
    def compile(self):
        generator = default_rng(seed = self.seed)
        seeds = generator.integers(0, len(self.layers)*100, (len(self.layers)-1,))

        self.weights = list()  # allows re-compilation
        for i in range(len(self.layers)-1):
            self.weights.append(Weights(self.layers[i], self.layers[i+1], seed = seeds[i]))
        
        for layer in self.layers:
            layer.init_biases()
    
    def weights_elements(self):
        vector = list()

        for weight in self.weights:
            vector.append(np.ravel(weight.matrix))
        for layer in self.layers:
            vector.append(np.ravel(layer.b_))
        
        return np.hstack(vector)

    def set_weights(self, vector):
        start = 0

        for weight in self.weights:
            weight.matrix = vector[start : start + weight.matrix.size].reshape(weight.matrix.shape)
            start += weight.matrix.size
        for layer in self.layers:
            layer.b_ = vector[start : start + layer.b_.size].reshape(layer.b_.shape)
            start += layer.b_.size
    
    def show_weights(self):
        for i in range(len(self.weights)):
            print(i, self.weights[i].matrix, sep = "\n")
    
    def show_biases(self):
        for i in range(1, len(self.layers)):
            print(i, self.layers[i].b_, sep = "\n")

    def save_weights(self, dir, name):
        # path = entire path of file to be created
        if not os.path.exists(dir):
            print(f'{dir} does not exist.')
            return
        
        path = dir + "/" + name
        weights_dict = dict()
        bias_dict = dict()

        for i in range(len(self.weights)):
            name = f'weights-{i}'
            weights_dict[name] = self.weights[i].matrix
        for i in range(len(self.layers)):
            name = f'bias-{i}'
            bias_dict[name] = self.layers[i].b_

        np.savez(path, allow_pickle = False, **weights_dict, **bias_dict)
    
    def load_weights(self, path):
        if not os.path.exists:
            print(f'{path} does not exist')
            return

        with np.load(path) as data:
            weights_keys = list(filter(lambda x: x.startswith("weights"), data.keys()))
            bias_keys = list(filter(lambda x: x.startswith("bias"), data.keys()))
            try:
                for i in range(len(weights_keys)):
                    name = f'weights-{i}'
                    self.weights[i].matrix = data[name]
                for i in range(len(bias_keys)):
                    name = f'bias-{i}'
                    self.layers[i].b_ = data[name]
            except IndexError:
                print("model structure does not match with loaded weights")
                return
        
        print("loaded successfully")