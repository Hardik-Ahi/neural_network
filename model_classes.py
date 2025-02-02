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
        self.activation = activation  # should be np.vectorize'd
        self.der_activation = der_activation  # also np.vectorize'd
        self.z_ = None
        self.del_ = np.array([])
        self.a_ = None
        self.b_ = np.zeros((n_neurons, 1))
    
    def compute_output_error(self, label, der_loss_function):
        self.del_ = der_loss_function(label, self.a_[0][0]) * self.der_activation(self.z_)
    
    def compute_output_error_batch(self, label, der_loss_function, batch_size = 1):
        if self.del_.size == 0:
            self.del_ = der_loss_function(label, self.a_[0][0]) * self.der_activation(self.z_) / batch_size
        else:
            self.del_ += der_loss_function(label, self.a_[0][0]) * self.der_activation(self.z_)

    def compute_error(self, weights, next_layer):  # weights matrix connecting this layer to the next layer
        self.del_ = np.matmul(np.transpose(weights), next_layer.del_) * self.der_activation(self.z_)
    
    def compute_error_batch(self, weights, next_layer, batch_size = 1):
        if self.del_.size == 0:
            self.del_ = np.matmul(np.transpose(weights), next_layer.del_) * self.der_activation(self.z_) / batch_size
        else:
            self.del_ += np.matmul(np.transpose(weights), next_layer.del_) * self.der_activation(self.z_) / batch_size
    
    def update_biases(self, learning_rate = 0.001):  # larger learning rate to provide the brute force assist in learning
        self.b_ -= ((learning_rate) * self.del_)
    
    def update_biases_batch(self, learning_rate = 0.1):
        self.b_ -= ((learning_rate) * self.del_)

class Weights:

    def __init__(self, layer_1, layer_2, seed = 1000):
        self.rows = layer_2.n_neurons
        self.cols = layer_1.n_neurons
        self.seed = seed
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.matrix = init_weights(self.rows, self.cols, self.seed)
        self.gradients = np.array([])
    
    def calc_gradient(self):  # no accumulation of gradients
        self.gradients = np.matmul(self.layer_2.del_, np.transpose(self.layer_1.a_))
    
    def calc_gradient_batch(self, batch_size = 1):
        if self.gradients.size == 0:
            self.gradients = np.matmul(self.layer_2.del_, np.transpose(self.layer_1.a_)) / batch_size
        else:
            self.gradients += np.matmul(self.layer_2.del_, np.transpose(self.layer_1.a_)) / batch_size

    def update_weights(self, learning_rate = 0.001):
        self.matrix = self.matrix - ((learning_rate) * self.gradients)
    
    def update_weights_batch(self, learning_rate = 0.1):
        self.matrix -= ((learning_rate) * self.gradients)

class Model:

    def __init__(self, loss_function, der_loss_function):
        self.layers = list()
        self.weights = list()
        self.loss_function = loss_function  # for single instance => args = (label, output)
        self.der_loss_function = der_loss_function
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def init_weights(self):
        for i in range(len(self.layers)-1):
            self.weights.append(Weights(self.layers[i], self.layers[i+1], seed = (i*100) + 5))

    def forward_pass(self, input_data):  # for single training instance
        # input_data shape = (1, n_cols) where n_cols = no.of features
        self.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.layers)):
            self.layers[i].z_ = np.matmul(self.weights[i-1].matrix, self.layers[i-1].a_) + self.layers[i].b_
            self.layers[i].a_ = self.layers[i].activation(self.layers[i].z_)
    
    def forward_pass_batch(self, input_data):
        self.layers[0].a_ = input_data.reshape((input_data.shape[1], input_data.shape[0]))
        for i in range(1, len(self.layers)):
            self.layers[i].z_ = np.matmul(self.weights[i-1].matrix, self.layers[i-1].a_) + self.layers[i].b_
            self.layers[i].a_ = self.layers[i].activation(self.layers[i].z_)
        
    
    def backward_pass(self, label):  # single instance
        self.layers[-1].compute_output_error(label, self.der_loss_function)
        for i in range(len(self.layers)-2, 0, -1):  # except the input layer
            self.layers[i].compute_error(self.weights[i].matrix, self.layers[i+1])
        
        for i in range(len(self.weights)):
            self.weights[i].calc_gradient()
    
    def backward_pass_batch(self, label, batch_size = 1):
        self.layers[-1].compute_output_error_batch(label, self.der_loss_function, batch_size = batch_size)
        for i in range(len(self.layers)-2, 0, -1):  # except the input layer
            self.layers[i].compute_error_batch(self.weights[i].matrix, self.layers[i+1], batch_size = batch_size)
        
        for i in range(len(self.weights)):
            self.weights[i].calc_gradient_batch(batch_size = batch_size)
    
    def show_weights(self):
        for i in range(len(self.weights)):
            print(i, self.weights[i].matrix, sep = "\n")
    
    def show_biases(self):
        for i in range(1, len(self.layers)):
            print(i, self.layers[i].b_, sep = "\n")
    
    def clean_up_pass(self):
        # reset gradients
        for weight in self.weights:
            weight.gradients = np.array([])
        
        # reset errors
        for layer in self.layers:
            if layer.del_.size != 0:
                layer.del_ = np.array([])

    def train(self, features, targets, epochs = 1):
        self.init_weights()
        print("weights:")
        self.show_weights()
        for epoch in range(epochs):
            print("-------------epoch:", epoch, "-------------", sep="")
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass(targets[i][0])
                for j in range(1, len(self.layers)):
                    self.layers[j].update_biases()
                for j in range(len(self.weights)):
                    self.weights[j].update_weights()
                prediction = self.layers[-1].a_[0]
                loss = self.loss_function(targets[i], prediction)
                print("loss:", loss)
                self.clean_up_pass()

            self.forward_pass_batch(features)
            predictions = self.layers[-1].a_  # shape = (1, m)
            temp_targets = targets.reshape((1, targets.shape[0])).astype(int)
            print("\naccuracy: ", accuracy(temp_targets, round(predictions)))
            print("predictions: ", round(predictions))
            print("targets:     ", temp_targets, "\n")
    
    def train_batch(self, features, targets, epochs = 1):
        self.init_weights()
        print("weights:")
        self.show_weights()
        for epoch in range(epochs):
            print("------epoch-", epoch, "------", sep="")
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass_batch(targets[i][0], batch_size = features.shape[0])

                prediction = self.layers[-1].a_[0]
                loss = self.loss_function(targets[i], prediction)
                print("loss:", loss)

            for j in range(1, len(self.layers)):
                self.layers[j].update_biases_batch()
            for j in range(len(self.weights)):
                self.weights[j].update_weights_batch()

            self.forward_pass_batch(features)
            predictions = self.layers[-1].a_  # shape = (1, m)
            temp_targets = targets.reshape((1, targets.shape[0])).astype(int)
            print("\naccuracy: ", accuracy(temp_targets, round(predictions)))
            print("predictions: ", round(predictions))
            print("targets:     ", temp_targets, "\n")
            self.clean_up_pass()
    
    def train_minibatch(self, features, targets, batch_size = 1, epochs = 1):
        self.init_weights()
        print("weights:")
        self.show_weights()
        learning_rate = 1
        for epoch in range(epochs):
            print("------epoch-", epoch, "------", sep="")
            if epoch > 500:
                learning_rate = 0.5
            start_index = 0
            while start_index < features.shape[0]:
                real_features, real_targets = get_minibatch(features, targets, batch_size, start_index)
                for i in range(real_features.shape[0]):
                    self.forward_pass(real_features[i])
                    self.backward_pass_batch(real_targets[i][0], batch_size = batch_size)

                for j in range(1, len(self.layers)):
                    self.layers[j].update_biases_batch(learning_rate = learning_rate)
                for j in range(len(self.weights)):
                    self.weights[j].update_weights_batch(learning_rate = learning_rate)
                
                self.clean_up_pass()
                start_index += batch_size
            
            self.forward_pass_batch(features)
            predictions = self.layers[-1].a_  # shape = (1, m)
            temp_targets = targets.reshape((1, targets.shape[0])).astype(int)
            print("\naccuracy: ", accuracy(temp_targets, round(predictions)))
            print("loss: ", binary_loss(targets, predictions.reshape((predictions.shape[-1], 1))))