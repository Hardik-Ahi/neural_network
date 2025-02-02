import numpy as np
from numpy.random import default_rng  # rng = random number generator
from model_classes import Layer, Model
from dataset_utils import and_gate_dataset, standardize_data
from model_functions import der_relu, der_sigmoid, relu, sigmoid, der_binary_cross_entropy, binary_loss


model = Model(binary_loss, der_binary_cross_entropy)  # for batch
model.add_layer(Layer(2))  # input layer
model.add_layer(Layer(2, relu, der_relu))
model.add_layer(Layer(1, sigmoid, der_sigmoid))

dataset = and_gate_dataset()
default_rng(seed = 100).shuffle(dataset)
features = dataset[:, [0, 1]]
labels = dataset[:, 2].reshape((dataset.shape[0], 1))
features = standardize_data(features)
model.train_minibatch(features, labels, 32, epochs = 100)