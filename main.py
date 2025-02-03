import numpy as np
from numpy.random import default_rng
from model_classes import Layer, Model
from dataset_utils import and_gate_dataset, standardize_data
from model_functions import der_relu, der_sigmoid, relu, sigmoid, der_binary_cross_entropy, binary_loss
from optimizers import SGD
from trainers import BatchTrainer, MiniBatchTrainer


model = Model(binary_loss, der_binary_cross_entropy, 100)  # for batch
model.add_layer(Layer(2))  # input layer
model.add_layer(Layer(2, relu, der_relu))
model.add_layer(Layer(1, sigmoid, der_sigmoid))

dataset = and_gate_dataset()
default_rng(seed = 100).shuffle(dataset)
features = dataset[:, [0, 1]]
labels = dataset[:, 2].reshape((dataset.shape[0], 1))

trainer = MiniBatchTrainer(model, SGD(), 16)
trainer.train(features, labels, epochs = 10)