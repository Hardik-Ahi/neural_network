import numpy as np
from nn.model_classes import Layer, Model
from nn.dataset_utils import and_gate_dataset
from nn.functions import der_leaky_relu, der_sigmoid, leaky_relu, sigmoid, der_binary_cross_entropy, binary_loss
from nn.optimizers import SGD
from nn.trainer import Trainer
from nn.plotter import Plotter

model = Model(binary_loss, der_binary_cross_entropy, 100)  # for batch
model.add_layer(Layer(2))  # input layer
model.add_layer(Layer(2, leaky_relu(), der_leaky_relu()))
model.add_layer(Layer(1, sigmoid, der_sigmoid))
model.compile()

trainer = Trainer(model, SGD())

model.load_weights(r'models\batch_size_full.npz')


plotter = Plotter()
X_test, y_test = and_gate_dataset(20, seed = 8)
plotter.plot_contours(trainer, X_test, y_test, "./plots", name = "batch_size_full")