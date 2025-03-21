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

X_train, y_train = and_gate_dataset(100)

trainer = Trainer(model, SGD())  # YES! got 100% accuracy in 1500 epochs using sigmoid activation function all-around!
model.show_weights()
model.show_biases()

#print(f"dataset:{train}")
trainer.train(X_train, y_train, 1, 0.02, 100, 30)  # double YES!! got 100% accuracy in 250 epochs using leaky_relu with leak = 0.4!!
#trainer.save_history("./logs", "test")

plotter = Plotter()
plotter.set_model_layers(3)
plotter.read_file('./logs/latest.txt')
plotter.plot_gradients("./plots", "latest")
