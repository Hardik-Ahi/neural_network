import numpy as np
from nn.model_classes import Layer, Model
from nn.dataset_utils import and_gate_dataset, linear_regression_dataset, standardize_data
from nn.functions import der_leaky_relu, der_sigmoid, leaky_relu, sigmoid, mean_square_error, der_mean_square_error
from nn.optimizers import SGD
from nn.trainer import Trainer, RegressionTrainer
from nn.plotter import Plotter

model = Model(mean_square_error, der_mean_square_error, 100)
model.add_layer(Layer(1))  # input layer
model.add_layer(Layer(2, leaky_relu(), der_leaky_relu()))
model.compile()

X_train, y_train = linear_regression_dataset(100)
X_train = standardize_data(X_train)  # important!!!
y_train = standardize_data(y_train)
Plotter.plot_points(X_train, y_train)

trainer = RegressionTrainer(model, SGD())
model.show_weights()
model.show_biases()


trainer.train(X_train, y_train, 1, 0.02, 10, use_inputs = True)
trainer.save_history("./logs", "regression")

plotter = Plotter()
plotter.set_model_layers(2)
plotter.read_file('./logs/regression.txt')
plotter.plot_gradients("./plots", "regression", 700)
plotter.plot_weights("./plots", "regression", 700)
plotter.plot_score("./plots", "regression", 700, confusion_matrix = False)
