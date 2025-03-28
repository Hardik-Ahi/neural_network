import numpy as np
from nn.model_classes import Layer, Model
from nn.dataset_utils import and_gate_dataset, linear_regression_dataset, standardize_data
from nn.functions import der_leaky_relu, der_sigmoid, leaky_relu, sigmoid, mean_square_error, der_mean_square_error, mirror, der_mirror
from nn.optimizers import SGD
from nn.trainer import Trainer, RegressionTrainer
from nn.plotter import Plotter

model = Model(mean_square_error, der_mean_square_error, 100)
model.add_layer(Layer(1))  # input layer
model.add_layer(Layer(1, mirror, der_mirror))
model.compile()

X_train, y_train = linear_regression_dataset(100)
X_train = standardize_data(X_train)  # important!!!
y_train = standardize_data(y_train)
Plotter.plot_points(X_train, y_train)

trainer = RegressionTrainer(model, SGD())
model.show_weights()
model.show_biases()

trainer.train(X_train, y_train, 1, 0.02, 50)
trainer.save_history("./logs", "regression")

plotter = Plotter()
plotter.set_model_layers(2)
plotter.read_file('./logs/regression.txt')
plotter.plot_regression(X_train, y_train, "./plots")

X_test, y_test = linear_regression_dataset(50, slope = 1, intercept = -4)
X_test = standardize_data(X_test)
y_test = standardize_data(y_test)
trainer.predict(X_test, y_test)