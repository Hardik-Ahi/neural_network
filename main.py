import numpy as np
from nn.model_classes import Layer, Model
from nn.dataset_utils import and_gate_dataset, linear_regression_dataset, standardize_data, equalize_classes, split_classes, split_data
from nn.functions import der_leaky_relu, der_sigmoid, leaky_relu, sigmoid, mean_square_error, der_mean_square_error, mirror, der_mirror
from nn.optimizers import SGD
from nn.trainer import Trainer, RegressionTrainer
from nn.plotter import Plotter
import pandas as pd

X_train, y_train = and_gate_dataset()
df = pd.DataFrame(np.hstack((X_train, y_train)), columns = ['input_0', 'input_1', 'output'])

df = pd.DataFrame([[1, 1, 'a'], [2, 2, 'b'], [3, 3, 'c'], [4, 4, 'a'], [5, 5, 'b'], [6, 6, 'c'], [7, 7, 'a']], columns = ['1', '2', 'output'])


X_train, y_train = linear_regression_dataset(100)
df = pd.DataFrame(np.hstack((X_train, y_train)), columns = ['x', 'y'])
print(df)
train, test = split_data(df)
print(train)
print(test)
