import numpy as np
from numpy.random import default_rng
from model_classes import Layer, Model
from dataset_utils import and_gate_dataset, standardize_data
from model_functions import der_relu, der_sigmoid, relu, sigmoid, der_binary_cross_entropy, binary_loss
from optimizers import SGD, Adam
from trainers import BatchTrainer, InstanceTrainer, MiniBatchTrainer


model = Model(binary_loss, der_binary_cross_entropy, 100)  # for batch
model.add_layer(Layer(2))  # input layer
model.add_layer(Layer(2, relu, der_relu))
model.add_layer(Layer(1, sigmoid, der_sigmoid))

train = and_gate_dataset()
test = and_gate_dataset(positive_samples = 10)
X_train = train[:, [0, 1]]
y_train = train[:, 2].reshape((train.shape[0], 1))

trainer = BatchTrainer(model, SGD())
trainer.train(X_train, y_train, epochs = 100)

X_test = test[:, [0, 1]]
y_test = test[:, 2].reshape((test.shape[0], 1))

trainer.predict(X_test, y_test, True)