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


train = and_gate_dataset(positive_samples = 100)
# train = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])  # all activations = sigmoid, reached 100% accuracy in 4000 epochs! (XOR problem solved)
X_train = train[:, [0, 1]]
y_train = train[:, 2].reshape((train.shape[0], 1))

trainer = InstanceTrainer(model, SGD())  # YES! got 100% accuracy in 1500 epochs using sigmoid activation function all-around!
model.show_weights()
model.show_biases()

#print(f"dataset:{train}")
trainer.train(X_train, y_train, epochs = 15)  # double YES!! got 100% accuracy in 250 epochs using leaky_relu with leak = 0.4!!

test = and_gate_dataset(positive_samples = 200)  # this is some serious testing! gets 100% accuracy here!!!!
X_test = test[:, [0, 1]]
y_test = test[:, 2]
print("testing:")
trainer.predict(X_test, y_test)

# what the ?! is happening: got 100% accuracy in 12 epochs using InstanceTrainer!!!
# and MiniBatchTrainer() performs more like InstanceTrainer with lower batch sizes (4, 8, 16) and more like BatchTrainer with higher sizes (32 etc).
# biases are now all zeros. we read something about them being able to contribute to training non-intrusively (and with lesser impact) 
# if they start at the zero point, instead of starting at a high positive / negative point.
# ADAM has something wrong; it is increasing the loss as training goes on.
# all this means that no need to get exact same numbers as in keras, and clipping to avoid inf and 0 loss is good enough!