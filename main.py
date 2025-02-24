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


# showing that when output falls in negative part of relu, with leak = 0, no gradients are observed throughout training.
'''model.weights[0].matrix = np.array([
    [-0.1, -0.01],
    [-0.06, -0.04]
])'''

train = and_gate_dataset(positive_samples = 100, seed = 1)
# train = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]])  # all activations = sigmoid, reached 100% accuracy in 4000 epochs! (XOR problem solved)
X_train = train[:, [0, 1]]
y_train = train[:, 2].reshape((train.shape[0], 1))

trainer = Trainer(model, SGD())  # YES! got 100% accuracy in 1500 epochs using sigmoid activation function all-around!
model.show_weights()
model.show_biases()

#print(f"dataset:{train}")
trainer.train(X_train, y_train, 1, 0.025, epochs = 300)  # double YES!! got 100% accuracy in 250 epochs using leaky_relu with leak = 0.4!!
trainer.save_history("./logs", "test")

test = and_gate_dataset(positive_samples = 100, seed = 2)  # this is some serious testing! gets 100% accuracy here!!!!
X_test = test[:, [0, 1]]
y_test = test[:, 2]

# load weights
model.load_weights('./models/test.npz')
model.show_weights()
model.show_biases()

print("testing with loaded weights")
trainer.predict(X_test, y_test)
# what the ?! is happening: got 100% accuracy in 12 epochs using InstanceTrainer!!!
# and MiniBatchTrainer() performs more like InstanceTrainer with lower batch sizes (4, 8, 16) and more like BatchTrainer with higher sizes (32 etc).
# biases are now all zeros. we read something about them being able to contribute to training non-intrusively (and with lesser impact) 
# if they start at the zero point, instead of starting at a high positive / negative point.
# ADAM has something wrong; it is increasing the loss as training goes on.
# all this means that no need to get exact same numbers as in keras, and clipping to avoid inf and 0 loss is good enough!

# further info: leaky_relu's leak, the higher it is, the worse the accuracy. leak = 0.1 gives 100% accuracy in merely 5 epochs!!!!
# but reducing leak further to 0.01 also performs worse; what's so special about the leak of 0.1? (maybe plug it in a derivation on paper)

plotter = Plotter()
plotter.set_model_layers(3)
plotter.read_file(r'logs\test.txt')
plotter.plot_gradients('./plots')
plotter.plot_weights('./plots')
plotter.plot_accuracy('./plots')