import numpy as np
from numpy.random import default_rng  # rng = random number generator
from math import sqrt

'''
structure: input nodes = 2, hidden nodes = 2, output nodes = 1. fully connected.
activation = relu, output activation = sigmoid
loss function = binary cross entropy
'''
@np.vectorize  # decorator, allows this func to execute on all elements of an ndarray
def relu(x):
    return x if x > 0 else 0

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def round(x):
    return 1 if x >= 0.5 else 0

def init_weights(fan_in, prev_neurons, seed):  # fan_in to this layer = fan_out of previous layer
    std = sqrt(2 / fan_in)  # standard deviation for 'He' initialization (use it if you use ReLU functions)
    generator = default_rng(seed)
    weights = generator.standard_normal((fan_in, prev_neurons)) * std  # z = x-mu / sigma, therefore x = z . sigma + mu; z = standard normal
    # here we need mu (mean) = 0, but standard deviation as per we calculated using fan_in.
    return weights

def binary_loss(labels, predictions):
    # predictions[i] is in the range [0, 1]
    total = predictions.size
    summation = 0

    for label, prediction in np.nditer([labels, predictions]):
        summation -= label * np.log(prediction) + (1 - label) * np.log(1 - prediction)
    
    return (1 / total) * summation

def accuracy(labels, predictions):
    confusion_matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

    for label, prediction in np.nditer([labels, predictions]):
        if label == 1:
            if prediction == 1:
                confusion_matrix['tp'] += 1
            else:
                confusion_matrix['fn'] += 1
        else:
            if prediction == 1:
                confusion_matrix['fp'] += 1
            else:
                confusion_matrix['tn'] += 1

    total = 0
    for i in confusion_matrix.values():
        total += i
    
    # print(confusion_matrix)
    return (confusion_matrix['tp'] + confusion_matrix['tn']) / total

def get_vector(seed = 12345, upper_bound = 100, n_samples = 10, zeros = False):
    generator = default_rng(seed)
    vector = None
    if zeros:
        vector = generator.integers(0, 1, size = (n_samples, 1))
    else:
        vector = generator.integers(1, upper_bound, (n_samples, 1))

    return vector


# rows = no.of fan_out from all neurons in previous layer = no.of neurons in current layer
# cols = no.of neurons in the previous layer
# cells = weight values for (rows * cols) weights

weights = init_weights(2, 2, 1000)
bias_weights = np.zeros((2, 1))

outputs = init_weights(1, 2, 4000)
bias_outputs = np.zeros((1, 1))

# generate sub-arrays (features) for classes '1' and '0'
both_positive = np.hstack((get_vector(seed = 187, n_samples = 12), get_vector(seed = 9, n_samples = 12)))
one_zero_1 = np.hstack((get_vector(zeros = True, n_samples = 6), get_vector(seed = 45, n_samples = 6)))
one_zero_2 = np.hstack((get_vector(seed = 987, n_samples = 6), get_vector(zeros = True, n_samples = 6)))
both_zero = np.array([0, 0])

features = np.vstack((both_positive, one_zero_1, one_zero_2, both_zero))
#print(features)

# generate labels
labels_positive = np.ones((12, 1), dtype = int)
labels_negative = np.zeros((13, 1), dtype = int)

labels = np.vstack((labels_positive, labels_negative))
#print(labels)

# combined view
dataset = np.hstack((features, labels))
#print(dataset)

# matrix multiplication: new_result = weight_matrix * previous_result

features = features.reshape((2, features.shape[0]))  # now can use entire dataset in one go
labels = labels.reshape((1, labels.shape[0]))

print("forward pass")
middle_1 = np.matmul(weights, features) + bias_weights  # this is called 'broadcasting' => bias_weights will stretch across multiple columns
print("before relu: ", middle_1)

middle_1 = relu(middle_1)
print("after relu: ", middle_1)

middle_2 = np.matmul(outputs, middle_1) + bias_outputs
print("output before sigmoid: ", middle_2)

middle_2 = sigmoid(middle_2)
print("output after sigmoid: ", middle_2)

print("loss: ", binary_loss(labels, middle_2))
middle_2 = round(middle_2)
print("rounded results: ", middle_2)
print("actual labels: ", labels)
print("accuracy: ", accuracy(labels, middle_2))

def der_binary_cross_entropy(label, output):  # output belongs to range [0, 1]
    return ((1 - label)/(1 - output)) - (label / output)

def sigmoid_func(x):
    return 1 / (1 + np.exp(-x))

@np.vectorize
def der_sigmoid(x):
    return sigmoid_func(x) * (1 - sigmoid_func(x))

@np.vectorize
def der_relu(x):
    return 1 if x > 0 else 0

class Layer:

    def __init__(self, n_neurons, activation, der_activation):
        self.n_neurons = n_neurons
        self.activation = activation  # should be np.vectorize'd
        self.der_activation = der_activation  # also np.vectorize'd
        self.z_ = None
        self.del_ = None
        self.a_ = None
        self.b_ = np.zeros((n_neurons, 1))
    
    def compute_output_error(self, label, der_loss_function):
        self.del_ = der_loss_function(label, self.a_[0][0]) * self.der_activation(self.z_)

    def compute_error(self, weights, next_layer):  # weights matrix connecting this layer to the next layer
        self.del_ = np.matmul(np.transpose(weights), next_layer.del_) * self.der_activation(self.z_)
    
    def update_biases(self, learning_rate = 0.2):  # larger learning rate to provide the brute force assist in learning
        self.b_ = self.b_ - (learning_rate * self.del_)

class Weights:

    def __init__(self, layer_1, layer_2, seed = 1000):
        self.rows = layer_2.n_neurons
        self.cols = layer_1.n_neurons
        self.seed = seed
        self.layer_1 = layer_1
        self.layer_2 = layer_2
        self.matrix = init_weights(self.rows, self.cols, self.seed)
        self.gradients = np.zeros((self.rows, self.cols))
    
    def calc_gradient(self):
        self.gradients = np.matmul(self.layer_2.del_, np.transpose(self.layer_1.a_))
    
    def update_weights(self, learning_rate = 0.1):
        self.weights = self.weights - (learning_rate * self.gradients)

class Model:

    def __init__(self):
        self.layers = list()
        self.weights = list()
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def init_weights(self):
        for i in range(len(self.layers)-1):
            weights.append(Weights(self.layers[i], self.layers[i+1], seed = (i*100) + 5))

    def forward_pass(self, input_data):  # for single training instance
        try:
            self.layers[0].a_ = input_data
            for i in range(1, len(self.layers)):
                self.layers[i].z_ = np.matmul(self.weights[i-1], self.layers[i-1].a_) + self.layers[i].b_
                self.layers[i].a_ = self.layers[i].activation(self.layers[i].z_)
        except IndexError:
            print("invalid index used in forward pass")
        except:
            print("numpy error in forward pass")
    
    def calc_loss(self, loss_function, label):  # for single instance only
        try:
            return loss_function(label, self.layers[-1].a_)
        except IndexError:
            print("invalid index in calc_loss")
        except:
            print("error calculating loss in calc_loss")
    
    def backward_pass(self, label):
        try:
            self.layers[-1].compute_output_error(label)
            for i in range(len(self.layers)-2, -1, -1):
                self.layers[i].compute_error(self.weights[i], self.layers[i+1])
                self.layers[i].update_biases()
            
            for i in range(len(self.weights)):
                self.weights[i].calc_gradient()
                self.weights[i].update_weights()
        except IndexError:
            print("invalid index in backward pass")
        except:
            print("error in backward pass")