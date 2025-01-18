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
