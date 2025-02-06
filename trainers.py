import numpy as np
from dataset_utils import get_minibatch
from model_functions import round

l_rate = 0.05

class InstanceTrainer():

    def __init__(self, model, optimizer):
        self.model = model
        self.model.init_weights()
        self.optimizer = optimizer
        self.optimizer.set_model(self.model)
    
    def update_biases(self, layer_index, learning_rate = l_rate):
        self.model.layers[layer_index].b_ -= ((learning_rate) * self.optimizer.current_gradient_biases(layer_index))
    
    def update_weights(self, weight_index, learning_rate = l_rate):
        self.model.weights[weight_index].matrix -= ((learning_rate) * self.optimizer.current_gradient_weights(weight_index))
    
    def forward_pass(self, input_data):
        # input_data shape = (1, n_cols)
        self.model.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.model.layers)):
            self.model.layers[i].z_ = np.matmul(self.model.weights[i-1].matrix, self.model.layers[i-1].a_) + self.model.layers[i].b_
            self.model.layers[i].a_ = self.model.layers[i].activation(self.model.layers[i].z_)
    
    def backward_pass(self, label):
        self.optimizer.error_output_layer(-1, label, True)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.optimizer.error_layer(i, i, True)
        
        for i in range(len(self.model.weights)):
            self.optimizer.gradients(i, True)
    
    def show_epoch(self, epoch):
        print(f"epoch-{epoch}:")

    def train(self, features, targets, epochs = 1):
        targets.reshape((targets.size,))

        for epoch in range(epochs):
            self.show_epoch(epoch)
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass(targets[i])

                for j in range(1, len(self.model.layers)):
                    self.update_biases(j)
                for j in range(len(self.model.weights)):
                    self.update_weights(j)

                self.optimizer.on_pass()

            self.predict(features, targets)
    
    def confusion_matrix(self, labels, predictions):
        predictions = predictions.reshape((predictions.size, 1))
        labels = labels.reshape((labels.size, 1))
        stack = np.hstack((labels, predictions))
        matrix = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}

        for i in range(stack.shape[0]):
            if stack[i][0] == 1:
                if stack[i][1] == 1:
                    matrix['tp'] += 1
                else:
                    matrix['fn'] += 1
            else:
                if stack[i][1] == 1:
                    matrix['fp'] += 1
                else:
                    matrix['tn'] += 1
        return matrix
    
    def accuracy(self, labels, predictions):
        confusion_matrix = self.confusion_matrix(labels, predictions)
        total = 0
        for i in confusion_matrix.values():
            total += i
        return (confusion_matrix['tp'] + confusion_matrix['tn']) / total
    
    def predict(self, features, targets, output_predictions = False):
        targets = targets.reshape((targets.size,))
        predictions = list()

        for i in range(features.shape[0]):
            self.forward_pass(features[i])
            predictions.append(self.model.layers[-1].a_[0][0])
        
        predictions = np.array(predictions)
        loss = self.model.loss_function(targets, predictions.reshape((predictions.size,)))
        print("Loss:", loss)
        predictions = round(predictions)
        score = self.accuracy(targets, predictions)
        print("Accuracy:", score)
        print("Confusion matrix:", self.confusion_matrix(targets, round(predictions)))  # why suddenly getting 85% accuracy?
        if output_predictions:
            print("predictions:", predictions)
            print("targets:    ", targets)
        '''print("weights:")
        self.model.show_weights()
        print("biases:")
        self.model.show_biases()'''

# you got something right here! (it's working)
class BatchTrainer(InstanceTrainer):

    def error_output_layer(self, layer_index, label):
        value = self.optimizer.error_output_layer(layer_index, label)
        self.model.layers[layer_index].b_gradients += value / self.batch_size
        self.model.layers[layer_index].del_ = value

    def error_layer(self, this_index, weight_index):
        value = self.optimizer.error_layer(this_index, weight_index)
        self.model.layers[this_index].b_gradients += value / self.batch_size
        self.model.layers[this_index].del_ = value

    def backward_pass(self, label):  # single instance
        self.error_output_layer(-1, label)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.error_layer(i, i)
        
        for i in range(len(self.model.weights)):
            self.gradients(i)

    def train(self, features, targets, epochs = 1):
        targets.reshape((targets.size,))
        self.batch_size = features.shape[0]
        
        for epoch in range(epochs):
            self.show_epoch(epoch)
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass(targets[i])

            for j in range(1, len(self.model.layers)):
                self.update_biases(j)
            for j in range(len(self.model.weights)):
                self.update_weights(j)
            
            self.predict(features, targets)
            self.optimizer.on_pass()
    
    def gradients(self, weight_index):
        self.model.weights[weight_index].gradients += self.optimizer.gradients(weight_index) / self.batch_size

class MiniBatchTrainer(BatchTrainer):

    def __init__(self, model, optimizer, minibatch_size):
        super().__init__(model, optimizer)
        self.minibatch_size = minibatch_size
    
    def train(self, features, targets, epochs = 1):
        targets.reshape((targets.size,))
        
        for epoch in range(epochs):
            self.show_epoch(epoch)
            start_index = 0
            while start_index < features.shape[0]:
                real_features, real_targets = get_minibatch(features, targets, self.minibatch_size, start_index)
                real_targets = real_targets.reshape((real_targets.size,))
                self.batch_size = real_features.shape[0]  # giving this minibatch_size to BatchTrainer's error_computing functions
                for i in range(real_features.shape[0]):
                    self.forward_pass(real_features[i])
                    self.backward_pass(real_targets[i])

                for j in range(1, len(self.model.layers)):
                    self.update_biases(j)
                for j in range(len(self.model.weights)):
                    self.update_weights(j)
                
                self.optimizer.on_pass()
                start_index += self.batch_size
            self.predict(features, targets)