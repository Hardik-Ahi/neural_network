import numpy as np
from dataset_utils import get_minibatch
from model_functions import round

l_rate = 0.01

class SGDTrainer():

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.set_model(self.model)
    
    def update_biases(self, layer, learning_rate = l_rate):
        layer.b_ += ((learning_rate) * layer.del_)
    
    def update_weights(self, weights, learning_rate = l_rate):
        weights.matrix += ((learning_rate) * weights.gradients)
    
    def forward_pass(self, input_data):
        # input_data shape = (1, n_cols)
        self.model.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.model.layers)):
            self.model.layers[i].z_ = np.matmul(self.model.weights[i-1].matrix, self.model.layers[i-1].a_) + self.model.layers[i].b_
            self.model.layers[i].a_ = self.model.layers[i].activation(self.model.layers[i].z_)
    
    def backward_pass(self, label):
        self.optimizer.error_output_layer(self.model.layers[-1], label, True)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.optimizer.error_layer(self.model.layers[i], self.model.weights[i].matrix, self.model.layers[i+1], True)
        
        for i in range(len(self.model.weights)):
            self.optimizer.gradients(self.model.weights[i], True)
    
    def show_epoch(self, epoch):
        print(f"epoch-{epoch}:")

    def train(self, features, targets, epochs = 1):
        targets.reshape((targets.size,))
        self.model.init_weights()

        for epoch in range(epochs):
            self.show_epoch(epoch)
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass(targets[i])

                for j in range(1, len(self.model.layers)):
                    self.update_biases(self.model.layers[j])
                for j in range(len(self.model.weights)):
                    self.update_weights(self.model.weights[j])

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
    
    def predict(self, features, targets):
        targets = targets.reshape((targets.size,))
        predictions = list()

        for i in range(features.shape[0]):
            self.forward_pass(features[i])
            predictions.append(self.model.layers[-1].a_[0][0])
        
        predictions = np.array(predictions)
        loss = self.model.loss_function(targets, predictions.reshape((predictions.size,)))
        print("Loss:", loss)
        score = self.accuracy(targets, round(predictions))
        print("Accuracy:", score)
        print("Confusion matrix:", self.confusion_matrix(targets, round(predictions)))


class BatchTrainer(SGDTrainer):

    def error_output_layer(self, layer, label):
        value = self.optimizer.error_output_layer(layer, label)
        layer.b_gradients += value / self.batch_size
        layer.del_ = value

    def error_layer(self, this_layer, weights, next_layer):
        value = self.optimizer.error_layer(this_layer, weights, next_layer)
        this_layer.b_gradients += value / self.batch_size
        this_layer.del_ = value
    
    def update_biases(self, layer, learning_rate = l_rate):
        layer.b_ += ((learning_rate) * layer.b_gradients)

    def backward_pass(self, label):  # single instance
        self.error_output_layer(self.model.layers[-1], label)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.error_layer(self.model.layers[i], self.model.weights[i].matrix, self.model.layers[i+1])
        
        for i in range(len(self.model.weights)):
            self.gradients(self.model.weights[i])

    def train(self, features, targets, epochs = 1):
        targets.reshape((targets.size,))
        self.model.init_weights()
        self.batch_size = features.shape[0]
        
        for epoch in range(epochs):
            self.show_epoch(epoch)
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass(targets[i])

            for j in range(1, len(self.model.layers)):
                self.update_biases(self.model.layers[j])
            for j in range(len(self.model.weights)):
                self.update_weights(self.model.weights[j])
            
            self.predict(features, targets)
            self.optimizer.on_pass()
    
    def gradients(self, weights):
        weights.gradients += self.optimizer.gradients(weights) / self.batch_size
    
    def update_weights(self, weights, learning_rate = l_rate):
        weights.matrix += ((learning_rate) * weights.gradients)

class MiniBatchTrainer(BatchTrainer):

    def __init__(self, model, optimizer, minibatch_size):
        super().__init__(model, optimizer)
        self.minibatch_size = minibatch_size
    
    def train(self, features, targets, epochs = 1):
        self.model.init_weights()
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
                    self.update_biases(self.model.layers[j])
                for j in range(len(self.model.weights)):
                    self.update_weights(self.model.weights[j])
                
                self.optimizer.on_pass()
                start_index += self.batch_size
            self.predict(features, targets)