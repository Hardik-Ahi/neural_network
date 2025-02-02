import numpy as np

from dataset_utils import get_minibatch
from model_classes import accuracy
from model_functions import binary_loss

class SGDTrainer():

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.optimizer.set_model(self.model)
    
    def update_biases(self, layer, learning_rate = 0.01):
        layer.b_ += ((learning_rate) * layer.del_)
    
    def update_weights(self, weights, learning_rate = 0.01):
        weights.matrix += ((learning_rate) * weights.gradients)
    
    def forward_pass(self, input_data):  # for single training instance
        # input_data shape = (1, n_cols)
        self.model.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.model.layers)):
            self.model.layers[i].z_ = np.matmul(self.model.weights[i-1].matrix, self.model.layers[i-1].a_) + self.model.layers[i].b_
            self.model.layers[i].a_ = self.model.layers[i].activation(self.model.layers[i].z_)
    
    def backward_pass(self, label):  # single instance
        self.model.layers[-1].del_ = self.optimizer.error_output_layer(self.model.layers[-1], label)
        for i in range(len(self.model.layers)-2, 0, -1):  # except the input layer
            self.model.layers[i].del_ = self.optimizer.error_layer(self.model.layers[i], self.model.weights[i].matrix, self.model.layers[i+1])
        
        for i in range(len(self.model.weights)):
            self.model.weights[i].gradients = self.optimizer.gradients(self.model.weights[i])
    
    def train(self, features, targets, epochs = 1):
        self.model.init_weights()

        for epoch in range(epochs):
            for i in range(features.shape[0]):
                self.forward_pass(features[i])
                self.backward_pass(targets[i][0])

                for j in range(1, len(self.model.layers)):
                    self.update_biases(self.model.layers[j])
                for j in range(len(self.model.weights)):
                    self.update_weights(self.model.weights[j])

                prediction = self.model.layers[-1].a_[0]
                loss = self.model.loss_function(targets[i], prediction)
                print("loss:", loss)
                self.optimizer.on_pass()
    
    # another method: predict() for a batch of features / single instance.
    # but this is independent from the Trainer used. so where should we place it?

# very ambiguous, uncertain method definitions
class BatchTrainer(SGDTrainer):

    # extending of parent's __init__
    def __init__(self, model, batch_size):
        super().__init__(model)  # super() just returns a reference to the parent class
        self.batch_size = batch_size

    def error_output_layer(self, layer, label):
        layer.del_ += self.model.der_loss_function(label, layer.a_[0][0]) * layer.der_activation(layer.z_) / self.batch_size

    def error_layer(self, this_layer, weights, next_layer):
        this_layer.del_ += np.matmul(np.transpose(weights), next_layer.del_) * this_layer.der_activation(this_layer.z_) / self.batch_size
    
    def update_biases(self, layer, learning_rate = 0.1):
        layer.b_ -= ((learning_rate) * layer.del_)

    def backward_pass(self, labels):
        # write this func
        pass

    def train(self, features, targets, epochs = 1):
        self.init_weights()
        # write this func
    
    # profound refactoring needed here
    def calc_gradient_batch(self, batch_size = 1):
        if self.gradients.size == 0:
            self.gradients = np.matmul(self.layer_2.del_, np.transpose(self.layer_1.a_)) / batch_size
        else:
            self.gradients += np.matmul(self.layer_2.del_, np.transpose(self.layer_1.a_)) / batch_size
    
    def update_weights_batch(self, learning_rate = 0.1):
        self.matrix -= ((learning_rate) * self.gradients)

    def forward_pass_batch(self, input_data):
        self.layers[0].a_ = input_data.reshape((input_data.shape[1], input_data.shape[0]))
        for i in range(1, len(self.layers)):
            self.layers[i].z_ = np.matmul(self.weights[i-1].matrix, self.layers[i-1].a_) + self.layers[i].b_
            self.layers[i].a_ = self.layers[i].activation(self.layers[i].z_)

class MiniBatchTrainer(SGDTrainer):

    def __init__(self, model, minibatch_size):
        super().__init__(model)
        self.minibatch_size = minibatch_size
    
    def train_minibatch(self, features, targets, batch_size = 1, epochs = 1):
        self.init_weights()
        print("weights:")
        self.show_weights()
        learning_rate = 1
        for epoch in range(epochs):
            print("------epoch-", epoch, "------", sep="")
            if epoch > 500:
                learning_rate = 0.5
            start_index = 0
            while start_index < features.shape[0]:
                real_features, real_targets = get_minibatch(features, targets, batch_size, start_index)
                for i in range(real_features.shape[0]):
                    self.forward_pass(real_features[i])
                    self.backward_pass_batch(real_targets[i][0], batch_size = batch_size)

                for j in range(1, len(self.layers)):
                    self.layers[j].update_biases_batch(learning_rate = learning_rate)
                for j in range(len(self.weights)):
                    self.weights[j].update_weights_batch(learning_rate = learning_rate)
                
                self.clean_up_pass()
                start_index += batch_size
            
            self.forward_pass_batch(features)
            predictions = self.layers[-1].a_  # shape = (1, m)
            temp_targets = targets.reshape((1, targets.shape[0])).astype(int)
            print("\naccuracy: ", accuracy(temp_targets, round(predictions)))
            print("loss: ", binary_loss(targets, predictions.reshape((predictions.shape[-1], 1))))