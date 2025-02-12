import numpy as np
from dataset_utils import get_minibatch
from functions import round_off
import json, os, time

l_rate = 0.05

class Trainer:

    def __init__(self, model, optimizer, minibatch_size = None):  # None means batch GD. for stochastic, specify '1'.
        self.model = model
        self.model.init_weights()
        self.minibatch_size = minibatch_size
        self.optimizer = optimizer
        self.optimizer.set_model(self.model)
        self.logger = Logger()
    
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
            self.current_gradient(i)
    
    def current_gradient(self, weight_index):
        self.model.weights[weight_index].gradients += self.optimizer.current_gradient(weight_index) / self.batch_size
    
    def update_biases(self, layer_index, learning_rate = l_rate):
        value = self.optimizer.gradient_biases(layer_index)
        self.model.layers[layer_index].b_ -= ((learning_rate) * value)
        self.logger.log_update_bias(layer_index, value)
    
    def update_weights(self, weight_index, learning_rate = l_rate):
        value = self.optimizer.gradient_weights(weight_index)
        self.model.weights[weight_index].matrix -= ((learning_rate) * value)
        self.logger.log_update_weights(weight_index, value)
    
    def forward_pass(self, input_data):
        # input_data shape = (1, n_cols)
        self.model.layers[0].a_ = input_data.reshape((input_data.shape[-1], 1))
        for i in range(1, len(self.model.layers)):
            self.model.layers[i].z_ = np.matmul(self.model.weights[i-1].matrix, self.model.layers[i-1].a_) + self.model.layers[i].b_
            self.model.layers[i].a_ = self.model.layers[i].activation(self.model.layers[i].z_)
        
        
    
    def train(self, features, targets, epochs = 1):
        targets.reshape((targets.size,))
        if self.minibatch_size is None:
            self.minibatch_size = features.shape[0]

        self.logger.log_init(self.model.weights, self.model.layers)
        for epoch in range(epochs):
            print(f"epoch-{epoch}:")
            self.logger.log_epoch(epoch)
            start_index = 0
            batch = 0
            while start_index < features.shape[0]:
                self.logger.log_batch(batch)
                real_features, real_targets = get_minibatch(features, targets, self.minibatch_size, start_index)
                real_targets = real_targets.reshape((real_targets.size,))
                self.batch_size = real_features.shape[0]  # giving this minibatch_size to BatchTrainer's error_computing functions
                for i in range(real_features.shape[0]):
                    self.forward_pass(real_features[i])
                    self.logger.log_fp(i, self.model.weights, self.model.layers)
                    self.backward_pass(real_targets[i])
                    self.logger.log_bp(i, self.model.layers[-1].del_, self.model.weights, self.model.layers)

                self.logger.log_updates_init(batch)
                for j in range(1, len(self.model.layers)):
                    self.update_biases(j)
                for j in range(len(self.model.weights)):
                    self.update_weights(j)
                self.logger.log_updates_finish(self.model.weights, self.model.layers)

                self.optimizer.on_pass()
                start_index += self.batch_size
                batch += 1
            self.predict(features, targets)
        self.logger.write_log()
    
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
        predictions = round_off(predictions)
        score = self.accuracy(targets, predictions)
        print("Accuracy:", score)
        print("Confusion matrix:", self.confusion_matrix(targets, round_off(predictions)))
        if output_predictions:
            print("predictions:", predictions)
            print("targets:    ", targets)

class Logger:
    def __init__(self):
        self.object = dict()
        self.epoch = 0
        self.batch = 0
        self.fp = 0
        self.bp = 0
        self.update = 0
    
    def log_init(self, model_weights, model_layers):
        self.object['init'] = dict()
        dict_ = self.object['init']
        dict_['weights'] = dict()
        dict_['bias'] = dict()
        
        for i in range(len(model_weights)):
            dict_['weights'][f'weights-{i}'] = model_weights[i].matrix.tolist()
        for i in range(len(model_layers)):
            dict_['bias'][f'bias-{i}'] = model_layers[i].b_.tolist()
    
    def log_epoch(self, epoch):
        self.object[f'epoch-{epoch}'] = dict()
        self.epoch = epoch
    
    def log_batch(self, batch):
        self.object[f'epoch-{self.epoch}'][f'batch-{batch}'] = dict()
        self.batch = batch
    
    def log_fp(self, fp, model_weights, model_layers):
        ref = self.object[f'epoch-{self.epoch}'][f'batch-{self.batch}']
        ref[f'fp-{fp}'] = dict()
        self.fp = fp

        for i in range(len(model_layers)):
            ref[f'fp-{fp}'][f'activation-{i}'] = model_layers[i].a_.tolist()
            ref[f'fp-{fp}'][f'bias-{i}'] = model_layers[i].b_.tolist()
        for i in range(len(model_weights)):
            ref[f'fp-{fp}'][f'weights-{i}'] = model_weights[i].matrix.tolist()
    
    def log_bp(self, bp, error_output, model_weights, model_layers):
        ref = self.object[f'epoch-{self.epoch}'][f'batch-{self.batch}']
        ref[f'bp-{bp}'] = dict()
        self.bp = bp

        ref[f'bp-{bp}']['err-output'] = error_output.tolist()
        for i in range(len(model_layers)):
            ref[f'bp-{bp}'][f'err-{i}'] = model_layers[i].del_.tolist()
        for i in range(len(model_weights)):
            ref[f'bp-{bp}'][f'current-gradient-{i}'] = model_weights[i].gradients.tolist()
    
    def log_updates_init(self, update):
        ref = self.object[f'epoch-{self.epoch}']
        ref[f'update-{update}'] = dict()
        self.update = update

    def log_update_weights(self, index, weights_gradient):
        self.object[f'epoch-{self.epoch}'][f'update-{self.update}'][f'weights-gradient-{index}'] = weights_gradient.tolist()
    
    def log_update_bias(self, index, bias_gradient):
        self.object[f'epoch-{self.epoch}'][f'update-{self.update}'][f'bias-gradient-{index}'] = bias_gradient.tolist()
    
    def log_updates_finish(self, model_weights, model_layers):
        ref = self.object[f'epoch-{self.epoch}'][f'update-{self.update}']

        for i in range(len(model_weights)):
            ref[f'weights-{i}'] = model_weights[i].matrix.tolist()
        for i in range(len(model_layers)):
            ref[f'bias-{i}'] = model_layers[i].b_.tolist()
        
    def write_log(self, directory = "./logs", name = None):
        if not os.access(directory, os.F_OK):
            print(f"access to {directory} not allowed")
            return
        
        path = f'{directory + ("/output_" + str(time.time_ns()) if name is None else name)}.txt'
        string = json.dumps(self.object)
        with open(path, 'w') as file:
            file.write(string)

        print(f'output written to {path}')
        self.object = dict()
    
    @staticmethod
    def load_data(path):
        if not os.path.exists(path):
            print(f"{path} does not exist.")
            return
        with open(path, 'r') as f:
            data = json.loads(f.read())
        return data