import numpy as np

# optimizer will give the trainer the gradients[] matrix to apply to the weights.
class SGD():

    def set_model(self, model):
        self.model = model
    
    def gradients(self, weight_index, apply = False):
        value = -np.matmul(self.model.weights[weight_index].layer_2.del_, np.transpose(self.model.weights[weight_index].layer_1.a_))
        if not apply:
            return value
        else:
            self.model.weights[weight_index].gradients = value
    
    def error_output_layer(self, layer_index, label, apply = False):
        value = -self.model.der_loss_function(
            label, self.model.layers[layer_index].a_[0][0]) * self.model.layers[layer_index].der_activation(self.model.layers[layer_index].z_)
        if not apply:
            return value
        else:
            self.model.layers[layer_index].del_ = value
            self.model.layers[layer_index].b_gradients = value

    def error_layer(self, this_index, weight_index, apply = False):  # weights connecting this layer to next layer
        value = -np.matmul(
            np.transpose(self.model.weights[weight_index].matrix),
            self.model.layers[this_index+1].del_) * self.model.layers[this_index].der_activation(self.model.layers[this_index].z_)
        if not apply:
            return value
        else:
            self.model.layers[this_index].del_ = value
            self.model.layers[this_index].b_gradients = value
        
    def on_pass(self):
        # reset gradients
        for weight in self.model.weights:
            weight.gradients = np.zeros((weight.rows, weight.cols))
        
        # reset errors
        for layer in self.model.layers:
            layer.del_ = np.zeros((layer.n_neurons, 1))
            layer.b_gradients = np.zeros((layer.n_neurons, 1))

# make every layer update its biases using layer.b_gradients rather than layer.del_
# layer.del_ is the error due to the present training sample; it needs to remain unmodified by optimizers to pass back to
# the weights and biases so that THEY can calculate their gradients (modified or not) correctly.
class Adam(SGD):

    def set_model(self, model):
        super().set_model(model)
        self.weight_mean = list()
        self.weight_std = list()
        self.bias_mean = list()
        self.bias_std = list()

        for i in self.model.weights:
            self.weight_mean.append(np.zeros(i.gradients.shape))
            self.weight_std.append(np.zeros(i.gradients.shape))
        for i in self.model.layers:
            self.bias_mean.append(np.zeros(i.b_gradients.shape))
            self.bias_std.append(np.zeros(i.b_gradients.shape))
        
        self.iteration = 0  # increment this in self.on_pass()
    
    def reset(self):
        self.iteration = 0

    # time-step increments whenever we do update_biases() or update_weights() from the Trainer.
    
