import numpy as np

# optimizer will give the trainer the gradients[] matrix to apply to the weights.
class SGD():
    def set_model(self, model):
        self.model = model
    
    def gradients(self, weights, apply = False):
        value = -np.matmul(weights.layer_2.del_, np.transpose(weights.layer_1.a_))
        if not apply:
            return value
        else:
            weights.gradients = value
    
    def error_output_layer(self, layer, label, apply = False):
        value = -self.model.der_loss_function(label, layer.a_[0][0]) * layer.der_activation(layer.z_)
        if not apply:
            return value
        else:
            layer.del_ = value

    def error_layer(self, this_layer, weights, next_layer, apply = False):  # weights connecting this layer to next layer
        value = -np.matmul(np.transpose(weights), next_layer.del_) * this_layer.der_activation(this_layer.z_)
        if not apply:
            return value
        else:
            this_layer.del_ = value
        
    
    def on_pass(self):
        # reset gradients
        for weight in self.model.weights:
            weight.gradients = np.zeros((weight.rows, weight.cols))
        
        # reset errors
        for layer in self.model.layers:
            layer.del_ = np.zeros((layer.n_neurons, 1))
            layer.b_gradients = np.zeros((layer.n_neurons, 1))
    