import numpy as np
import matplotlib.pyplot as plt
from trainer import Logger
import os, time

class Plotter:
    def read_file(self, path):
        self.data = Logger.load_data(path)
    
    def set_model_info(self, n_layers, n_weights):
        self.n_layers = n_layers
        self.n_weights = n_weights
    
    def plot_gradients(self, directory = "./plots", name = None):
        if not os.access(directory, os.F_OK):
            print(f'cannot access {directory}')
            return
        name = f'{"output_" + str(time.time_ns()) if name is None else name}'

        weights_gradients = dict()
        bias_gradients = dict()

        for i in range(len(self.data)):
            epoch = f'epoch-{i}'
            for j in range(len(self.data[epoch])):
                update = f'update-{j}'
                for w in range(self.n_weights):
                    weight = f'weights-gradient-{w}'
                    if weights_gradients[w] is None:
                        weights_gradients[w] = list()
                    weights_gradients[w].append(np.asarray(self.data[epoch][update][weight]))
                for b in range(self.n_layers):
                    bias = f'bias-gradient-{b}'
                    if bias_gradients[b] is None:
                        bias_gradients[b] = list()
                    bias_gradients[b].append(np.asarray(self.data[epoch][update][bias]))
