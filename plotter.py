import numpy as np
import matplotlib.pyplot as plt
from trainer import Logger
import os, time

class Plotter:
    def read_file(self, path):
        self.data = Logger.load_data(path)
        if self.data is None:
            print("unsucessful.")
        else:
            print("file read.")
    
    def set_model_info(self, n_layers):
        self.n_layers = n_layers
    
    def plot_gradients(self, directory = "./plots", name = None):
        if not os.access(directory, os.F_OK):
            print(f'cannot access {directory}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = f'/{"output_" + time_str if name is None else name}.png'

        weights_gradients = dict()
        bias_gradients = dict()

        for i in range(len(self.data)-1):
            epoch = f'epoch-{i}'
            for j in range(len(self.data[epoch])-1):
                try:
                    update = f'update-{j}'
                    for w in range(self.n_layers-1):
                        weight = f'weights-gradient-{w}'
                        if weight not in weights_gradients:
                            weights_gradients[weight] = list()
                        weights_gradients[weight].append(np.asarray(self.data[epoch][update][weight]))
                    for b in range(1, self.n_layers):
                        bias = f'bias-gradient-{b}'
                        if bias not in bias_gradients:
                            bias_gradients[bias] = list()
                        bias_gradients[bias].append(np.asarray(self.data[epoch][update][bias]))
                except KeyError:
                    break
        
        fig, axs = plt.subplots(2, self.n_layers-1, figsize = (12, 7), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle(f'Applied Gradients', size = 'xx-large')
        # weights
        for ax in range(self.n_layers-1):
            weight = f'weights-gradient-{ax}'
            weights_gradients[weight] = np.asarray(weights_gradients[weight])
            for r in range(weights_gradients[weight][0].shape[0]):
                for c in range(weights_gradients[weight][0].shape[1]):
                    axs[0, ax].plot(weights_gradients[weight][:, r, c], label = f'[{r}, {c}]')
                    axs[0, ax].legend(title = 'elements')
                    axs[0, ax].set_title(f'weights-{ax}')
        
        # bias
        for ax in range(1, self.n_layers):
            bias = f'bias-gradient-{ax}'
            bias_gradients[bias] = np.asarray(bias_gradients[bias])
            for r in range(bias_gradients[bias][0].shape[0]):
                    axs[1, ax-1].plot(bias_gradients[bias][:, r, 0], label = f'[{r}]')
                    axs[1, ax-1].legend(title = 'elements')
                    axs[1, ax-1].set_title(f'bias-{ax-1}')
        
        fig.savefig(directory + name, bbox_inches = "tight")

plotter = Plotter()
plotter.set_model_info(3)
plotter.read_file("logs\output_07-54-08_PM.txt")
plotter.plot_gradients()