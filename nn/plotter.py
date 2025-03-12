import numpy as np
import matplotlib.pyplot as plt
from nn.trainer import Logger
import os, time

class Plotter:
    def read_file(self, path):
        self.data = Logger.load_data(path)
        if self.data is None:
            print("unsucessful.")
        else:
            print("file read.")
    
    def set_model_layers(self, n_layers):
        self.n_layers = n_layers
    
    @staticmethod
    def skip_samples(array, limit):
        # can discard a few points from the end of the array
        has = array.shape[0]
        initial_skip = has // limit
        if initial_skip == 0:
            return array
        array = array[::initial_skip]

        first = array.shape[0] - 2 * (array.shape[0] - limit)
        if first == array.shape[0]:
            return array
        array = np.vstack((array[:first], array[first::2]))
        return array

    @staticmethod
    def select_samples(array, limit):
        if limit > array.shape[0]:
            return array
        return array[:limit]
    
    def plot_gradients(self, dir, name = None, n_points = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/" + ("gradients_" + time_str if name is None else name) + ".png"

        weights_gradients = dict()
        bias_gradients = dict()

        for i in range(self.data['n-epochs']):
            epoch = f'epoch-{i}'
            for j in range(self.data[epoch]['n-batches']):
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
        
        fig, axs = plt.subplots(2, self.n_layers-1, figsize = (15, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle(f'Gradients', size = 'xx-large')

        # weights
        for ax in range(self.n_layers-1):
            weight = f'weights-gradient-{ax}'
            weights_gradients[weight] = np.asarray(weights_gradients[weight])  # convert entire history of gradients into one numpy array
            if n_points is not None:
                weights_gradients[weight] = Plotter.select_samples(weights_gradients[weight], n_points)
            for r in range(weights_gradients[weight][0].shape[0]):
                for c in range(weights_gradients[weight][0].shape[1]):
                    axs[0, ax].plot(weights_gradients[weight][:, r, c], label = f'[{r}, {c}]', linewidth = 0.7)
                    axs[0, ax].legend(title = 'elements')
                    axs[0, ax].set_title(f'weights-{ax}')
                    axs[0, ax].set_xlabel("Updates")
        
        # bias
        for ax in range(1, self.n_layers):
            bias = f'bias-gradient-{ax}'
            bias_gradients[bias] = np.asarray(bias_gradients[bias])
            if n_points is not None:
                bias_gradients[bias] = Plotter.select_samples(bias_gradients[bias], n_points)
            for r in range(bias_gradients[bias][0].shape[0]):
                    axs[1, ax-1].plot(bias_gradients[bias][:, r, 0], label = f'[{r}]', linewidth = 0.7)
                    axs[1, ax-1].legend(title = 'elements')
                    axs[1, ax-1].set_title(f'bias-{ax-1}')
                    axs[1, ax-1].set_xlabel("Updates")
        
        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()  # awesome pop-up window when using vs code!
    
    def plot_weights(self, dir, name = None, n_points = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/" + ("weights_" + time_str if name is None else name) + ".png"

        weights = dict()
        bias = dict()

        init = self.data['init']
        for i in range(self.n_layers-1):
            weight = f'weights-{i}'
            if weight not in weights:
                weights[weight] = list()
            weights[weight].append(np.asarray(init['weights'][weight]))
        for i in range(1, self.n_layers):
            bias_name = f'bias-{i}'
            if bias_name not in bias:
                bias[bias_name] = list()
            bias[bias_name].append(np.asarray(init['bias'][bias_name]))

        for i in range(self.data['n-epochs']):
            epoch = f'epoch-{i}'
            for j in range(self.data[epoch]['n-batches']):
                update = f'update-{j}'
                for w in range(self.n_layers-1):
                    weight = f'weights-{w}'
                    weights[weight].append(np.asarray(self.data[epoch][update][weight]))
                for b in range(1, self.n_layers):
                    bias_name = f'bias-{b}'
                    bias[bias_name].append(np.asarray(self.data[epoch][update][bias_name]))
        
        fig, axs = plt.subplots(2, self.n_layers-1, figsize = (15, 8), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle("Weights & Bias Values", size = 'xx-large')
        # weights
        for ax in range(self.n_layers-1):
            weight = f'weights-{ax}'
            weights[weight] = np.asarray(weights[weight])
            if n_points is not None:
                weights[weight] = Plotter.select_samples(weights[weight], n_points)
            for r in range(weights[weight][0].shape[0]):
                for c in range(weights[weight][0].shape[1]):
                    axs[0, ax].plot(weights[weight][:, r, c], label = f'[{r}, {c}]', linewidth = 0.7)
                    axs[0, ax].legend(title = 'elements')
                    axs[0, ax].set_title(f'weights-{ax}')
                    axs[0, ax].set_xlabel("Updates")
        
        # bias
        for ax in range(1, self.n_layers):
            bias_name = f'bias-{ax}'
            bias[bias_name] = np.asarray(bias[bias_name])
            if n_points is not None:
                bias[bias_name] = Plotter.select_samples(bias[bias_name], n_points)
            for r in range(bias[bias_name][0].shape[0]):
                    axs[1, ax-1].plot(bias[bias_name][:, r, 0], label = f'[{r}]', linewidth = 0.7)
                    axs[1, ax-1].legend(title = 'elements')
                    axs[1, ax-1].set_title(f'bias-{ax-1}')
                    axs[1, ax-1].set_xlabel("Updates")
        
        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()
    
    def plot_accuracy(self, dir, name = None, n_points = None):
        if not os.access(dir, os.F_OK):
            print(f'cannot access {dir}')
            return
        time_str = time.strftime("%I-%M-%S_%p", time.localtime(time.time()))
        name = "/" + ("accuracy_" + time_str if name is None else name) + ".png"

        accuracy_list = list()
        loss_list = list()
        confusion_matrix = {'tp': list(), 'tn': list(), 'fp': list(), 'fn': list()}

        for i in range(self.data['n-epochs']):
            epoch = f'epoch-{i}'
            accuracy_list.append(self.data[epoch]['accuracy'])
            loss_list.append(self.data[epoch]['loss'])
            matrix = self.data[epoch]['confusion-matrix']
            for key in matrix.keys():
                confusion_matrix[key].append(matrix[key])

        fig, axs = plt.subplots(1, 2, figsize = (15, 5), gridspec_kw = {'wspace': 0.2, 'hspace': 0.3})
        fig.suptitle(f"Metrics", size = 'xx-large')

        loss_list = np.asarray(loss_list)
        accuracy_list = np.asarray(accuracy_list)
        if n_points is not None:
                loss_list = Plotter.select_samples(loss_list, n_points)
                accuracy_list = Plotter.select_samples(accuracy_list, n_points)
        axs[0].plot(loss_list, label = "Loss")
        axs[0].plot(accuracy_list, label = "Accuracy")
        axs[0].set_xlabel("Epochs")
        axs[0].set_title("Loss & Accuracy")
        axs[0].legend()

        for key in confusion_matrix.keys():
            confusion_matrix[key] = np.asarray(confusion_matrix[key])
            if n_points is not None:
                confusion_matrix[key] = Plotter.select_samples(confusion_matrix[key], n_points)
            axs[1].plot(confusion_matrix[key], label = key)
        axs[1].set_xlabel("Epochs")
        axs[1].set_title("Confusion Matrix")
        axs[1].legend()

        fig.savefig(dir + name, bbox_inches = "tight")
        print(f'plot saved at {dir + name}')
        plt.show()
    
    @staticmethod
    def plot_points(x, y):
        fig, ax = plt.subplots(figsize = (7, 5))
        fig.suptitle(f"Points", size = 'xx-large')

        ax.scatter(x, y)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        plt.show()