## What is this project?
This is an implementation of neural networks from scratch, using only their mathematical foundations and the NumPy library to assist in matrix operations.

## Why did I do it?
Modern-day frameworks for machine learning, such as Tensorflow, Keras, PyTorch etc make it super easy to build, customize, train and test your neural network models. In order to make this possible, they abstract away the core mathematical representations and operations that drive the training process.
The goal with this project was to break that abstraction and see how a neural network can 'learn from the data'. More specifically, the backpropagation process was the core focus.

## What did I actually do?
In no specific order:

1. Use articles, blogs on the internet to get a grasp of the math: matrix representations, forward and backward pass algorithms, optimizers, principal component analysis (PCA), activation & loss functions (with their derivatives), 'He' initialization of weights, contour plots for loss landscape visualization.
2. Use NumPy to facilitate the implementation of that math.
3. Use Matplotlib to show as much information about the training process as possible, through plots such as:
    1. Gradients per update
    2. Parameter values per update
    3. Loss & score per epoch, with confusion matrix
    4. Loss landscape views through contour plots
    5. Change in the predictions of the model through the epochs.
4. Use object-oriented programming (where possible) for separation of concerns and simplified interactions between components.
5. Use jupyter notebooks as the 'client' to access the API offered by this system (package), and demonstrate its working.
6. Achieved ~95% accuracy on an 'AND gate dataset' which was generated programmatically, and ~85% accuracy on two datasets from Kaggle. Shows that the learning process actually happened, and thus the from-scratch implementation was a success.