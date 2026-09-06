import streamlit as st
import numpy as np
import pandas as pd

st.title("Neural Network from Scratch", 
  text_alignment="center", 
  icon=":material/network_node:")

@st.cache_data
def load_train():
  X_train, y_train = and_gate_dataset(100, 1)
  dataset = pd.DataFrame(np.hstack((X_train, y_train)), columns=["Input 1", "Input 2", "Output"])
  return dataset

@st.cache_data
def load_test():
  X_test, y_test = and_gate_dataset(50, 2)
  dataset = pd.DataFrame(np.hstack((X_test, y_test)), columns=["Input 1", "Input 2", "Output"])
  return dataset

# DATASET
from nn.dataset_utils import and_gate_dataset

st.header("AND gate dataset")  # convert to dropdown
# map load_train(), load_test() to different datasets - Kaggle, AND gate.
# like a dropdown to select which dataset to load.

st.subheader("Training set")
train_data = load_train()
st.dataframe(train_data)

st.subheader("Testing set")
test_data = load_test()
st.dataframe(test_data)

# MODEL
from nn.model_classes import Model, Layer
from nn.functions import BinaryLoss, leaky_relu, der_leaky_relu, sigmoid, der_sigmoid

st.header("Model")

model = Model(BinaryLoss(), 1)
model.add_layer(Layer(2))
model.add_layer(Layer(2, leaky_relu(), der_leaky_relu()))
model.add_layer(Layer(1, sigmoid(), der_sigmoid))
model.compile()

# Display model layers as table
layers = pd.DataFrame(columns=["Layer", "Number of Neurons", "Activation Function"])
for i, layer in enumerate(model.layers):
  layers.loc[i] = [i+1, layer.n_neurons, layer.activation.__name__ if layer.activation else "None"]

st.dataframe(layers, hide_index=True)

# show weights and biases of model via checkbox
show_weights_biases = st.checkbox("Show Weights and Biases")

if show_weights_biases:
  st.subheader("Weights and Biases")
  
  # collect weights and biases from model
  biases = []
  weights = []
  for i, layer in enumerate(model.layers):
    biases.append(layer.b_.T)
  for i, weight in enumerate(model.weights):
    weights.append(weight.matrix)

  # display
  for i in range(len(weights)):
    st.write(f"Layer {i+1} Biases:")
    biases[i]
    st.write(f"Next Weights:")
    weights[i]
  st.write(f"Layer {len(weights)+1} Biases:")
  biases[-1]

# TRAIN
from nn.trainer import Trainer
from nn.optimizers import SGD

st.header("Training")

trainer = Trainer(model, SGD())
X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values

# input fields for training
batch_size = st.number_input("Batch Size (1 - 32)", min_value=1, max_value=32, value=1, step=1)
learning_rate = st.number_input("Learning Rate (0.001 - 1.0)", min_value=0.001, max_value=1.0, value=0.02, step=0.001, format="%.3f")
epochs = st.number_input("Epochs (1 - 500)", min_value=1, max_value=500, value=120, step=1)

if st.button("Train Model"):
  with st.spinner("Training in progress..."):
    trainer.train(X_train, y_train, batch_size, learning_rate, epochs = epochs)
  st.success("Training completed!")