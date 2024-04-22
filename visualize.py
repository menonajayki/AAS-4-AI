import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.utils import plot_model

# Define the input shape
input_shape = (10, 1)  # sequence length = 10, number of features = 1

# Define the input layer
inputs = Input(shape=input_shape, name='Input')

# Define the LSTM layer
num_units = 16
lstm_layer = LSTM(num_units, name='LSTM')

# Pass the input through the LSTM layer
lstm_output = lstm_layer(inputs)

# Define the output layer
output_layer = Dense(1, name='Output')(lstm_output)

# Define the model
model = tf.keras.Model(inputs=inputs, outputs=output_layer)

# Visualize the model architecture
plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)
