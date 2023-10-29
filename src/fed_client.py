import os
import argparse

import flwr as fl
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense

# loggers output all messages to stdout in addition to log file
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load model and data (MobileNetV2, CIFAR-10)
DATA_FRACTION = 0.1
def get_random_subset(x, y, fraction):
    indices = np.random.choice(len(x), int(len(x) * fraction), replace=False)
    return x[indices], y[indices]
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() 

# normalize 
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train_client, y_train_client = get_random_subset(x_train, y_train, DATA_FRACTION) # take a fraction of the dataset

model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])



# Load the MNIST dataset
#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data(); x_train = x_train / 255.0; x_test = x_test / 255.0

# Define the model architecture
#model = Sequential([
#    Flatten(input_shape=(28, 28)),
#    Dense(64, activation='relu'),
#    Dense(10, activation='softmax')
#])
# Compile the model
#model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train_client, y_train_client, epochs=3, batch_size=5)  # use the client's data subset
        return model.get_weights(), len(x_train_client), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test)
        return loss, len(x_test), {"accuracy": accuracy}


def start_flower_client(ip_server_address, port = "8080"):
    # Create the full server address
    server_address = ip_server_address+":"+port
    
    # Start Flower client
    fl.client.start_numpy_client(server_address=server_address, client=CifarClient())

def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument('--ip_address', type=str, default="0.0.0.0", help='IP address of the server.')
    

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    server_address = args.ip_address

    start_flower_client(server_address)

if __name__ == "__main__":
    main()
