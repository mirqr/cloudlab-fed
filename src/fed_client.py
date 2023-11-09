import os
import sys
import random
import string
import datetime

import argparse

import logging


import flwr as fl
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense



logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Make TensorFlow log less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data() # all the dataset

# normalize 
#x_train, x_test = x_train / 255.0, x_test / 255.0
# x_train_client, y_train_client = x_train, y_train # use all the dataset
#x_train_client, y_train_client = get_random_subset(x_train, y_train, DATA_FRACTION) # take a fraction of the dataset



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
# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        # generate a random id for the client using datetime and random # use some dashes in the date
        now = datetime.datetime.now() 
        self.id = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self, config):
        """Get parameters of client."""
        return self.model.get_weights()



    def fit(self, parameters, config):
        
        self.model.set_weights(parameters)

        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        server_round: int = config["server_round"]

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size,
            epochs,
            validation_split=0.0,
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)

        # write on file (with id client) how many examples were used for training
        #with open('log_client_'+self.id+'.txt', 'a') as f:
            #f.write('\nclient: ' + self.id + ' in round '+str(server_round)+' used ' + str(num_examples_train) + ' samples for training')
            #f.write('\nclient: ' + self.id + ' used ' + str(batch_size) + ' batch size')
            #f.write('\nclient: ' + self.id + ' used ' + str(epochs) + ' epochs')
            #f.write('\nclient: ' + self.id + ' used ' + str(config) + ' config')
            #f.write('\nclient: ' + self.id + ' used ' + str(history.history) + ' history')
            #f.write('\n----------------------------------')
            #f.write('\n\n')

        # print the same info on the terminal
        print('\nclient: ' + self.id + ' in round '+str(server_round)+' used ' + str(num_examples_train) + ' samples for training')


        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            #"val_loss": history.history["val_loss"][0],
            #"val_accuracy": history.history["val_accuracy"][0],
        }
        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}





def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument('--ip_address', type=str, default="0.0.0.0", help='IP address of the server.')
    

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    

    # model and data
    #model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # a model for mnist
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # a convolutional model for fashion mnist
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])        
    

    #(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data(); 
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train / 255.0; x_test = x_test / 255.0
    
    #DATA_FRACTION=0.1
    #(x_train, y_train), (x_test, y_test) = get_random_subset(x_train, y_train, DATA_FRACTION), (x_test, y_test)
    
    
    n = 10   # 
    #x_train, y_train = get_partition(x_train, y_train, i=0, num_partitions=n) # take a fraction of the dataset
    x_train, y_train = get_random_subset(x_train, y_train, fraction=1.0/n) # take a fraction of the dataset
    print('x_train.shape: ', x_train.shape)

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test)

    server_ip = args.ip_address
    port = "8080"
    server_address = server_ip+":"+port
    fl.client.start_numpy_client(
        server_address=server_address, 
        client=client)


def get_random_subset(x, y, fraction=0.1):
    indices = np.random.choice(len(x), int(len(x) * fraction), replace=False)
    return x[indices], y[indices]

def get_partition(x, y, i, num_partitions):
    size = len(x) // num_partitions
    start = i * size
    end = start + size
    return x[start:end], y[start:end]





if __name__ == "__main__":
    main()
