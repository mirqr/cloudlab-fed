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
# gpu growth
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
# disable gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"




class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, id=0):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

        
        now = datetime.datetime.now() 
        self.id_name = now.strftime("%Y-%m-%d_%H-%M-%S") + '_' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
        self.id = id

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
        num_clients_last_round = config["num_clients_last_round"]
        num_clients_current_round = config["num_clients_current_round"]
        
        # client have all the data, select a fraction of it
        #x_train, y_train = get_random_subset(self.x_train, self.y_train, fraction=1.0/num_clients_current_round) # take a fraction of the dataset
        
        # NEVER use on aws (cannot assign increment id to clients). use get_random_subset 
        x_train, y_train = get_partition(self.x_train, self.y_train, i=self.id, num_partitions=num_clients_current_round) # take the i-th partition of the dataset given the number of partitions

        num_examples_train = len(x_train)
        # write on file (with id_name client) how many examples were used for training
        #with open('log_client_'+self.id_name+'.txt', 'a') as f:
            #f.write('\nclient: ' + self.id_name + ' in round '+str(server_round)+' used ' + str(num_examples_train) + ' samples for training')
            #f.write('\n----------------------------------')
            #f.write('\n\n')

        
        print('\nclient: ' + self.id_name + ' in round '+str(server_round)+' used ' + str(num_examples_train) + ' samples for training')
        print('num_clients_last_round: ', num_clients_last_round, 'num_clients_current_round: ', num_clients_current_round)


        history = self.model.fit(
            x_train,
            y_train,
            batch_size,
            epochs,
            validation_split=0.0, # no validation
        )

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()

        

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
    # id client (a number)
    parser.add_argument('--client_id', type=int, default="0", help='ID of the client.')
    

    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    
    # a convolutional model for fashion mnist
    # MAKE SURE TO USE THE SAME MODEL in the client and the server
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
    

    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train / 255.0; x_test = x_test / 255.0
    

    # Start Flower client
    client = CifarClient(model, x_train, y_train, x_test, y_test, id=args.client_id)

    server_ip = args.ip_address
    port = "8089"
    server_address = server_ip+":"+port
    fl.client.start_numpy_client(
        server_address=server_address, 
        client=client)


def get_random_subset(x, y, fraction=0.1):
    indices = np.random.choice(len(x), int(len(x) * fraction), replace=False)

    # do the same but with seed of np
    #indices = np.random.RandomState(seed=42).choice(len(x), int(len(x) * fraction), replace=False)

    return x[indices], y[indices]

def get_partition(x, y, i, num_partitions): # take the i-th partition of the dataset given the number of partitions
    size = len(x) // num_partitions
    start = i * size
    end = start + size
    # print take dataset from to
    print('----->take dataset from ', start, ' to ', end)
    return x[start:end], y[start:end]




if __name__ == "__main__":
    main()
