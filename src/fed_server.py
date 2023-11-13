import logging
import sys
from datetime import datetime

import tensorflow as tf

from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import argparse

# loggers output all messages to stdout in addition to log file


from flwr.common import FitRes, Parameters, Scalar, FitIns
from flwr.server.client_proxy import ClientProxy


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


num_clients_last_round = 0
start_time = None

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    global start_time
    if server_round == 1:
        start_time = datetime.now()
        with open('log_server.txt', 'a') as f:
            f.write('\n start (first round): ' + str(start_time))
    config = {
        'server_round': server_round, # send the server round to the client
        "batch_size": 8,
        "local_epochs": 2,
        #"local_epochs": 2 if server_round < 2 else 5,
        "num_clients_last_round": num_clients_last_round,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def get_evaluate_fn(model):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    _ , (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    #(x_train, y_train), _ = tf.keras.datasets.fashion_mnist.load_data()

    # Use test data for evaluation
    x_val, y_val = x_test, y_test

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        model.set_weights(parameters)  # Update model with the latest parameters
        loss, accuracy = model.evaluate(x_val, y_val)
        
        #print(f"---------------------------round {server_round} - loss: {loss}, accuracy: {accuracy}")
        with open('log_server.txt', 'a') as f:
            f.write('\nserver_round: ' + str(server_round) + ' (server-side evalutation) loss: ' + str(loss) + ' accuracy: ' + str(accuracy))
        
        
        return loss, {"accuracy": accuracy}
    
    return evaluate



# Custom strategy to capture the number of clients selected for the current round
class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # override configure_fit to add the number of clients selected for the current round
    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
        # start FedAvg code 
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        # end FedAvg code 
        # add information about the number of clients selected for the current round in the config

        config['num_clients_current_round'] = len(clients)
        with open('log_server.txt', 'a') as f:
            f.write('\nserver_round: ' + str(server_round) + ' used ' + str(len(clients)) + ' clients for training')

        return [(client, fit_ins) for client in clients]

    
    def aggregate_fit(self, rnd, results, failures):
        # override aggregate_fit to capture the number of clients selected in the previous round (it is called after the round is finished)
        global num_clients_last_round
        num_clients_last_round = len(results)
        return super().aggregate_fit(rnd, results, failures)
    
def start_flower_server(ip_address, port = "8080", rounds = 3, clients = 2):
    # Create the full server address
    server_address = ip_address+":"+port
    print("----> Server address: "+server_address, 'num_rounds: ', rounds)

    # Load and compile model for
    # 1. server-side parameter initialization (not configured in this example)
    # 2. server-side parameter evaluation
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

    
    
    strategy = MyStrategy(
        # Fraction of clients used during training. In case min_fit_clients > fraction_fit * available_clients, min_fit_clients will still be sampled. Defaults to 1.0.
        fraction_fit=1, # 0.1,  
        #fraction_eval=0.1,
        min_fit_clients=clients,       # Minimum number of clients used during training. Default 2. Always >= min_available_clients.
        min_available_clients=clients, # Minimum number of total clients in the system. server will wait until at least 2 clients are connected.
        evaluate_fn=get_evaluate_fn(model), # server-side evaluation function
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        #initial_parameters=None,
    )


    
    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=rounds),
    )
    
    
    # write a file to indicate that the server has finished. and the time (date and time it finished)
    end_time = datetime.now()
    with open('log_server.txt', 'a') as f:
        f.write('\nserver finished at: ' + str(end_time))
        f.write('\nserver took: ' + str(end_time - start_time))
        f.write('\n----------------------------------')
        f.write('\n\n')

    # print server.log
    with open('log_server.txt', 'r') as f:
        print(f.read())

# Define a function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument('--ip_address', type=str, default="0.0.0.0", help='IP address of the server.')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds to train.')
    # number of clients
    parser.add_argument('--clients', type=int, default=2, help='Number of clients.')
    return parser.parse_args()



def main():
    # Parse command line arguments
    args = parse_arguments()

    ip = args.ip_address
    rnds = args.rounds
    clients = args.clients # default 2
    
    # Start the server
    start_flower_server(ip_address = ip, rounds=rnds, clients=clients)

if __name__ == "__main__":
    print("STARTTT SERVERRRR")
    main()
