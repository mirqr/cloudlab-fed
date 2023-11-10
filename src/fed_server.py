from typing import Dict, List, Optional, Tuple, Union
import flwr as fl
import argparse

# loggers output all messages to stdout in addition to log file
import logging
import sys
from datetime import datetime

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



# Custom strategy to capture the number of clients selected for the current round
class MyStrategy(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def configure_fit(self, server_round, parameters, client_manager):
        """Configure the next round of training."""
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
        
        # add information about the number of clients selected for the current round in the config

        config['num_clients_current_round'] = len(clients)
        # write on file (with id client) how many clients were selected for the current round
        with open('log_server.txt', 'a') as f:
            f.write('\nserver_round: ' + str(server_round) + ' used ' + str(len(clients)) + ' clients for training')
        #print ('--------fffff----clients selected', len(clients))
        #print ('--------fffff----clients available', len(client_manager.all().values()))
        # Return client/config pairs
        return [(client, fit_ins) for client in clients]


        
    
    def aggregate_fit(self, rnd, results, failures):
        # Update the number of clients that successfully completed the fit round
        print('overriding aggregate_fit')
        global num_clients_last_round
        num_clients_last_round = len(results)
        return super().aggregate_fit(rnd, results, failures)
    
def start_flower_server(ip_address, port = "8080", rounds = 3, clients = 2):
    # Create the full server address
    server_address = ip_address+":"+port
    print("----> Server address: "+server_address, 'num_rounds: ', rounds)

    strategy = MyStrategy(
        # Fraction of clients used during training. In case min_fit_clients > fraction_fit * available_clients, min_fit_clients will still be sampled. Defaults to 1.0.
        fraction_fit=1, # 0.1,  
        #fraction_eval=0.1,
        min_fit_clients=clients,       # Minimum number of clients used during training. Default 2. Always >= min_available_clients.
        min_available_clients=clients, # Minimum number of total clients in the system. server will wait until at least 2 clients are connected.
        #eval_fn=None,
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
