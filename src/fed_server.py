import flwr as fl
import argparse

# loggers output all messages to stdout in addition to log file
import logging
import sys
from datetime import datetime
from dataclasses import dataclass, field


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

@dataclass
class ServerState:
    num_clients_last_round: int = 0

# Initialize the server state
server_state = ServerState()

current_round_num_clients = 0


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        'server_round': server_round, # send the server round to the client
        "batch_size": 8,
        "local_epochs": 2,
        #"local_epochs": 2 if server_round < 2 else 5,
        "num_clients_last_round": server_state.num_clients_last_round,
        "num_clients_current_round": current_round_num_clients,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

class MyFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def aggregate_fit(self, rnd, results, failures):
        # Update the number of clients that successfully completed the fit round
        server_state.num_clients_last_round = len(results)
        return super().aggregate_fit(rnd, results, failures)

# Custom strategy to capture the number of clients selected for the current round
class MyStrategy(fl.server.strategy.FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        global current_round_num_clients
        print('overriding configure_fit')
        # Get the number of available clients for the current round
        current_round_num_clients = len(client_manager.all().values())
        return super().configure_fit(server_round, parameters, client_manager)
    
def start_flower_server(ip_address, port = "8080", rounds = 3, clients = 2):
    # Create the full server address
    server_address = ip_address+":"+port
    print("----> Server address: "+server_address, 'num_rounds: ', rounds)

    strategy = MyStrategy(
        # Fraction of clients used during training. In case min_fit_clients > fraction_fit * available_clients, min_fit_clients will still be sampled. Defaults to 1.0.
        fraction_fit=1, # 0.1,  
        #fraction_eval=0.1,
        min_fit_clients=2,       # Minimum number of clients used during training. Default 2. Always >= min_available_clients.
        min_available_clients=2, # Minimum number of total clients in the system. server will wait until at least 2 clients are connected.
        #eval_fn=None,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        #initial_parameters=None,
    )

    start_time = datetime.now()
    with open('log_server.txt', 'a') as f:
        f.write('\nserver started at: ' + str(start_time))

    server_state.num_clients_last_round = clients
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
