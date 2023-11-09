import flwr as fl
import argparse

# loggers output all messages to stdout in addition to log file
import logging
import sys
from datetime import datetime

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 4,
        "local_epochs": 2
        #"local_epochs": 2 if server_round < 2 else 5
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.

    Perform five local evaluation steps on each client (i.e., use five batches) during
    rounds one to three, then increase to ten local evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def start_flower_server(ip_address, port = "8080", rounds = 3):
    # Create the full server address
    server_address = ip_address+":"+port
    print("----> Server address: "+server_address, 'num_rounds: ', rounds)

    strategy = fl.server.strategy.FedAvg(
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
    return parser.parse_args()


    

def main():
    # Parse command line arguments
    args = parse_arguments()

    ip = args.ip_address
    rnds = args.rounds
    
    # Start the server
    start_flower_server(ip_address = ip, rounds=rnds)

if __name__ == "__main__":
    print("STARTTT SERVERRRR")
    main()
