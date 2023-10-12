import flwr as fl
import argparse

# loggers output all messages to stdout in addition to log file
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

# Define a function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument('--ip_address', type=str, default="0.0.0.0", help='IP address of the server.')
    parser.add_argument('--rounds', type=int, default=3, help='Number of rounds to train.')
    return parser.parse_args()

def start_flower_server(ip_address, port = "8080", rounds = 3):
    # Create the full server address
    server_address = ip_address+":"+port
    
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.1,
        #fraction_eval=0.1,
        min_fit_clients=2,
        min_available_clients=2,
        #eval_fn=None,
        #on_fit_config_fn=None,
        #on_evaluate_config_fn=None,
        #initial_parameters=None,
    )

    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=rounds),
    )

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
