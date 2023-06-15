import flwr as fl
import argparse

# Define a function to parse command line arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Flower Server")
    parser.add_argument('--ip_address', type=str, default="0.0.0.0", help='IP address of the server.')
    return parser.parse_args()

def start_flower_server(ip_address, port = "8080"):
    # Create the full server address
    server_address = ip_address+":"+port
    
    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=3),
    )

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Start the server
    start_flower_server(args.ip_address)

if __name__ == "__main__":
    main()
