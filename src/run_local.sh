#!/bin/bash


ip="127.0.0.1"

echo "Starting server"
python fed_server.py --ip_address $ip &
sleep 3  # Sleep for 3s to give the server enough time to start

for i in `seq 0 1`; do
    echo "Starting client $i"
    python fed_client.py --ip_address $ip &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

