#!/bin/bash


ip="127.0.0.1"
rounds=5
clients=5

echo "Starting server"
python3 fed_server.py --ip_address $ip --rounds $rounds --clients $clients  &
sleep 5  # Sleep for 3s to give the server enough time to start

# for in clients
for i in `seq 0 $((clients-1))`; do
    echo "Starting client $i"
    python3 fed_client.py --ip_address $ip --client_id $i &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

