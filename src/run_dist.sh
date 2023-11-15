#!/bin/bash

while [ $# -gt 0 ]; do
    if [ "$1" == "-h" -o "$1" == "--help" ]; then
        echo "Usage: run_dist.sh [-h|--help] [-cph|--clients_per_host n] Client-IPs..."
        exit 0
    elif [ "$1" == "-cph" -o "$1" == "--clients-per-host" ]; then
        clients_per_host="$2"
        shift
    else
        if [ "$HOSTS" == "" ]; then
            HOSTS="$1"
        else
            HOSTS="$HOSTS $1"
        fi
    fi
    shift
done

if [ "$HOSTS" == "" ]; then
    echo "Error: need to define client HOSTS either as an environment variable, or as command-line arguments"
    exit 1
fi

clients_per_host=${clients_per_host:-2}
clients=$[ $(echo $HOSTS | tr " " "\n" | wc -l) * $clients_per_host ]

echo HOSTS=$HOSTS
echo clients_per_host=$clients_per_host

ip=$(route -n | grep '^0\.0\.0\.0' | head -1 | sed -e 's/.* \([a-z0-9]\+\)$/\1/')
rounds=5

for h in $HOSTS; do
    rsync -avz -e ssh $HOME/cloudlab-fed $h:
done

echo "Starting server with $clients clients..."
python3 fed_server.py --ip_address $ip --rounds $rounds --clients $clients  &
sleep 5  # Sleep for 3s to give the server enough time to start

# for in clients
for h in $HOSTS; do
    for i in `seq 0 $((clients_per_host-1))`; do
	echo "Starting client $i on host $h..."
	ssh $h "cd cloudlab-fed/src/; python3 fed_client.py --ip_address $ip" &
    done
done

# This will allow you to use CTRL+C to stop all background processes
# TODO: check stop also of remote clients
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

mv log_server.txt log_server_${clients}_${clients_per_host}.txt
