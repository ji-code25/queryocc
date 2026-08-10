#!/bin/bash
# Open a shell inside the running queryocc container (start it first with containers/start_docker.sh).

if ! docker ps --format '{{.Names}}' | grep -qx queryocc; then
    echo "Container 'queryocc' is not running. Start it with: ./containers/start_docker.sh"
    exit 1
fi

docker exec -it -w /queryocc queryocc /bin/bash -l
