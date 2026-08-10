#!/bin/bash
# Stop (and auto-remove) the queryocc container.

if ! docker ps --format '{{.Names}}' | grep -qx queryocc; then
    echo "Container 'queryocc' is not running."
    exit 0
fi

docker stop queryocc
echo "Container 'queryocc' stopped."
