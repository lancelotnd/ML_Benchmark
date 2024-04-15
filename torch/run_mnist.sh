#!/bin/bash

# Script to run elastic_mnist.py with varying number of processes per node
# Redirect output to a file while also displaying it

for i in {1..6}
do
    echo "Running with nproc_per_node=$i"
    time torchrun --nnodes=1 --nproc_per_node=$i --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29400 elastic_mnist.py 2>&1 | tee -a run_mnist_logs_$i.txt
    echo "Completed run with nproc_per_node=$i"
done
