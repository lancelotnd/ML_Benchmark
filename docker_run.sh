#! /bin/bash
docker build -t mlbench:latest .
docker run --gpus all -it mlbench:latest bash
