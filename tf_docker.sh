docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:devel-gpu bash