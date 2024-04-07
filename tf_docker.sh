docker run --gpus all -it -w /tensorflow -v /data:/data -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" \
    tensorflow/tensorflow:devel-gpu bash