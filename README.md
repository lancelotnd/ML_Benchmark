# ML_Benchmark


### To build the container from the Dockerfile
```sh
docker build -t mlbench:latest .
```
Should you need to delete the image to recreate it you can find its id with `docker images` and then `docker rmi -f 
<image_id>`


### To run the container 

```sh
docker run --gpus all -it mlbench:latest bash
```

