FROM tensorflow/tensorflow:devel-gpu
RUN mkdir data
RUN cd /data && \
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz && \
tar -xvf cifar-10-python.tar.gz && \
rm cifar-10-python.tar.gz
RUN virtualenv env
RUN source env/bin/activate && \
python -m pip install --upgrade pip && \
python -m pip install tensorflow==2.11.0 scipy pycuda GPUtil psutil
RUN git clone https://github.com/lancelotnd/ML_Benchmark.git
RUN cp ML_Benchmark/data/* /data/


