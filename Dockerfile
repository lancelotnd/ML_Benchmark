FROM tensorflow/tensorflow:devel-gpu
RUN virtualenv env
RUN source env/bin/activate
RUN python -m pip install --upgrade pip
RUN python -m pip install tensorflow==2.11.0 scipy
RUN git clone https://github.com/lancelotnd/ML_Benchmark.git


