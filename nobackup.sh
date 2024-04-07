#to add the cuda drivers to path when cuda is installed in /nobackup like in bk-gpu-1
#Currently does not work
export PATH=/nobackup/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/nobackup/usr/local/cuda/lib64:$LD_LIBRARY_PATH
