import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank,size):
    """ Blocking point to point communication """
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        dist.send(tensor=tensor, dst=1)
    else:
        dist.recv(tensor=tensor, src=0)
    print("Rank", rank, 'has data', tensor[0])


def init_process(rank, size, fn, backend='gloo'):
    """ Init the dist env """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,size)


if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank,size,run))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()