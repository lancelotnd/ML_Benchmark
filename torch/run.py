import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


"""Blocking point-to-point communication."""
def run_blocking_p2p(rank, size):
    tensor = torch.zeros(1)
    if rank == 0:
        tensor += 1
        # Send the tensor to process 1
        dist.send(tensor=tensor, dst=1)
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
    print('Rank ', rank, ' has data ', tensor[0])


""" Non Blocking point to point communication """
def run_non_blocking(rank,size):
    tensor = torch.zeros(1)
    req = None
    if rank == 0:
        tensor += 1
        # Send to process one
        req = dist.isend(tensor=tensor, dst=1)
    else:
        req = dist.irecv(tensor=tensor, src=0)
    req.wait()
    print("Rank", rank, 'has data', tensor[0])


""" All-Reduce example."""
def run_all_reduce(rank,size):
    """ Simple collective communication. """
    group = dist.new_group([0, 1])
    tensor = torch.ones(1)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    print('Rank ', rank, ' has data ', tensor[0])

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
        p = mp.Process(target=init_process, args=(rank,size,run_all_reduce))
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()