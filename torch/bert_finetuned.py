import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim as optim
import os
import argparse
import functools
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

def setup():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl")

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):


    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Load dataset
    dataset = load_dataset('imdb')
    train_dataset = dataset['train'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Ensure the DataLoader shuffles data and is aware of the distributed training setup
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    model.train()
    for epoch in range(3):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs = {k: v.to(rank) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(rank)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()



def fsdp_main():
    print("Starting FSDP")
    setup()

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    model.to(rank)
    model = FSDP(model)

    # Load dataset
    dataset = load_dataset('imdb')
    train_dataset = dataset['train'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Ensure the DataLoader shuffles data and is aware of the distributed training setup
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-5)

    # Training loop
    model.train()
    for epoch in range(3):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            inputs = {k: v.to(rank) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(rank)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Rank {rank}, Epoch {epoch}, Loss: {loss.item()}")

    cleanup()




    

if __name__ == '__main__':

    dist.init_process_group("nccl")
    rank = dist.get_rank()    
    world_size = dist.get_world_size()    
    train(rank, world_size=world_size)
    
