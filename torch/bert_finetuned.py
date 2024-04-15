import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):

    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    # Load dataset
    dataset = load_dataset('imdb')
    train_dataset = dataset['train'].map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=512), batched=True)
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Training loop
    model.train()
    for epoch in range(3):  # Loop over the dataset multiple times
        for i, batch in enumerate(train_loader, 0):
            inputs = {k: v.to(rank) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(rank)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f'Rank {rank}, Epoch {epoch}, Step {i}, Loss: {loss.item()}')

    cleanup()

if __name__ == '__main__':

    dist.init_process_group("nccl")
    rank = dist.get_rank()    
    world_size = dist.get_world_size()    
    train(rank, world_size=world_size)
    
