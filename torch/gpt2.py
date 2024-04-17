import torch
from torch.utils.data import DataLoader
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from torch.distributed import init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import wrap
from torch.optim import Adam

# Initialization
init_process_group(backend='nccl')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

# Load and preprocess dataset
dataset = load_dataset('openwebtext', trust_remote_code=True)
dataset = dataset.map(tokenize_function,batched=True)
dataset.set_format(type='torch', columns=['input_ids'])

dataloader = DataLoader(dataset['train'], batch_size=4, shuffle=True)

# Setup model
config = GPT2Config.from_pretrained('gpt2')
model = GPT2LMHeadModel(config)
model = wrap(model)
model = FSDP(model)

# Setup optimizer
optimizer = Adam(model.parameters(), lr=0.0001)

# Training loop
model.train()
for epoch in range(3):
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch['input_ids'].to(torch.cuda.current_device())
        outputs = model(inputs, labels=inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss {loss.item()}")

# Save the model
if torch.distributed.get_rank() == 0:
    model.save_pretrained('my_gpt2')
