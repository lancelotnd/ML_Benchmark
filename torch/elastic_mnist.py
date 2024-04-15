import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.net(x)

def demo_mnist_ddp(rank, world_size):
    print(f"Start running DDP example on rank {rank}.")

    # Set up the device
    device_id = rank % torch.cuda.device_count()
    torch.cuda.set_device(device_id)

    # MNIST Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = DataLoader(dataset, batch_size=64, sampler=sampler)

    # Model, loss, and optimizer
    model = MNISTModel().to(device_id)
    ddp_model = DDP(model, device_ids=[device_id])
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Training loop
    ddp_model.train()
    for epoch in range(10):  
        running_loss = 0.0
        for i, data in enumerate(data_loader,0):
            data, target = data[0].to(device_id), data[1].to(device_id)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    dist.destroy_process_group()

if __name__ == "__main__":
    dist.init_process_group("nccl")
    rank = dist.get_rank()    
    world_size = 2
    demo_mnist_ddp(rank, world_size)
