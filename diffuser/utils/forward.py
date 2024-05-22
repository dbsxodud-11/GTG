import sys
import os
import argparse

from tqdm import tqdm
from contextlib import contextmanager, redirect_stderr, redirect_stdout

@contextmanager
def suppress_output():
    """
        A context manager that redirects stdout and stderr to devnull
        https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

with suppress_output():
    import design_bench

    from design_bench.datasets.discrete.tf_bind_8_dataset import TFBind8Dataset
    from design_bench.datasets.discrete.tf_bind_10_dataset import TFBind10Dataset
    # from design_bench.datasets.discrete.cifar_nas_dataset import CIFARNASDataset

    from design_bench.datasets.continuous.ant_morphology_dataset import AntMorphologyDataset
    from design_bench.datasets.continuous.dkitty_morphology_dataset import DKittyMorphologyDataset
    from design_bench.datasets.continuous.superconductor_dataset import SuperconductorDataset

import torch
import torch.nn as nn
import pickle as pkl
import numpy as np

from torch.nn import functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

TASKNAME2TASK = {
        'dkitty': 'DKittyMorphology-Exact-v0',
        'ant': 'AntMorphology-Exact-v0',
        'tf-bind-8': 'TFBind8-Exact-v0',
        'tf-bind-10': 'TFBind10-Exact-v0',
        'superconductor': 'Superconductor-RandomForest-v0',
        # 'nas': 'CIFARNAS-Exact-v0',
        }

TASKNAME2CLASS = {
        'dkitty': DKittyMorphologyDataset,
        'ant': AntMorphologyDataset,
        'tf-bind-8': TFBind8Dataset,
        'tf-bind-10': TFBind10Dataset,
        'superconductor': SuperconductorDataset,
        # 'nas': CIFARNASDataset,
        }

class CustomDataset(Dataset):
    def __init__(self, task, x, y):
        self.task = task
        self.x = x
        self.y = y
    
    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.float32)

class ForwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)

        return x 

class ProbabilisticForwardModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.layers = []
        self.layer1 = nn.Linear(self.input_size, self.hidden_size)
        self.layer2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer3 = nn.Linear(self.hidden_size, 2)

    def forward(self, x):
        x = self.layer1(x)
        x = F.leaky_relu(x)
        x = self.layer2(x)
        x = F.leaky_relu(x)
        x = self.layer3(x)

        return x

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for batch_idx, (data, target) in pbar:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # loss = F.mse_loss(output, target, reduction='mean')
        loss = F.gaussian_nll_loss(output[:,0], target, var=torch.exp(output[:,1]), reduction='mean')
        loss.backward()
        optimizer.step()
        pbar.set_description(f"Train epoch {epoch} batch_idx {batch_idx}: train loss {loss.item():.5f}")

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.gaussian_nll_loss(output[:,0], target, var=torch.exp(output[:,1]), reduction='mean').item()
            # test_loss += F.mse_loss(output, target, reduction='mean').item()  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}'.format(test_loss))

    return test_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--task', type=str, default='ant')
    parser.add_argument('--checkpoint', type=str, default='default')

    args = parser.parse_args()

    task = design_bench.make(TASKNAME2TASK[args.task])
    fully_observed_task = TASKNAME2CLASS[args.task]()
    dim = task.x.shape[-1]

    # model = ProbabilisticForwardModel(input_size=dim, num_layers=1, hidden_size=args.hidden_size)
    model = ForwardModel(input_size=dim, num_layers=1, hidden_size=args.hidden_size)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = torch.nn.DataParallel(model).to(device)

    print(args.train)
    if (not args.train):
        raise NotImplementedError()

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}

    betas = (0.9, 0.95)
    train_size = int(9 * task.x.shape[0] // 10)
    print(train_size)
    normalised_y = (task.y - fully_observed_task.y.min()) / (fully_observed_task.y.max() - fully_observed_task.y.min())

    train_dataset = CustomDataset(task=task, x=task.x[:train_size], y=normalised_y[:train_size])
    test_dataset = CustomDataset(task=task, x=task.x[train_size:], y=normalised_y[train_size:])

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=betas)

    best_loss = float('inf')
    for epoch in range(args.epochs):
        train(args, model, device, train_loader, optimizer, epoch+1)
        test_loss = test(model, device, test_loader)
        if test_loss < best_loss:
            # save checkpoint
            torch.save(model.state_dict(), "forward_checkpoints/{}_best".format(args.checkpoint))
