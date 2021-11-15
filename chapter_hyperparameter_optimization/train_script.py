import os
import torch
import torchvision

from torchvision import transforms
from torchvision import datasets
from torchvision import models


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = models.resnet18(pretrained=False, num_classes=10)
        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = torch.nn.functional.log_softmax(x, dim=1)
        return x

def get_CIFAR10(root):
    input_size = 32
    num_classes = 10
    normalize = [(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)]
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
        ]
    )
    local_path = os.path.join(root, 'CIFAR10')
    train_dataset = datasets.CIFAR10(
        local_path, train=True, transform=train_transform, download=True)

    valid_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(*normalize),
        ]
    )
    valid_dataset = datasets.CIFAR10(
        local_path, train=True, transform=valid_transform, download=True)

    return input_size, num_classes, train_dataset, valid_dataset



def train(model, train_loader, optimizer):
    model.train()
    total_loss = []
    for data, target in train_loader:
        if torch.cuda.is_available():
            data = data.cuda()
            target = target.cuda()
        optimizer.zero_grad()
        prediction = model(data)
        loss = torch.nn.functional.nll_loss(prediction, target)
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())
    avg_loss = sum(total_loss) / len(total_loss)

def valid(model, valid_loader):
    model.eval()
    loss = 0
    correct = 0
    for data, target in valid_loader:
        with torch.no_grad():
            if torch.cuda.is_available():
                data = data.cuda()
                target = target.cuda()
            prediction = model(data)
            prediction = prediction.max(1)[1]
            correct += prediction.eq(
                target.view_as(prediction)).sum().item()
    n_valid = len(valid_loader.sampler)
    percentage_correct = 100.0 * correct / n_valid
    return 1 - percentage_correct / 100


def objective(config, epochs=3, path='./'):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    weight_decay = config['weight_decay']
    
    
    input_size, num_classes, train_dataset, valid_dataset = get_CIFAR10(
            root=path)

    indices = list(range(train_dataset.data.shape[0]))
    train_idx, valid_idx = indices[:100], indices[-100:]
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               # shuffle=True,
                                               num_workers=0,
                                               sampler=train_sampler,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=128,
                                               # shuffle=False,
                                               num_workers=0,
                                               sampler=valid_sampler,
                                               pin_memory=True)

    model = Model()
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")
        # print(device)
        model = torch.nn.DataParallel(
            model, device_ids=[i for i in range(num_gpus)]).to(device)
    milestones = [25, 40]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum,
        weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=0.1)

    for epoch in range(epochs):
        train(model, train_loader, optimizer)
        validation_error = valid(model, valid_loader)
        scheduler.step()

    return validation_error
