# API

We provide three base classes, inspired by pytorch lightning:
- `DataModule` to download data and return train, val data loaders
- `Module` contains the neural network, loss, and optimization method
- `Trainer` provide a fit function to train the model

The goal is to improve reusablity. For example, all chapter can use `trainer = d2l.Trainer(...); trainer.fit(model, data)` to train the model. No need to reimplement everywhere. Though we may still show how to implement the Trainer for GPU, multi-GPU, with hidden state, ... but every chapter calls `d2l.Trainer`, instead of `train_ch3`.

Some utility functions, can move to appendix.


```{.python .input}
#@tab all
# a place holder
```

```{.python .input}
#@tab pytorch

import inspect
from d2l import torch as d2l
from IPython import display

class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self'])}
        for k, v in self.hparams.items():
            setattr(self, k, v)

class Animator(HyperParameters):
    """For plotting data in animation."""
    def __init__(self, xlabel=None, ylabel=None, legend=[], xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 ls=[], colors=[], fig=None, axes=None,
                 figsize=(3.5, 2.5), draw_interval=5):
        if not ls:
            ls = ['-', '--', '-.', ':']*3
        if not colors:
            colors = [f'C{i}' for i in range(10)]
        self.save_hyperparameters()
        self.points = []
        self.lines = {}

    def _update_lines(self):
        lines = {}
        for p in self.points:
            x = float(p['x'])
            for k, v in p.items():
                if k == 'x': continue
                if k not in lines:
                    lines[k] = ([], [])
                lines[k][0].append(x)
                lines[k][1].append(float(v))
        for k, v in lines.items():
            if k not in self.lines:
                self.lines[k] = ([], [])
            self.lines[k][0].append(v[0][-1])
            self.lines[k][1].append(sum(v[1])/len(v[1]))
        self.points = []

    def draw(self, points, avg_n=1):
        assert 'x' in points, 'must specify the x-axis value'
        self.points.append(points)
        if len(self.points) != avg_n:
            return
        self._update_lines()
        d2l.use_svg_display()
        if self.fig is None:
            self.fig = d2l.plt.figure(figsize=self.figsize)
        for (k, v), ls, color in zip(self.lines.items(), self.ls, self.colors):
            d2l.plt.plot(v[0], v[1], linestyle=ls, color=color, label=k)
        axes = self.axes if self.axes else d2l.plt.gca()
        if self.xlim: axes.set_xlim(self.xlim)
        if self.ylim: axes.set_ylim(self.ylim)
        axes.legend(self.lines.keys())
        display.display(self.fig)
        display.clear_output(wait=True)

```

```{.python .input}
#@tab pytorch

draw = Animator()
draw.draw({'x':1, 'y':2})
draw.draw({'x':2, 'y':4})
draw.draw({'x':3, 'y':4})
draw.draw({'x':1, 'w':3})
draw.draw({'x':2, 'w':5})


```

##  Data API

```{.python .input}
#@tab pytorch
class DataModule(HyperParameters):
    def __init__(self, batch_size=1, train_transforms=None, val_transforms=None,
                 num_workers=4, root='../data', ):
        self.save_hyperparameters()

    def prepare_data(self):
        """Downloads and prepare the data."""

    def train_dataloader(self):
        """Returns the dataloader for the training dataset."""
        raise NotImplemented()

    def val_dataloader(self):
        """Returns the dataloader for the validation dataset."""
```

Example: the Image Classification Dataset

```{.python .input}
#@tab pytorch

from torchvision import transforms
import torchvision
import torch
from torch import nn


class FashionMNIST(DataModule):
    def __init__(self, batch_size, resize=(28, 28)):
        trans = transforms.Compose([transforms.Resize(resize),
                                    transforms.ToTensor()])
        super().__init__(batch_size,
                         train_transforms=trans,
                         val_transforms=trans)

    def prepare_data(self):
        self.mnist_train = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=self.train_transforms,
            download=True)
        self.mnist_val = torchvision.datasets.FashionMNIST(
            root=self.root, train=True, transform=self.val_transforms,
            download=False)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_train, self.batch_size, shuffle=True,
            num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.mnist_val, self.batch_size,
            num_workers=self.num_workers,)

data = FashionMNIST(batch_size=64)
data.prepare_data()
X, y = next(iter(data.train_dataloader()))
X.shape, y.shape
```

## Model API

```{.python .input}
#@tab pytorch
class Module(nn.Module, HyperParameters):
    def __init__(self):
        """"""
        super().__init__()
        self.board = Animator()
        self.board.xlabel = 'steps'

    def training_step(self, batch, batch_idx):
        raise NotImplemented()

    def validaton_step():
        """"""

    def configure_optimizers(self):
        raise NotImplemented()



```

Implement Softmax concise

```{.python .input}
#@tab pytorch

def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum()) / float(y.numel())

class SoftmaxRegression(Module):
    def __init__(self, input_size, output_size, lr):
        super().__init__()
        self.save_hyperparameters()
        self.linear = nn.Linear(self.input_size, self.output_size)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        return self.linear(X.reshape(X.shape[0], -1))

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        l = self.loss(y_hat, y)
        self.board.draw(
            {'x':batch_idx,  'train_loss':float(l),
             'train_acc':accuracy(y_hat, y)}, avg_n=100)
        return l

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

model = SoftmaxRegression(784, 10, 0.1)
model.forward(torch.randn(2, 1, 28, 28)).shape
```

Implement softmax from scratch

```{.python .input}
#@tab pytorch
class SGD:
    def __init__(self, params, lr):
        self.params, self.lr = params, lr

    def step(self, batch_size):
        for param in self.params:
            param -= self.lr * param.grad / batch_size

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()

def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition

def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)), y]).mean()

class SoftmaxRegressionScratch(Module):
    def __init__(self, input_size, output_size, lr):
        super().__init__()
        self.save_hyperparameters()
        self.W = torch.normal(0, 0.01, size=(input_size, output_size), requires_grad=True)
        self.b = torch.zeros(output_size, requires_grad=True)

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        return softmax(X.reshape(X.shape[0], -1) @ self.W + self.b)

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        l = cross_entropy(y_hat, y)
        self.board.draw(
            {'x':batch_idx,  'train_loss':float(l),
             'train_acc':accuracy(y_hat, y)}, avg_n=100)
        return l

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), self.lr)

scratch_model = SoftmaxRegressionScratch(784, 10, 0.1)
scratch_model.forward(torch.randn(2, 1, 28, 28)).shape
```

## Training API

```{.python .input}
#@tab pytorch

class Trainer(HyperParameters):
    def __init__(self, gpus=0, num_epochs=1):
        """"""
        self.save_hyperparameters()

    def fit(self, model, data_module):
        data_module.prepare_data()
        train_dataloader = data_module.train_dataloader()
        val_dataloader = data_module.val_dataloader()
        batch_idx = 0
        optim = model.configure_optimizers()
        model.board.xlim = [0, len(train_dataloader)*self.num_epochs]
        for epoch in range(self.num_epochs):
            for batch in train_dataloader:
                loss = model.training_step(batch, batch_idx)
                optim.zero_grad()
                with torch.no_grad():
                    loss.backward()
                    optim.step()
                batch_idx += 1
```

```{.python .input}
#@tab pytorch
trainer = Trainer(num_epochs=10)
trainer.fit(model, data)
```

```{.python .input}
#@tab pytorch
trainer.fit(scratch_model, data)
```
