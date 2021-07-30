# Utility Functions and Classes
:label:`sec_utils`


This section contains the implementations of utility functions and classes used in this book.

```{.python .input}
import inspect
from d2l import mxnet as d2l
from IPython import display
```

```{.python .input  n=2}
#@tab pytorch
import inspect
from d2l import torch as d2l
from IPython import display
```

hyper parameters 

```{.python .input}
#@tab all
@d2l.add_to_class(d2l.HyperParameters)  #@save
def save_hyperparameters(self, ignore=[]):
    """Save function arguments into class attributes."""
    frame = inspect.currentframe().f_back
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k:v for k, v in local_vars.items()
                    if k not in set(ignore+['self'])}
    for k, v in self.hparams.items():
        setattr(self, k, v)
```

progress bar

```{.python .input  n=3}
#@tab all
@d2l.add_to_class(d2l.ProgressBoard)  #@save
def _update_lines(self):
    lines = {}
    for p in self.points:
        x = float(p[self.x])
        for k, v in p.items():
            if k == self.x: continue
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

@d2l.add_to_class(d2l.ProgressBoard)  #@save
def draw(self, points, every_n=1):
    assert self.x in points, 'must specify the x-axis value'
    self.points.append(points)
    if len(self.points) != every_n:
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
    if not self.xlabel: self.xlabel = self.x
    axes.set_xlabel(self.xlabel)
    axes.set_ylabel(self.ylabel)
    axes.legend(self.lines.keys())
    axes.grid()
    display.display(self.fig)
    display.clear_output(wait=True)
```

trainer 

```{.python .input}
@d2l.add_to_class(d2l.Trainer)  #@save
def fit(self, model, data):
    train_dataloader = data.train_dataloader()
    val_dataloader = data.val_dataloader()
    self.train_batch_idx = 0
    optim = model.configure_optimizers()
    try:
        model.board.xlim = [0, len(train_dataloader) * self.max_epochs]
    except:
        pass

    for epoch in range(self.max_epochs):
        for batch in train_dataloader:
            loss = model.training_step(batch, self.train_batch_idx)
            optim.zero_grad()
            with torch.no_grad():
                loss.backward()
                optim.step()
            self.train_batch_idx += 1
```
