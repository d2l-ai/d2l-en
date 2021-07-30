# Utility Functions and Classes
:label:`sec_utils`


This section contains the implementations of utility functions and classes used in this book.

```{.python .input}
import inspect
from d2l import mxnet as d2l
from IPython import display
from mxnet import autograd, np, npx
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
import inspect
from d2l import torch as d2l
from IPython import display
```

```{.python .input}
#@tab tensorflow
from d2l import tensorflow as d2l
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

```{.python .input}
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
    self.train_batch_idx = 0
    optim = model.configure_optimizers()
    for epoch in range(self.max_epochs):
        for batch in train_dataloader:
            with autograd.record():
                loss = model.training_step(batch, self.train_batch_idx)
            loss.backward()
            if isinstance(optim, gluon.Trainer):
                optim.step(1)
            else:
                optim.step()
            self.train_batch_idx += 1
```

```{.python .input}
#@tab pytorch
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

```{.python .input}
#@tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit(self, model, data):
    train_dataloader = data.train_dataloader()
    self.train_batch_idx = 0
    optim = model.configure_optimizers()
    for epoch in range(self.max_epochs):
        for batch in train_dataloader:
            with tf.GradientTape() as tape:
                loss = model.training_step(batch, self.train_batch_idx)
            grads = tape.gradient(loss, model.trainable_variables)
            optim.apply_gradients(zip(grads, model.trainable_variables))
            self.train_batch_idx += 1
```

a bunch of functions that will be deprecated

```{.python .input}
#@tab mxnet
def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a Gluon data iterator."""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))

def sgd(params, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

```{.python .input}
#@tab pytorch

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = d2l.normal(0, 1, (num_examples, len(w)))
    y = d2l.matmul(X, w) + b
    y += d2l.normal(0, 0.01, y.shape)
    return X, d2l.reshape(y, (-1, 1))

def sgd(params, lr, batch_size):
    """Minibatch stochastic gradient descent."""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

```

```{.python .input}
#@tab tensorflow

def load_array(data_arrays, batch_size, is_train=True):  #@save
    """Construct a TensorFlow data iterator."""
    dataset = tf.data.Dataset.from_tensor_slices(data_arrays)
    if is_train:
        dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)
    return dataset

def synthetic_data(w, b, num_examples):  #@save
    """Generate y = Xw + b + noise."""
    X = tf.zeros((num_examples, w.shape[0]))
    X += tf.random.normal(shape=X.shape)
    y = tf.matmul(X, tf.reshape(w, (-1, 1))) + b
    y += tf.random.normal(shape=y.shape, stddev=0.01)
    y = tf.reshape(y, (-1, 1))
    return X, y


def sgd(params, grads, lr, batch_size):  #@save
    """Minibatch stochastic gradient descent."""
    for param, grad in zip(params, grads):
        param.assign_sub(lr * grad / batch_size)
```

```{.python .input}
#@tab all

def linreg(X, w, b):  #@save
    """The linear regression model."""
    return d2l.matmul(X, w) + b

def squared_loss(y_hat, y):  #@save
    """Squared loss."""
    return (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
```
