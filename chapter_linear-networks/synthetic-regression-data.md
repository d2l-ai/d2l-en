# Synthetic Regression Data
:label:`synthetic_data`

In this section, we will construct an artificial dataset according to a linear model with additive noise. We will use it to train the linear model.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Generating the Dataset

We keep the data low-dimensional so we can visualize it easily.
In the following code snippet, we generate
1000 examples, each consisting of 2 features
drawn from a standard normal distribution.
Thus the matrix of features is
$\mathbf{X}\in \mathbb{R}^{1000 \times 2}$.

The synthetic labels are corrupted by additive noise $\epsilon$, drawn independently for each $\mathbf{x}$:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**)

For convenience we assume that $\epsilon$ arises from a normal distribution with mean of 0.
To make our problem easy, we will set its standard deviation to $\sigma = 0.01$.

We put the code into the `__init__` method of a subclass of `DataModule`, and allow the hyper-parameters configurable. The addition `batch_size` will be used later.

```{.python .input}
#@tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    def __init__(self, w, b, noise=0.01, num_examples=1000, 
                 batch_size=8):
        super().__init__()
        self.save_hyperparameters()        
        if d2l.USE_PYTORCH or d2l.USE_MXNET:                
            self.X = d2l.randn(num_examples, len(w))
            noise = d2l.randn(num_examples, 1) * noise
        if d2l.USE_TENSORFLOW:
            self.X = tf.random.normal((num_examples, w.shape[0]))
            noise = tf.random.normal((num_examples, 1)) * noise            
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

We set the true parameters to 
$\mathbf{w} = [2, -3.4]^\top$ and $b = 4.2$ to generate the data

```{.python .input}
#@tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Each row in `features` consists of a vector in $\mathbb{R}^2$ and each row in `labels` is a scalar.**]

```{.python .input}
#@tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Reading the Dataset from Scratch

Training models consists of
making multiple passes over the dataset,
grabbing one minibatch of examples at a time,
and using them to update our model.

In the following code, we [**define the `train_dataloader` function**] 
to demonstrate one possible implementation of this functionality.
It (**takes a batch size, a matrix of features,
and a vector of labels, yielding minibatches of size `batch_size`.**)
Each minibatch consists of a tuple of features and labels.

```{.python .input}
#@tab all
@d2l.add_to_class(SyntheticRegressionData)
def train_dataloader(self):
    indices = list(range(self.num_examples))
    # The examples are read at random, in no particular order
    random.shuffle(indices)
    for i in range(0, self.num_examples, self.batch_size):
        if d2l.USE_MXNET or d2l.USE_PYTORCH:
            batch_indices = d2l.tensor(
                indices[i: min(i + self.batch_size, self.num_examples)])
            yield self.X[batch_indices], self.y[batch_indices]
        if d2l.USE_TENSORFLOW:
            j = tf.constant(indices[
                i : min(i+self.batch_size, self.num_examples)])
            yield tf.gather(self.X, j), tf.gather(self.y, j)            
```

To build some intuition, let's inspect the first minibatch of
data. Each minibatch of features provides us both with its size and the dimensionality of input features.
Likewise, our minibatch of labels will have a matching shape given by `batch_size`.

```{.python .input}
#@tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

While seemingly quite innocuous, the invocation of `iter(data.train_dataloader())`
illustrates the power of Python's object oriented design. We ended up adding
a method to the `SyntheticRegressionData` class *after* creating the `data`
object. Nonetheless, the object benefits from the a-posteriori addition of
functionality to the class.

As we run the iteration, we obtain distinct minibatches
until the entire dataset has been exhausted (try this).
While the iteration implemented above is good for didactic purposes,
it is inefficient in ways that might get us in trouble on real problems.
For example, it requires that we load all the data in memory
and that we perform lots of random memory access.
The built-in iterators implemented in a deep learning framework
are considerably more efficient and they can deal
with sources such as data stored in files, data received via a stream, or data
generated/processed on the fly. Next let's try to implement the same function using built-in iterators.

## Concise Implementation of Loading Data

Rather than writing our own iterator,
we can [**call the existing API in a framework to load data.**] We first create a dataset with `X` and `y`, then specify the `batch_size`, and let the data loader to shuffle examples.

```{.python .input}
#@tab all
def tensorloader(tensors, batch_size, shuffle):  #@save
    if d2l.USE_MXNET:
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, batch_size, shuffle=shuffle)        
    if d2l.USE_PYTORCH:
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle)
    if d2l.USE_TENSORFLOW:
        shuffle_buffer = tensors[0].shape[0] if shuffle else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)        

@d2l.add_to_class(SyntheticRegressionData)  #@save
def train_dataloader(self):
    return tensorloader((self.X, self.y), self.batch_size, shuffle=True)
```

The usage is much the same way as before.

```{.python .input  n=4}
#@tab all
next(iter(data.train_dataloader()))
```

Additional benefit is that the framework API implemented the built-in `__len__` method, so we can get the length, i.e. the number of batches.

```{.python .input}
#@tab all
len(data.train_dataloader())
```

## Summary

- We can use a Python generator to yield a batch as a data loader.
- We can use framework's APIs to build a dataset and then load it by batches

## Exercises

1. What will happen if the number of examples cannot be divided by the batch size. How to change this behavior by specifying a different argument by using framework's API.
2. What if we want to generate a huge dataset, such as both the length of `w` and `num_examples` are very big numbers, that cannot hold in memory?
