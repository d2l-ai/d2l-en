```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Synthetic Regression Data
:label:`sec_synthetic-regression-data`


Machine learning is all about extracting information from data.
So you might wonder, what could we possibly learn from synthetic data?
While we might not care intrinsically about the patterns 
that we ourselves baked into an artificial data generating model,
such datasets are nevertheless useful for didactic purposes,
helping us to evaluate the properties of our learning 
algorithms and to confirm that our implementations work as expected.
For example, if we create data for which the correct parameters are known *a priori*,
then we can verify that our model can in fact recover them.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx, gluon
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
import random
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
import random
```

## Generating the Dataset

For this example, we will work low-dimensional
for succinctness.
The following code snippet generates 1000 examples
with 2-dimensional features drawn 
from a standard normal distribution.
The resulting design matrix $\mathbf{X}$
belongs to $\mathbb{R}^{1000 \times 2}$. 
We generate each label by applying 
a *ground truth* linear function, 
corrupted them via additive noise $\epsilon$, 
drawn independently and identically for each example:

(**$$\mathbf{y}= \mathbf{X} \mathbf{w} + b + \mathbf\epsilon.$$**)

For convenience we assume that $\epsilon$ is drawn 
from a normal distribution with mean $\mu= 0$ 
and standard deviation $\sigma = 0.01$.
Note that for object-oriented design
we add the code to the `__init__` method of a subclass of `d2l.DataModule` (introduced in :numref:`oo-design-data`). 
It's good practice to allow setting any additional hyperparameters. 
We accomplish this with `save_hyperparameters()`. 
The `batch_size` will be determined later on.

```{.python .input}
%%tab all
class SyntheticRegressionData(d2l.DataModule):  #@save
    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, 
                 batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        if tab.selected('pytorch') or tab.selected('mxnet'):                
            self.X = d2l.randn(n, len(w))
            noise = d2l.randn(n, 1) * noise
        if tab.selected('tensorflow'):
            self.X = tf.random.normal((n, w.shape[0]))
            noise = tf.random.normal((n, 1)) * noise            
        self.y = d2l.matmul(self.X, d2l.reshape(w, (-1, 1))) + b + noise
```

Below, we set the true parameters to $\mathbf{w} = [2, -3.4]^\top$ and $b = 4.2$.
Later, we can check our estimated parameters against these *ground truth* values.

```{.python .input}
%%tab all
data = SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
```

[**Each row in `features` consists of a vector in $\mathbb{R}^2$ and each row in `labels` is a scalar.**] Let's have a look at the first entry.

```{.python .input}
%%tab all
print('features:', data.X[0],'\nlabel:', data.y[0])
```

## Reading the Dataset

Training machine learning models often requires multiple passes over a dataset, 
grabbing one minibatch of examples at a time. 
This data is then used to update the model. 
To illustrate how this works, we 
[**implement the `get_dataloader` function,**] 
registering it as a method in the `SyntheticRegressionData` class via `add_to_class` (introduced in :numref:`oo-design-utilities`).
It (**takes a batch size, a matrix of features,
and a vector of labels, and generates minibatches of size `batch_size`.**)
As such, each minibatch consists of a tuple of features and labels. 
Note that we need to be mindful of whether we're in training or validation mode: 
in the former, we will want to read the data in random order, 
whereas for the latter, being able to read data in a pre-defined order 
may be important for debugging purposes.

```{.python .input}
%%tab all
@d2l.add_to_class(SyntheticRegressionData)
def get_dataloader(self, train):
    if train:
        indices = list(range(0, self.num_train))
        # The examples are read in random order
        random.shuffle(indices)
    else:
        indices = list(range(self.num_train, self.num_train+self.num_val))
    for i in range(0, len(indices), self.batch_size):
        if tab.selected('mxnet') or tab.selected('pytorch'):
            batch_indices = d2l.tensor(indices[i: i+self.batch_size])
            yield self.X[batch_indices], self.y[batch_indices]
        if tab.selected('tensorflow'):
            j = tf.constant(indices[i : i+self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)
```

To build some intuition, let's inspect the first minibatch of
data. Each minibatch of features provides us with both its size and the dimensionality of input features.
Likewise, our minibatch of labels will have a matching shape given by `batch_size`.

```{.python .input}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

While seemingly innocuous, the invocation 
of `iter(data.train_dataloader())` 
illustrates the power of Python's object-oriented design. 
Note that we added a method to the `SyntheticRegressionData` class
*after* creating the `data` object. 
Nonetheless, the object benefits from 
the *ex post facto* addition of functionality to the class.

Throughout the iteration we obtain distinct minibatches
until the entire dataset has been exhausted (try this).
While the iteration implemented above is good for didactic purposes,
it is inefficient in ways that might get us in trouble on real problems.
For example, it requires that we load all the data in memory
and that we perform lots of random memory access.
The built-in iterators implemented in a deep learning framework
are considerably more efficient and they can deal
with sources such as data stored in files, 
data received via a stream, 
and data generated or processed on the fly. 
Next let's try to implement the same function using built-in iterators.

## Concise Implementation of the Data Loader

Rather than writing our own iterator,
we can [**call the existing API in a framework to load data.**]
As before, we need a dataset with features `X` and labels `y`. 
Beyond that, we set `batch_size` in the built-in data loader 
and let it take care of shuffling examples  efficiently.

```{.python .input}
%%tab all
@d2l.add_to_class(d2l.DataModule)  #@save
def get_tensorloader(self, tensors, train, indices=slice(0, None)):
    tensors = tuple(a[indices] for a in tensors)
    if tab.selected('mxnet'):
        dataset = gluon.data.ArrayDataset(*tensors)
        return gluon.data.DataLoader(dataset, self.batch_size,
                                     shuffle=train)
    if tab.selected('pytorch'):
        dataset = torch.utils.data.TensorDataset(*tensors)
        return torch.utils.data.DataLoader(dataset, self.batch_size,
                                           shuffle=train)
    if tab.selected('tensorflow'):
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return tf.data.Dataset.from_tensor_slices(tensors).shuffle(
            buffer_size=shuffle_buffer).batch(self.batch_size)

@d2l.add_to_class(SyntheticRegressionData)  #@save
def get_dataloader(self, train):
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader((self.X, self.y), train, i)
```

The new data loader behaves just as the previous one, except that it is more efficient and has some added functionality.

```{.python .input  n=4}
%%tab all
X, y = next(iter(data.train_dataloader()))
print('X shape:', X.shape, '\ny shape:', y.shape)
```

For instance, the data loader provided by the framework API 
supports the built-in `__len__` method, 
so we can query its length, 
i.e., the number of batches.

```{.python .input}
%%tab all
len(data.train_dataloader())
```

## Summary

Data loaders are a convenient way of abstracting out 
the process of loading and manipulating data. 
This way the same machine learning *algorithm* 
is capable of processing many different types and sources of data 
without the need for modification. 
One of the nice things about data loaders 
is that they can be composed. 
For instance, we might be loading images 
and then have a post-processing filter 
that crops them or modifies them otherwise. 
As such, data loaders can be used 
to describe an entire data processing pipeline. 

As for the model itself, the two-dimensional linear model 
is about as simple a model as we might encounter. 
It lets us test out the accuracy of regression models 
without worry about having insufficient amounts of data 
or an underdetermined system of equations. 
We will put this to good use in the next section.  


## Exercises

1. What will happen if the number of examples cannot be divided by the batch size. How to change this behavior by specifying a different argument by using framework's API?
1. What if we want to generate a huge dataset, where both the size of the parameter vector `w` and the number of examples `num_examples` are large? 
    1. What happens if we cannot hold all data in memory?
    1. How would you shuffle the data if data is held on disk? Your task is to design an *efficient* algorithm that does not require too many random reads or writes. Hint: [pseudorandom permutation generators](https://en.wikipedia.org/wiki/Pseudorandom_permutation) allow you to design a reshuffle without the need to store the permutation table explicitly :cite:`Naor.Reingold.1999`. 
1. Implement a data generator that produces new data on the fly, every time the iterator is called. 
1. How would you design a random data generator that generates *the same* data each time it's called?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6662)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6663)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6664)
:end_tab:
