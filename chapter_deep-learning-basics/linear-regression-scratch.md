# Linear regression implementation from scratch

After getting to know the background of linear regression, we are now ready for a hands-on implementation.  While a powerful deep learning framework minimizes repetitive work, relying on it too much to make things easy can make it hard to properly understand how deep learning works.    Because of this, this section will describe how to implement linear regression training using only NDArray and `autograd`.

Before we begin, let’s import the package or module required for this section’s experiment; where the matplotlib package will be used for plotting and will be set to embed in the GUI.

```{.python .input  n=1}
%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
```

## Generating Data Sets

By constructing a simple artificial training data set, we can visually compare the differences between the parameters we have learned and the actual model parameters.  Set the number of examples in the training data set as 1000 and the number of inputs (feature number) as 2.  Using the randomly generated batch example feature $\boldsymbol{X}\in \mathbb{R}^{1000 \times 2}$, we use the actual weight $\boldsymbol{w} = [2, -3.4]^\top$ and bias $b = 4.2$ of the linear regression model, as well as a random noise item $\epsilon$ to generate the tag

$$\boldsymbol{y}= \boldsymbol{X}\boldsymbol{w} + b + \epsilon,$$

The noise term $\epsilon$ obeys a normal distribution with a mean of 0 and a standard deviation of 0.01. Next, we’ll generate the data set.

```{.python .input  n=2}
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
```

Note that each `features` line is a vector with a length of 2, whereas each  `labels` line is a vector with a length of 1 (scalar).

```{.python .input  n=3}
features[0], labels[0]
```

By generating a scatter plot using the second `features[:, 1]` and `labels`, we can clearly observe the linear correlation between the two.

```{.python .input  n=4}
def use_svg_display():
    # Displayed in vector graphics.
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # Set the size of the graph to be plotted.
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);
```

The plotting function `plt` as well as the `use_svg_display` and `set_figsize` functions are defined in the `gluonbook` package. We will call `gluonbook.plt` directly for future plotting. To print the vector diagram and set its size, we only need to call  `gluonbook.set_figsize()`  before plotting, because `plt` is a global variable in the `gluonbook` package.


## Reading Data

We need to go through the entire data set and continuously examine mini-batches of data examples when training the model.   Here we define a function: its purpose is to return the features and tags of random `batch_size` (batch size) examples every time it’s called.

```{.python .input  n=5}
# This function has been saved in the gluonbook package for future use.
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # The examples are read at random, in no particular order. 
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # The “take” function will then return the corresponding element based on the indices. 
```

Let's read and print out the first small batch of data examples. Each batch’s feature shape, (10, 2), corresponds to the batch size and the number of inputs, with the label shape also representing the batch size.

```{.python .input  n=6}
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break
```

## Initialize Model Parameters

Weights are initialized to normal random numbers using a mean of 0 and a standard deviation of 0.01, with the deviation  initialized to zero.

```{.python .input  n=7}
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))
```

In the subsequent model training, we need to create the gradients for the parameters in order for them to iterate their values.

```{.python .input  n=8}
w.attach_grad()
b.attach_grad()
```

## Define the Model

The following is an example of the implementation of a vector calculation expression used in linear regression.   We use the `dot` function for the purpose of matrix multiplication.

```{.python .input  n=9}
def linreg(X, w, b):  # This function has been saved in the gluonbook package for future use.
    return nd.dot(X, w) + b
```

## Defining the Loss Function

We will use the squared loss function described in the previous section to define the linear regression loss. In the implementation, we need to transform the true value `y` into the predicted value’s shape `y_hat`.  The result returned by the following function will also be the same as the `y_hat` shape.

```{.python .input  n=10}
def squared_loss(y_hat, y):  # This function has been saved in the gluonbook package for future use.
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

## Defining the Optimization Algorithm

The following `sgd` function makes use of the mini-batch stochastic gradient descent algorithm outlined in the previous section. The algorithm optimizes the loss function by iterating over the model parameters. Here, the gradient calculated by the automatic differentiation module is the gradient sum of a batch of examples.   We divide it by the batch size to obtain the average.

```{.python .input  n=11}
def sgd(params, lr, batch_size):  # This function has been saved in the gluonbook package for future use.
    for param in params:
        param[:] = param - lr * param.grad / batch_size
```

## To train a model

In training, we will iterate the model parameters. In each iteration, the mini-batch stochastic gradient is calculated by first calling the inverse function `backward` depending on the currently read mini-batch data examples (feature `X` and label `y`), and then calling the optimization algorithm `sgd` to iterate the model parameters. Since we previously set the batch size `batch_size` to 10, the loss shape `l` for each small batch is (10, 1). Refer to the section ["Automatic Gradient"](../chapter_prerequisite/autograd.md). Since `l` is not a scalar variable, running `l.backward()` will add together the elements in `l` to obtain the new variable, and then calculate the variable model parameters’ gradient.

In an iterative cycle (epoch), we will iterate through the `data_iter` function once and use it for all the examples in the training data set (assuming the number of examples is divisible by the batch size). In this case, the number of epoch iterations `num_epochs` and the learning rate `lr` are both hyper-parameters and are set to 3 and 0.03, respectively. In practice, the majority of the hyper-parameters require constant adjustment through trial and error.  Although the model may become more efficient, the training time may be longer when the number of epoch iterations is raised.  The impact of the learning rate on the model will be covered in further detail later on in the “Optimization Algorithms” chapter.

```{.python .input  n=12}
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # The total epoch iterations required by the training model is calculated by num_epochs.
    # Assuming the number of examples can be divided by the batch size, all the examples in the training data set are used once in one epoch iteration. 
    # The features and tags of mini-batch examples are represented by X and Y respectively.  
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # Small batch loss in X and Y is represented by I.
        l.backward()  # The gradient of the model parameters is calculated by the small batch loss. 
        sgd([w, b], lr, batch_size)  # Use the small batch stochastic gradient descent to iterate the model parameters. 
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))
```

To generate the training set, we can compare the actual parameters used with the parameters we have learned after the training has been completed.  They should be very close to each other.

```{.python .input  n=13}
true_w, w
```

```{.python .input  n=14}
true_b, b
```

## Summary

* It can be observed that a model can be easily implemented using just the NDArray and `autograd`. In the following sections, we will be describing additional deep learning models based on what we have just learned and you will learn how to implement them using more concise codes (such as those in the next section).


## exercise

* Why is the `reshape` function needed in the `squared_loss` function?
* Experiment using different learning rates to find out how fast the loss function value drops. 
* If the number of examples cannot be divided by the batch size, what happens to the `data_iter` function’s behavior?


## Scan the QR code to access the [forum](https://discuss.gluon.ai/t/topic/743)

![](../img/qr_linear-regression-scratch.svg)
