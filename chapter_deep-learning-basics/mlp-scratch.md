# Implementation of the Multilayer Perceptron starting from scratch

We have already learned the multilayer perceptron principle from the previous section.  Let’s work through the following together in order to implement a multilayer perceptron.  First, import the required packages or modules.

```{.python .input  n=9}
%matplotlib inline
import gluonbook as gb
from mxnet import nd
from mxnet.gluon import loss as gloss
```

## Retrieve and read the data

We continue to use the Fashion-MNIST data set. We will use the Multilayer Perceptron for image classification

```{.python .input  n=2}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## Defining Model Parameters

In the section titled “Implementation of Softmax Regression Starting From Scratch”, we have already stated the image shape in the  []()Fashion-MNIST data set is $28 \times 28$, and the number of categories is 10. In this section, we still use a vector length of $28 \times 28 = 784$ to represent each image. Therefore, the number of inputs is 784 and the number of outputs is 10. In this experiment, we set the number of hyper-parameter hidden units at 256. 256

```{.python .input  n=3}
num_inputs, num_outputs, num_hiddens = 784, 10, 256

W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)
params = [W1, b1, W2, b2]

for param in params:
    param.attach_grad()
```

## Defining Activation Function

Here, we use the underlying `maximum` function to implement the ReLU process, instead of instructing `ReLU` to directly function.

```{.python .input  n=4}
def relu(X):
    return nd.maximum(X, 0)
```

## Define the model

As softmax regression, with a`reshape` function, we changed each original image to a length vector of  `num_inputs`. We then implemented a Multilayer Perceptron calculation expression in the previous section.

```{.python .input  n=5}
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(nd.dot(X, W1) + b1)
    return nd.dot(H, W2) + b2
```

## Define loss function

For better numerical stability, we use Gluon's functions, including softmax calculation and cross-entropy loss calculation.

```{.python .input  n=6}
loss = gloss.SoftmaxCrossEntropyLoss()
```

## To train a model

Steps for training the Multilayer Perceptron are no different from Softmax Regression training steps.  In the `gluonbook` package, we directly call the `train_ch3` function, whose implementation was introduced in [ "Implementation of Softmax Regression Starting from Scratch" ](softmax-regression-scratch.md) section.  Here we set the number of hyper-parameter epochs to 5 and the learning rate to 0.5.

```{.python .input  n=7}
num_epochs, lr = 5, 0.5
gb.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
             params, lr)
```

## Summary

* We can implement a simple multilayer perceptron by manually defining the model and its parameters. 
* When there is a large number of layers in the Multilayer Perceptron, the implementation of this section will be more complicated, such as the ability to define model parameters.  

## exercise

* Change the value of the hyper-parameter `num_hiddens` in order to see the result effects. 
* Try adding a new hidden layer to see how it affects the results.  

## Scan the QR code to access the  [  ](> forum. 

![](../img/qr_mlp-scratch.svg)
