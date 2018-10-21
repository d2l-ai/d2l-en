# Softmax Regression from Scratch

In this section, we will try to manually implement Softmax regression.  First, we will import the packages or modules required for the implementation in this section.

```{.python .input}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, nd
```

## Retrieve and Read Data

We will use the Fashion-MNIST data set and set the batch size to 256.

```{.python .input}
batch_size = 256
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
```

## Initialize Model Parameters

Just as we did for linear regression, vectors will be used to represent each example. We know that each example input is an image with a height and a width of 28 pixels. The vector length of the model is $28 \times 28 = 784$ and each element of the vector corresponds to each pixel in the image. Since there are 10 categories of images, the single-layer neural network output layer generates 10 outputs. Therefore, the weight and bias parameters of Softmax regression will be matrices of $784 \times 10$ and $1 \times 10$, respectively.

```{.python .input  n=9}
num_inputs = 784
num_outputs = 10

W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
```

As before, we have to attach a gradient to the model parameters.

```{.python .input  n=10}
W.attach_grad()
b.attach_grad()
```

## Implement the Softmax Operation

Before actually define Softmax regression, we should talk about operates by dimension on a multidimensional NDArray. In the following example, we are given an NDArray matrix `X`. We can just sum the elements in the same column (`axis=0`) or the same row (`axis=1`), and retain the column and row dimensions in the result (`keepdims=True`).

```{.python .input  n=11}
X = nd.array([[1, 2, 3], [4, 5, 6]])
X.sum(axis=0, keepdims=True), X.sum(axis=1, keepdims=True)
```

Now we can go ahead and define the softmax operation from the previous section.  In the function below, the number of rows in matrix `X` is the number of examples, and the number of columns is the number of outputs. In order to express the probability that examples predict each output, the softmax operation will first perform an exponential operation on each element through the `exp` function, sum the elements from the same row for matrix `exp`, and then divide each element by that sum of the row.  In this way, the sum of elements from each row of the resulting matrix will be 1, not negative. Therefore, each row of the matrix is a valid probability distribution. In softmax operations, any element from a random row of the output matrix represents the predicted probability of an example on every output category.

```{.python .input  n=12}
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # The broadcast mechanism is applied here.
```

As you can see, for any random input, we turn each element into a non-negative number. 1 is the sum of each row.

```{.python .input  n=13}
X = nd.random.normal(shape=(2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(axis=1)
```

## Define the Model

With the softmax operation, we can define the softmax regression model discussed in the last section. We change each original image into a vector with length `num inputs` through the `reshape` function.

```{.python .input  n=14}
def net(X):
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)
```

## Define the Loss Function

In the last section, we introduced the cross-entropy loss function used by softmax regression. In order to get the label's predicted probability, we can use the `pick` function. In the example below, the variable `y_hat` is the predicted probability of 2 examples in 3 different categories, and variable `y` is the label category of those 2 examples.  Through the `pick` function, we get the label's predicted probability of 2 examples. In the ["Softmax regression"](softmax-regression.md) section, the label category's discrete value increases by 1 from 1. This mathematical expression is different in the code, where the increment of label category's discrete value starts from 0.

```{.python .input  n=15}
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y = nd.array([0, 2])
nd.pick(y_hat, y)
```

Here, we have an implementation of the cross-entropy loss function introduced in the [Softmax regression](softmax-regression.md) section.

```{.python .input  n=16}
def cross_entropy(y_hat, y):
    return - nd.pick(y_hat, y).log()
```

## Calculate the Classification Accuracy

Given a class of predicted probability distributions `y_hat`, we use the one with the highest predicted probability as the output category. If it is consistent with the actual category `y`, then this prediction is correct.  The classification accuracy rate is the ratio between the number of correct predictions and the total number of predictions made. 

The accuracy rate function `accuracy` is defined as follows. Here, `y_hat.argmax(axis=1)`returns the largest element index to matrix `y_hat`, the result has the same shape as variable `y`. We have mentioned in the [data operation](../chapter_prerequisite/ndarray.md) section that the conditional equality formula `(y_hat.argmax(axis=1) == y)` is an NDArray with a value of 0 (equal to false) or 1 (equal to true). Since the label type is an integer, we need to convert the variable `y` into a floating point number before deciding the equality condition.

```{.python .input  n=17}
# 
def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()
```

We will continue to use the variables `y_hat` and `y` defined in the `pick` function, as the predicted probability distribution and label, respectively. We can see that the first example's prediction category is 2 (the largest element of the row is 0.6 with an index of 2), which is inconsistent with the actual label, 0. The second example's prediction category is 2 (the largest element of the row is 0.5 with an index of 2), which is consistent with the actual label, 2. Therefore, the classification accuracy rate for these two examples is 0.5.

```{.python .input  n=18}
accuracy(y_hat, y)
```

Similarly, we can evaluate the accuracy rate for model `net` on data set `data_iter`.

```{.python .input  n=19}
#  The function will be gradually improved: the complete implementation will be
# discussed in the "Image Augmentation" section.
def evaluate_accuracy(data_iter, net):
    acc = 0
    for X, y in data_iter:
        acc += accuracy(net(X), y)
    return acc / len(data_iter)
```

Because we randomly initialized the `net` model, the accuracy rate of this stochastic model should be close to 0.1, the reciprocal of 10, which is the number of categories.

```{.python .input  n=20}
evaluate_accuracy(test_iter, net)
```

## To train a model

The implementation for training softmax regression is very similar to the implementation of linear regression discussed earlier. We still use the mini-batch stochastic gradient descent to optimize the loss function of the model. When training the model, the number of epochs, `num_epochs`, and learning rate `lr` are both adjustable hyper-parameters. By changing their values, we may be able to increase the classification accuracy of the model.

```{.python .input  n=21}
num_epochs, lr = 5, 0.1

# 
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum = 0
        train_acc_sum = 0
        for X, y in train_iter:
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y)
            l.backward()
            if trainer is None:
                gb.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # This will be illustrated in the next section.
            train_l_sum += l.mean().asscalar()
            train_acc_sum += accuracy(y_hat, y)
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / len(train_iter),
                 train_acc_sum / len(train_iter), test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs,
          batch_size, [W, b], lr)
```

## PREDICTION

Now that the training is complete, we can demonstrate how to classify the image. Given a series of images (third line image output), we will compare their actual labels (first line of text output) and the model predictions (second line of text output).

```{.python .input}
for X, y in test_iter:
    break

true_labels = gb.get_fashion_mnist_labels(y.asnumpy())
pred_labels = gb.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

gb.show_fashion_mnist(X[0:9], titles[0:9])
```

## Summary
 
* We can use softmax regression to carry out multi-category classification. You will find that softmax regression training is very similar to the training of linear regression: retrieve and read data, define models and loss functions, then train models using optimization algorithms. In fact, most deep learning models have a similar training procedure.

## exercise

* In this section, we directly implement the softmax function based on the mathematical definition of the softmax operation. What problems might this cause? (Hint: Try to calculate the size of $\exp(50)$. )
* The function `cross_entropy` in this section is implemented according to the definition of the cross-entropy loss function.  What could be the problem with this implementation? (Hint: Consider the domain of the logarithmic function. )
* What solutions you can think of to fix the two problems above?

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/741)

![](../img/qr_softmax-regression-scratch.svg)
