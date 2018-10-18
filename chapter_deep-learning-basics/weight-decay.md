# Weight Decay

In the previous section, we observed the overfitting issue, which occurs when the training error of the model is much smaller than the error on the test set. While increasing the training data set may mitigate overfitting, obtaining additional training data is often costly. This section describes a common method for dealing with overfitting problems: weight decay.


## Technique

Weight decay is equivalent to $L_2$ norm regularization. Regularization is a common method of dealing with overfitting by adding a penalty term to the model loss function to reduce the size of the learned model parameter values. We will first describe $L_2$ norm regularization and then explain why it is also called weight decay.

$L_2$ norm regularization adds a $L_2$ norm penalty to the original loss function of the model to obtain the minimum function required for training. The $L_2$ norm penalty term refers to the product of the sum of the squares of each element of the model weight parameter and a positive constant. Here, we use the linear regression loss function in the [“Linear Regression”](linear-regression.md) section

$\ell(w_1, w_2, b) = \frac{1}{n}\sum_{i=1}^n \frac{1}{2}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right)^2$

as an example. Here, $w_1, w_2$ are the weight parameters, $b$ is the bias parameter, and the input of example $i$ is $x_1^{(i)}, x_2^{(i)}$. The label is $y^{(i)}$, and the example size is $n$. The weight parameter is represented by the vector $\boldsymbol{w}= [w_1, w_2]$, the new loss function with the $L_2$ norm penalty term is

$$\ell(w_1, w_2, b) + \frac{\lambda}{2n} \|\boldsymbol{w}\|^2,$$

其中超参数$\lambda > 0$。当权重参数均为0时，惩罚项最小。当$\lambda$较大时，惩罚项在损失函数中的比重较大，这通常会使学到的权重参数的元素较接近0。当$\lambda$设为0时，惩罚项完全不起作用。上式中$L_2$范数平方$\|\boldsymbol{w}\|^2$展开后得到$w_1^2 + w_2^2$。有了$L_2$范数惩罚项后，在小批量随机梯度下降中，我们将[“线性回归”](linear-regression.md)一节中权重$w_1$和$w_2$的迭代方式更改为

$$
\begin{aligned}
w_1 &\leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right)w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_1^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
w_2 &\leftarrow \left(1- \frac{\eta\lambda}{|\mathcal{B}|} \right)w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_2^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right).
\end{aligned}
$$

We can see that $L_2$ norm regularization multiplies the weights $w_1$ and $w_2$ by a number less than one and then subtracts the gradient without the penalty term. Therefore, $L_2$ norm regularization is also called weight decay. Weight decay adds a limit to the model that needs to be learned using a penalty for the model parameter with a larger absolute value. This can be an effective solution for overfitting. In real-world use, we sometimes add the sum of the squares of the bias elements to the penalty term.

## High-dimensional Linear Regression Experiment

Below, we will look at an overfitting problem using high-dimensional linear regression as an example, and try to cope with overfitting using weight decay. We assume the dimension of the data example feature is $p$. For any example with features $x_1, x_2, \ldots, and x_p$ in the training data set and test data set, we use the following linear function to generate the label for the example:

$y = 0.05 + \sum_{i = 1}^p 0.01x_i +  \epsilon,$

The noise term $\epsilon$ obeys a normal distribution with a mean of 0 and a standard deviation of 0.01. In order to observe overfitting more easily, we consider a high-dimensional linear regression problem, such as setting the dimension $p=200$. At the same time, we deliberately set a low number of examples in the training data, for example, 20.

```{.python .input  n=2}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

n_train, n_test, num_inputs = 20, 100, 200
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

features = nd.random.normal(shape=(n_train + n_test, num_inputs))
labels = nd.dot(features, true_w) + true_b
labels += nd.random.normal(scale=0.01, shape=labels.shape)
train_features, test_features = features[:n_train, :], features[n_train:, :]
train_labels, test_labels = labels[:n_train], labels[n_train:]
```

## Weight Decay Implementation from Scratch

Next, we will show how to implement weight decay from scratch. We achieve weight decay by adding an $L_2$ norm penalty term after the target function.

### Initialize Model Parameters

First, define a function that randomly initializes model parameters. This function attaches a gradient to each parameter.

```{.python .input  n=5}
def init_params():
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    return [w, b]
```

### Define $L_2$ Norm Penalty Term

The $L_2$ norm penalty term is defined below. Here, only the weight parameters of the model are used for the penalty.

```{.python .input  n=6}
def l2_penalty(w):
    return (w**2).sum() / 2
```

### Define Training and Testing

The following defines how to train and test the model separately on the training data set and the test data set. Unlike the previous sections, here, the $L_2$ norm penalty term is added when calculating the final loss function.

```{.python .input  n=7}
batch_size, num_epochs, lr = 1, 100, 0.003
net, loss = gb.linreg, gb.squared_loss
train_iter = gdata.DataLoader(gdata.ArrayDataset(
    train_features, train_labels), batch_size, shuffle=True)

def fit_and_plot(lambd):
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                # The L2 norm penalty term has been added.
                l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            l.backward()
            gb.sgd([w, b], lr, batch_size)
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    gb.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', w.norm().asscalar())
```

### Observe Overfitting

Next, we will train and test the high-dimensional linear regression model. When `lambd` is set to 0, we do not use weight decay. As a result, the training error is much smaller than the error on the test set. This is a typical case of overfitting.

```{.python .input  n=8}
fit_and_plot(lambd=0)
```

### Use Weight Decay

Next, we use weight decay. As we can see, even though the training error has increased, the error on the test set has decreased. The overfitting issue has been mitigated to some extent. In addition, the $L_2$ norm of the weight parameter is smaller than when weight decay is not used: the weight parameter at this time is closer to zero.

```{.python .input  n=9}
fit_and_plot(lambd=3)
```

## Weight Decay Gluon Implementation

Here, we directly specify the weight decay hyper-parameter through the `wd` parameter when constructing the `Trainer` instance. By default, Gluon decays weight and bias simultaneously. We can construct `Trainer` instances for weight and bias respectively to only decay the weight.

```{.python .input}
def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # The weight parameter has been decayed. Weight names generally end with "weight".
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                              {'learning_rate': lr, 'wd': wd})
    # The bias parameter has not decayed. Bias names generally end with "bias".
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                              {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # Call the step function on each of the two Trainer instances to update the weight and bias separately.
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    gb.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())
```

Similar to the experimental phenomena in the "Weight Decay Starting from Scratch" section, the use of weight decay can mitigate the overfitting problem to some extent.

```{.python .input}
fit_and_plot_gluon(0)
```

```{.python .input}
fit_and_plot_gluon(3)
```

## Summary

* Regularization is a common method for dealing with overfitting. It adds a penalty term to the model loss function to reduce the size of the learned model parameter values.
* Weight decay is equivalent to $L_2$ norm regularization, which usually makes the elements of the learned weight parameter closer to zero.
* Weight decay can be specified by Gluon's `wd` hyper-parameter.
* We can define multiple `Trainer` instances to use different iteration methods for different model parameters.

## exercise

* Review the relationship between training error and generalization error. In addition to weight decay, increased training, and the use of a model of suitable complexity, what other ways can you think of to deal with overfitting?
* If you understand Bayesian statistics, which important concept in Bayesian statistics do you think weight decay corresponds to?
* Adjust the weight decay hyper-parameter in the experiment and observe and analyze the experimental results.

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/984)

![](../img/qr_weight-decay.svg)
