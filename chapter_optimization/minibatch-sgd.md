# Mini-Batch Stochastic Gradient Descent

In each iteration, the gradient descent uses the entire training data set to compute the gradient, so it is sometimes referred to as batch gradient descent. Stochastic gradient descent (SGD) only randomly select one example in each iteration to compute the gradient. Just like in the previous chapters, we can perform random uniform sampling for each iteration to form a mini-batch and then use this mini-batch to compute the gradient. Now, we are going to discuss mini-batch stochastic gradient descent.


Set objective function $f(\boldsymbol{x}): \mathbb{R}^d \rightarrow \mathbb{R}$. The time step before the start of iteration is set to 0. The independent variable of this time step is $\boldsymbol{x}_0\in \mathbb{R}^d$ and is usually obtained by random initialization. In each subsequent time step $t>0$, mini-batch SGD uses random uniform sampling to get a mini-batch $\mathcal{B}_t$ made of example indices from the training data set. We can use sampling with replacement or sampling without replacement to get a mini-batch example. The former method allows duplicate examples in the same mini-batch, the latter does not and is more commonly used. We can use either of the two methods

$$\boldsymbol{g}_t \leftarrow \nabla f_{\mathcal{B}_t}(\boldsymbol{x}_{t-1}) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}_t}\nabla f_i(\boldsymbol{x}_{t-1})$$

to compute the gradient $\boldsymbol{g}_t$ of the objective function at $\boldsymbol{x}_{t-1}$ with mini-batch $\mathcal{B}_t$ at time step $t$. Here, $|\mathcal{B}|$ is the size of the batch, which is the number of examples in the mini-batch. This is a hyper-parameter. Just like the stochastic gradient, the mini-batch SGD $\boldsymbol{g}_t$ obtained by sampling with replacement is also the unbiased estimate of the gradient $\nabla f(\boldsymbol{x}_{t-1})$. Given the learning rate $\eta_t$ (positive), the iteration of the mini-batch SGD on the independent variable is as follows:

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \eta_t \boldsymbol{g}_t.$$

The variance of the gradient based on random sampling cannot be reduced during the iterative process, so in practice, the learning rate of the (mini-batch) SGD can self-decay during the iteration, such as $\eta_t=\eta t^\alpha $ (usually $\alpha=-1$ or $-0.5$), $\eta_t = \eta \alpha^t$ (e.g $\alpha=0.95$), or learning rate decay once per iteration or after several iterations. As a result, the variance of the learning rate and the (mini-batch) SGD will decrease. Gradient descent always uses the true gradient of the objective function during the iteration, without the need to self-decay the learning rate.


The cost for computing each iteration is $\mathcal{O}(|\mathcal{B}|)$. When the batch size is 1, the algorithm is an SGD; when the batch size equals the example size of the training data, the algorithm is a gradient descent. When the batch size is small, fewer examples are used in each iteration, which will result in parallel processing and reduce the RAM usage efficiency. This makes it more time consuming to compute examples of the same size than using larger batches. When the batch size increases, each mini-batch gradient may contain more redundant information. To get a better solution, we need to compute more examples for a larger batch size, such as increasing the number of epochs.


## Reading Data

In this chapter, we will use a data set developed by NASA to test the wing noise from different aircraft to compare these optimization algorithms[1]. We will use the first 1500 examples of the data set, 5 features, and a normalization method to preprocess the data.

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn, data as gdata, loss as gloss
import numpy as np
import time

def get_data_ch7():  # This function is saved in the gluonbook package for future use.
    data = np.genfromtxt('../data/airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])

features, labels = get_data_ch7()
features.shape
```

## Implementation from Scratch

We have already implemented the mini-batch SGD algorithm in the [Linear Regression Implemented From Scratch](../chapter_deep-learning-basics/linear-regression-scratch.md) section. We have made its input parameters more generic here, so that we can conveniently use the same input for the other optimization algorithms introduced later in this chapter. Specifically, we add the status input `states` and place the hyper-parameter in dictionary `hyperparams`. In addition, we will average the loss of each mini-batch example in the training function, so the gradient in the optimization algorithm does not need to be divided by the batch size.

```{.python .input  n=3}
def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad
```

Next, we are going to implement a generic training function to facilitate the use of the other optimization algorithms introduced later in this chapter. It initializes a linear regression model and can then be used to train the model with the mini-batch SGD and other algorithms introduced in subsequent sections.

```{.python .input  n=4}
# This function is saved in the gluonbook package for future use.
def train_ch7(trainer_fn, states, hyperparams, features, labels,
              batch_size=10, num_epochs=2):
    # Initialize model parameters.
    net, loss = gb.linreg, gb.squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w, b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()  # Average the loss.
            l.backward()
            trainer_fn([w, b], states, hyperparams)  # Update model parameter(s).
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())  # Record the current training error for every 100 examples.
    # Print and plot the results.
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')
```

When the batch size equals 1500 (the total number of examples), we use gradient descent for optimization. The model parameters will be iterated only once for each epoch of the gradient descent. As we can see, the downward trend of the value of the objective function (training loss) flattened out after 6 iterations.

```{.python .input  n=5}
def train_sgd(lr, batch_size, num_epochs=2):
    train_ch7(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)

train_sgd(1, 1500, 6)
```

When the batch size equals 1, we use SGD for optimization. In order to simplify the implementation, we did not self-decay the learning rate. Instead, we simply used a small constant for the learning rate in the (mini-batch) SGD experiment. In SGD, the independent variable (model parameter) is updated whenever an example is processed. Thus it is updated 1500 times in one epoch. As we can see, the decline in the value of the objective function slows down after one epoch.

Although both the procedures processed 1500 examples within one epoch, SGD consumes more time than gradient descent in our experiment. This is because SGD performed more iterations on the independent variable within one epoch, and it is harder for single-example gradient computation to use parallel computing effectively.

```{.python .input  n=6}
train_sgd(0.005, 1)
```

When the batch size equals 10, we use mini-batch SGD for optimization. The time required for one epoch is between the time needed for gradient descent and SGD to complete the same epoch.

```{.python .input  n=7}
train_sgd(0.05, 10)
```

## Implementation with Gluon

In Gluon, we can use the `Trainer` class to call optimization algorithms. Next, we are going to implement a generic training function that uses the optimization name `trainer name` and hyperparameter `trainer_hyperparameter` to create the instance `Trainer`.

```{.python .input  n=8}
# This function is saved in the gluonbook package for future use.
def train_gluon_ch7(trainer_name, trainer_hyperparams, features, labels,
                    batch_size=10, num_epochs=2):
    # Initialize model parameters.
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(
        gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    # Create the instance "Trainer" to update model parameter(s).
    trainer = gluon.Trainer(
        net.collect_params(), trainer_name, trainer_hyperparams)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)  # Average the gradient in the "Trainer" instance.
            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    # Print and plot the results.
    print('loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    gb.set_figsize()
    gb.plt.plot(np.linspace(0, num_epochs, len(ls)), ls)
    gb.plt.xlabel('epoch')
    gb.plt.ylabel('loss')
```

Use Gluon to repeat the last experiment.

```{.python .input  n=9}
train_gluon_ch7('sgd', {'learning_rate': 0.05}, features, labels, 10)
```

## Summary

* Mini-batch stochastic gradient uses random uniform sampling to get a mini-batch training example for gradient computation.
* In practice, learning rates of the (mini-batch) SGD can self-decay during iteration.
* In general, the time consumption per epoch for mini-batch stochastic gradient is between what takes for gradient descent and SGD to complete the same epoch.

## exercise

* Modify the batch size and learning rate and observe the rate of decline for the value of the objective function and the time consumed in each epoch.
* Read the MXNet documentation and use the `Trainer` class `set_learning_rate` function to reduce the learning rate of the mini-batch SGD to 1/10 of its previous value after each epoch.


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1877)

![](../img/qr_minibatch-sgd.svg)

## Reference

[1] Aircraft wing noise data set. https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise
