# Batch Normalization

Training very deep models is difficult and it can be tricky to get the models to converge (or converge within a reasonable amount of time) when training. It can be equally challenging to ensure that they do not overfit. This is one of the reasons why it took a long time for very deep networks with over 100 layers to gain popularity.

## Training Deep Networks

Let's review some of the practical challenges when training deep networks.

1. Data preprocessing is a key aspect of effective statistical modeling. Recall our discussion when we applied deep networks to [Predicting House Prices](../chapter_deep-learning-basics/kaggle-house-price.md). There we standardized input data to zero mean and unit variance. Standardizing input data makes the distribution of features similar, which generally makes it easier to train effective models since parameters are a-priori at a similar scale.
1. As we train the model, the activations in intermediate layers of the network will assume rather different orders of magnitude. This can lead to issues with the convergence of the network due to scale of activations - if one layer has activation values that are 100x that of another layer, we need to adjust learning rates adaptively per layer (or even per parameter group per layer).
1. Deeper networks are fairly complex and they are more prone to overfitting. This means that regularization becomes more critical. That said dropout is nontrivial to use in convolutional layers and does not perform as well, hence we need a more appropriate type of regularization.
1. When training deep networks the last layers will converge first, at which point the layers below start converging. Unfortunately, once this happens, the weights for the last layers are no longer optimal and they need to converge again. As training progresses, this gets worse.

Batch normalization (BN), as proposed by [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167), can be used to cope with the challenges of deep model training. During training, BN continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the mini-batch. In effect that causes the optimization landscape of the model to be smoother, hence allowing the model to reach a local minimum and to be trained faster. That being said, one has to be careful in oder to avoid the already troubling trends in machine learning ([Lipton et al, 2018](https://arxiv.org/abs/1807.03341)). Batch normalization has been shown ([Santukar et al., 2018](https://arxiv.org/abs/1805.11604)) to have no relation at all with internal covariate shift, as a matter in fact it has been shown that it actually causes the opposite result from what it was originally intended, pointed by [Lipton et al., 2018](https://arxiv.org/abs/1807.03341) as well. In a nutshell, the idea in Batch Normalization is to transform the activation at a given layer from $\mathbf{x}$ to

$$\mathrm{BN}(\mathbf{x}) = \mathbf{\gamma} \odot \frac{\mathbf{x} - \hat{\mathbf{\mu}}}{\hat\sigma} + \mathbf{\beta}$$

Here $\hat{\mathbf{\mu}}$ is the estimate of the mean and $\hat{\mathbf{\sigma}}$ is the estimate of the variance. The result is that the activations are approximately rescaled to zero mean and unit variance. Since this may not be quite what we want, we allow for a coordinate-wise scaling coefficient $\mathbf{\gamma}$ and an offset $\mathbf{\beta}$. Consequently the activations for intermediate layers cannot diverge any longer: we are actively rescaling it back to a given order of magnitude via $\mathbf{\mu}$ and $\sigma$. Consequently we can be more aggressive in picking large learning rates on the data. To address the fact that in some cases the activations actually *need* to differ from standardized data, we need to introduce scaling coefficients $\mathbf{\gamma}$ and an offset $\mathbf{\beta}$.

We use training data to estimate mean and variance. Unfortunately, the statistics change as we train our model. To address this, we use the current minibatch also for estimating $\hat{\mathbf{\mu}}$ and $\hat\sigma$. This is fairly straightforward. All we need to do is aggregate over a small set of activations, such as a minibatch of data. Hence the name *Batch* Normalization. To indicate which minibatch $\mathcal{B}$ we draw this from, we denote the quantities with $\hat{\mathbf{\mu}}_\mathcal{B}$ and $\hat\sigma_\mathcal{B}$.

$$\hat{\mathbf{\mu}}_\mathcal{B} \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\text{ and }
\hat{\mathbf{\sigma}}_\mathcal{B}^2 \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \mathbf{\mu}_{\mathcal{B}})^2 + \epsilon$$

Note that we add a small constant $\epsilon > 0$ to the variance estimate to ensure that we never end up dividing by zero, even in cases where the empirical variance estimate might vanish by accident.
The estimates $\hat{\mathbf{\mu}}_\mathcal{B}$ and $\hat{\mathbf{\sigma}}_\mathcal{B}$ counteract the scaling issue by using unbiased but potentially very noisy estimates of mean and variance. Normally we would consider this a problem. After all, each minibatch has different data, different labels and with it, different activations, predictions and errors. As it turns out, this is actually beneficial. This natural variation acts as *regularization* which prevents models from overfitting too badly. There is some preliminary work by [Teye, Azizpour and Smith, 2018](https://arxiv.org/pdf/1802.06455.pdf) and by [Luo et al, 2018](https://arxiv.org/pdf/1809.00846.pdf) which relate the properties of Batch Normalization (BN) to Bayesian Priors and penalties respectively. In particular, this resolves the puzzle why BN works best for moderate sizes of minibatches, i.e. of size 50-100.

Lastly, let us briefly review the original motivation of BN, namely covariate shift correction due to training. Obviously, rescaling activations to zero mean and unit variance does not entirely remove covariate shift (in fact, recent work suggests that it actually increases it). In fact, if it did, it would render deep networks entirely useless. After all, we want the activations become more meaningful for solving estimation problems. However, at least, it prevents mean and variance from diverging and thus decouples one of the more problematic aspects from training and inference.

After a lot of theory, let's look at how BN works in practice. Empirically it appears to stabilize the gradient (less exploding or vanishing values) and batch-normalized models appear to overfit less. In fact, batch-normalized models seldom even use dropout.

## Batch Normalization Layers

The batch normalization methods for fully connected layers and convolutional layers are slightly different. This is due to the dimensionality of the data generated by convolutional layers. We discuss both cases below. Note that one of the key differences between BN and other layers is that BN operates on a a full minibatch at a time (otherwise it cannot compute the mean and variance parameters per batch).

### Fully Connected Layers

Usually we apply the batch normalization layer between the affine transformation and the activation function in a fully connected layer. In the following we denote by $\mathbf{u}$ the input and by $\mathbf{x} = \mathbf{W}\mathbf{u} + \mathbf{b}$ the output of the linear transform. This yields the following variant of the batch norm:

$$\mathbf{y} = \phi(\mathrm{BN}(\mathbf{x})) =  \phi(\mathrm{BN}(\mathbf{W}\mathbf{u} + \mathbf{b}))$$

Recall that mean and variance are computed on the *same* minibatch $\mathcal{B}$ on which this transformation is applied to. Also recall that the scaling coefficient $\mathbf{\gamma}$ and the offset $\mathbf{\beta}$ are parameters that need to be learned. They ensure that the effect of batch normalization can be neutralized as needed.

### Convolutional Layers

For convolutional layers, batch normalization occurs after the convolution computation and before the application of the activation function. If the convolution computation outputs multiple channels, we need to carry out batch normalization for *each* of the outputs of these channels, and each channel has an independent scale parameter and shift parameter, both of which are scalars. Assume that there are $m$ examples in the mini-batch. On a single channel, we assume that the height and width of the convolution computation output are $p$ and $q$, respectively. We need to carry out batch normalization for $m \times p \times q$ elements in this channel simultaneously. While carrying out the standardization computation for these elements, we use the same mean and variance. In other words, we use the means and variances of the $m \times p \times q$ elements in this channel rather than one per pixel.


### Batch Normalization During Prediction

At prediction time we might not have the luxury of computing offsets per batch - we might be required to make one prediction at a time. Secondly, the uncertainty in $\mathbf{\mu}$ and $\mathbf{\sigma}$, as arising from a minibatch are highly undesirable once we've trained the model. One way to mitigate this is to compute more stable estimates on a larger set for once (e.g. via a moving average) and then fix them at prediction time. Consequently, Batch Normalization behaves differently during training and test time (just like we already saw in the case of Dropout).

## Implementation from Scratch

Next, we will implement the batch normalization layer via the NDArray from scratch.

```{.python .input  n=72}
import sys
sys.path.insert(0, '..')

import d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is the prediction mode, directly use the mean and variance
        # obtained from the incoming moving average
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of X, so that the broadcast operation
            # can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # Update the mean and variance of the moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

Next, we will customize a `BatchNorm` layer. This retains the scale parameter `gamma` and the shift parameter `beta` involved in gradient finding and iteration, and it also maintains the mean and variance obtained from the moving average, so that they can be used during model prediction. The `num_features` parameter required by the `BatchNorm` instance is the number of outputs for a fully connected layer and the number of output channels for a convolutional layer. The `num_dims` parameter also required by this instance is 2 for a fully connected layer and 4 for a convolutional layer.

Besides the algorithm per se, also note the design pattern in implementing layers. Typically one defines the math in a separate function, say `batch_norm`. This is then integrated into a custom layer that mostly focuses on bookkeeping, such as moving data to the right device context, ensuring that variables are properly initialized, keeping track of the running averages for mean and variance, etc.; That way we achieve a clean separation of math and boilerplate code. Also note that for the sake of convenience we did not add automagic size inference here, hence we will need to specify the number of features throughout (the Gluon version takes care of this).

```{.python .input  n=73}
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter involved in gradient
        # finding and iteration are initialized to 0 and 1 respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # All the variables not involved in gradient finding and iteration are
        # initialized to 0 on the CPU
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X):
        # If X is not on the CPU, copy moving_mean and moving_var to the
        # device where X is located
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        # Save the updated moving_mean and moving_var
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

## Use a Batch Normalization LeNet

Next, we will modify the [LeNet model](lenet.md) in order to apply the batch normalization layer. We add the batch normalization layer after all the convolutional layers and after all fully connected layers. As discussed, we add it  before the activation layer.

```{.python .input  n=74}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

Next we train the modified model, again on Fashion-MNIST. The code is virtually identical to that in previous steps. The main difference is the considerably larger learning rate.

```{.python .input  n=77}
lr, num_epochs, batch_size, ctx = 1.0, 5, 256, d2l.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs)
```

Let's have a look at the scale parameter `gamma` and the shift parameter `beta` learned from the first batch normalization layer.

```{.python .input  n=60}
net[1].gamma.data().reshape((-1,)), net[1].beta.data().reshape((-1,))
```

## Gluon Implementation for Batch Normalization

Compared with the `BatchNorm` class, which we just defined ourselves, the `BatchNorm` class defined by the `nn` model in Gluon is easier to use. In Gluon, we do not have to define the `num_features` and `num_dims` parameter values required in the `BatchNorm` class. Instead, these parameter values will be obtained automatically by delayed initialization. The code looks virtually identical (save for the lack of an explicit specification of the dimensionality of the features for the Batch Normalization layers).

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.MaxPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

Use the same hyper-parameter to carry out the training. Note that as always the Gluon variant runs a lot faster since the code that is being executed is compiled C++/CUDA rather than interpreted Python.

```{.python .input}
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs)
```

## Summary

* During model training, batch normalization continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the mini-batch, so that the values of the intermediate output in each layer throughout the neural network are more stable.
* The batch normalization methods for fully connected layers and convolutional layers are slightly different.
* Like a dropout layer, batch normalization layers have different computation results in training mode and prediction mode.
* Batch Normalization has many beneficial side effects, primarily that of regularization. On the other hand, the original motivation of reducing covariate shift seems not to be a valid explanation.

## Problems

1. Can we remove the fully connected affine transformation before the batch normalization or the bias parameter in convolution computation?
    * Find an equivalent transformation that applies prior to the fully connected layer.
    * Is this reformulation effective. Why (not)?
1. Compare the learning rates for LeNet with and without batch normalization.
    * Plot the decrease in training and test error.
    * What about the region of convergence? How large can you make the learning rate?
1. Do we need Batch Normalization in every layer? Experiment with it?
1. Can you replace Dropout by Batch Normalization? How does the behavior change?
1. Fix the coefficients `beta` and `gamma` (add the parameter `grad_req='null'` at the time of construction to avoid calculating the gradient), and observe and analyze the results.
1. Review the Gluon documentation for `BatchNorm` to see the other applications for Batch Normalization.
1. Research ideas - think of other normalization transforms that you can apply? Can you apply the probability integral transform? How about a full rank covariance estimate?


## References

[1] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.

## Discuss on our Forum

<div id="discuss" topic_id="2358"></div>
