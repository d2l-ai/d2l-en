# Batch Normalization

Training very deep models is difficult and it can be tricky to get the models to converge or to ensure that they do not overfit. This is one of the reasons why it took a long time for very deep networks with over 100 layers to become popular. 

## Training deep networks

1. Data preprocessing is a key aspect of effective statistical modeling. Recall our discussion when we applied deep networks to [Predicting House Prices](../chapter_deep-learning-basics/kaggle-house-price.md). Here we standardized input data to zero mean and unit variance. Standardizing input data makes the distribution of features similar, which generally makes it easier to train effective models. 
1. As we train the model, the activations in intermediate layers of the network will assume rather different orders of magnitude. This can lead to issues with the convergence of the network due to scale of activations - if one layer has activation values that are 100x that of another layer, we need to adjust learning rates adaptively per layer (or even per parameter group per layer). 
1. Deeper networks are fairly complex and they are more prone to overfitting. This means that regularization becomes more critical. That said dropout is nontrivial to use in convolutional layers and does not perform as well, hence we need a more appropriate type of regularization. 
1. When training deep networks the last layers will converge first, at which point the layers below start converging. Unfortunately, once this happens, the weights for the last layers are no longer optimal and they need to converge again. As training progresses, this gets worse. 

All of this requires fixing in order for deep network training to become feasible. Batch Normalization, as proposed by [Ioffe and Szegedy, 2015](https://arxiv.org/abs/1502.03167) is an effective technique to alleviate these problems. The ingenious thing is that it does all of this with one very simple trick. Let's discuss this first before going into details as to why it is a good idea after all. In a nutshell, the idea is to transform the activation at a given layer from $\mathbf{x}$ to 

$$\bar{\mathbf{x}} = \alpha \frac{\mathbf{x} - \mathbf{\hat\mu}}{\hat\sigma} + \mathbf{\beta}$$

Here $\mathbf{\hat\mu}$ is the estimate of the mean and $\mathbf{\hat\sigma}$ is the estimate of the variance. The result is that the activations are approximately rescaled to zero mean and unit variance. Since this may not be quite what we want, we allow for a scaling coefficient $\alpha$ and an offset $\beta$. Consequently the activations for intermediate layers cannot diverge any longer, thus solving problem 1: we are actively rescaling it back to a given order of magnitude via $\mathbf{\mu}$ and $\sigma$. Consequently we can be more aggressive in picking large learning rates on the data. To address the fact that in some cases the activations actually *need* to differ from standardized data, we need to introduce scaling coefficients $\alpha$ and an offset $\mathbf{\beta}$. 

So far we didn't discuss how we actually obtain the estimates for mean and variance. The canonical way is to estimate them from data. Unfortunately, the statistics change as we train our model. Hence one option is to use the current minibatch also for estimating $\mathbf{\hat\mu}$ and $\hat\sigma$. This is fairly straightforward. All we need to do is aggregate over a small set of activations, such as a minibatch of data. Hence the name *Batch Normalization*. To indicate which minibatch $\mathcal{B}$ we draw this from, we denote the quantities with $\mathbf{\hat\mu}_\mathcal{B}$ and $\hat\sigma_\mathcal{B}$. 

$$\mathbf{\hat\mu}_\mathcal{B} \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x} 
\text{ and }
\mathbf{\hat\sigma}_\mathcal{B}^2 \leftarrow \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \mathbf{\mu}_{\mathcal{B}})^2$$

This estimate counteracts the scaling issue by using unbiased but potentially very noisy estimates of mean and variance. Normally we would consider this a problem. After all, each minibatch has different data, different labels and with it, different activations, predictions and errors. As it turns out, this is actually beneficial. This natural variation acts as *regularization* which prevents models from overfitting too badly. There is some preliminary work by [Teye, Azizpour and Smith, 2018](https://arxiv.org/pdf/1802.06455.pdf) and by [Luo et al, 2018](https://arxiv.org/pdf/1809.00846.pdf) which relate the properties of Batch Normalization (BN) to Bayesian Priors and penalties respectively. In particular, this resolves the puzzle why BN works best for moderate sizes of minibatches, i.e. of size 50-100. 

Lastly, let us briefly review the original motivation of BN, namely covariate shift correction due to training. Obviously, rescaling activations to zero mean and unit variance does not entirely remove covariate shift. In fact, if it did, it would render deep networks entirely useless. After all, we want the activations become more meaningful for solving estimation problems. However, at least, it prevents mean and variance from diverging and thus decouples one of the more problematic aspects from training and inference. 

After a lot of theory, let's look at how BN works in practice. Empirically it appears to stabilize the gradient (less exploding or vanishing values) and batch-normalized models appear to overfit less. In fact, batch-normalized models seldom even use dropout. 

## Batch Normalization Layers

The batch normalization methods for fully connected layers and convolutional layers are slightly different. This is due to the dimensionality of the data generated by convolutional layers. We discuss both cases below. Note that one of the key differences between BN and other layers is that BN operates on a a full minibatch at a time (otherwise it cannot compute the mean and variance parameters per batch).

### Fully Connected Layers

Usually we apply the batch normalization layer between the affine transformation and the activation function in the fully connected layer. Denote the input of the fully connected layer by $\mathbf{u}$, let the weight parameter and the bias parameter be $\mathbf{W}$ and $\mathbf{b}$ respectively, and let the activation function by $\phi$. In this case we have by definition

$$\mathbf{y} = \phi(\mathrm{BN}(\mathbf{x})) = \phi(\mathrm{BN}(\mathbf{W}\mathbf{u} + \mathbf{b}))$$



得到。考虑一个由$m$个样本组成的小批量，仿射变换的输出为一个新的小批量$\mathcal{B} = \{\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(m)} \}$。它们正是批量归一化层的输入。对于小批量$\mathcal{B}$中任意样本$\mathbf{x}^{(i)} \in \mathbb{R}^d, 1 \leq  i \leq m$，批量归一化层的输出同样是$d$维向量

$$\mathbf{y}^{(i)} = \text{BN}(\mathbf{x}^{(i)}),$$

and it is obtained by the following steps. First, calculate the mean and the variance of the mini-batch $\mathcal{B}$:

$$\mathbf{\mu}_\mathcal{B} \leftarrow \frac{1}{m}\sum_{i = 1}^{m} \mathbf{x}^{(i)},$$
$$\mathbf{\sigma}_\mathcal{B}^2 \leftarrow \frac{1}{m} \sum_{i=1}^{m}(\mathbf{x}^{(i)} - \mathbf{\mu}_\mathcal{B})^2,$$

Here, square computation is carried out by squaring by element. Next, we use squaring by element and division by element to standardize $\mathbf{x}^{(i)}$:

$$\hat{\mathbf{x}}^{(i)} \leftarrow \frac{\mathbf{x}^{(i)} - \mathbf{\mu}_\mathcal{B}}{\sqrt{\mathbf{\sigma}_\mathcal{B}^2 + \epsilon}},$$

Here, $\epsilon > 0$ is a very small constant, and the denominator is guaranteed to be greater than 0. Based on the above standardization, the batch normalization layer introduces two model parameters that can be learned: the scale parameter $\mathbf{\gamma}$ and the shift parameter $\mathbf{\beta}$. These two parameters have the same shape as $\mathbf{x}^{(i)}$, and both are $d$-dimensional vectors. They are calculated respectively with $\mathbf{x}^{(i)}$ by multiplication by element (symbol $\odot$) and addition by element:

$${\mathbf{y}}^{(i)} \leftarrow \mathbf{\gamma} \odot \hat{\mathbf{x}}^{(i)} + \mathbf{\beta}.$$

At this point, we have obtained the batch normalization output for $\mathbf{x}^{(i)}$: $\mathbf{y}^{(i)}$.
It is worth noting that the learnable scale and shift parameters still may not be able to perform batch normalization on $\hat{\mathbf{x}}^{(i)}$. In this case, it is necessary to learn $\mathbf{\gamma} = \sqrt{\mathbf{\sigma}_\mathcal{B}^2 + \epsilon}$ and $\mathbf{\beta} = \mathbf{\mu}_\mathcal{B}$. We can understand it like this: If the batch normalization is not beneficial, theoretically, the learned model does not have to use batch normalization.


### Batch Normalization for Convolutional Layers

For convolutional layers, batch normalization occurs after the convolution computation and before the application of the activation function. If the convolution computation outputs multiple channels, we need to carry out batch normalization for each of the outputs of these channels, and each channel has an independent scale parameter and shift parameter, both of which are scalars. Assume that there are $m$ examples in the mini-batch. On a single channel, we assume that the height and width of the convolution computation output are $p$ and $q$, respectively. We need to carry out batch normalization for $m \times p \times q$ elements in this channel simultaneously. While carrying out the standardization computation for these elements, we use the same mean and variance. In other words, we use the means and variances of the $m \times p \times q$ elements in this channel.


### Batch Normalization During Prediction

When using batch normalization training, we can set the batch size to be a bit larger, so that the computation of the mean and variance of the examples in the batch will be more accurate. When using the trained model for prediction, we want the model to have definite output for any input. Therefore, the output of a single example should not depend on the mean and variance in the random mini-batch required by the batch normalization. A common method is to estimate the mean and variance of the examples for the entire training data set via moving average, and use them to obtain definite output at the time of prediction. As we can see, like the dropout layer, the batch normalization layer has different computation results in the training mode and the prediction mode.


## Implementation Starting from Scratch

Next, we will implement the batch normalization layer via the NDArray.

```{.python .input  n=72}
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use autograd to determine whether the current mode is training mode or prediction mode.
    if not autograd.is_training():
        # If it is the prediction mode, directly use the mean and variance obtained from the incoming moving average.
        X_hat = (X - moving_mean) / nd.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and variance on the feature dimension.
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the mean and variance on the channel dimension (axis=1). Here we need to 
            # maintain the shape of X, so that the broadcast operation can be carried out later.
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the standardization.
        X_hat = (X - mean) / nd.sqrt(var + eps)
        # Update the mean and variance of the moving average.
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift.
    return Y, moving_mean, moving_var
```

Next, we will customize a `BatchNorm` layer. This retains the scale parameter `gamma` and the shift parameter `beta` involved in gradient finding and iteration, and it also maintains the mean and variance obtained from the moving average, so that they can be used during model prediction. The `num_features` parameter required by the `BatchNorm` instance is the number of outputs for a fully connected layer and the number of output channels for a convolutional layer. The `num_dims` parameter also required by this instance is 2 for a fully connected layer and 4 for a convolutional layer.

```{.python .input  n=73}
class BatchNorm(nn.Block):
    def __init__(self, num_features, num_dims, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter involved in gradient finding and iteration are initialized to 0 and 1 respectively.
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # All the variables not involved in gradient finding and iteration are initialized to 0 on the CPU.
        self.moving_mean = nd.zeros(shape)
        self.moving_var = nd.zeros(shape)

    def forward(self, X):
        # If X is not on the CPU, copy moving_mean and moving_var to the device where X is located.
        if self.moving_mean.context != X.context:
            self.moving_mean = self.moving_mean.copyto(X.context)
            self.moving_var = self.moving_var.copyto(X.context)
        # Save the updated moving_mean and moving_var.
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

## Use a Batch Normalization LeNet

Next, we will modify the LeNet model introduced in the section ["Convolutional Neural Network (LeNet)"](lenet.md) in order to apply the batch normalization layer. We add the batch normalization layer after all the convolutional layers and fully connected layers and before the activation layer.

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

Next, we will train the modified model.

```{.python .input  n=77}
lr, num_epochs, batch_size, ctx = 1.0, 5, 256, gb.try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = gb.load_data_fashion_mnist(batch_size)
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

Finally, we check the scale parameter `gamma` and the shift parameter `beta` learned from the first batch normalization layer.

```{.python .input  n=60}
net[1].gamma.data().reshape((-1,)), net[1].beta.data().reshape((-1,))
```

## Gluon Implementation for Batch Normalization

Compared with the `BatchNorm` class, which we just defined ourselves, the `BatchNorm` class defined by the `nn` model in Gluon is easier to use. In Gluon, we do not have to define the `num_features` and `num_dims` parameter values required in the `BatchNorm` class. Instead, these parameter values will be obtained automatically by delayed initialization. Next, we will use Gluon to implement the batch normalization LeNet.

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

Use the same hyper-parameter to carry out the training.

```{.python .input}
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
gb.train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)
```

## Summary

* During model training, batch normalization continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the mini-batch, so that the values of the intermediate output in each layer throughout the neural network are more stable.
* The batch normalization methods for fully connected layers and convolutional layers are slightly different.
* Like a dropout layer, batch normalization layers have different computation results in training mode and prediction mode.
* The BatchNorm function provided by Gluon is easy and convenient.

## exercise

* Can we remove the fully connected affine transformation before the batch normalization or the bias parameter in convolution computation? Why? (Hint: Recall the definition of standardization in batch normalization. )
* Try to increase the learning rate. Compared with the previous LeNet, which does not use batch normalization, is it now possible to use a bigger learning rate?
* Try to insert the batch normalization layer somewhere else in the LeNet, and observe and analyze the changes to the results.
* Try not to learn `beta` and `gamma` (add the parameter `grad_req='null'` at the time of construction to avoid calculating the gradient), and observe and analyze the results.
* To learn about more application methods, such as how to use the mean and variance of the global average during training, view the documentation for the `BatchNorm` class.


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1253)

![](../img/qr_batch-norm.svg)

## References

[1] Ioffe, S., & Szegedy, C. (2015). Batch normalization: Accelerating deep network training by reducing internal covariate shift. arXiv preprint arXiv:1502.03167.
