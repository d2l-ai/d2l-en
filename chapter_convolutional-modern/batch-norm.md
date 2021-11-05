```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "The d2lbook.tab extension is already loaded. To reload it, use:\n  %reload_ext d2lbook.tab\n"
 },
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "b2726fa7dc4a4a2cba6c63433e0743f6",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "interactive(children=(Dropdown(description='tab', index=1, options=('mxnet', 'pytorch', 'tensorflow'), value='\u2026"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

# Batch Normalization
:label:`sec_batch_norm`

Training deep neural networks is difficult.
Getting them to converge in a reasonable amount of time can be tricky.
In this section, we describe *batch normalization*, a popular and effective technique
that consistently accelerates the convergence of deep networks :cite:`Ioffe.Szegedy.2015`.
Together with residual blocks---covered later in :numref:`sec_resnet`---batch normalization
has made it possible for practitioners to routinely train networks with over 100 layers.
A secondary (serendipitous) benefit of Batch Normalization is its inherent regularization. 

## Training Deep Networks

When working with data, we often preprocess before training. 
Choices regarding data preprocessing often make an enormous difference in the final results.
Recall our application of MLPs to predicting house prices (:numref:`sec_kaggle_house`).
Our first step when working with real data
was to standardize our input features to have 
zero mean $\mathbf{\mu} = 0$ and unit variance $\mathbf{\Sigma} = \mathbf{1}$ across multiple observations :cite:`friedman1987exploratory`.
At a mimimum, one frequently rescales it such that the diagonal is unity, i.e., $\Sigma_{ii} = 1$. 
Yet another strategy is to rescale vectors to unit length, possibly zero mean *per observation*. 
This can work well e.g., for spatial sensor data. These preprocessing techniques and many more are 
beneficial to keep the estimation problem well controlled. 

Intuitively, this standardization plays nicely with our optimizers
because it puts the parameters *a priori* at a similar scale.
As such, it is only natural to ask whether a corresponding normalization step *inside* a deep network
might not be beneficial. While this isn't quite the reasoning that led to the invention of Batch Normalization :cite:`Ioffe.Szegedy.2015`, it is a useful way of understanding it and its cousin, Layer Normalization :cite:`Ba.Kiros.Hinton.2016` within a unified framework. 

Second, for a typical MLP or CNN, as we train,
the variables (e.g., affine transformation outputs in MLP)
in intermediate layers 
may take values with widely varying magnitudes:
both along the layers from the input to the output, across units in the same layer,
and over time due to our updates to the model parameters.
The inventors of batch normalization postulated informally
that this drift in the distribution of such variables could hamper the convergence of the network.
Intuitively, we might conjecture that if one
layer has variable values that are 100 times that of another layer,
this might necessitate compensatory adjustments in the learning rates.
   
Third, deeper networks are complex and easily capable of overfitting.
This means that regularization becomes more critical.

Batch normalization is applied to individual layers
(optionally, to all of them) and works as follows:
In each training iteration,
we first normalize the inputs (of batch normalization)
by subtracting their mean and
dividing by their standard deviation,
where both are estimated based on the statistics of the current minibatch.
Next, we apply a scale coefficient and a scale offset.
It is precisely due to this *normalization* based on *batch* statistics
that *batch normalization* derives its name.

Note that if we tried to apply batch normalization with minibatches of size 1,
we would not be able to learn anything.
That is because after subtracting the means,
each hidden unit would take value 0!
As you might guess, since we are devoting a whole section to batch normalization,
with large enough minibatches, the approach proves effective and stable.
One takeaway here is that when applying batch normalization,
the choice of batch size may be
even more significant than without batch normalization, or at least, 
suitable calibration is needed as we might adjust it.

Formally, denote by $\mathbf{x} \in \mathcal{B}$ an input to batch normalization ($\mathrm{BN}$)
that is from a minibatch $\mathcal{B}$,
batch normalization transforms $\mathbf{x}$
according to the following expression:

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$
:eqlabel:`eq_batchnorm`

In :eqref:`eq_batchnorm`,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ is the  sample mean
and $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ is the sample standard deviation of the minibatch $\mathcal{B}$.
After applying standardization,
the resulting minibatch
has zero mean and unit variance.
Because the choice of unit variance
(vs. some other magic number) is an arbitrary choice,
we commonly include elementwise
*scale parameter* $\boldsymbol{\gamma}$ and *shift parameter* $\boldsymbol{\beta}$
that have the same shape as $\mathbf{x}$.
Note that $\boldsymbol{\gamma}$ and $\boldsymbol{\beta}$ are
 parameters that need to be learned jointly with the other model parameters.

Consequently, the variable magnitudes
for intermediate layers cannot diverge during training
because batch normalization actively centers and rescales them back
to a given mean and size (via $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
One piece of practitioner's intuition or wisdom
is that batch normalization seems to allow for more aggressive learning rates.


Formally, 
we calculate $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ in :eqref:`eq_batchnorm` as follows:

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\text{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Note that we add a small constant $\epsilon > 0$
to the variance estimate
to ensure that we never attempt division by zero,
even in cases where the empirical variance estimate might vanish.
The estimates $\hat{\boldsymbol{\mu}}_\mathcal{B}$ and ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ counteract the scaling issue
by using noisy estimates of mean and variance.
You might think that this noisiness should be a problem.
As it turns out, this is actually beneficial.

This turns out to be a recurring theme in deep learning.
For reasons that are not yet well-characterized theoretically,
various sources of noise in optimization
often lead to faster training and less overfitting:
this variation appears to act as a form of regularization.
:cite:`Teye.Azizpour.Smith.2018` and :cite:`Luo.Wang.Shao.ea.2018`
relate the properties of batch normalization to Bayesian priors and penalties respectively.
In particular, this sheds some light on the puzzle
of why batch normalization works best for moderate minibatches sizes in the $50 \sim 100$ range.

Fixing a trained model, you might think
that we would prefer using the entire dataset
to estimate the mean and variance.
Once training is complete, why would we want
the same image to be classified differently,
depending on the batch in which it happens to reside?
During training, such exact calculation is infeasible
because the intermediate variables
for all data examples
change every time we update our model.
However, once the model is trained,
we can calculate the means and variances
of each layer's variables based on the entire dataset.
Indeed this is standard practice for
models employing batch normalization
and thus batch normalization layers function differently
in *training mode* (normalizing by minibatch statistics)
and in *prediction mode* (normalizing by dataset statistics).

We are now ready to take a look at how batch normalization works in practice.


## Batch Normalization Layers

Batch normalization implementations for fully connected layers
and convolutional layers are slightly different.
We discuss both cases below.
Recall that one key differences between batch normalization and other layers
is that because batch normalization operates on a full minibatch at a time,
we cannot just ignore the batch dimension
as we did before when introducing other layers.

### Fully Connected Layers

When applying batch normalization to fully connected layers,
the original paper inserts batch normalization after the affine transformation
and before the nonlinear activation function (later applications may insert batch normalization right after activation functions) :cite:`Ioffe.Szegedy.2015`.
Denoting the input to the fully connected layer by $\mathbf{x}$,
the affine transformation
by $\mathbf{W}\mathbf{x} + \mathbf{b}$ (with the weight parameter $\mathbf{W}$ and the bias parameter $\mathbf{b}$),
and the activation function by $\phi$,
we can express the computation of a batch-normalization-enabled,
fully connected layer output $\mathbf{h}$ as follows:

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Recall that mean and variance are computed
on the *same* minibatch 
on which the transformation is applied.

### Convolutional Layers

Similarly, with convolutional layers,
we can apply batch normalization after the convolution
and before the nonlinear activation function. The key difference to batch norms 
in fully-connected layers is that we apply the operation on a per-channel basis 
*across all locations*. 

Assume that our minibatches contain $m$ examples
and that for each channel,
the output of the convolution has height $p$ and width $q$.
For convolutional layers, we carry out each batch normalization
over the $m \cdot p \cdot q$ elements per output channel simultaneously.
Thus, we collect the values over all spatial locations
when computing the mean and variance
and consequently 
apply the same mean and variance
within a given channel
to normalize the value at each spatial location.
Each channel has its own scale and shift parameters,
both of which are scalars.

### Batch Normalization During Prediction

As we mentioned earlier, batch normalization typically behaves differently
in training mode and prediction mode.
First, the noise in the sample mean and the sample variance
arising from estimating each on minibatches
are no longer desirable once we have trained the model.
Second, we might not have the luxury
of computing per-batch normalization statistics.
For example,
we might need to apply our model to make one prediction at a time.

Typically, after training, we use the entire dataset
to compute stable estimates of the variable statistics
and then fix them at prediction time.
Consequently, batch normalization behaves differently during training and at test time.
Recall that dropout also exhibits this characteristic.

## (**Implementation from Scratch**)

Below, we implement a batch normalization layer with tensors from scratch.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
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
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.json .output n=2}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

We can now [**create a proper `BatchNorm` layer.**]
Our layer will maintain proper parameters
for scale `gamma` and shift `beta`,
both of which will be updated in the course of training.
Additionally, our layer will maintain
moving averages of the means and variances
for subsequent use during model prediction.

Putting aside the algorithmic details,
note the design pattern underlying our implementation of the layer.
Typically, we define the mathematics in a separate function, say `batch_norm`.
We then integrate this functionality into a custom layer,
whose code mostly addresses bookkeeping matters,
such as moving data to the right device context,
allocating and initializing any required variables,
keeping track of moving averages (here for mean and variance), and so on.
This pattern enables a clean separation of mathematics from boilerplate code.
Also note that for the sake of convenience
we did not worry about automatically inferring the input shape here,
thus we need to specify the number of features throughout.
Do not worry, the high-level batch normalization APIs in the deep learning framework will care of this for us and we will demonstrate that later.

```{.python .input  n=5}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.json .output n=5}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=8}
%%tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input  n=7}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

```{.json .output n=7}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

## [**Applying Batch Normalization in LeNet**]

To see how to apply `BatchNorm` in context,
below we apply it to a traditional LeNet model (:numref:`sec_lenet`).
Recall that batch normalization is applied
after the convolutional layers or fully connected layers
but before the corresponding activation functions.

```{.python .input  n=9}
%%tab mxnet
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        BatchNorm(6, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        BatchNorm(16, num_dims=4),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        BatchNorm(120, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        BatchNorm(84, num_dims=2),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.json .output n=9}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=10}
%%tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), BatchNorm(6, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), BatchNorm(16, num_dims=4), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(16*4*4, 120), BatchNorm(120, num_dims=2), nn.Sigmoid(),
    nn.Linear(120, 84), BatchNorm(84, num_dims=2), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input  n=11}
%%tab tensorflow
# Recall that this has to be a function that will be passed to `d2l.train_ch6`
# so that model building or compiling need to be within `strategy.scope()` in
# order to utilize the CPU/GPU devices that we have
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        BatchNorm(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10)]
    )
```

```{.json .output n=11}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

As before, we will [**train our network on the Fashion-MNIST dataset**].
This code is virtually identical to that when we first trained LeNet (:numref:`sec_lenet`).
The main difference is the larger learning rate.

```{.python .input  n=12}
%%tab mxnet, pytorch
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=12}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "loss 0.263, train acc 0.904, test acc 0.888\n10653.9 examples/sec on cpu\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 238.965625 180.65625\" width=\"238.965625pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2021-10-24T17:39:04.755344</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.4.3, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 238.965625 180.65625 \nL 238.965625 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \nL 225.403125 7.2 \nL 30.103125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 51.803125 143.1 \nL 51.803125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m05d406af88\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"51.803125\" xlink:href=\"#m05d406af88\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 2 -->\n      <g transform=\"translate(48.621875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 1228 531 \nL 3431 531 \nL 3431 0 \nL 469 0 \nL 469 531 \nQ 828 903 1448 1529 \nQ 2069 2156 2228 2338 \nQ 2531 2678 2651 2914 \nQ 2772 3150 2772 3378 \nQ 2772 3750 2511 3984 \nQ 2250 4219 1831 4219 \nQ 1534 4219 1204 4116 \nQ 875 4013 500 3803 \nL 500 4441 \nQ 881 4594 1212 4672 \nQ 1544 4750 1819 4750 \nQ 2544 4750 2975 4387 \nQ 3406 4025 3406 3419 \nQ 3406 3131 3298 2873 \nQ 3191 2616 2906 2266 \nQ 2828 2175 2409 1742 \nQ 1991 1309 1228 531 \nz\n\" id=\"DejaVuSans-32\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 95.203125 143.1 \nL 95.203125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"95.203125\" xlink:href=\"#m05d406af88\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 4 -->\n      <g transform=\"translate(92.021875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2419 4116 \nL 825 1625 \nL 2419 1625 \nL 2419 4116 \nz\nM 2253 4666 \nL 3047 4666 \nL 3047 1625 \nL 3713 1625 \nL 3713 1100 \nL 3047 1100 \nL 3047 0 \nL 2419 0 \nL 2419 1100 \nL 313 1100 \nL 313 1709 \nL 2253 4666 \nz\n\" id=\"DejaVuSans-34\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 138.603125 143.1 \nL 138.603125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"138.603125\" xlink:href=\"#m05d406af88\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 6 -->\n      <g transform=\"translate(135.421875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2113 2584 \nQ 1688 2584 1439 2293 \nQ 1191 2003 1191 1497 \nQ 1191 994 1439 701 \nQ 1688 409 2113 409 \nQ 2538 409 2786 701 \nQ 3034 994 3034 1497 \nQ 3034 2003 2786 2293 \nQ 2538 2584 2113 2584 \nz\nM 3366 4563 \nL 3366 3988 \nQ 3128 4100 2886 4159 \nQ 2644 4219 2406 4219 \nQ 1781 4219 1451 3797 \nQ 1122 3375 1075 2522 \nQ 1259 2794 1537 2939 \nQ 1816 3084 2150 3084 \nQ 2853 3084 3261 2657 \nQ 3669 2231 3669 1497 \nQ 3669 778 3244 343 \nQ 2819 -91 2113 -91 \nQ 1303 -91 875 529 \nQ 447 1150 447 2328 \nQ 447 3434 972 4092 \nQ 1497 4750 2381 4750 \nQ 2619 4750 2861 4703 \nQ 3103 4656 3366 4563 \nz\n\" id=\"DejaVuSans-36\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 182.003125 143.1 \nL 182.003125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"182.003125\" xlink:href=\"#m05d406af88\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 8 -->\n      <g transform=\"translate(178.821875 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 2034 2216 \nQ 1584 2216 1326 1975 \nQ 1069 1734 1069 1313 \nQ 1069 891 1326 650 \nQ 1584 409 2034 409 \nQ 2484 409 2743 651 \nQ 3003 894 3003 1313 \nQ 3003 1734 2745 1975 \nQ 2488 2216 2034 2216 \nz\nM 1403 2484 \nQ 997 2584 770 2862 \nQ 544 3141 544 3541 \nQ 544 4100 942 4425 \nQ 1341 4750 2034 4750 \nQ 2731 4750 3128 4425 \nQ 3525 4100 3525 3541 \nQ 3525 3141 3298 2862 \nQ 3072 2584 2669 2484 \nQ 3125 2378 3379 2068 \nQ 3634 1759 3634 1313 \nQ 3634 634 3220 271 \nQ 2806 -91 2034 -91 \nQ 1263 -91 848 271 \nQ 434 634 434 1313 \nQ 434 1759 690 2068 \nQ 947 2378 1403 2484 \nz\nM 1172 3481 \nQ 1172 3119 1398 2916 \nQ 1625 2713 2034 2713 \nQ 2441 2713 2670 2916 \nQ 2900 3119 2900 3481 \nQ 2900 3844 2670 4047 \nQ 2441 4250 2034 4250 \nQ 1625 4250 1398 4047 \nQ 1172 3844 1172 3481 \nz\n\" id=\"DejaVuSans-38\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#m05d406af88\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 10 -->\n      <g transform=\"translate(219.040625 157.698438)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 794 531 \nL 1825 531 \nL 1825 4091 \nL 703 3866 \nL 703 4441 \nL 1819 4666 \nL 2450 4666 \nL 2450 531 \nL 3481 531 \nL 3481 0 \nL 794 0 \nL 794 531 \nz\n\" id=\"DejaVuSans-31\" transform=\"scale(0.015625)\"/>\n        <path d=\"M 2034 4250 \nQ 1547 4250 1301 3770 \nQ 1056 3291 1056 2328 \nQ 1056 1369 1301 889 \nQ 1547 409 2034 409 \nQ 2525 409 2770 889 \nQ 3016 1369 3016 2328 \nQ 3016 3291 2770 3770 \nQ 2525 4250 2034 4250 \nz\nM 2034 4750 \nQ 2819 4750 3233 4129 \nQ 3647 3509 3647 2328 \nQ 3647 1150 3233 529 \nQ 2819 -91 2034 -91 \nQ 1250 -91 836 529 \nQ 422 1150 422 2328 \nQ 422 3509 836 4129 \nQ 1250 4750 2034 4750 \nz\n\" id=\"DejaVuSans-30\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_6\">\n     <!-- epoch -->\n     <g transform=\"translate(112.525 171.376563)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 3597 1894 \nL 3597 1613 \nL 953 1613 \nQ 991 1019 1311 708 \nQ 1631 397 2203 397 \nQ 2534 397 2845 478 \nQ 3156 559 3463 722 \nL 3463 178 \nQ 3153 47 2828 -22 \nQ 2503 -91 2169 -91 \nQ 1331 -91 842 396 \nQ 353 884 353 1716 \nQ 353 2575 817 3079 \nQ 1281 3584 2069 3584 \nQ 2775 3584 3186 3129 \nQ 3597 2675 3597 1894 \nz\nM 3022 2063 \nQ 3016 2534 2758 2815 \nQ 2500 3097 2075 3097 \nQ 1594 3097 1305 2825 \nQ 1016 2553 972 2059 \nL 3022 2063 \nz\n\" id=\"DejaVuSans-65\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1159 525 \nL 1159 -1331 \nL 581 -1331 \nL 581 3500 \nL 1159 3500 \nL 1159 2969 \nQ 1341 3281 1617 3432 \nQ 1894 3584 2278 3584 \nQ 2916 3584 3314 3078 \nQ 3713 2572 3713 1747 \nQ 3713 922 3314 415 \nQ 2916 -91 2278 -91 \nQ 1894 -91 1617 61 \nQ 1341 213 1159 525 \nz\nM 3116 1747 \nQ 3116 2381 2855 2742 \nQ 2594 3103 2138 3103 \nQ 1681 3103 1420 2742 \nQ 1159 2381 1159 1747 \nQ 1159 1113 1420 752 \nQ 1681 391 2138 391 \nQ 2594 391 2855 752 \nQ 3116 1113 3116 1747 \nz\n\" id=\"DejaVuSans-70\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 1959 3097 \nQ 1497 3097 1228 2736 \nQ 959 2375 959 1747 \nQ 959 1119 1226 758 \nQ 1494 397 1959 397 \nQ 2419 397 2687 759 \nQ 2956 1122 2956 1747 \nQ 2956 2369 2687 2733 \nQ 2419 3097 1959 3097 \nz\nM 1959 3584 \nQ 2709 3584 3137 3096 \nQ 3566 2609 3566 1747 \nQ 3566 888 3137 398 \nQ 2709 -91 1959 -91 \nQ 1206 -91 779 398 \nQ 353 888 353 1747 \nQ 353 2609 779 3096 \nQ 1206 3584 1959 3584 \nz\n\" id=\"DejaVuSans-6f\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3122 3366 \nL 3122 2828 \nQ 2878 2963 2633 3030 \nQ 2388 3097 2138 3097 \nQ 1578 3097 1268 2742 \nQ 959 2388 959 1747 \nQ 959 1106 1268 751 \nQ 1578 397 2138 397 \nQ 2388 397 2633 464 \nQ 2878 531 3122 666 \nL 3122 134 \nQ 2881 22 2623 -34 \nQ 2366 -91 2075 -91 \nQ 1284 -91 818 406 \nQ 353 903 353 1747 \nQ 353 2603 823 3093 \nQ 1294 3584 2113 3584 \nQ 2378 3584 2631 3529 \nQ 2884 3475 3122 3366 \nz\n\" id=\"DejaVuSans-63\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 4863 \nL 1159 4863 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" id=\"DejaVuSans-68\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-70\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-68\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 30.103125 118.87473 \nL 225.403125 118.87473 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m5364f498e3\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m5364f498e3\" y=\"118.87473\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.4 -->\n      <g transform=\"translate(7.2 122.673948)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 684 794 \nL 1344 794 \nL 1344 0 \nL 684 0 \nL 684 794 \nz\n\" id=\"DejaVuSans-2e\" transform=\"scale(0.015625)\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-34\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 30.103125 92.61683 \nL 225.403125 92.61683 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m5364f498e3\" y=\"92.61683\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.6 -->\n      <g transform=\"translate(7.2 96.416049)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-36\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_15\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 30.103125 66.358931 \nL 225.403125 66.358931 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m5364f498e3\" y=\"66.358931\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.8 -->\n      <g transform=\"translate(7.2 70.158149)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-30\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-38\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 30.103125 40.101031 \nL 225.403125 40.101031 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m5364f498e3\" y=\"40.101031\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 43.90025)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-30\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_19\">\n      <path clip-path=\"url(#pa75969a859)\" d=\"M 30.103125 13.843132 \nL 225.403125 13.843132 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_20\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m5364f498e3\" y=\"13.843132\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 1.2 -->\n      <g transform=\"translate(7.2 17.642351)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-31\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-2e\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-32\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_21\">\n    <path clip-path=\"url(#pa75969a859)\" d=\"M 12.743125 13.377273 \nL 17.083125 44.900063 \nL 21.423125 59.732598 \nL 25.763125 68.803932 \nL 30.103125 75.240211 \nL 34.443125 105.884058 \nL 38.783125 105.002605 \nL 43.123125 107.419428 \nL 47.463125 108.231763 \nL 51.803125 109.230641 \nL 56.143125 114.543468 \nL 60.483125 117.124425 \nL 64.823125 117.666476 \nL 69.163125 117.920761 \nL 73.503125 118.588435 \nL 77.843125 123.704095 \nL 82.183125 123.101705 \nL 86.523125 123.704915 \nL 90.863125 123.685907 \nL 95.203125 124.026883 \nL 99.543125 127.366095 \nL 103.883125 127.782386 \nL 108.223125 127.682727 \nL 112.563125 127.778292 \nL 116.903125 127.718312 \nL 121.243125 129.923565 \nL 125.583125 129.850492 \nL 129.923125 130.481009 \nL 134.263125 130.812406 \nL 138.603125 130.580723 \nL 142.943125 133.442904 \nL 147.283125 132.736571 \nL 151.623125 132.662576 \nL 155.963125 132.626737 \nL 160.303125 132.617967 \nL 164.643125 132.198538 \nL 168.983125 132.764385 \nL 173.323125 133.097689 \nL 177.663125 133.673915 \nL 182.003125 133.823114 \nL 186.343125 135.475741 \nL 190.683125 135.576659 \nL 195.023125 135.672777 \nL 199.363125 135.271427 \nL 203.703125 135.411201 \nL 208.043125 136.922727 \nL 212.383125 136.901237 \nL 216.723125 136.7311 \nL 221.063125 136.659528 \nL 225.403125 136.82133 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_22\">\n    <path clip-path=\"url(#pa75969a859)\" d=\"M 12.743125 94.048444 \nL 17.083125 84.326125 \nL 21.423125 79.794135 \nL 25.763125 76.627926 \nL 30.103125 74.417918 \nL 34.443125 64.226786 \nL 38.783125 64.537769 \nL 43.123125 63.444781 \nL 47.463125 63.146528 \nL 51.803125 62.835996 \nL 56.143125 60.778691 \nL 60.483125 59.911211 \nL 64.823125 59.774815 \nL 69.163125 59.717528 \nL 73.503125 59.450915 \nL 77.843125 57.439712 \nL 82.183125 57.74524 \nL 86.523125 57.530643 \nL 90.863125 57.420617 \nL 95.203125 57.313084 \nL 99.543125 56.435837 \nL 103.883125 55.933899 \nL 108.223125 56.05029 \nL 112.563125 56.015736 \nL 116.903125 56.043953 \nL 121.243125 55.322844 \nL 125.583125 55.050052 \nL 129.923125 54.882739 \nL 134.263125 54.760892 \nL 138.603125 54.963002 \nL 142.943125 53.991617 \nL 147.283125 54.220763 \nL 151.623125 54.286233 \nL 155.963125 54.25077 \nL 160.303125 54.229969 \nL 164.643125 54.537202 \nL 168.983125 54.286233 \nL 173.323125 54.202577 \nL 177.663125 53.983433 \nL 182.003125 53.882052 \nL 186.343125 53.195064 \nL 190.683125 53.020476 \nL 195.023125 53.155054 \nL 199.363125 53.366923 \nL 203.703125 53.286873 \nL 208.043125 52.39851 \nL 212.383125 52.507627 \nL 216.723125 52.605832 \nL 221.063125 52.755868 \nL 225.403125 52.691694 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_23\">\n    <path clip-path=\"url(#pa75969a859)\" d=\"M 30.103125 69.221042 \nL 51.803125 91.724062 \nL 73.503125 62.013248 \nL 95.203125 65.466162 \nL 116.903125 64.323944 \nL 138.603125 62.774727 \nL 160.303125 79.028367 \nL 182.003125 67.067894 \nL 203.703125 61.107351 \nL 225.403125 54.752939 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 143.1 \nL 30.103125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 143.1 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 143.1 \nL 225.403125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 7.2 \nL 225.403125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 37.103125 59.234375 \nL 114.871875 59.234375 \nQ 116.871875 59.234375 116.871875 57.234375 \nL 116.871875 14.2 \nQ 116.871875 12.2 114.871875 12.2 \nL 37.103125 12.2 \nQ 35.103125 12.2 35.103125 14.2 \nL 35.103125 57.234375 \nQ 35.103125 59.234375 37.103125 59.234375 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_24\">\n     <path d=\"M 39.103125 20.298438 \nL 59.103125 20.298438 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_25\"/>\n    <g id=\"text_12\">\n     <!-- train loss -->\n     <g transform=\"translate(67.103125 23.798438)scale(0.1 -0.1)\">\n      <defs>\n       <path d=\"M 1172 4494 \nL 1172 3500 \nL 2356 3500 \nL 2356 3053 \nL 1172 3053 \nL 1172 1153 \nQ 1172 725 1289 603 \nQ 1406 481 1766 481 \nL 2356 481 \nL 2356 0 \nL 1766 0 \nQ 1100 0 847 248 \nQ 594 497 594 1153 \nL 594 3053 \nL 172 3053 \nL 172 3500 \nL 594 3500 \nL 594 4494 \nL 1172 4494 \nz\n\" id=\"DejaVuSans-74\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2631 2963 \nQ 2534 3019 2420 3045 \nQ 2306 3072 2169 3072 \nQ 1681 3072 1420 2755 \nQ 1159 2438 1159 1844 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1341 3275 1631 3429 \nQ 1922 3584 2338 3584 \nQ 2397 3584 2469 3576 \nQ 2541 3569 2628 3553 \nL 2631 2963 \nz\n\" id=\"DejaVuSans-72\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2194 1759 \nQ 1497 1759 1228 1600 \nQ 959 1441 959 1056 \nQ 959 750 1161 570 \nQ 1363 391 1709 391 \nQ 2188 391 2477 730 \nQ 2766 1069 2766 1631 \nL 2766 1759 \nL 2194 1759 \nz\nM 3341 1997 \nL 3341 0 \nL 2766 0 \nL 2766 531 \nQ 2569 213 2275 61 \nQ 1981 -91 1556 -91 \nQ 1019 -91 701 211 \nQ 384 513 384 1019 \nQ 384 1609 779 1909 \nQ 1175 2209 1959 2209 \nL 2766 2209 \nL 2766 2266 \nQ 2766 2663 2505 2880 \nQ 2244 3097 1772 3097 \nQ 1472 3097 1187 3025 \nQ 903 2953 641 2809 \nL 641 3341 \nQ 956 3463 1253 3523 \nQ 1550 3584 1831 3584 \nQ 2591 3584 2966 3190 \nQ 3341 2797 3341 1997 \nz\n\" id=\"DejaVuSans-61\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 603 3500 \nL 1178 3500 \nL 1178 0 \nL 603 0 \nL 603 3500 \nz\nM 603 4863 \nL 1178 4863 \nL 1178 4134 \nL 603 4134 \nL 603 4863 \nz\n\" id=\"DejaVuSans-69\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 3513 2113 \nL 3513 0 \nL 2938 0 \nL 2938 2094 \nQ 2938 2591 2744 2837 \nQ 2550 3084 2163 3084 \nQ 1697 3084 1428 2787 \nQ 1159 2491 1159 1978 \nL 1159 0 \nL 581 0 \nL 581 3500 \nL 1159 3500 \nL 1159 2956 \nQ 1366 3272 1645 3428 \nQ 1925 3584 2291 3584 \nQ 2894 3584 3203 3211 \nQ 3513 2838 3513 2113 \nz\n\" id=\"DejaVuSans-6e\" transform=\"scale(0.015625)\"/>\n       <path id=\"DejaVuSans-20\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 603 4863 \nL 1178 4863 \nL 1178 0 \nL 603 0 \nL 603 4863 \nz\n\" id=\"DejaVuSans-6c\" transform=\"scale(0.015625)\"/>\n       <path d=\"M 2834 3397 \nL 2834 2853 \nQ 2591 2978 2328 3040 \nQ 2066 3103 1784 3103 \nQ 1356 3103 1142 2972 \nQ 928 2841 928 2578 \nQ 928 2378 1081 2264 \nQ 1234 2150 1697 2047 \nL 1894 2003 \nQ 2506 1872 2764 1633 \nQ 3022 1394 3022 966 \nQ 3022 478 2636 193 \nQ 2250 -91 1575 -91 \nQ 1294 -91 989 -36 \nQ 684 19 347 128 \nL 347 722 \nQ 666 556 975 473 \nQ 1284 391 1588 391 \nQ 1994 391 2212 530 \nQ 2431 669 2431 922 \nQ 2431 1156 2273 1281 \nQ 2116 1406 1581 1522 \nL 1381 1569 \nQ 847 1681 609 1914 \nQ 372 2147 372 2553 \nQ 372 3047 722 3315 \nQ 1072 3584 1716 3584 \nQ 2034 3584 2315 3537 \nQ 2597 3491 2834 3397 \nz\n\" id=\"DejaVuSans-73\" transform=\"scale(0.015625)\"/>\n      </defs>\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-69\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-6c\"/>\n      <use x=\"292.333984\" xlink:href=\"#DejaVuSans-6f\"/>\n      <use x=\"353.515625\" xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"405.615234\" xlink:href=\"#DejaVuSans-73\"/>\n     </g>\n    </g>\n    <g id=\"line2d_26\">\n     <path d=\"M 39.103125 34.976562 \nL 59.103125 34.976562 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_27\"/>\n    <g id=\"text_13\">\n     <!-- train acc -->\n     <g transform=\"translate(67.103125 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-72\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-69\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-6e\"/>\n      <use x=\"232.763672\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"264.550781\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"325.830078\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"380.810547\" xlink:href=\"#DejaVuSans-63\"/>\n     </g>\n    </g>\n    <g id=\"line2d_28\">\n     <path d=\"M 39.103125 49.654688 \nL 59.103125 49.654688 \n\" style=\"fill:none;stroke:#008000;stroke-dasharray:9.6,2.4,1.5,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_29\"/>\n    <g id=\"text_14\">\n     <!-- test acc -->\n     <g transform=\"translate(67.103125 53.154688)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-65\"/>\n      <use x=\"100.732422\" xlink:href=\"#DejaVuSans-73\"/>\n      <use x=\"152.832031\" xlink:href=\"#DejaVuSans-74\"/>\n      <use x=\"192.041016\" xlink:href=\"#DejaVuSans-20\"/>\n      <use x=\"223.828125\" xlink:href=\"#DejaVuSans-61\"/>\n      <use x=\"285.107422\" xlink:href=\"#DejaVuSans-63\"/>\n      <use x=\"340.087891\" xlink:href=\"#DejaVuSans-63\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pa75969a859\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"30.103125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=13}
%%tab tensorflow
lr, num_epochs, batch_size = 1.0, 10, 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
net = d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

```{.json .output n=13}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

Let's [**have a look at the scale parameter `gamma`
and the shift parameter `beta`**] learned
from the first batch normalization layer.

```{.python .input  n=14}
%%tab mxnet
net[1].gamma.data().reshape(-1,), net[1].beta.data().reshape(-1,)
```

```{.json .output n=14}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=15}
%%tab pytorch
net[1].gamma.reshape((-1,)), net[1].beta.reshape((-1,))
```

```{.json .output n=15}
[
 {
  "data": {
   "text/plain": "(tensor([3.1785, 3.4545, 2.9430, 1.6547, 3.8373, 1.5187],\n        grad_fn=<ViewBackward>),\n tensor([-0.0658, -3.9745, -2.1778,  1.6972,  2.5361,  1.7197],\n        grad_fn=<ViewBackward>))"
  },
  "execution_count": 15,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=16}
%%tab tensorflow
tf.reshape(net.layers[1].gamma, (-1,)), tf.reshape(net.layers[1].beta, (-1,))
```

```{.json .output n=16}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

## [**Concise Implementation**]

Compared with the `BatchNorm` class,
which we just defined ourselves,
we can use the `BatchNorm` class defined in high-level APIs from the deep learning framework directly.
The code looks virtually identical
to our implementation above.

```{.python .input}
net = nn.Sequential()
net.add(nn.Conv2D(6, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Conv2D(16, kernel_size=5),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.AvgPool2D(pool_size=2, strides=2),
        nn.Dense(120),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(84),
        nn.BatchNorm(),
        nn.Activation('sigmoid'),
        nn.Dense(10))
```

```{.python .input}
#@tab pytorch
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5), nn.BatchNorm2d(6), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5), nn.BatchNorm2d(16), nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2), nn.Flatten(),
    nn.Linear(256, 120), nn.BatchNorm1d(120), nn.Sigmoid(),
    nn.Linear(120, 84), nn.BatchNorm1d(84), nn.Sigmoid(),
    nn.Linear(84, 10))
```

```{.python .input}
#@tab tensorflow
def net():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                               input_shape=(28, 28, 1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('sigmoid'),
        tf.keras.layers.Dense(10),
    ])
```

Below, we [**use the same hyperparameters to train our model.**]
Note that as usual, the high-level API variant runs much faster
because its code has been compiled to C++ or CUDA
while our custom implementation must be interpreted by Python.

```{.python .input}
#@tab all
d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
```

## Controversy

Intuitively, batch normalization is thought
to make the optimization landscape smoother.
However, we must be careful to distinguish between
speculative intuitions and true explanations
for the phenomena that we observe when training deep models.
Recall that we do not even know why simpler
deep neural networks (MLPs and conventional CNNs)
generalize well in the first place.
Even with dropout and weight decay,
they remain so flexible that their ability to generalize to unseen data
cannot be explained via conventional learning-theoretic generalization guarantees.

In the original paper proposing batch normalization,
the authors, in addition to introducing a powerful and useful tool,
offered an explanation for why it works:
by reducing *internal covariate shift*.
Presumably by *internal covariate shift* the authors
meant something like the intuition expressed above---the
notion that the distribution of variable values changes
over the course of training.
However, there were two problems with this explanation:
i) This drift is very different from *covariate shift*,
rendering the name a misnomer.
ii) The explanation offers an under-specified intuition
but leaves the question of *why precisely this technique works*
an open question wanting for a rigorous explanation.
Throughout this book, we aim to convey the intuitions that practitioners
use to guide their development of deep neural networks.
However, we believe that it is important
to separate these guiding intuitions
from established scientific fact.
Eventually, when you master this material
and start writing your own research papers
you will want to be clear to delineate
between technical claims and hunches.

Following the success of batch normalization,
its explanation in terms of *internal covariate shift*
has repeatedly surfaced in debates in the technical literature
and broader discourse about how to present machine learning research.
In a memorable speech given while accepting a Test of Time Award
at the 2017 NeurIPS conference,
Ali Rahimi used *internal covariate shift*
as a focal point in an argument likening
the modern practice of deep learning to alchemy.
Subsequently, the example was revisited in detail
in a position paper outlining
troubling trends in machine learning :cite:`Lipton.Steinhardt.2018`.
Other authors
have proposed alternative explanations for the success of batch normalization,
some claiming that batch normalization's success comes despite exhibiting behavior
that is in some ways opposite to those claimed in the original paper :cite:`Santurkar.Tsipras.Ilyas.ea.2018`.

We note that the *internal covariate shift*
is no more worthy of criticism than any of
thousands of similarly vague claims
made every year in the technical machine learning literature.
Likely, its resonance as a focal point of these debates
owes to its broad recognizability to the target audience.
Batch normalization has proven an indispensable method,
applied in nearly all deployed image classifiers,
earning the paper that introduced the technique
tens of thousands of citations.


## Summary

* During model training, batch normalization continuously adjusts the intermediate output of the neural network by utilizing the mean and standard deviation of the minibatch, so that the values of the intermediate output in each layer throughout the neural network are more stable.
* The batch normalization methods for fully connected layers and convolutional layers are slightly different.
* Like a dropout layer, batch normalization layers have different computation results in training mode and prediction mode.
* Batch normalization has many beneficial side effects, primarily that of regularization. On the other hand, the original motivation of reducing internal covariate shift seems not to be a valid explanation.

## Exercises

1. Can we remove the bias parameter from the fully connected layer or the convolutional layer before the batch normalization? Why?
1. Compare the learning rates for LeNet with and without batch normalization.
    1. Plot the increase in training and test accuracy.
    1. How large can you make the learning rate?
1. Do we need batch normalization in every layer? Experiment with it?
1. Can you replace dropout by batch normalization? How does the behavior change?
1. Fix the parameters `beta` and `gamma`, and observe and analyze the results.
1. Review the online documentation for `BatchNorm` from the high-level APIs to see the other applications for batch normalization.
1. Research ideas: think of other normalization transforms that you can apply? Can you apply the probability integral transform? How about a full rank covariance estimate?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
