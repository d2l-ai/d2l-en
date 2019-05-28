# Generative Adversarial Networks
:label:`chapter_basic_gan`

Throughout most of this book, we've talked about how to make predictions. In some form or another, we used deep neural networks learned mappings from data points to labels. This kind of learning is called discriminative learning, as in, we'd like to be able to discriminate between photos cats and photos of dogs. Classifiers and regressors are both examples of discriminative learning. And neural networks trained by backpropagation have upended everything we thought we knew about discriminative learning on large complicated datasets. Classification accuracies on high-res images has gone from useless to human-level (with some caveats) in just 5-6 years. We'll spare you another spiel about all the other discriminative tasks where deep neural networks do astoundingly well.

But there's more to machine learning than just solving discriminative tasks. For example, given a large dataset, without any labels, we might want to learn a model that concisely captures the characteristics of this data. Given such a model, we could sample synthetic data points that resemble the distribution of the training data. For example, given a large corpus of photographs of faces, we might want to be able to generate a new photorealistic image that looks like it might plausibly have come from the same dataset. This kind of learning is called generative modeling.

Until recently, we had no method that could synthesize novel photorealistic images. But the success of deep neural networks for discriminative learning opened up new possibilities. One big trend over the last three years has been the application of discriminative deep nets to overcome challenges in problems that we don't generally think of as supervised learning problems. The recurrent neural network language models are one example of using a discriminative network (trained to predict the next character) that once trained can act as a generative model.

In 2014, a breakthrough paper introduced Generative Adversarial Networks (GANs) :cite:`Goodfellow.Pouget-Abadie.Mirza.ea.2014`, a clever new way to leverage the power of discriminative models to get good generative models. At their heart, GANs rely on the idea that a data generator is good if we cannot tell fake data apart from real data. In statistics, this is called a two-sample test - a test to answer the question whether datasets $X=\{x_1,\ldots,x_n\}$ and $X'=\{x'_1,\ldots,x'_n\}$ were drawn from the same distribution. The main difference between most statistics papers and GANs is that the latter use this idea in a constructive way. In other words, rather than just training a model to say "hey, these two datasets don't look like they came from the same distribution", they use the [two-sample test](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) to provide training signal to a generative model. This allows us to improve the data generator until it generates something that resembles the real data. At the very least, it needs to fool the classifier. And if our classifier is a state of the art deep neural network.

![Generative Adversarial Networks](../img/gan.svg)
:label:`fig:gan`

The GANs architecture is illustrated in :numref:`fig:gan`.
As you can see, there are two pieces to GANs - first off, we need a device (say, a deep network but it really could be anything, such as a game rendering engine) that might potentially be able to generate data that looks just like the real thing. If we are dealing with images, this needs to generate images. If we're dealing with speech, it needs to generate audio sequences, and so on. We call this the generator network. The second component is the discriminator network. It attempts to distinguish fake and real data from each other. Both networks are in competition with each other. The generator network attempts to fool the discriminator network. At that point, the discriminator network adapts to the new fake data. This information, in turn is used to improve the generator network, and so on.

The discriminator is a binary classifier to distinguish if the input $x$ is real (from real data) or fake (from the generator). Typically, the discriminator outputs a scalar prediction $o\in\mathbb R$ for input $\mathbf x$, such as using a dense layer with hidden size 1, and then applies sigmoid function to obtain the predicted probability $D(\mathbf x) = 1/(1+e^{-o})$. Assume the label $y$ for true data is $1$ and $0$ for fake data. We train the discriminator to minimize the cross entropy loss, i.e.

$$ \min - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x)),$$

For the generator, it first draws some parameter $\mathbf z\in\mathbb R^d$ from a source of randomness, e.g. a normal distribution $\mathbf z\sim\mathcal(0,1)$. We often call $\mathbf z$ the latent variable.
It then applies a function to generate $\mathbf x'=G(\mathbf z)$. The goal of the generator is to fool the discriminator to classify $\mathbf x'$ as true data. In other words, we update the parameters of the generator to maximize the cross entropy loss when $y=0$, i.e.

$$ \max - \log(1-D(\mathbf x')).$$

If the discriminator does a perfect job, then $D(\mathbf x')\approx 1$ so the above loss near 0, which results the gradients are too small to make a good progress for the discriminator. So commonly we minimize the following loss

$$ \max \log(D(\mathbf x')), $$

which is just feed $\mathbf x'$ into the discriminator but giving label $y=1$.


Many of the GANs applications are in the context of images. As a demonstration purpose, we're going to content ourselves with fitting a much simpler distribution first. We will illustrate what happens if we use GANs to build the world's most inefficient estimator of parameters for a Gaussian. Let's get started.

```{.python .input  n=1}
%matplotlib inline

import d2l
from mxnet import nd, gluon, autograd, init
from mxnet.gluon import nn
from IPython import display
```

## Generate some "real" data

Since this is going to be the world's lamest example, we simply generate data drawn from a Gaussian.

```{.python .input  n=2}
X = nd.random.normal(shape=(1000, 2))
A = nd.array([[1, 2], [-0.1, 0.5]])
b = nd.array([1, 2])
data = nd.dot(X, A) + b
```

Let's see what we got. This should be a Gaussian shifted in some rather arbitrary way with mean $b$ and covariance matrix $A^TA$.

```{.python .input  n=3}
d2l.set_figsize((6, 3))
d2l.plt.scatter(data[:100,0].asnumpy(), data[:100,1].asnumpy());
print("The covariance matrix is", nd.dot(A.T,A))
```

```{.python .input  n=4}
batch_size = 8
dataset = gluon.data.ArrayDataset(data)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)
```

## Generator

Our generator network will be the simplest network possible - a single layer linear model. This is since we'll be driving that linear network with a Gaussian data generator. Hence, it literally only needs to learn the parameters to fake things perfectly.

```{.python .input  n=5}
net_G = nn.Sequential()
net_G.add(nn.Dense(2))
```

## Discriminator

For the discriminator we will be a bit more discriminating: we will use an MLP with 3 layers to make things a bit more interesting.

```{.python .input  n=6}
net_D = nn.Sequential()
net_D.add(nn.Dense(5, activation='tanh'),
          nn.Dense(3, activation='tanh'),
          nn.Dense(1))
```

## Training

First we define a function to update the discriminator.

```{.python .input  n=7}
def update_D(X, Z, net_D, net_G, loss, trainer_D):
    """Update discriminator"""
    batch_size = X.shape[0]
    ones = nd.ones((batch_size,), ctx=X.context)
    zeros = nd.zeros((batch_size,), ctx=X.context)
    with autograd.record():
        real_Y = net_D(X)
        fake_X = net_G(Z)
        # Don't need to compute gradient for net_G, detach it from
        # computing gradients.
        fake_Y = net_D(fake_X.detach())
        loss_D = (loss(real_Y, ones) + loss(fake_Y, zeros)) / 2
    loss_D.backward()
    trainer_D.step(batch_size)
    return loss_D.mean().asscalar()
```

The generator is updated similarly. Here we reuse the cross entropy loss but change the label of the fake data from $0$ to $1$.

```{.python .input  n=8}
def update_G(Z, net_D, net_G, loss, trainer_G):
    """Update generator"""
    batch_size = Z.shape[0]
    ones = nd.ones((batch_size,), ctx=X.context)
    with autograd.record():
        # We could reuse fake_X from update_D to save computation.
        fake_X = net_G(Z)
        # Recomputing fake_Y is needed since net_D is changed.
        fake_Y = net_D(fake_X)
        loss_G = loss(fake_Y, ones)
    loss_G.backward()
    trainer_G.step(batch_size)
    return loss_G.mean().asscalar()
```

In each iteration, we first update the discriminator and then the generator. We visualize losses for each network, and also show the generated data from the generator.

```{.python .input  n=9}
def train():
    all_loss_D, all_loss_G = [], []
    fig, (ax1, ax2) = d2l.plt.subplots(2, 1, figsize=(6,6))
    fig.subplots_adjust(hspace=0.3)
    for epoch in range(1, num_epochs+1):
        total_loss_D, total_loss_G = 0.0, 0.0
        for i, X in enumerate(data_iter):
            Z = nd.random.normal(0, 1, shape=(batch_size, latent_dim))
            total_loss_D += update_D(X, Z, net_D, net_G, loss, trainer_D)
            total_loss_G += update_G(Z, net_D, net_G, loss, trainer_G)
        # Show progress.
        all_loss_D.append(total_loss_D/len(data_iter))
        all_loss_G.append(total_loss_G/len(data_iter))
        d2l.plot(list(range(1, epoch+1)), [all_loss_G, all_loss_D],
                 'epoch', 'loss', ['generator', 'discriminator'],
                 xlim=[0, num_epochs+1], axes=ax1)
        # Show generated examples
        Z = nd.random.normal(0, 1, shape=(100, latent_dim))
        fake_X = net_G(Z)
        d2l.scatter([data[:100,0], fake_X[:,0]], [data[:100,1], fake_X[:,1]],
                    'x', 'y', ['real', 'generated'], axes=ax2)
        d2l.show(fig)
```

Now we specify the hyper-parameters to fit the Gaussian distribution.

```{.python .input  n=10}
lr_D = 0.05
lr_G = 0.005
latent_dim = 2
num_epochs = 20

loss = gluon.loss.SigmoidBCELoss()
net_D.initialize(init=init.Normal(0.02), force_reinit=True)
net_G.initialize(init=init.Normal(0.02), force_reinit=True)
trainer_D = gluon.Trainer(net_D.collect_params(),
                          'adam', {'learning_rate': lr_D})
trainer_G = gluon.Trainer(net_G.collect_params(),
                          'adam', {'learning_rate': lr_G})

train()
```

## Summary

* Generative Adversarial Networks (GANs) composes of two deep networks, the generator and the discriminator.
* The generator generates the image as much closer to the true image as possible to fool the discriminator, via maximizing the cross entropy loss, i.e. $ \max \log(D(\mathbf x'))$. 
* The discriminator tries to distinguish the generated images from the true images, via minimizing the cross entropy loss, i.e. $ \min - y \log D(\mathbf x) - (1-y)\log(1-D(\mathbf x))$.


## Reference

[1] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. In Advances in neural information processing systems, 2672–2680. 2014.