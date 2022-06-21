# Working with Sequences
:label:`sec_sequence`



Up until now, we've focused on models whose inputs 
consisted of a single feature vector $\mathbf{x} \in \mathbb{R}^d$.
The main change of perspective when developing models
capable of processing sequences is that we now 
focus on inputs that consist of an ordered list 
of feature vectors $\mathbf{x}_1, \dots, \mathbf{x}_T$,
where each feature vector $x_t$
indexed by a sequence step $t \in \mathbb{Z}^+$
lies in $\mathbb{R}^d$.

Some datasets consist of a single massive sequence.
Consider, for example, the extremely long streams
of sensor readings that might be available to climate scientists. 
In such cases, we might create training datasets
by randomly sampling subsequences of some predetermined length.
More often, our data arrive as a collection of sequences.
Consider the following examples: 
(i) a collection of documents,
each represented as its own sequence of words,
and each having its own length $T_i$;
(ii) sequence representation of 
patient stays in the hospital,
where each stay consists of a number of events
and the sequence length depends roughly 
on the length of the stay.


Previously, when dealing with individual inputs,
we assumed that they were sampled independently 
from the same underlying distribution $P(X)$.
While we still assume that entire sequences 
(e.g., entire documents or patient trajectories)
are sampled independently,
we cannot assume that the data arriving 
at each sequence step are independent of each other. 
For example, what words are likely to appear later in a document
depends heavily on what words occurred earlier in the document. 
What medicine a patient is likely to receive 
on the 10th day of a hospital visit 
depends heavily on what transpired 
in the previous nine days. 

This should come as no surprise.
If we didn't believe that the elements in a sequence were related,
we wouldn't have bothered to model them as a sequence in the first place. 
Consider the usefulness of the auto-fill features
that are popular on search tools and modern email clients.
They are useful precisely because it is often possible 
to predict (imperfectly, but better than random guessing)
what likely continuations of a sequence might be,
given some initial prefix. 
For most sequence models,
we don't require independence,
or even stationarity, of our sequences. 
Instead, we require only that 
the sequences themselves are sampled 
from some fixed underlying distribution 
over entire sequences. 

This flexible approach, allows for such phenomena
as (i) documents looking significantly different 
at the beginning than at the end,
or (ii) patient status evolving either 
towards recovery or towards death 
over the course of a hospital stay;
and (iii) customer taste evolving in predictable ways
over course of continued interaction with a recommender system.


We sometimes wish to predict a fixed target $y$
given sequentially structured input
(e.g., sentiment classification based on a movie review). 
At other times, we wish to predict a sequentially structured target
($y_1, \cdots, y_T$)
given a fixed input (e.g., image captioning).
Still other times, out goal is to predict sequentially structured targets
based on sequentially structured inputs 
(e.g., machine translation or video captioning).
Such sequence-to-sequence tasks take two forms:
(a) **aligned:** where the input at each sequence step
aligns with a corresponding target (e.g., part of speech taggin);
(b) **unaligned** where the input and target 
do not necessarily exhibit a step-for-step correspondence
(e.g. machine translation). 

But before we worry about handling targets of any kind,
we can tackle the most straightforward problem: 
unsupervised density modeling (also called *sequence modeling*).
Here, given a collection of sequences, 
our goal is to estimate the probability mass function
that tells us how likely we are to see any given sequence,
i.e. $p(\mathbf{x}_1, \cdots, \mathbf{x}_T)$.





## Basic Tools


Before we start introducing specialized neural networks 
designed to handle sequentially structured data,
let's focus.

We need statistical tools and new deep neural network architectures to deal with sequence data. 
To keep things simple, we use the stock price (FTSE 100 index) 
illustrated in :numref:`fig_ftse100` as an example.

![FTSE 100 index over about 30 years.](../img/ftse100.png)
:width:`400px`
:label:`fig_ftse100`


Let's denote the prices by $x_t$, i.e., at *time step* 
$t \in \mathbb{Z}^+$, we observe price $x_t$.
Note that for sequences in this text,
$t$ will typically be discrete 
and take only positive integers as values. 
Suppose that
a trader who wants to do well in the stock market on day $t$ predicts $x_t$ via

$$x_t \sim P(x_t \mid x_{t-1}, \ldots, x_1).$$



### Autoregressive Models

In order to achieve this, our trader could use a regression model 
such as the one that we trained in :numref:`sec_linear_concise`.
There is just one major problem: 
the number of inputs, $x_{t-1}, \ldots, x_1$ varies, depending on $t$.
That is to say, the number increases with the amount of data that we encounter, 
and we will need an approximation to make this computationally tractable.
Much of what follows in this chapter will revolve around 
how to estimate $P(x_t \mid x_{t-1}, \ldots, x_1)$ efficiently. 
In a nutshell, it boils down to two strategies, as follows.

First, assume that the potentially rather long sequence 
$x_{t-1}, \ldots, x_1$ is not really necessary.
In this case we might content ourselves 
with some timespan of length $\tau$ 
and only use $x_{t-1}, \ldots, x_{t-\tau}$ observations. 
The immediate benefit is that now the number of arguments 
is always the same, at least for $t > \tau$. 
This allows us to train a deep network as indicated above. 
Such models will be called *autoregressive models*, 
as they quite literally perform regression on themselves.

The second strategy, shown in :numref:`fig_sequence-model`, 
is to keep some summary $h_t$ of the past observations, 
and at the same time update $h_t$ in addition to the prediction $\hat{x}_t$.
This leads to models that estimate $x_t$ with $\hat{x}_t = P(x_t \mid h_{t})$ 
and moreover updates of the form  $h_t = g(h_{t-1}, x_{t-1})$. 
Since $h_t$ is never observed, these models are also called *latent autoregressive models*.

![A latent autoregressive model.](../img/sequence-model.svg)
:label:`fig_sequence-model`

Both cases raise the obvious question 
of how to generate training data. 
One typically uses historical observations to predict 
the next observation given the ones up to right now. 
Obviously we do not expect time to stand still. 
However, a common assumption is that while 
the specific values of $x_t$ might change, 
at least the dynamics of the sequence itself will not. 
This is reasonable, since novel dynamics are just that, novel,
and thus not predictable using data that we have so far. 
Statisticians call dynamics that do not change *stationary*.
Regardless of what we do, we will thus get an estimate of the entire sequence via

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}, \ldots, x_1).$$

Note that the above considerations still hold 
if we deal with discrete objects, such as words, 
rather than continuous numbers. 
The only difference is that in such a situation 
we need to use a classifier rather than a regression model 
to estimate $P(x_t \mid  x_{t-1}, \ldots, x_1)$.


### Markov Models
:label:`subsec_markov-models`

Recall the approximation that in an autoregressive model
we use only $x_{t-1}, \ldots, x_{t-\tau}$,
instead of $x_{t-1}, \ldots, x_1$ to estimate $x_t$. 
Whenever this approximation is accurate we say 
that the sequence satisfies a *Markov condition*. 
In particular, if $\tau = 1$, we have a *first-order Markov model*,
and $P(x)$ is given by

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t \mid x_{t-1}) \text{ where } P(x_1 \mid x_0) = P(x_1).$$

Such models are particularly nice whenever $x_t$ assumes only a discrete value, 
since in this case dynamic programming can be used 
to compute values along the chain exactly. 
For instance, we can compute $P(x_{t+1} \mid x_{t-1})$ efficiently:

$$\begin{aligned}
P(x_{t+1} \mid x_{t-1})
&= \frac{\sum_{x_t} P(x_{t+1}, x_t, x_{t-1})}{P(x_{t-1})}\\
&= \frac{\sum_{x_t} P(x_{t+1} \mid x_t, x_{t-1}) P(x_t, x_{t-1})}{P(x_{t-1})}\\
&= \sum_{x_t} P(x_{t+1} \mid x_t) P(x_t \mid x_{t-1})
\end{aligned}
$$

by using the fact that we only need to take into account 
for a very short history of past observations: 
$P(x_{t+1} \mid x_t, x_{t-1}) = P(x_{t+1} \mid x_t)$.
Going into details of dynamic programming 
is beyond the scope of this section. 
Control and reinforcement learning algorithms use such tools extensively.



### Causality

<!--  cut? -->
In principle, there is nothing wrong with unfolding 
$P(x_1, \ldots, x_T)$ in reverse order. 
After all, by conditioning we can always write it via

$$P(x_1, \ldots, x_T) = \prod_{t=T}^1 P(x_t \mid x_{t+1}, \ldots, x_T).$$

In fact, if we have a Markov model, we can obtain 
a reverse conditional probability distribution, too.
In many cases, however, there exists a natural direction for the data,
namely going forward in time. 
It is clear that future events cannot influence the past. 
Hence, if we change $x_t$, we may be able to influence 
what happens for $x_{t+1}$ going forward but not the converse. 
That is, if we change $x_t$, the distribution over past events will not change. 
Consequently, it ought to be easier to explain $P(x_{t+1} \mid x_t)$ 
rather than $P(x_t \mid x_{t+1})$. For instance, 
it has been shown that in some cases we can find 
$x_{t+1} = f(x_t) + \epsilon$ for some additive noise $\epsilon$, 
whereas the converse is not true :cite:`Hoyer.Janzing.Mooij.ea.2009`. 
This is great news, since it is typically the forward direction 
that we are interested in estimating.
The book by Peters et al. has explained more on this topic 
:cite:`Peters.Janzing.Scholkopf.2017`.
We are barely scratching the surface of it.


## Training
<!-- fix -->
After reviewing many different statistical tools, let's try this out in practice.

```{.python .input  n=6}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input  n=7}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon, init
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input  n=8}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input  n=9}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

We begin by generating some data.
To keep things simple we 
(**generate our sequence data by using a sine function
with some additive noise for time steps $1, 2, \ldots, 1000$.**)

```{.python .input  n=10}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = d2l.arange(1, T + 1, dtype=d2l.float32)
        if tab.selected('mxnet', 'pytorch'):
            self.x = d2l.sin(0.01 * self.time) + d2l.randn(T) * 0.2
        if tab.selected('tensorflow'):    
            self.x = d2l.sin(0.01 * self.time) + d2l.normal([T]) * 0.2

data = Data()
d2l.plot(data.time, data.x, 'time', 'x', xlim=[1, 1000], figsize=(6, 3))
```

Next, we need to turn such a sequence into features and labels 
<!-- language -->
that our model can train on.
With a Markov assumption that $x_t$ only depends 
on observations at the past $\tau$ time steps,
we [**construct examples with labels $y_t = x_t$ and features 
$\mathbf{x}_t = [x_{t-\tau}, \ldots, x_{t-1}]$.**]
The astute reader might have noticed that 
this gives us $\tau$ fewer data examples,
since we do not have sufficient history for $y_1, \ldots, y_\tau$. 
While we could pad the first $\tau$ sequences with zeros,
to keep things simple, we drop them for now. 
The resulting dataset contains $T - \tau$ examples,
where each input to the model
has sequence length $\tau$.
We (**create a data iterator on the first 600 examples**),
covering a period of the sine function.

```{.python .input}
%%tab all
@d2l.add_to_class(Data)
def get_dataloader(self, train):
    features = [self.x[i : self.T-self.tau+i] for i in range(self.tau)]
    self.features = d2l.stack(features, 1)
    self.labels = d2l.reshape(self.x[self.tau:], (-1, 1))
    i = slice(0, self.num_train) if train else slice(self.num_train, None)
    return self.get_tensorloader([self.features, self.labels], train, i)
```

The model to train is simple: just linear regression.

```{.python .input}
%%tab all
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

## Prediction

Let's see how well the model predicts. 
The first thing to check is 
[**predicting what happens just in the next time step**],
namely the *one-step-ahead prediction*.

```{.python .input}
%%tab all
onestep_preds = d2l.numpy(model(data.features))
d2l.plot(data.time[data.tau:], [data.labels, onestep_preds], 'time', 'x', 
         legend=['labels', '1-step preds'], figsize=(6, 3))
```

The one-step-ahead predictions look nice. 
Even near the end $t=1000$ the predictions still look trustworthy.
However, there is just one little problem to this:
if we observe sequence data only until time step 604 (`n_train + tau`), 
we cannot hope to receive the inputs 
for all the future one-step-ahead predictions.
Instead, we need to use earlier predictions 
as input to our model for these future predictions, 
one step at a time:

$$
\hat{x}_{605} = f(x_{601}, x_{602}, x_{603}, x_{604}), \\
\hat{x}_{606} = f(x_{602}, x_{603}, x_{604}, \hat{x}_{605}), \\
\hat{x}_{607} = f(x_{603}, x_{604}, \hat{x}_{605}, \hat{x}_{606}),\\
\hat{x}_{608} = f(x_{604}, \hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}),\\
\hat{x}_{609} = f(\hat{x}_{605}, \hat{x}_{606}, \hat{x}_{607}, \hat{x}_{608}),\\
\ldots
$$

Generally, for an observed sequence $x_1, \ldots, x_t$, 
its predicted output $\hat{x}_{t+k}$ at time step $t+k$ 
is called the $k$*-step-ahead prediction*. 
Since we have observed up to $x_{604}$, 
its $k$-step-ahead prediction is $\hat{x}_{604+k}$.
In other words, we will have to 
keep on using our own predictions
to make multistep-ahead predictions.
Let's see how well this goes.

```{.python .input}
%%tab mxnet, pytorch
multistep_preds = d2l.zeros(data.T)
multistep_preds[:] = data.x
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i] = model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1)))
multistep_preds = d2l.numpy(multistep_preds)    
```

```{.python .input}
%%tab tensorflow
multistep_preds = tf.Variable(d2l.zeros(data.T))
multistep_preds[:].assign(data.x)
for i in range(data.num_train + data.tau, data.T):
    multistep_preds[i].assign(d2l.reshape(model(
        d2l.reshape(multistep_preds[i-data.tau : i], (1, -1))), ()))
```

```{.python .input}
%%tab all
d2l.plot([data.time[data.tau:], data.time[data.num_train+data.tau:]], 
         [onestep_preds, multistep_preds[data.num_train+data.tau:]], 'time', 
         'x', legend=['1-step preds', 'multistep preds'], figsize=(6, 3))
```

As the above example shows, this is a spectacular failure. 
The predictions decay to a constant 
pretty quickly after a few prediction steps.
Why did the algorithm work so poorly?
This is ultimately due to the fact that the errors build up.
Let's say that after step 1 we have some error $\epsilon_1 = \bar\epsilon$.
Now the *input* for step 2 is perturbed by $\epsilon_1$,
hence we suffer some error in the order of 
$\epsilon_2 = \bar\epsilon + c \epsilon_1$ for some constant $c$, and so on. 
The error can diverge rather rapidly from the true observations. 
This is a common phenomenon. 
For instance, weather forecasts for the next 24 hours tend to be pretty accurate 
but beyond that the accuracy declines rapidly. 
We will discuss methods for improving this 
throughout this chapter and beyond.

Let's [**take a closer look at the difficulties in $k$-step-ahead predictions**]
by computing predictions on the entire sequence for $k = 1, 4, 16, 64$.

```{.python .input}
%%tab all
def k_step_pred(k):
    features = []
    for i in range(data.tau):
        features.append(data.x[i : i+data.T-data.tau-k+1])
    # The (i+tau)-th element stores the (i+1)-step-ahead predictions
    for i in range(k):
        preds = model(d2l.stack(features[i : i+data.tau], 1))
        features.append(d2l.reshape(preds, -1))
    return features[data.tau:]
```

```{.python .input}
%%tab all
steps = (1, 4, 16, 64)
preds = k_step_pred(steps[-1])
d2l.plot(data.time[data.tau+steps[-1]-1:], 
         [d2l.numpy(preds[k-1]) for k in steps], 'time', 'x', 
         legend=[f'{k}-step preds' for k in steps], figsize=(6, 3))
```

This clearly illustrates how the quality of the prediction changes 
as we try to predict further into the future.
While the 4-step-ahead predictions still look good, 
anything beyond that is almost useless.

## Summary

* There is quite a difference in difficulty between interpolation and extrapolation. 
  Consequently, if you have a sequence, always respect 
  the temporal order of the data when training, 
  i.e., never train on future data.
* Sequence models require specialized statistical tools for estimation. 
  Two popular choices are autoregressive models 
  and latent-variable autoregressive models.
* For causal models (e.g., time going forward), 
  estimating the forward direction is typically 
  a lot easier than the reverse direction.
* For an observed sequence up to time step $t$, 
  its predicted output at time step $t+k$ 
  is the $k$*-step-ahead prediction*. 
  As we predict further in time by increasing $k$, 
  the errors accumulate and the quality of the prediction degrades,
  often dramatically.

## Exercises

1. Improve the model in the experiment of this section.
    1. Incorporate more than the past 4 observations? How many do you really need?
    1. How many past observations would you need if there was no noise? Hint: you can write $\sin$ and $\cos$ as a differential equation.
    1. Can you incorporate older observations while keeping the total number of features constant? Does this improve accuracy? Why?
    1. Change the neural network architecture and evaluate the performance. You may train the new model with more epochs. What do you observe?
1. An investor wants to find a good security to buy. 
   He looks at past returns to decide which one is likely to do well. 
   What could possibly go wrong with this strategy?
1. Does causality also apply to text? To which extent?
1. Give an example for when a latent autoregressive model 
   might be needed to capture the dynamic of the data.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/113)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/114)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1048)
:end_tab:
