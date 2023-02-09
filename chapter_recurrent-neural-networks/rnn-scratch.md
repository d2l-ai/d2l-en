# Recurrent Neural Network Implementation from Scratch
:label:`sec_rnn-scratch`

We are now ready to implement an RNN from scratch.
In particular, we will train this RNN to function
as a character-level language model
(see :numref:`sec_rnn`)
and train it on a corpus consisting of 
the entire text of H. G. Wells' *The Time Machine*,
following the data processing steps 
outlined in :numref:`sec_text-sequence`.
We start by loading the dataset.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow', 'jax')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

```{.python .input  n=5}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from flax import linen as nn
import jax
from jax import numpy as jnp
import math
```

## RNN Model

We begin by defining a class 
to implement the RNN model
(:numref:`subsec_rnn_w_hidden_states`).
Note that the number of hidden units `num_hiddens` 
is a tunable hyperparameter.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
class RNNScratch(d2l.Module):  #@save
    """The RNN model implemented from scratch."""
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.W_xh = d2l.randn(num_inputs, num_hiddens) * sigma
            self.W_hh = d2l.randn(
                num_hiddens, num_hiddens) * sigma
            self.b_h = d2l.zeros(num_hiddens)
        if tab.selected('pytorch'):
            self.W_xh = nn.Parameter(
                d2l.randn(num_inputs, num_hiddens) * sigma)
            self.W_hh = nn.Parameter(
                d2l.randn(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))
```

```{.python .input  n=7}
%%tab jax
class RNNScratch(nn.Module):  #@save
    """The RNN model implemented from scratch."""
    num_inputs: int
    num_hiddens: int
    sigma: float = 0.01

    def setup(self):
        self.W_xh = self.param('W_xh', nn.initializers.normal(self.sigma),
                               (self.num_inputs, self.num_hiddens))
        self.W_hh = self.param('W_hh', nn.initializers.normal(self.sigma),
                               (self.num_hiddens, self.num_hiddens))
        self.b_h = self.param('b_h', nn.initializers.zeros, (self.num_hiddens))
```

[**The `forward` method below defines how to compute 
the output and hidden state at any time step,
given the current input and the state of the model
at the previous time step.**]
Note that the RNN model loops through 
the outermost dimension of `inputs`,
updating the hidden state 
one time step at a time.
The model here uses a $\tanh$ activation function (:numref:`subsec_tanh`).

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is None:
        # Initial state with shape: (batch_size, num_hiddens)
        if tab.selected('mxnet'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              ctx=inputs.ctx)
        if tab.selected('pytorch'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens),
                              device=inputs.device)
        if tab.selected('tensorflow'):
            state = d2l.zeros((inputs.shape[1], self.num_hiddens))
    else:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.num_hiddens))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) +
                         d2l.matmul(state, self.W_hh) + self.b_h)
        outputs.append(state)
    return outputs, state
```

```{.python .input  n=9}
%%tab jax
@d2l.add_to_class(RNNScratch)  #@save
def __call__(self, inputs, state=None):
    if state is not None:
        state, = state
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

We can feed a minibatch of input sequences into an RNN model as follows.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

```{.python .input  n=11}
%%tab jax
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
(outputs, state), _ = rnn.init_with_output(d2l.get_key(), X)
```

Let's check whether the RNN model
produces results of the correct shapes
to ensure that the dimensionality 
of the hidden state remains unchanged.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    """Check the length of a list."""
    assert len(a) == n, f'list\'s length {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    """Check the shape of a tensor."""
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

check_len(outputs, num_steps)
check_shape(outputs[0], (batch_size, num_hiddens))
check_shape(state, (batch_size, num_hiddens))
```

## RNN-based Language Model

The following `RNNLMScratch` class defines 
an RNN-based language model,
where we pass in our RNN 
via the `rnn` argument
of the `__init__` method.
When training language models, 
the inputs and outputs are 
from the same vocabulary. 
Hence, they have the same dimension,
which is equal to the vocabulary size.
Note that we use perplexity to evaluate the model. 
As discussed in :numref:`subsec_perplexity`, this ensures 
that sequences of different length are comparable.

```{.python .input}
%%tab pytorch
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        self.W_hq = nn.Parameter(
            d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 

    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input}
%%tab mxnet, tensorflow
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    def __init__(self, rnn, vocab_size, lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        
    def init_params(self):
        if tab.selected('mxnet'):
            self.W_hq = d2l.randn(
                self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma
            self.b_q = d2l.zeros(self.vocab_size)        
            for param in self.get_scratch_params():
                param.attach_grad()
        if tab.selected('tensorflow'):
            self.W_hq = tf.Variable(d2l.normal(
                (self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma)
            self.b_q = tf.Variable(d2l.zeros(self.vocab_size))
        
    def training_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=True)
        return l
        
    def validation_step(self, batch):
        l = self.loss(self(*batch[:-1]), batch[-1])
        self.plot('ppl', d2l.exp(l), train=False)
```

```{.python .input  n=14}
%%tab jax
class RNNLMScratch(d2l.Classifier):  #@save
    """The RNN-based language model implemented from scratch."""
    rnn: nn.Module
    vocab_size: int
    lr: float = 0.01

    def setup(self):
        self.W_hq = self.param('W_hq', nn.initializers.normal(self.rnn.sigma),
                               (self.rnn.num_hiddens, self.vocab_size))
        self.b_q = self.param('b_q', nn.initializers.zeros, (self.vocab_size))

    def training_step(self, params, batch, state):
        value, grads = jax.value_and_grad(
            self.loss, has_aux=True)(params, batch[:-1], batch[-1], state)
        l, _ = value
        self.plot('ppl', d2l.exp(l), train=True)
        return value, grads

    def validation_step(self, params, batch, state):
        l, _ = self.loss(params, batch[:-1], batch[-1], state)
        self.plot('ppl', d2l.exp(l), train=False)
```

### [**One-Hot Encoding**]

Recall that each token is represented 
by a numerical index indicating the
position in the vocabulary of the 
corresponding word/character/word-piece.
You might be tempted to build a neural network
with a single input node (at each time step),
where the index could be fed in as a scalar value.
This works when we are dealing with numerical inputs 
like price or temperature, where any two values
sufficiently close together
should be treated similarly.
But this does not quite make sense. 
The $45^{\mathrm{th}}$ and $46^{\mathrm{th}}$ words 
in our vocabulary happen to be "their" and "said",
whose meanings are not remotely similar.

When dealing with such categorical data,
the most common strategy is to represent
each item by a *one-hot encoding*
(recall from :numref:`subsec_classification-problem`).
A one-hot encoding is a vector whose length
is given by the size of the vocabulary $N$,
where all entries are set to $0$,
except for the entry corresponding 
to our token, which is set to $1$.
For example, if the vocabulary had 5 elements,
then the one-hot vectors corresponding 
to indices 0 and 2 would be the following.

```{.python .input}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

```{.python .input  n=18}
%%tab jax
jax.nn.one_hot(jnp.array([0, 2]), 5)
```

(**The minibatches that we sample at each iteration
will take the shape (batch size, number of time steps).
Once representing each input as a one-hot vector,
we can think of each minibatch as a three-dimensional tensor, 
where the length along the third axis 
is given by the vocabulary size (`len(vocab)`).**)
We often transpose the input so that we will obtain an output 
of shape (number of time steps, batch size, vocabulary size).
This will allow us to more conveniently
loop through the outermost dimension
for updating hidden states of a minibatch,
time step by time step
(e.g., in the above `forward` method).

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def one_hot(self, X):    
    # Output shape: (num_steps, batch_size, vocab_size)    
    if tab.selected('mxnet'):
        return npx.one_hot(X.T, self.vocab_size)
    if tab.selected('pytorch'):
        return F.one_hot(X.T, self.vocab_size).type(torch.float32)
    if tab.selected('tensorflow'):
        return tf.one_hot(tf.transpose(X), self.vocab_size)
    if tab.selected('jax'):
        return jax.nn.one_hot(X.T, self.vocab_size)
```

### Transforming RNN Outputs

The language model uses a fully connected output layer
to transform RNN outputs into token predictions at each time step.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNLMScratch)  #@save
def output_layer(self, rnn_outputs):
    outputs = [d2l.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
    return d2l.stack(outputs, 1)

@d2l.add_to_class(RNNLMScratch)  #@save
def forward(self, X, state=None):
    embs = self.one_hot(X)
    rnn_outputs, _ = self.rnn(embs, state)
    return self.output_layer(rnn_outputs)
```

Let's [**check whether the forward computation
produces outputs with the correct shape.**]

```{.python .input}
%%tab pytorch, mxnet, tensorflow
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

```{.python .input  n=23}
%%tab jax
model = RNNLMScratch(rnn, num_inputs)
outputs, _ = model.init_with_output(d2l.get_key(),
                                    d2l.ones((batch_size, num_steps),
                                             dtype=d2l.int32))
check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradient Clipping**]


While you are already used to thinking of neural networks
as "deep" in the sense that many layers
separate the input and output 
even within a single time step,
the length of the sequence introduces
a new notion of depth.
In addition to the passing through the network
in the input-to-output direction,
inputs at the first time step
must pass through a chain of $T$ layers
along the time steps in order 
to influence the output of the model
at the final time step.
Taking the backwards view, in each iteration,
we backpropagate gradients through time,
resulting in a chain of matrix-products 
with length  $\mathcal{O}(T)$.
As mentioned in :numref:`sec_numerical_stability`, 
this can result in numerical instability, 
causing the gradients to either explode or vanish
depending on the properties of the weight matrices. 

Dealing with vanishing and exploding gradients 
is a fundamental problem when designing RNNs
and has inspired some of the biggest advances
in modern neural network architectures.
In the next chapter, we will talk about
specialized architectures that were designed
in hopes of mitigating the vanishing gradient problem.
However, even modern RNNs still often suffer
from exploding gradients.
One inelegant but ubiquitous solution
is to simply clip the gradients 
forcing the resulting "clipped" gradients
to take smaller values. 


Generally speaking, when optimizing some objective
by gradient descent, we iteratively update
the parameter of interest, say a vector $\mathbf{x}$,
but pushing it in the direction of the 
negative gradient $\mathbf{g}$
(in stochastic gradient descent, 
we calculate this gradient
on a randomly sampled minibatch).
For example, with learning rate $\eta > 0$,
each update takes the form 
$\mathbf{x} \gets \mathbf{x} - \eta \mathbf{g}$.
Let's further assume that the objective function $f$
is sufficiently smooth. 
Formally, we say that the objective 
is *Lipschitz continuous* with constant $L$,
meaning that for any $\mathbf{x}$ and $\mathbf{y}$, we have

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

As you can see, when we update the parameter vector by subtracting $\eta \mathbf{g}$,
the change in the value of the objective
depends on the learning rate,
the norm of the gradient and $L$ as follows:

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|.$$

In other words, the objective cannot
change by more than $L \eta \|\mathbf{g}\|$. 
Having a small value for this upper bound 
might be viewed as a good thing or a bad thing.
On the downside, we are limiting the speed
at which we can reduce the value of the objective.
On the bright side, this limits just how much
we can go wrong in any one gradient step.


When we say that gradients explode, 
we mean that $\|\mathbf{g}\|$ 
becomes excessively large.
In this worst case, we might do so much
damage in a single gradient step that we
could undo all of the progress made over
the course of thousands of training iterations.
When gradients can be so large,
neural network training often diverges,
failing to reduce the value of the objective.
At other times, training eventually converges
but is unstable owing to massive spikes in the loss.


One way to limit the size of $L \eta \|\mathbf{g}\|$ 
is to shrink the learning rate $\eta$ to tiny values.
One advantage here is that we do not bias the updates.
But what if we only *rarely* get large gradients?
This drastic move slows down our progress at all steps,
just to deal with the rare exploding gradient events.
A popular alternative is to adopt a *gradient clipping* heuristic
projecting the gradients $\mathbf{g}$ onto a ball 
of some given radius $\theta$ as follows:

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

This ensures that the gradient norm never exceeds $\theta$ 
and that the updated gradient is entirely aligned 
with the original direction of $\mathbf{g}$.
It also has the desirable side-effect 
of limiting the influence any given minibatch 
(and within it any given sample) 
can exert on the parameter vector. 
This bestows a certain degree of robustness to the model. 
To be clear, it is a hack. 
Gradient clipping means that we are not always
following the true gradient and it is hard 
to reason analytically about the possible side effects.
However, it is a very useful hack,
and is widely adopted in RNN implementations
in most deep learning frameworks.


Below we define a method to clip gradients,
which is invoked by the `fit_epoch` method of
the `d2l.Trainer` class (see :numref:`sec_linear_scratch`).
Note that when computing the gradient norm,
we are concatenating all model parameters,
treating them as a single giant parameter vector.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.parameters()
    if not isinstance(params, list):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = [p for p in model.parameters() if p.requires_grad]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_clip_val = tf.constant(grad_clip_val, dtype=tf.float32)
    new_grads = [tf.convert_to_tensor(grad) if isinstance(
        grad, tf.IndexedSlices) else grad for grad in grads]    
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)) for grad in new_grads))
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=27}
%%tab jax
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, grads):
    grad_leaves, _ = jax.tree_util.tree_flatten(grads)
    norm = jnp.sqrt(sum(jnp.vdot(x, x) for x in grad_leaves))
    clip = lambda grad: jnp.where(norm < grad_clip_val,
                                  grad, grad * (grad_clip_val / norm))
    return jax.tree_util.tree_map(clip, grads)
```

## Training

Using *The Time Machine* dataset (`data`),
we train a character-level language model (`model`)
based on the RNN (`rnn`) implemented from scratch.
Note that we first calculate the gradients,
then clip them, and finally 
update the model parameters
using the clipped gradients.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch', 'jax'):
    rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
    model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        rnn = RNNScratch(num_inputs=len(data.vocab), num_hiddens=32)
        model = RNNLMScratch(rnn, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

## Decoding

Once a language model has been learned,
we can use it not only to predict the next token
but to continue predicting each subsequent token,
treating the previously predicted token as though
it were the next token in the input. 
Sometimes we will just want to generate text
as though we were starting at the beginning 
of a document. 
However, it is often useful to condition
the language model on a user-supplied prefix.
For example, if we were developing an
autocomplete feature for search engine
or to assist users in writing emails,
we would want to feed in what they 
had written so far (the prefix), 
and then generate a likely continuation.


[**The following `predict` method
generates a continuation, one character at a time,
after ingesting a user-provided `prefix`**],
When looping through the characters in `prefix`,
we keep passing the hidden state
to the next time step 
but do not generate any output.
This is called the *warm-up* period.
After ingesting the prefix, we are now
ready to begin emitting the subsequent characters,
each of which will be fed back into the model 
as the input at the subsequent time step.

```{.python .input}
%%tab pytorch, mxnet, tensorflow
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, device=None):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        if tab.selected('mxnet'):
            X = d2l.tensor([[outputs[-1]]], ctx=device)
        if tab.selected('pytorch'):
            X = d2l.tensor([[outputs[-1]]], device=device)
        if tab.selected('tensorflow'):
            X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn(embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
%%tab jax
@d2l.add_to_class(RNNLMScratch)  #@save
def predict(self, prefix, num_preds, vocab, params):
    state, outputs = None, [vocab[prefix[0]]]
    for i in range(len(prefix) + num_preds - 1):
        X = d2l.tensor([[outputs[-1]]])
        embs = self.one_hot(X)
        rnn_outputs, state = self.rnn.apply({'params': params['rnn']},
                                            embs, state)
        if i < len(prefix) - 1:  # Warm-up period
            outputs.append(vocab[prefix[i + 1]])
        else:  # Predict num_preds steps
            Y = self.apply({'params': params}, rnn_outputs,
                           method=self.output_layer)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

In the following, we specify the prefix 
and have it generate 20 additional characters.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

```{.python .input}
%%tab jax
model.predict('it has', 20, data.vocab, trainer.state.params)
```

While implementing the above RNN model from scratch is instructive, it is not convenient.
In the next section, we will see how to leverage deep learning frameworks to whip up RNNs
using standard architectures, and to reap performance gains 
by relying on highly optimized library functions.


## Summary

We can train RNN-based language models to generate text following the user-provided text prefix. 
A simple RNN language model consists of input encoding, RNN modeling, and output generation.
During training, gradient clipping can mitigate the problem of exploding gradients but does not address the problem of vanishing gradients. In the experiment, we implemented a simple RNN language model and trained it with gradient clipping on sequences of text, tokenized at the character level. By conditioning on a prefix, we can use a language model to generate likely continuations, which proves useful in many applications, e.g., autocomplete features.


## Exercises

1. Does the implemented language model predict the next token based on all the past tokens up to the very first token in *The Time Machine*? 
1. Which hyperparameter controls the length of history used for prediction?
1. Show that one-hot encoding is equivalent to picking a different embedding for each object.
1. Adjust the hyperparameters (e.g., number of epochs, number of hidden units, number of time steps in a minibatch, and learning rate) to improve the perplexity. How low can you go while sticking with this simple architecture?
1. Replace one-hot encoding with learnable embeddings. Does this lead to better performance?
1. Conduct an experiment to determine how well this language model 
   trained on *The Time Machine* works on other books by H. G. Wells,
   e.g., [The War of the Worlds](http://www.gutenberg.org/ebooks/36).
1. Conduct another experiment to evaluate the perplexity of this model
   on books written by other authors. 
1. Modify the prediction method such as to use sampling 
   rather than picking the most likely next character.
    * What happens?
    * Bias the model towards more likely outputs, e.g., 
    by sampling from $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ for $\alpha > 1$.
1. Run the code in this section without clipping the gradient. What happens?
1. Replace the activation function used in this section with ReLU 
   and repeat the experiments in this section. Do we still need gradient clipping? Why?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
