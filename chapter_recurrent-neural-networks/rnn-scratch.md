# Recurrent Neural Network Implementation from Scratch
:label:`sec_rnn-scratch`

In this section we will implement an RNN
from scratch
for a character-level language model,
according to our descriptions
in :numref:`sec_rnn`.
Such a model
will be trained on H. G. Wells' *The Time Machine*.
As before, we start by reading the dataset first, which is introduced in :numref:`sec_text-sequence`.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf
```

## RNN Model

Following the descriptions in
:numref:`subsec_rnn_w_hidden_states`,
we begin by defining the class for the RNN model
with its model parameters only.
The number of hidden units `num_hiddens` is a tunable hyperparameter.

```{.python .input  n=5}
%%tab all
class RNNScratch(d2l.Module):  #@save
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
                d2l.rand(num_hiddens, num_hiddens) * sigma)
            self.b_h = nn.Parameter(d2l.zeros(num_hiddens))
        if tab.selected('tensorflow'):
            self.W_xh = tf.Variable(d2l.normal(
                (num_inputs, num_hiddens)) * sigma)
            self.W_hh = tf.Variable(d2l.normal(
                (num_hiddens, num_hiddens)) * sigma)
            self.b_h = tf.Variable(d2l.zeros(num_hiddens))        
```

[**The following `forward` method defines how to compute the output and hidden state at a time step.**]
Note that
the RNN model
loops through the outermost dimension of `inputs`
so that it updates hidden state of a minibatch,
time step by time step.
Besides,
the activation function here uses the $\tanh$ function.
As
described in :numref:`sec_mlp`, the
mean value of the $\tanh$ function is 0, when the elements are uniformly
distributed over the real numbers.

```{.python .input}
%%tab all
@d2l.add_to_class(RNNScratch)  #@save
def forward(self, inputs, state=None):
    if state is not None:
        state, = state
        if tab.selected('tensorflow'):
            state = d2l.reshape(state, (-1, self.W_hh.shape[0]))
    outputs = []
    for X in inputs:  # Shape of inputs: (num_steps, batch_size, num_inputs) 
        state = d2l.tanh(d2l.matmul(X, self.W_xh) + (
            d2l.matmul(state, self.W_hh) if state is not None else 0)
                         + self.b_h)
        outputs.append(state)
    return outputs, state
```

For example, we can feed a minibatch of input sequences
into an RNN model as follows.

```{.python .input}
%%tab all
batch_size, num_inputs, num_hiddens, num_steps = 2, 16, 32, 100
rnn = RNNScratch(num_inputs, num_hiddens)
X = d2l.ones((num_steps, batch_size, num_inputs))
outputs, state = rnn(X)
```

Let's check whether the RNN model
produces results of the correct shapes,
e.g., to ensure that the dimensionality of the hidden state remains unchanged.

```{.python .input}
%%tab all
def check_len(a, n):  #@save
    assert len(a) == n, f'list\'s len {len(a)} != expected length {n}'
    
def check_shape(a, shape):  #@save
    assert a.shape == shape, \
            f'tensor\'s shape {a.shape} != expected shape {shape}'

d2l.check_len(outputs, num_steps)
d2l.check_shape(outputs[0], (batch_size, num_hiddens))
d2l.check_shape(state, (batch_size, num_hiddens))
```

## RNN-based Language Model

The following `RNNLMScratch` class defines 
an RNN-based language model,
where an RNN implemented from scratch
is passed via the `rnn` argument
of the `__init__` method.
When training language models, the inputs and outputs are from the same vocabulary. Hence, they have the same dimension, which is equal to the vocabulary size.
Note that we use perplexity to evaluate the model. As discussed in :numref:`subsec_perplexity`, this ensures that sequences of different length are comparable.

```{.python .input}
%%tab all
class RNNLMScratch(d2l.Classification):  #@save
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
        if tab.selected('pytorch'):
            self.W_hq = nn.Parameter(
                d2l.randn(
                    self.rnn.num_hiddens, self.vocab_size) * self.rnn.sigma)
            self.b_q = nn.Parameter(d2l.zeros(self.vocab_size)) 
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

### [**One-Hot Encoding**]

Recall that each token is represented as a numerical index in the vocabulary.
Feeding these indices directly to a neural network might make it hard to
learn.
We often represent each token as a more expressive feature vector.
The easiest representation is called *one-hot encoding*,
which is introduced
in :numref:`subsec_classification-problem`.

In a nutshell, we map each index to a different unit vector: assume that the number of different tokens in the vocabulary is $N$ (`len(vocab)`) and the token indices range from $0$ to $N-1$.
If the index of a token is the integer $i$, then we create a vector of all 0s with a length of $N$ and set the element at position $i$ to 1.
This vector is the one-hot vector of the original token. The one-hot vectors with indices 0 and 2 are shown below.

```{.python .input  n=6}
%%tab mxnet
npx.one_hot(np.array([0, 2]), 5)
```

```{.python .input  n=11}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), 5)
```

```{.python .input  n=12}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), 5)
```

(**The shape of the minibatch**) that we sample each time (**is (batch size, number of time steps).
The `one_hot` method transforms such a minibatch into a three-dimensional tensor with the last dimension equals to the vocabulary size (`len(vocab)`).**)
We often transpose the input so that we will obtain an
output of shape
(number of time steps, batch size, vocabulary size).
This will allow us
to more conveniently
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
```

### Transforming RNN Outputs

The language model
uses a fully connected output layer
to transform RNN outputs 
into token predictions at each time step.

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

Let's [**check whether the forward computation outputs have the correct shape.**]

```{.python .input}
%%tab all
model = RNNLMScratch(rnn, num_inputs)
outputs = model(d2l.ones((batch_size, num_steps), dtype=d2l.int64))
d2l.check_shape(outputs, (batch_size, num_steps, num_inputs))
```

## [**Gradient Clipping**]

For a sequence of length $T$,
we compute the gradients over these $T$ time steps in an iteration, which results in a chain of matrix-products with length  $\mathcal{O}(T)$ during backpropagation.
As mentioned in :numref:`sec_numerical_stability`, it might result in numerical instability, e.g., the gradients may either explode or vanish, when $T$ is large. Therefore, RNN models often need extra help to stabilize the training.

Generally speaking,
when solving an optimization problem,
we take update steps for the model parameter,
say in the vector form
$\mathbf{x}$,
in the direction of the negative gradient $\mathbf{g}$ on a minibatch.
For example,
with $\eta > 0$ as the learning rate,
in one iteration we update
$\mathbf{x}$
as $\mathbf{x} - \eta \mathbf{g}$.
Let's further assume that the objective function $f$
is well behaved, say, *Lipschitz continuous* with constant $L$.
That is to say,
for any $\mathbf{x}$ and $\mathbf{y}$ we have

$$|f(\mathbf{x}) - f(\mathbf{y})| \leq L \|\mathbf{x} - \mathbf{y}\|.$$

In this case we can safely assume that if we update the parameter vector by $\eta \mathbf{g}$, then

$$|f(\mathbf{x}) - f(\mathbf{x} - \eta\mathbf{g})| \leq L \eta\|\mathbf{g}\|,$$

which means that
we will not observe a change by more than $L \eta \|\mathbf{g}\|$. This is both a curse and a blessing.
On the curse side,
it limits the speed of making progress;
whereas on the blessing side,
it limits the extent to which things can go wrong if we move in the wrong direction.

Sometimes the gradients can be quite large and the optimization algorithm may fail to converge. We could address this by reducing the learning rate $\eta$. But what if we only *rarely* get large gradients? In this case such an approach may appear entirely unwarranted. One popular alternative is to clip the gradient $\mathbf{g}$ by projecting them back to a ball of a given radius, say $\theta$ via

(**$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$**)

By doing so we know that the gradient norm never exceeds $\theta$ and that the
updated gradient is entirely aligned with the original direction of $\mathbf{g}$.
It also has the desirable side-effect of limiting the influence any given
minibatch (and within it any given sample) can exert on the parameter vector. This
bestows a certain degree of robustness to the model. Gradient clipping provides
a quick fix to the gradient exploding. While it does not entirely solve the problem, it is one of the many techniques to alleviate it.

Below we define a function to clip the gradients of
a model.
Also note that we compute the gradient norm over all the model parameters.

```{.python .input  n=43}
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

```{.python .input  n=17}
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

## Training

Using *The Time Machine* dataset (`data`),
we train a character-level language model (`model`)
based on the RNN (`rnn`) implemented from scratch.
Note that we clip the gradients before updating the model parameters. This ensures that the model does not diverge even when gradients blow up at some point during the training process.

```{.python .input  n=26}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'pytorch'):
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

## Prediction

We need to [**define the prediction function
to generate new characters following
the user-provided `prefix`**],
which is a string containing several characters.
When looping through these beginning characters in `prefix`,
we keep passing the hidden state
to the next time step without
generating any output.
This is called the *warm-up* period,
during which the model updates itself
(e.g., update the hidden state)
but does not make predictions.
After the warm-up period,
the hidden state is generally better than
its initialized value at the beginning.
So we generate the predicted characters and emit them.

```{.python .input}
%%tab all
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
        else:  # Predict `num_preds` steps
            Y = self.output_layer(rnn_outputs)
            outputs.append(int(d2l.reshape(d2l.argmax(Y, axis=2), 1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

In the following, we specify the prefix 
and have it generate 20 additional characters.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input  n=22}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

While implementing the above RNN model from scratch is instructive, it is not convenient.
In the next section we will see how to improve the RNN model,
such as how to make it easier to implement
and make it run faster.


## Summary

* We can train an RNN-based character-level language model to generate text following the user-provided text prefix.
* A simple RNN language model consists of input encoding, RNN modeling, and output generation.
* Gradient clipping prevents gradient explosion, but it cannot fix vanishing gradients.
* A warm-up period allows a model to update itself (e.g., obtain a better hidden state than its initialized value) before making any prediction.


## Exercises

1. Does the implemented language model predict the next token based on all the past tokens up to the very first token in *The Time Machine*? Which hyperparameter controls the length of history used for prediction?
1. Show that one-hot encoding is equivalent to picking a different embedding for each object.
1. Adjust the hyperparameters (e.g., number of epochs, number of hidden units, number of time steps in a minibatch, and learning rate) to improve the perplexity.
    * How low can you go?
    * Replace one-hot encoding with learnable embeddings. Does this lead to better performance?
    * How well will it work on other books by H. G. Wells, e.g., [*The War of the Worlds*](http://www.gutenberg.org/ebooks/36)?
1. Modify the prediction function such as to use sampling rather than picking the most likely next character.
    * What happens?
    * Bias the model towards more likely outputs, e.g., by sampling from $q(x_t \mid x_{t-1}, \ldots, x_1) \propto P(x_t \mid x_{t-1}, \ldots, x_1)^\alpha$ for $\alpha > 1$.
1. Run the code in this section without clipping the gradient. What happens?
1. Replace the activation function used in this section with ReLU and repeat the experiments in this section. Do we still need gradient clipping? Why?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1052)
:end_tab:
