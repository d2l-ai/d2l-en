# Recurrent Neural Network Implementation from Scratch
:label:`sec_rnn_scratch`

In this section we will implement an RNN
from scratch
for a character-level language model,
according to our descriptions
in :numref:`sec_rnn`.
Such a model
will be trained on H. G. Wells' *The Time Machine*.
As before, we start by reading the dataset first, which is introduced in :numref:`sec_language_model`.

```{.python .input  n=2}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.json .output n=2}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "4a201b6ce9154672b642b6b3b967d20d",
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

```{.python .input  n=3}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
npx.set_np()

data = d2l.TimeMachine(batch_size=32, num_steps=35)
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=4}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F

data = d2l.TimeMachine(batch_size=32, num_steps=35)
```

```{.python .input  n=5}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import tensorflow as tf

data = d2l.TimeMachine(batch_size=32, num_steps=35)
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

## [**One-Hot Encoding**]

Recall that each token is represented as a numerical index in `train_iter`.
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
npx.one_hot(np.array([0, 2]), len(data.vocab))
```

```{.json .output n=6}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=7}
%%tab pytorch
F.one_hot(torch.tensor([0, 2]), len(data.vocab))
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "tensor([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n         0, 0, 0, 0],\n        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,\n         0, 0, 0, 0]])"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
%%tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(data.vocab))
```

```{.json .output n=8}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

(**The shape of the minibatch**) that we sample each time (**is (batch size, number of time steps).
The `one_hot` function transforms such a minibatch into a three-dimensional tensor with the last dimension equals to the vocabulary size (`len(vocab)`).**)
We often transpose the input so that we will obtain an
output of shape
(number of time steps, batch size, vocabulary size).
This will allow us
to more conveniently
loop through the outermost dimension
for updating hidden states of a minibatch,
time step by time step.

```{.python .input  n=9}
%%tab mxnet
X = d2l.reshape(d2l.arange(10), (2, 5))
npx.one_hot(X.T, len(data.vocab)).shape
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
X = d2l.reshape(d2l.arange(10), (2, 5))
F.one_hot(X.T, len(data.vocab)).shape
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "torch.Size([5, 2, 28])"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=11}
%%tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), len(data.vocab)).shape
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

## RNN Model

Next, we define the model class.
The number of hidden units `num_hiddens` is a tunable hyperparameter.
When training language models,
the inputs and outputs are from the same vocabulary.
The dataset is relatively small, we will train with hundreds of epochs, so we choose to plot for every 10 epochs.  

```{.python .input  n=12}
%%tab mxnet
class RNNScratch(d2l.Classification):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__(plot_train_per_epoch=0.1, plot_train_per_epoch=0.1)
        self.save_hyperparameters()
        self.init_params()
        if tab.selected('mxnet'):
            for param in self._params:
                param.attach_grad()
        if tab.selected('pytorch'):
            for param in self._params:
                param.requires_grad_(True)        
```

```{.json .output n=12}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=16}
%%tab pytorch
class RNNScratch(d2l.Classification):
    def __init__(self, num_inputs, num_outputs, num_hiddens, lr, sigma=0.01):
        super().__init__(plot_train_per_epoch=0.1, plot_valid_per_epoch=0.1)
        self.save_hyperparameters()
        self.init_params()

    def init_params(self):
        # Hidden layer parameters
        self.W_xh = nn.Parameter(d2l.randn(
            self.num_inputs, self.num_hiddens) * self.sigma)
        self.W_hh = nn.Parameter(
            d2l.rand(self.num_hiddens, self.num_hiddens) * self.sigma)
        self.b_h = nn.Parameter(d2l.zeros(self.num_hiddens))
        # Output layer parameters
        self.W_hq = nn.Parameter(d2l.randn(
            self.num_hiddens, self.num_outputs) * self.sigma)
        self.b_q = nn.Parameter(d2l.zeros(self.num_outputs))
```

We initialize all learnable parameters. 

```{.python .input  n=17}
%%tab mxnet
@d2l.add_to_class(RNNScratch):
def init_params(self)
    # Hidden layer parameters
    self.W_xh = d2l.randn(self.num_inputs, self.num_hiddens) * self.sigma
    self.W_hh = normal(self.num_hiddens, self.num_hiddens)
    self.b_h = d2l.zeros(self.num_hiddens, ctx=device)
    # Output layer parameters
    self.W_hq = normal(self.num_hiddens, self.num_outputs)
    self.b_q = d2l.zeros(self.num_outputs)
    self._params = [self.W_xh, self.W_hh, self.b_h, self.W_hq, self.b_q]

```

```{.json .output n=17}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

```{.python .input  n=18}
%%tab tensorflow
def get_params(vocab_size, num_hiddens):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return d2l.normal(shape=shape,stddev=0.01,mean=0,dtype=tf.float32)

    # Hidden layer parameters
    W_xh = tf.Variable(normal((num_inputs, num_hiddens)), dtype=tf.float32)
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens)), dtype=tf.float32)
    b_h = tf.Variable(d2l.zeros(num_hiddens), dtype=tf.float32)
    # Output layer parameters
    W_hq = tf.Variable(normal((num_hiddens, num_outputs)), dtype=tf.float32)
    b_q = tf.Variable(d2l.zeros(num_outputs), dtype=tf.float32)
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    return params
```

```{.json .output n=18}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

To define an RNN model,
we first need [**an `init_rnn_state` function
to return the hidden state at initialization.**]
It returns a tensor filled with 0 and with a shape of (batch size, number of hidden units).
Using tuples makes it easier to handle situations where the hidden state contains multiple variables,
which we will encounter in later sections.

```{.python .input  n=19}
%%tab mxnet, pytorch
@d2l.add_to_class(RNNScratch)
def init_state(self, batch_size):
    return (d2l.zeros((batch_size, self.num_hiddens)), )
```

```{.python .input  n=20}
%%tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

```{.json .output n=20}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"pytorch\" cell."
 }
]
```

[**The following `rnn` function defines how to compute the hidden state and output
at a time step.**]
Note that
the RNN model
loops through the outermost dimension of `inputs`
so that it updates hidden states `H` of a minibatch,
time step by time step.
Besides,
the activation function here uses the $\tanh$ function.
As
described in :numref:`sec_mlp`, the
mean value of the $\tanh$ function is 0, when the elements are uniformly
distributed over the real numbers.

```{.python .input  n=21}
%%tab mxnet, pytorch
@d2l.add_to_class(RNNScratch)
def forward(self, X, state=None):
    if state is None:
        state = self.init_state(X.shape[0])
    # Shape of X: (batch_size, num_steps)
    # Shape of embs: (num_steps, batch_size, num_inputs)
    if tab.selected('pytorch'):
        embs = F.one_hot(X.T, self.num_inputs).type(torch.float32)
    if tab.selected('mxnet'):
        embs = npx.one_hot(X.T, self.num_inputs)
    H, = state
    outputs = []
    for emb in embs:        
        H = d2l.tanh(d2l.matmul(emb, self.W_xh) + d2l.matmul(H, self.W_hh) + self.b_h)        
        Y = d2l.matmul(H, self.W_hq) + self.b_q
        outputs.append(Y)
    # Return shape (batch_size x num_steps, num_outputs)
    return d2l.concat(outputs, 0), (H,)
```

```{.python .input  n=22}
%%tab all
@d2l.add_to_class(RNNScratch)
def loss(self, outputs, Y):
    y_hat, _ = outputs
    return super(RNNScratch, self).loss(y_hat, d2l.reshape(Y.T, -1))

@d2l.add_to_class(RNNScratch)
def accuracy(self, outputs, y):
    y_hat, _ = outputs    
    return super(RNNScratch, self).accuracy(y_hat, d2l.reshape(Y.T, (-1,1)))
```

```{.python .input  n=21}
%%tab tensorflow
def rnn(inputs, state, params):
    # Here `inputs` shape: (`num_steps`, `batch_size`, `vocab_size`)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # Shape of `X`: (`batch_size`, `vocab_size`)
    for X in inputs:
        X = tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

With all the needed functions being defined,
next we [**create a class to wrap these functions and store parameters**] for an RNN model implemented from scratch.

Let's [**check whether the outputs have the correct shapes**], e.g., to ensure that the dimensionality of the hidden state remains unchanged.

```{.python .input  n=22}
%%tab mxnet, pytorch
model = RNNScratch(num_inputs=len(data.vocab), 
                   num_outputs=len(data.vocab), num_hiddens=512, lr=1)
X, Y = next(iter(data.train_dataloader()))
Y_hat, new_state = model(X)
Y_hat.shape, len(new_state), new_state[0].shape
```

We can see that the output shape is (number of time steps $\times$ batch size, vocabulary size), while the hidden state shape remains the same, i.e., (batch size, number of hidden units).




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
a model that is implemented from scratch or a model constructed by the high-level APIs.
Also note that we compute the gradient norm over all the model parameters.

```{.python .input  n=43}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def clip_gradients(self, grad_clip_val, model):
    params = model.collect_params()
    if not isinstance(params, (list, tuple)):
        params = [p.data() for p in params.values()]    
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > grad_clip_val:
        for param in params:
            param.grad[:] *= grad_clip_val / norm
```

```{.python .input  n=44}
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
    new_grads = [tf.convert_to_tensor(grad) for grad in grads
                 if isinstance(grad, tf.IndexedSlices) else grad]
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in new_grad))
    norm = tf.cast(norm, tf.float32)
    if tf.greater(norm, grad_clip_val):
        for i, grad in enumerate(new_grads):
            new_grads[i] = grad * grad_clip_val / norm
        return new_grads
    return grads
```

```{.python .input  n=26}
%%tab all
trainer = d2l.Trainer(max_epochs=10, gradient_clip_val=1)
trainer.fit(model, data)
```

## Training

Before training the model,
let's [**define a function to train the model in one epoch**]. It differs from how we train the model of :numref:`sec_softmax_scratch` in three places:

1. We iterate over sequential data with random sampling, where we re-initialize the hidden state for each iteration.
1. We clip the gradients before updating the model parameters. This ensures that the model does not diverge even when gradients blow up at some point during the training process.
1. We use perplexity to evaluate the model. As discussed in :numref:`subsec_perplexity`, this ensures that sequences of different length are comparable.

Same as the `train_epoch_ch3` function in :numref:`sec_softmax_scratch`,
`updater` is a general function
to update the model parameters.
It can be either the `d2l.sgd` function implemented from scratch or the built-in optimization function in
a deep learning framework.

```{.python .input  n=30}
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device):
    """Train a model within one epoch (defined in Chapter 8)."""
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        # With random sampling, initialize state for each iteration
        state = net.begin_state(batch_size=X.shape[0], ctx=device)
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            y_hat, state = net(X, state)
            l = loss(y_hat, y).mean()
        l.backward()
        grad_clipping(net, 1)
        updater(batch_size=1)  # Since the `mean` function has been invoked
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input  n=31}
#@tab pytorch
#@save
def train_epoch_ch8(net, train_iter, loss, updater, device):
    """Train a net within one epoch (defined in Chapter 8)."""
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        # With random sampling, initialize state for each iteration
        state = net.begin_state(batch_size=X.shape[0], device=device)
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        l = loss(y_hat, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # Since the `mean` function has been invoked
            updater(batch_size=1)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input  n=32}
#@tab tensorflow
#@save
def train_epoch_ch8(net, train_iter, loss, updater):
    """Train a model within one epoch (defined in Chapter 8)."""
    timer = d2l.Timer()
    metric = d2l.Accumulator(2)  # Sum of training loss, no. of tokens
    for X, Y in train_iter:
        # With random sampling, initialize state for each iteration
        state = net.begin_state(batch_size=X.shape[0], dtype=tf.float32)
        with tf.GradientTape(persistent=True) as g:
            y_hat, state = net(X, state)
            y = d2l.reshape(tf.transpose(Y), (-1))
            l = loss(y, y_hat)
        params = net.trainable_variables
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))

        # Keras loss by default returns the average loss in a batch
        # l_sum = l * float(d2l.size(y)) if isinstance(
        #     loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

[**The training function supports
an RNN model implemented
either from scratch
or using high-level APIs.**]

```{.python .input  n=33}
def train_ch8(net, train_iter, vocab, lr, num_epochs, device):  #@save
    """Train a model (defined in Chapter 8)."""
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, gluon.Block):
        net.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(net.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
```

```{.python .input  n=34}
#@tab pytorch
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, device):
    """Train a model (defined in Chapter 8)."""
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # Initialize
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
```

```{.python .input  n=35}
#@tab tensorflow
#@save
def train_ch8(net, train_iter, vocab, lr, num_epochs, strategy):
    """Train a model (defined in Chapter 8)."""
    with strategy.scope():
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        updater = tf.keras.optimizers.SGD(lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab)
    # Train and predict
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(net, train_iter, loss, updater)
        if (epoch + 1) % 10 == 0:
            animator.add(epoch + 1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
```

[**Now we can train the RNN model.**]
Since we only use 10000 tokens in the dataset, the model needs more epochs to converge better.

```{.python .input  n=36}
%%tab xx
num_epochs, lr = 500, 1.5
train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.python .input  n=37}
%%tab tensorflow
num_epochs, lr = 500, 1.5
train_ch8(net, train_iter, vocab, lr, num_epochs, strategy)
```

## Prediction

Let's [**first define the prediction function
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

```{.python .input  n=38}
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(
        d2l.tensor([outputs[-1]], ctx=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input  n=39}
#@tab pytorch
def predict_ch8(prefix, num_preds, net, vocab, device):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor(
        [outputs[-1]], device=device), (1, 1))
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input  n=40}
#@tab tensorflow
def predict_ch8(prefix, num_preds, net, vocab):  #@save
    """Generate new characters following the `prefix`."""
    state = net.begin_state(batch_size=1, dtype=tf.float32)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(d2l.tensor([outputs[-1]]), (1, 1)).numpy()
    for y in prefix[1:]:  # Warm-up period
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # Predict `num_preds` steps
        y, state = net(get_input(), state)
        outputs.append(int(y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

Now we can test the `predict_ch8` function.
We specify the prefix as `time traveller ` and have it generate 10 additional characters.
Given that we have not trained the network,
it will generate nonsensical predictions.

```{.python .input  n=41}
%%tab xx
predict_ch8('time traveller ', 10, net, vocab, d2l.try_gpu())
```

```{.python .input  n=42}
%%tab tensorflow
predict_ch8('time traveller ', 10, net, vocab)
```

While implementing the above RNN model from scratch is instructive, it is not convenient.
In the next section we will see how to improve the RNN model,
such as how to make it easier to implement
and make it run faster.


## Summary

* We can train an RNN-based character-level language model to generate text following the user-provided text prefix.
* A simple RNN language model consists of input encoding, RNN modeling, and output generation.
* We iterate over sequential data with random sampling, where we re-initialize the RNN hidden state for each iteration.
* A warm-up period allows a model to update itself (e.g., obtain a better hidden state than its initialized value) before making any prediction.
* Gradient clipping prevents gradient explosion, but it cannot fix vanishing gradients.


## Exercises

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
