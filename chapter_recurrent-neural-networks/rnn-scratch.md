# Implementation of Recurrent Neural Networks from Scratch
:label:`sec_rnn_scratch`

In this section we implement a language model introduced in :numref:`chap_rnn` from scratch. It is based on a character-level recurrent neural network trained on H. G. Wells' *The Time Machine*. As before, we start by reading the dataset first, which is introduced in :numref:`sec_language_model`.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import math
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input  n=2}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import math
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab mxnet,pytorch
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

```{.python .input  n=3}
#@tab tensorflow
batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
train_random_iter, vocab_random_iter = d2l.load_data_time_machine(batch_size,
                                            num_steps, use_random_iter=True)
```

## One-hot Encoding

Remember that each token is presented as a numerical index in `train_iter`.
Feeding these indices directly to the neural network might make it hard to
learn. We often present each token as a more expressive feature vector. The
easiest representation is called *one-hot encoding*.

In a nutshell, we map each index to a different unit vector: assume that the number of different tokens in the vocabulary is $N$ (the `len(vocab)`) and the token indices range from 0 to $N-1$. If the index of a token is the integer $i$, then we create a vector $\mathbf{e}_i$ of all 0s with a length of $N$ and set the element at position $i$ to 1. This vector is the one-hot vector of the original token. The one-hot vectors with indices 0 and 2 are shown below.

```{.python .input}
npx.one_hot(np.array([0, 2]), len(vocab))
```

```{.python .input}
#@tab pytorch
F.one_hot(torch.tensor([0, 2]), len(vocab))
```

```{.python .input  n=4}
#@tab tensorflow
tf.one_hot(tf.constant([0, 2]), len(vocab))
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "<tf.Tensor: shape=(2, 28), dtype=float32, numpy=\narray([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],\n       [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,\n        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]], dtype=float32)>"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

The shape of the minibatch we sample each time is (batch size, time step). The `one_hot` function transforms such a minibatch into a 3-D tensor with the last dimension equals to the vocabulary size. We often transpose the input so that we will obtain a (time step, batch size, vocabulary size) output that fits into a sequence model easier.

```{.python .input}
X = np.arange(10).reshape(2, 5)
npx.one_hot(X.T, 28).shape
```

```{.python .input}
#@tab pytorch
X = torch.arange(10).reshape(2, 5)
F.one_hot(X.T, 28).shape
```

```{.python .input  n=5}
#@tab tensorflow
X = d2l.reshape(d2l.arange(10), (2, 5))
tf.one_hot(tf.transpose(X), 28).shape
```

```{.json .output n=5}
[
 {
  "data": {
   "text/plain": "TensorShape([5, 2, 28])"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Initializing the Model Parameters

Next, we initialize the model parameters for a RNN model. The number of hidden units `num_hiddens` is a tunable parameter.

```{.python .input}
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return np.random.normal(scale=0.01, size=shape, ctx=device)
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, ctx=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, ctx=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

```{.python .input}
#@tab pytorch
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = d2l.zeros(num_hiddens, device=device)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = d2l.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params
```

```{.python .input  n=6}
#@tab tensorflow
def get_params(vocab_size, num_hidden):
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

## RNN Model

First, we need an `init_rnn_state` function to return the hidden state at initialization. It returns a tensor filled with 0 and with a shape of (batch size, number of hidden units). Using tuples makes it easier to handle situations where the hidden state contains multiple variables (e.g., when combining multiple layers in an RNN where each layer requires initializing).

```{.python .input}
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), ctx=device), )
```

```{.python .input}
#@tab pytorch
def init_rnn_state(batch_size, num_hiddens, device):
    return (d2l.zeros((batch_size, num_hiddens), device=device), )
```

```{.python .input  n=7}
#@tab tensorflow
def init_rnn_state(batch_size, num_hiddens):
    return (d2l.zeros((batch_size, num_hiddens)), )
```

The following `rnn` function defines how to compute the hidden state and output
in a time step. The activation function here uses the $\tanh$ function. As
described in :numref:`sec_mlp`, the
mean value of the $\tanh$ function is 0, when the elements are evenly
distributed over the real numbers.

```{.python .input}
def rnn(inputs, state, params):
    # Inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = np.tanh(np.dot(X, W_xh) + np.dot(H, W_hh) + b_h)
        Y = np.dot(H, W_hq) + b_q
        outputs.append(Y)
    return np.concatenate(outputs, axis=0), (H,)
```

```{.python .input}
#@tab pytorch
def rnn(inputs, state, params):
    # Inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```

```{.python .input  n=8}
#@tab tensorflow
def rnn(inputs, state, params):
    # Inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        X=tf.reshape(X,[-1,W_xh.shape[0]])
        H = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(H, W_hh) + b_h)
        Y = tf.matmul(H, W_hq) + b_q
        outputs.append(Y)
    return d2l.concat(outputs, axis=0), (H,)
```

Now we have all functions defined, next we create a class to wrap these functions and store parameters.

```{.python .input}
class RNNModelScratch:  #@save
    """A RNN Model based on scratch implementations."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = npx.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

```{.python .input}
#@tab pytorch
class RNNModelScratch: #@save
    """A RNN Model based on scratch implementations."""
    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)
```

```{.python .input  n=9}
#@tab tensorflow
class RNNModelScratch: #@save
    """A RNN Model based on scratch implementations."""
    def __init__(self, vocab_size, num_hiddens,
                 init_state, forward):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.init_state, self.forward_fn = init_state, forward

    def __call__(self, X, state, params):
        X = tf.one_hot(tf.transpose(X), self.vocab_size)
        X = tf.cast(X, tf.float32)
        return self.forward_fn(X, state, params)

    def begin_state(self, batch_size):
        return self.init_state(batch_size, self.num_hiddens)
```

Let us do a sanity check whether inputs and outputs have the correct dimensions, e.g., to ensure that the dimensionality of the hidden state has not changed.

```{.python .input}
#@tab mxnet
num_hiddens = 512
model = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                        init_rnn_state, rnn)
state = model.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = model(X.as_in_context(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input}
#@tab pytorch
num_hiddens = 512
model = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                        init_rnn_state, rnn)
state = model.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = model(X.to(d2l.try_gpu()), state)
Y.shape, len(new_state), new_state[0].shape
```

```{.python .input  n=10}
#@tab tensorflow
num_hiddens = 512
model = RNNModelScratch(len(vocab), num_hiddens, 
                        init_rnn_state, rnn)
state = model.begin_state(X.shape[0])
params = get_params(len(vocab), num_hiddens)
Y, new_state = model(X, state, params)
Y.shape, len(new_state), new_state[0].shape
```

```{.json .output n=10}
[
 {
  "data": {
   "text/plain": "(TensorShape([10, 28]), 1, TensorShape([2, 512]))"
  },
  "execution_count": 10,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

We can see that the output shape is (number steps $\times$ batch size, vocabulary size), while the hidden state shape remains the same, i.e., (batch size, number of hidden units).

## Prediction

We first explain the predicting function so we can regularly check the prediction during training. This function predicts the next `num_predicts` characters based on the `prefix` (a string containing several characters). For the beginning of the sequence, we only update the hidden state. After that we begin generating new characters and emitting them.

```{.python .input}
def predict_ch8(prefix, num_predicts, model, vocab, device):  #@save
    state = model.begin_state(batch_size=1, ctx=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: np.array([outputs[-1]], ctx=device).reshape(1, 1)
    for y in prefix[1:]:  # Warmup state with prefix
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input}
#@tab pytorch
def predict_ch8(prefix, num_predicts, model, vocab, device):  #@save
    state = model.begin_state(batch_size=1, device=device)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: torch.tensor(
        [outputs[-1]], device=device).reshape(1, 1)
    for y in prefix[1:]:  # Warmup state with prefix
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

```{.python .input  n=11}
#@tab tensorflow
def predict_ch8(prefix, num_predicts, model, vocab, params): #@save
    state = model.begin_state(batch_size=1)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: d2l.reshape(tf.constant([outputs[-1]]), (1,1)).numpy()
    for y in prefix[1:]: # Warmup state with prefix
        _, state = model(get_input(), state, params)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state, params)
        outputs.append(int(Y.numpy().argmax(axis=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

We test the `predict_ch8` function first. Given that we did not train the network, it will generate nonsensical predictions. We initialize it with the sequence `traveller ` and have it generate 10 additional characters.

```{.python .input}
#@tab mxnet,pytorch
predict_ch8('time traveller ', 10, model, vocab, d2l.try_gpu())
```

```{.python .input  n=12}
#@tab tensorflow
predict_ch8('time traveller ', 10, model, vocab, params)
```

```{.json .output n=12}
[
 {
  "data": {
   "text/plain": "'time traveller <unk>wbylofdkg'"
  },
  "execution_count": 12,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Gradient Clipping

For a sequence of length $T$, we compute the gradients over these $T$ time steps in an iteration, which results in a chain of matrix-products with length  $\mathcal{O}(T)$ during backpropagating. As mentioned in :numref:`sec_numerical_stability`, it might result in numerical instability, e.g., the gradients may either explode or vanish, when $T$ is large. Therefore, RNN models often need extra help to stabilize the training.

Recall that when solving an optimization problem, we take update steps for the weights $\mathbf{w}$ in the general direction of the negative gradient $\mathbf{g}_t$ on a minibatch, say $\mathbf{w} - \eta \cdot \mathbf{g}_t$. Let us further assume that the objective is well behaved, i.e., it is Lipschitz continuous with constant $L$, i.e.,

$$|l(\mathbf{w}) - l(\mathbf{w}')| \leq L \|\mathbf{w} - \mathbf{w}'\|.$$

In this case we can safely assume that if we update the weight vector by $\eta \cdot \mathbf{g}_t$, we will not observe a change by more than $L \eta \|\mathbf{g}_t\|$. This is both a curse and a blessing. A curse since it limits the speed of making progress, whereas a blessing since it limits the extent to which things can go wrong if we move in the wrong direction.

Sometimes the gradients can be quite large and the optimization algorithm may fail to converge. We could address this by reducing the learning rate $\eta$ or by some other higher order trick. But what if we only rarely get large gradients? In this case such an approach may appear entirely unwarranted. One alternative is to clip the gradients by projecting them back to a ball of a given radius, say $\theta$ via

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

By doing so we know that the gradient norm never exceeds $\theta$ and that the
updated gradient is entirely aligned with the original direction $\mathbf{g}$.
It also has the desirable side-effect of limiting the influence any given
minibatch (and within it any given sample) can exert on the weight vectors. This
bestows a certain degree of robustness to the model. Gradient clipping provides
a quick fix to the gradient exploding. While it does not entirely solve the problem, it is one of the many techniques to alleviate it.

Below we define a function to clip the gradients of a model that is either a building from scratch instance or a model constructed by the high-level APIs. Also note that we compute the gradient norm over all parameters.

```{.python .input}
def grad_clipping(model, theta):  #@save
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input}
#@tab pytorch
def grad_clipping(model, theta):  #@save
    if isinstance(model, nn.Module):
        params = [p for p in model.parameters() if p.requires_grad]
    else:
        params = model.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

```{.python .input  n=13}
#@tab tensorflow
def grad_clipping(grads, theta): #@save
    theta = tf.constant(theta, dtype=tf.float32)
    norm = tf.math.sqrt(sum((tf.reduce_sum(grad ** 2)).numpy()
                        for grad in grads))
    norm = tf.cast(norm, tf.float32)
    new_grad = []
    if tf.greater(norm, theta):
        for grad in grads:
            new_grad.append(grad * theta / norm)
    else:
        for grad in grads:
            new_grad.append(grad)
    return new_grad
```

## Training

Let us first define the function to train the model on one data epoch. It differs from the models training of :numref:`sec_softmax_scratch` in three places:

1. Different sampling methods for sequential data (random sampling and
   sequential partitioning) will result in differences in the initialization of
   hidden states.
1. We clip the gradients before updating the model parameters. This ensures that the model does not diverge even when gradients blow up at some point during the training process, and it effectively reduces the step size automatically.
1. We use perplexity to evaluate the model. This ensures that sequences of different length are comparable.


When the sequential partitioning is used, we initialize the hidden state at the beginning of each epoch. Since the $i^\mathrm{th}$ example in the next minibatch is adjacent to the current $i^\mathrm{th}$ example, so the next minibatch can use the current hidden state directly, we only detach the gradient so that we compute the gradients within a minibatch. When using the random sampling, we need to re-initialize the hidden state for each iteration since each example is sampled with a random position. Same as the `train_epoch_ch3` function in :numref:`sec_softmax_scratch`, we use generalized `updater`, which could be either a high-level API trainer or a scratched implementation.

```{.python .input}
def train_epoch_ch8(model, train_iter, loss, updater, device,  #@save
                    use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # loss_sum, num_examples
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize state when either it is the first iteration or
            # using random sampling.
            state = model.begin_state(batch_size=X.shape[0], ctx=device)
        else:
            for s in state:
                s.detach()
        y = Y.T.reshape(-1)
        X, y = X.as_in_ctx(device), y.as_in_ctx(device)
        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)  # Since used mean already
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()
```

```{.python .input}
#@tab pytorch
def train_epoch_ch8(model, train_iter, loss, updater, device,  #@save
                    use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # loss_sum, num_examples
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize state when either it is the first iteration or
            # using random sampling.
            state = model.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(model, nn.Module) and not isinstance(state, tuple):
                # state is a tensor for nn.GRU  
                state.detach_()
            else:
                # state is a tuple of tensors for nn.LSTM and
                # for our custom scratch implementation 
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        py, state = model(X, state)
        l = loss(py, y.long()).mean()
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(model, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(model, 1)
            updater(batch_size=1)  # Since used mean already
        metric.add(l * d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

```{.python .input  n=40}
#@tab tensorflow
def train_epoch_ch8(model, train_iter, loss, updater,  #@save
                    params, use_random_iter):
    state, timer = None, d2l.Timer()
    # initialize the state at the begining of the epoch
    # when not using random_iter
    metric = d2l.Accumulator(2) 
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize state when either it is the first iteration or
            # using random sampling.
            state = model.begin_state(batch_size=X.shape[0])
        with tf.GradientTape(persistent=True) as g:
            g.watch(params)
            py, state= model(X, state, params)
            y = d2l.reshape(Y, (-1))
            y = tf.keras.activations.softmax(py)
            l = tf.math.reduce_mean(loss(y, py))
        grads = g.gradient(l, params)
        grads = grad_clipping(grads, 1)
        updater.apply_gradients(zip(grads, params))
        
        
        # Keras loss by default returns the average loss in a batch
        #l_sum = l * float(d2l.size(y)) if isinstance(
            #loss, tf.keras.losses.Loss) else tf.reduce_sum(l)
        metric.add(l* d2l.size(y), d2l.size(y))
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()
```

The training function again supports either we implement the model from scratch or using high-level APIs.

```{.python .input  n=28}
def train_ch8(model, train_iter, vocab, lr, num_epochs, device,  #@save
              use_random_iter=False):
    # Initialize
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[1, num_epochs])
    if isinstance(model, gluon.Block):
        model.initialize(ctx=device, force_reinit=True,
                         init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(),
                                'sgd', {'learning_rate': lr})
        updater = lambda batch_size: trainer.step(batch_size)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, device)
    # Train and check the progress.
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, device, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input  n=29}
#@tab pytorch
#@save
def train_ch8(model, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    # Initialize
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[1, num_epochs])
    if isinstance(model, nn.Module):
        updater = torch.optim.SGD(model.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, device)
    # Train and check the progress.
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, device, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

```{.python .input  n=38}
#@tab tensorflow
#@save
def train_ch8(model, train_iter, vocab, num_hiddens, lr, num_epochs,
              use_random_iter=False):
    params = get_params(len(vocab), num_hiddens)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[1, num_epochs])
    updater = tf.keras.optimizers.SGD(lr)
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, params)
    # Train and check the progress.
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
             model, train_iter, loss, updater, params, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    device = d2l.try_gpu()._device_name
    print(f'perplexity {ppl:.1f}, {speed:.1f} tokens/sec on {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))
```

Now we can train a model. Since we only use $10,000$ tokens in the dataset, the model needs more epochs to converge.

```{.python .input  n=31}
#@tab mxnet,pytorch
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, d2l.try_gpu())
```

```{.json .output n=31}
[
 {
  "ename": "TypeError",
  "evalue": "'_EagerDeviceContext' object cannot be interpreted as an integer",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mTypeError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-31-db52f72f7589>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#@tab mxnet,pytorch\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m20\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mtrain_ch8\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvocab\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mtry_gpu\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m<ipython-input-30-041197592a78>\u001b[0m in \u001b[0;36mtrain_ch8\u001b[0;34m(model, train_iter, vocab, num_hiddens, lr, num_epochs, use_random_iter)\u001b[0m\n\u001b[1;32m     10\u001b[0m     \u001b[0mpredict\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;32mlambda\u001b[0m \u001b[0mprefix\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mpredict_ch8\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mprefix\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m50\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvocab\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mparams\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;31m# Train and check the progress.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 12\u001b[0;31m     \u001b[0;32mfor\u001b[0m \u001b[0mepoch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnum_epochs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     13\u001b[0m         ppl, speed = train_epoch_ch8(\n\u001b[1;32m     14\u001b[0m              model, train_iter, loss, updater, params, use_random_iter)\n",
   "\u001b[0;31mTypeError\u001b[0m: '_EagerDeviceContext' object cannot be interpreted as an integer"
  ]
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"170.777344pt\" version=\"1.1\" viewBox=\"0 0 240.554688 170.777344\" width=\"240.554688pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2020-09-06T19:53:21.043787</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.3.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 170.777344 \nL 240.554688 170.777344 \nL 240.554688 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 146.899219 \nL 225.403125 146.899219 \nL 225.403125 10.999219 \nL 30.103125 10.999219 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m28dca0299e\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m28dca0299e\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0.0 -->\n      <g transform=\"translate(22.151563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n        <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"69.163125\" xlink:href=\"#m28dca0299e\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 0.2 -->\n      <g transform=\"translate(61.211563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"108.223125\" xlink:href=\"#m28dca0299e\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 0.4 -->\n      <g transform=\"translate(100.271563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"147.283125\" xlink:href=\"#m28dca0299e\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 0.6 -->\n      <g transform=\"translate(139.331563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"186.343125\" xlink:href=\"#m28dca0299e\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 0.8 -->\n      <g transform=\"translate(178.391563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#m28dca0299e\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 1.0 -->\n      <g transform=\"translate(217.451563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_7\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"md1ed7d1af3\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#md1ed7d1af3\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.0 -->\n      <g transform=\"translate(7.2 150.698437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#md1ed7d1af3\" y=\"119.719219\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.2 -->\n      <g transform=\"translate(7.2 123.518437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#md1ed7d1af3\" y=\"92.539219\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.4 -->\n      <g transform=\"translate(7.2 96.338437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#md1ed7d1af3\" y=\"65.359219\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.6 -->\n      <g transform=\"translate(7.2 69.158437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#md1ed7d1af3\" y=\"38.179219\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0.8 -->\n      <g transform=\"translate(7.2 41.978437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#md1ed7d1af3\" y=\"10.999219\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 14.798437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 146.899219 \nL 30.103125 10.999219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 146.899219 \nL 225.403125 10.999219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 146.899219 \nL 225.403125 146.899219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 10.999219 \nL 225.403125 10.999219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n </g>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=41}
#@tab tensorflow
num_epochs, lr = 100, 1
train_ch8(model, train_iter, vocab, num_hiddens, lr, num_epochs)
```

```{.json .output n=41}
[
 {
  "ename": "ValueError",
  "evalue": "Shape mismatch: The shape of labels (received (31360,)) should equal the shape of logits except for the last dimension (received (1120, 28)).",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mValueError\u001b[0m                                Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-41-5f9968655514>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      1\u001b[0m \u001b[0;31m#@tab tensorflow\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      2\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m100\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 3\u001b[0;31m \u001b[0mtrain_ch8\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mtrain_iter\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mvocab\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_hiddens\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mnum_epochs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;32m<ipython-input-38-1222bc661cb1>\u001b[0m in \u001b[0;36mtrain_ch8\u001b[0;34m(model, train_iter, vocab, num_hiddens, lr, num_epochs, use_random_iter)\u001b[0m\n\u001b[1;32m     11\u001b[0m     \u001b[0;31m# Train and check the progress.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     12\u001b[0m     \u001b[0;32mfor\u001b[0m \u001b[0mepoch\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnum_epochs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 13\u001b[0;31m         ppl, speed = train_epoch_ch8(\n\u001b[0m\u001b[1;32m     14\u001b[0m              model, train_iter, loss, updater, params, use_random_iter)\n\u001b[1;32m     15\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mepoch\u001b[0m \u001b[0;34m%\u001b[0m \u001b[0;36m10\u001b[0m \u001b[0;34m==\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m<ipython-input-40-64889fd5c9c6>\u001b[0m in \u001b[0;36mtrain_epoch_ch8\u001b[0;34m(model, train_iter, loss, updater, params, use_random_iter)\u001b[0m\n\u001b[1;32m     16\u001b[0m             \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreshape\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mY\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m(\u001b[0m\u001b[0;34m-\u001b[0m\u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     17\u001b[0m             \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mkeras\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mactivations\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0msoftmax\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mpy\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 18\u001b[0;31m             \u001b[0ml\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mmath\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mreduce_mean\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mloss\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mpy\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     19\u001b[0m         \u001b[0mgrads\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mg\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mgradient\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0ml\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mparams\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     20\u001b[0m         \u001b[0mgrads\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mgrad_clipping\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mgrads\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;36m1\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.local/lib/python3.8/site-packages/tensorflow/python/keras/losses.py\u001b[0m in \u001b[0;36m__call__\u001b[0;34m(self, y_true, y_pred, sample_weight)\u001b[0m\n\u001b[1;32m    141\u001b[0m         y_true, y_pred, sample_weight)\n\u001b[1;32m    142\u001b[0m     \u001b[0;32mwith\u001b[0m \u001b[0mK\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mname_scope\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_name_scope\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mgraph_ctx\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 143\u001b[0;31m       \u001b[0mlosses\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcall\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_true\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    144\u001b[0m       return losses_utils.compute_weighted_loss(\n\u001b[1;32m    145\u001b[0m           losses, sample_weight, reduction=self._get_reduction())\n",
   "\u001b[0;32m~/.local/lib/python3.8/site-packages/tensorflow/python/keras/losses.py\u001b[0m in \u001b[0;36mcall\u001b[0;34m(self, y_true, y_pred)\u001b[0m\n\u001b[1;32m    244\u001b[0m       y_pred, y_true = tf_losses_util.squeeze_or_expand_dimensions(\n\u001b[1;32m    245\u001b[0m           y_pred, y_true)\n\u001b[0;32m--> 246\u001b[0;31m     \u001b[0;32mreturn\u001b[0m \u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfn\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_true\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0;34m**\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0m_fn_kwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    247\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    248\u001b[0m   \u001b[0;32mdef\u001b[0m \u001b[0mget_config\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.local/lib/python3.8/site-packages/tensorflow/python/keras/losses.py\u001b[0m in \u001b[0;36msparse_categorical_crossentropy\u001b[0;34m(y_true, y_pred, from_logits, axis)\u001b[0m\n\u001b[1;32m   1555\u001b[0m   \u001b[0my_pred\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mops\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mconvert_to_tensor_v2\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_pred\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1556\u001b[0m   \u001b[0my_true\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mmath_ops\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mcast\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my_true\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_pred\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mdtype\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1557\u001b[0;31m   return K.sparse_categorical_crossentropy(\n\u001b[0m\u001b[1;32m   1558\u001b[0m       y_true, y_pred, from_logits=from_logits, axis=axis)\n\u001b[1;32m   1559\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.local/lib/python3.8/site-packages/tensorflow/python/keras/backend.py\u001b[0m in \u001b[0;36msparse_categorical_crossentropy\u001b[0;34m(target, output, from_logits, axis)\u001b[0m\n\u001b[1;32m   4655\u001b[0m           labels=target, logits=output)\n\u001b[1;32m   4656\u001b[0m   \u001b[0;32melse\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 4657\u001b[0;31m     res = nn.sparse_softmax_cross_entropy_with_logits_v2(\n\u001b[0m\u001b[1;32m   4658\u001b[0m         labels=target, logits=output)\n\u001b[1;32m   4659\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.local/lib/python3.8/site-packages/tensorflow/python/ops/nn_ops.py\u001b[0m in \u001b[0;36msparse_softmax_cross_entropy_with_logits_v2\u001b[0;34m(labels, logits, name)\u001b[0m\n\u001b[1;32m   3588\u001b[0m       \u001b[0mof\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mlabels\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mnot\u001b[0m \u001b[0mequal\u001b[0m \u001b[0mto\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mrank\u001b[0m \u001b[0mof\u001b[0m \u001b[0mthe\u001b[0m \u001b[0mlogits\u001b[0m \u001b[0mminus\u001b[0m \u001b[0mone\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3589\u001b[0m   \"\"\"\n\u001b[0;32m-> 3590\u001b[0;31m   return sparse_softmax_cross_entropy_with_logits(\n\u001b[0m\u001b[1;32m   3591\u001b[0m       labels=labels, logits=logits, name=name)\n\u001b[1;32m   3592\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m~/.local/lib/python3.8/site-packages/tensorflow/python/ops/nn_ops.py\u001b[0m in \u001b[0;36msparse_softmax_cross_entropy_with_logits\u001b[0;34m(_sentinel, labels, logits, name)\u001b[0m\n\u001b[1;32m   3502\u001b[0m     if (static_shapes_fully_defined and\n\u001b[1;32m   3503\u001b[0m         labels_static_shape != logits.get_shape()[:-1]):\n\u001b[0;32m-> 3504\u001b[0;31m       raise ValueError(\"Shape mismatch: The shape of labels (received %s) \"\n\u001b[0m\u001b[1;32m   3505\u001b[0m                        \u001b[0;34m\"should equal the shape of logits except for the last \"\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   3506\u001b[0m                        \"dimension (received %s).\" % (labels_static_shape,\n",
   "\u001b[0;31mValueError\u001b[0m: Shape mismatch: The shape of labels (received (31360,)) should equal the shape of logits except for the last dimension (received (1120, 28))."
  ]
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"170.777344pt\" version=\"1.1\" viewBox=\"0 0 240.554688 170.777344\" width=\"240.554688pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <metadata>\n  <rdf:RDF xmlns:cc=\"http://creativecommons.org/ns#\" xmlns:dc=\"http://purl.org/dc/elements/1.1/\" xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n   <cc:Work>\n    <dc:type rdf:resource=\"http://purl.org/dc/dcmitype/StillImage\"/>\n    <dc:date>2020-09-06T21:30:28.576585</dc:date>\n    <dc:format>image/svg+xml</dc:format>\n    <dc:creator>\n     <cc:Agent>\n      <dc:title>Matplotlib v3.3.0, https://matplotlib.org/</dc:title>\n     </cc:Agent>\n    </dc:creator>\n   </cc:Work>\n  </rdf:RDF>\n </metadata>\n <defs>\n  <style type=\"text/css\">*{stroke-linecap:butt;stroke-linejoin:round;}</style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 170.777344 \nL 240.554688 170.777344 \nL 240.554688 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 30.103125 146.899219 \nL 225.403125 146.899219 \nL 225.403125 10.999219 \nL 30.103125 10.999219 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m64bd59b029\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#m64bd59b029\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0.0 -->\n      <g transform=\"translate(22.151563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n        <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_2\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"69.163125\" xlink:href=\"#m64bd59b029\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 0.2 -->\n      <g transform=\"translate(61.211563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_3\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"108.223125\" xlink:href=\"#m64bd59b029\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 0.4 -->\n      <g transform=\"translate(100.271563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"147.283125\" xlink:href=\"#m64bd59b029\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 0.6 -->\n      <g transform=\"translate(139.331563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_5\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"186.343125\" xlink:href=\"#m64bd59b029\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 0.8 -->\n      <g transform=\"translate(178.391563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"225.403125\" xlink:href=\"#m64bd59b029\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 1.0 -->\n      <g transform=\"translate(217.451563 161.497656)scale(0.1 -0.1)\">\n       <defs>\n        <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n       </defs>\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_7\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"me2a2de4c6a\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#me2a2de4c6a\" y=\"146.899219\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 0.0 -->\n      <g transform=\"translate(7.2 150.698437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#me2a2de4c6a\" y=\"119.719219\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- 0.2 -->\n      <g transform=\"translate(7.2 123.518437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_9\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#me2a2de4c6a\" y=\"92.539219\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.4 -->\n      <g transform=\"translate(7.2 96.338437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#me2a2de4c6a\" y=\"65.359219\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.6 -->\n      <g transform=\"translate(7.2 69.158437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_11\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#me2a2de4c6a\" y=\"38.179219\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0.8 -->\n      <g transform=\"translate(7.2 41.978437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-56\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"30.103125\" xlink:href=\"#me2a2de4c6a\" y=\"10.999219\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 1.0 -->\n      <g transform=\"translate(7.2 14.798437)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 30.103125 146.899219 \nL 30.103125 10.999219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 225.403125 146.899219 \nL 225.403125 10.999219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 30.103125 146.899219 \nL 225.403125 146.899219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 30.103125 10.999219 \nL 225.403125 10.999219 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n </g>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

Finally let us check the results to use a random sampling iterator.

```{.python .input}
#@tab mxnet,pytorch
train_ch8(model, train_iter, vocab, lr, num_epochs, d2l.try_gpu(),
          use_random_iter=True)
```

```{.python .input}
#@tab tensorflow
params = get_params(len(vocab_random_iter), num_hiddens)
train_ch8(model, train_random_iter, vocab_random_iter, num_hiddens,
          lr, num_epochs, use_random_iter=True)
```

```{.python .input  n=26}
i=0
for X, y in train_iter:
    print(X)
    print(d2l.reshape(y, (-1)))
    if i==4:
        break
    i+=1
    
```

```{.json .output n=26}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "tf.Tensor(\n[[ 5 13  2 ...  2  1  3]\n [ 3  1  3 ...  8  8  2]\n [ 4  1 12 ... 12  4 25]\n ...\n [ 7  6 26 ... 14  3 21]\n [ 8  3  4 ...  3  1 21]\n [15  9  4 ...  6 11  1]], shape=(32, 35), dtype=int32)\ntf.Tensor([13  2  1 ... 11  1  8], shape=(1120,), dtype=int32)\ntf.Tensor(\n[[10  4 22 ...  6  5  2]\n [11  1  5 ... 18  1  9]\n [ 5 12 19 ...  3  9  5]\n ...\n [ 2 16  7 ...  2  1  3]\n [ 2  1  2 ... 14  6 12]\n [ 8  4  3 ...  6  1  3]], shape=(32, 35), dtype=int32)\ntf.Tensor([ 4 22  2 ...  1  3  9], shape=(1120,), dtype=int32)\ntf.Tensor(\n[[ 6  3  1 ...  1  4  1]\n [ 5  8  1 ... 14  8  1]\n [ 8  1  6 ... 11  1  9]\n ...\n [ 9  2  1 ...  5 12 21]\n [ 2  8  8 ...  2 20  3]\n [ 9  2  3 ...  3  9  2]], shape=(32, 35), dtype=int32)\ntf.Tensor([3 1 3 ... 9 2 1], shape=(1120,), dtype=int32)\ntf.Tensor(\n[[10  2 15 ...  2 19  2]\n [10  4  3 ... 14 20  7]\n [ 5  8  1 ...  4 10  2]\n ...\n [19  1  8 ...  3  9  2]\n [ 2 11  1 ...  1  3  9]\n [ 1 21 10 ...  3  9  2]], shape=(32, 35), dtype=int32)\ntf.Tensor([ 2 15  7 ...  9  2  1], shape=(1120,), dtype=int32)\ntf.Tensor(\n[[ 8  1  8 ... 12 12 19]\n [ 6  1  4 ...  3  2 10]\n [16 14 12 ...  1  7  6]\n ...\n [ 1  3  5 ...  1 17  4]\n [ 5  6 18 ...  4 18  7]\n [ 1 13  7 ...  1 11  7]], shape=(32, 35), dtype=int32)\ntf.Tensor([ 1  8  9 ... 11  7 25], shape=(1120,), dtype=int32)\n"
 }
]
```

```{.python .input}
for X, y in 
```

While implementing the above RNN model from scratch is instructive, it is not convenient. In the next section we will see how to improve significantly on the current model and how to make it faster and easier to implement.


## Summary

* Sequence models need state initialization for training.
* Between sequential models you need to ensure to detach the gradients, to ensure that the automatic differentiation does not propagate effects beyond the current sample.
* A simple RNN language model consists of an encoder, an RNN model, and a decoder.
* Gradient clipping prevents gradient explosion (but it cannot fix vanishing gradients).
* Perplexity calibrates model performance across different sequence length. It is the exponentiated average of the cross-entropy loss.
* Sequential partitioning typically leads to better models.

## Exercises

1. Show that one-hot encoding is equivalent to picking a different embedding for each object.
1. Adjust the hyperparameters to improve the perplexity.
    * How low can you go? Adjust embeddings, hidden units, learning rate, etc.
    * How well will it work on other books by H. G. Wells, e.g., [The War of the Worlds](http://www.gutenberg.org/ebooks/36).
1. Modify the predict function such as to use sampling rather than picking the most likely next character.
    * What happens?
    * Bias the model towards more likely outputs, e.g., by sampling from $q(w_t \mid w_{t-1}, \ldots, w_1) \propto p^\alpha(w_t \mid w_{t-1}, \ldots, w_1)$ for $\alpha > 1$.
1. Run the code in this section without clipping the gradient. What happens?
1. Change sequential partitioning so that it does not separate hidden states from the computational graph. Does the running time change? How about the accuracy?
1. Replace the activation function used in this section with ReLU and repeat the experiments in this section.
1. Prove that the perplexity is the inverse of the harmonic mean of the conditional word probabilities.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/336)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/486)
:end_tab:
