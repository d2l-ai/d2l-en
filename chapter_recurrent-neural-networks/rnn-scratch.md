# Implementation of Recurrent Neural Networks from Scratch
:label:`chapter_rnn_scratch`

In this section we implement a language model from scratch. It is based on a character-level recurrent neural network trained on H. G. Wells' *The Time Machine*. As before, we start by reading the data set first, which is introduced in :numref:`chapter_text_preprocessing`.

```{.python .input  n=14}
import d2l
import math
from mxnet import autograd, nd, gluon

corpus, vocab = d2l.load_data_time_machine()
```

## One-hot Encoding

One-hot encoding vectors provide an easy way to express token indices as vectors in order to process them in a deep network. In a nutshell, we map each index to a different unit vector: assume that the number of different characters in the dictionary is $N$ (the `len(vocab)`) and the character indices range from 0 to $N-1$. If the index of a character is the integer $i$, then we create a vector $\mathbf{e}_i$ of all 0s with a length of $N$ and set the element at position $i$ to 1. This vector is the one-hot vector of the original character. The one-hot vectors with indices 0 and 2 are shown below.

```{.python .input  n=21}
nd.one_hot(nd.array([0, 2]), len(vocab))
```

Note that one-hot encodings are just a convenient way of separating the encoding (e.g. mapping the character `a` to $(1,0,0, \ldots)$ vector) from the embedding (i.e. multiplying the encoded vectors by some weight matrix $\mathbf{W}$). This simplifies the code greatly relative to storing an embedding matrix that the user needs to maintain.

The shape of the mini-batch we sample each time is (batch size, time step). The following function transforms such mini-batches into a number of matrices with the shape of (batch size, dictionary size) that can be entered into the network. The total number of matrices is equal to the number of time steps. That is, the input of time step $t$ is $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$, where $n$ is the batch size and $d$ is the number of inputs. That is the one-hot vector length (the dictionary size).

```{.python .input  n=18}
X = nd.arange(10).reshape((2, 5))
nd.one_hot(X.T, 28).shape
```

The code above generates 5 minibatches containing 2 vectors each. Since we have a total of 43 distinct symbols in "The Time Machine" we get 43-dimensional vectors.

## Initializing the Model Parameters

Next, we initialize the model parameters. The number of hidden units `num_hiddens` is a tunable parameter.

```{.python .input  n=19}
def get_params(vocab_size, num_hiddens, ctx):
    num_inputs = num_outputs = vocab_size
    normal = lambda shape: nd.random.normal(scale=0.01, shape=shape, ctx=ctx)
    # Hidden layer parameters
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = nd.zeros(num_outputs, ctx=ctx)
    # Attach a gradient
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.attach_grad()
    return params
```

## Sequence Modeling

### RNN Model

We implement this model based on the definition of an RNN. First, we need an `init_rnn_state` function to return the hidden state at initialization. It returns a tuple consisting of an NDArray with a value of 0 and a shape of (batch size, number of hidden units). Using tuples makes it easier to handle situations where the hidden state contains multiple NDArrays (e.g. when combining multiple layers in an RNN where each layers requires initializing).

```{.python .input  n=20}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

The following `rnn` function defines how to compute the hidden state and output
in a time step. The activation function here uses the tanh function. As
described in :numref:`chapter_mlp`, the
mean value of the $\tanh$ function values is 0 when the elements are evenly
distributed over the real numbers.

```{.python .input  n=6}
def rnn(inputs, state, params):
    # Both inputs and outputs are composed of num_steps matrices of the shape
    # (batch_size, len(vocab))
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)#.expand_dims(axis=0))
    #return nd.concatenate(outputs), (H,)
    return nd.concat(*outputs, dim=0), (H,)
```

```{.python .input}
# Save to the d2l package.
class RNNModelScratch(object):
    def __init__(self, vocab_size, num_hiddens,
                 get_params, init_state, forward):
        self.num_hiddens, self.vocab_size = num_hiddens, vocab_size
        self.get_params, self.init_state = get_params, init_state
        self.forward_fn = forward

    def initialize(self, ctx, **kwargs):
        """xxx"""
        self.params = get_params(self.vocab_size, self.num_hiddens, ctx)

    def __call__(self, X, state):
        X = nd.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)


```

Let's run a simple test to check whether the model makes any sense at all. In particular, let's check whether inputs and outputs have the correct dimensions, e.g. to ensure that the dimensionality of the hidden state hasn't changed.

```{.python .input}
vocab_size, num_hiddens, ctx = len(vocab), 512, d2l.try_gpu()

#state = init_rnn_state(X.shape[0], num_hiddens, ctx)
#inputs = nd.one_hot(X.as_in_context(ctx).T, len(vocab))
#params = get_params(vocab_size, num_hiddens, ctx)
#outputs, state_new = rnn(inputs, state, params)
#outputs.shape, outputs[0].shape, state_new[0].shape

model = RNNModelScratch(len(vocab), num_hiddens, get_params,
                        init_rnn_state, rnn)
model.initialize(ctx)
state = model.begin_state(X.shape[0], ctx)
Y, new_state = model(X.as_in_context(ctx), state)
Y.shape, len(new_state), new_state[0].shape
```

### Prediction Function

The following function predicts the next `num_chars` characters based on the `prefix` (a string containing several characters). This function is a bit more complicated. Whenever the actual sequence is known, i.e. for the beginning of the sequence, we only update the hidden state. After that we begin generating new characters and emitting them. For convenience we use the recurrent neural unit `rnn` as a function parameter, so that this function can be reused in the other recurrent neural networks described in following sections.

```{.python .input}
# Save to the d2l package.
def predict_ch9(prefix, num_predicts, model, vocab, ctx):
    state = model.begin_state(batch_size=1, ctx=ctx)
    outputs = [vocab[prefix[0]]]
    get_input = lambda: nd.array([outputs[-1]], ctx=ctx).reshape((1, 1))
    for y in prefix[1:]:  # Warmup state with prefix
        _, state = model(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_predicts):  # Predict num_predicts steps
        Y, state = model(get_input(), state)
        outputs.append(int(Y.argmax(axis=1).reshape(1).asscalar()))
    return ''.join([vocab.idx_to_token[i] for i in outputs])
```

We test the `predict_rnn` function first. Given that we didn't train the network it will generate nonsensical predictions. We initialize it with the sequence `traveller ` and have it generate 10 additional characters.

```{.python .input  n=9}
predict_ch9('traveller ', 10, model, vocab, ctx)
```

## Gradient Clipping

When solving an optimization problem we take update steps for the
weights $\mathbf{w}$ in the general direction of the negative gradient
$\mathbf{g}_t$ on a minibatch, say $\mathbf{w} - \eta \cdot \mathbf{g}_t$. Let's further assume that the objective is well behaved, i.e. it is Lipschitz continuous with constant $L$, i.e.

$$|l(\mathbf{w}) - l(\mathbf{w}')| \leq L \|\mathbf{w} - \mathbf{w}'\|.$$

In this case we can safely assume that if we update the weight vector by $\eta \cdot \mathbf{g}_t$ we will not observe a change by more than $L \eta \|\mathbf{g}_t\|$. This is both a curse and a blessing. A curse since it limits the speed with which we can make progress, a blessing since it limits the extent to which things can go wrong if we move in the wrong direction.

Sometimes the gradients can be quite large and the optimization algorithm may fail to converge. We could address this by reducing the learning rate $\eta$ or by some other higher order trick. But what if we only rarely get large gradients? In this case such an approach may appear entirely unwarranted. One alternative is to clip the gradients by projecting them back to a ball of a given radius, say $\theta$ via

$$\mathbf{g} \leftarrow \min\left(1, \frac{\theta}{\|\mathbf{g}\|}\right) \mathbf{g}.$$

By doing so we know that the gradient norm never exceeds $\theta$ and that the updated gradient is entirely aligned with the original direction $\mathbf{g}$. It also has the desirable side-effect of limiting the influence any given minibatch (and within it any given sample) can exert on the weight vectors. This bestows a certain degree of robustness to the model. Back to the case at hand - optimization in RNNs. One of the issues is that the gradients in an RNN may either explode or vanish. Consider the chain of matrix-products involved in backpropagation. If the largest eigenvalue of the matrices is typically larger than $1$, then the product of many such matrices can be much larger than $1$. As a result, the aggregate gradient might explode. Gradient clipping provides a quick fix. While it doesn't entire solve the problem, it is one of the many techniques to alleviate it.

```{.python .input  n=10}
# Save to the d2l package.
def grad_clipping(model, theta, ctx):
    if hasattr(model, 'params'):
        params = model.params
    else:  # model is a nn.Block object.
        params = [p.data(ctx) for p in model.collect_params().values()]
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## Training the Model

Training a sequence model proceeds quite different from previous codes. In particular we need to take care of the following changes due to the fact that the tokens appear in order:

1. We use perplexity to evaluate the model. This ensures that different tests are comparable.
1. We clip the gradient before updating the model parameters. This ensures that the model doesn't diverge even when gradients blow up at some point during the training process (effectively it reduces the stepsize automatically).
1. Different sampling methods for sequential data (independent sampling and
   sequential partitioning) will result in differences in the initialization of
   hidden states. We discussed these issues in detail when we covered
   :numref:`chapter_lang_model_dataset`.

### Optimization Loop

Now, we can use a hidden state of the last time step of a mini-batch to initialize the hidden state of the next mini-batch, so that the output of the next mini-batch is also dependent on the input of the mini-batch, with this pattern continuing in subsequent mini-batches. This has two effects on the implementation of a recurrent neural network. On the one hand,
when training the model, we only need to initialize the hidden state at the beginning of each epoch.
On the other hand, when multiple adjacent mini-batches are concatenated by passing hidden states, the gradient calculation of the model parameters will depend on all the mini-batch sequences that are concatenated. In the same epoch as the number of iterations increases, the costs of gradient calculation rise.
So that the model parameter gradient calculations only depend on the mini-batch sequence read by one iteration, we can separate the hidden state from the computational graph before reading the mini-batch (this can be done by detaching the graph). We will gain a deeper understand this approach in the following sections.

```{.python .input}
# Save to the d2l package.
def train_epoch_ch9(model, train_iter, loss, updater, batch_size, ctx, use_random_iter):
    timer = d2l.Timer()
    if not use_random_iter:
        # Hidden state is initialized at the beginning of each epoch
        # for the consecutive sampling. Otherwise, will initialize for each
        # iteration.
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
    metric = d2l.Accumulator(2)  # loss_sum, num_examples
    for X, Y in train_iter:
        if use_random_iter:
            state = model.begin_state(batch_size=batch_size, ctx=ctx)
        else:
            # Detach gradient to avoid backpropagation beyond the
            # current batch for the consecutive sampling.
            for s in state: s.detach()
        y = Y.T.reshape((-1,))
        with autograd.record():
            py, state = model(X, state)
            #print(py.shape, Y.T.shape)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1, ctx)
        updater(batch_size=1)  # Since used mean already.
        metric.add((l.asscalar() * y.size, y.size))
    print(timer.stop())
    return math.exp(metric[0]/metric[1])
```

```{.python .input  n=11}
# Save to the d2l package.
def train_ch9(model, corpus, vocab, updater, num_epochs, batch_size, num_steps, 
              ctx, use_random_iter=False):
    if use_random_iter:
        data_iter_fn = d2l.seq_data_iter_random
    else:
        data_iter_fn = d2l.seq_data_iter_consecutive
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='loss', legend=['train'],
                            xlim=[1, num_epochs])
    if hasattr(updater, 'step'): updater = updater.step  # It's Gluon Trainer.
    for epoch in range(num_epochs):
        data_iter = data_iter_fn(corpus, batch_size, num_steps, ctx)
        train_err = train_epoch_ch9(
            model, data_iter, loss, updater, batch_size, ctx, use_random_iter)
        if epoch % 10 == 0:
            print(predict_ch9('time traveller', 50, model, vocab, ctx))
            animator.add(epoch+1, [train_err])

```

```{.python .input}
num_epochs, num_steps, batch_size, lr = 50, 64, 32, 1
updater = lambda batch_size: d2l.sgd(model.params, lr, batch_size)
train_ch9(model, corpus, vocab, updater,  num_epochs, batch_size, num_steps,
              ctx, use_random_iter=False)
```

In the following we will see how to improve significantly on the current model and how to make it faster and easier to implement.

## Summary

* Sequence models need state initialization for training.
* Between sequential models you need to ensure to detach the gradient, to ensure that the automatic differentiation does not propagate effects beyond the current sample.
* A simple RNN language model consists of an encoder, an RNN model and a decoder.
* Gradient clipping prevents gradient explosion (but it cannot fix vanishing gradients).
* Perplexity calibrates model performance across variable sequence length. It is the exponentiated average of the cross-entropy loss.
* Sequential partitioning typically leads to better models.

## Exercises

1. Show that one-hot encoding is equivalent to picking a different embedding for each object.
1. Adjust the hyperparameters to improve the perplexity.
    * How low can you go? Adjust embeddings, hidden units, learning rate, etc.
    * How well will it work on other books by H. G. Wells, e.g. [The War of the Worlds](http://www.gutenberg.org/ebooks/36).
1. Run the code in this section without clipping the gradient. What happens?
1. Set the `pred_period` variable to 1 to observe how the under-trained model (high perplexity) writes lyrics. What can you learn from this?
1. Change adjacent sampling so that it does not separate hidden states from the computational graph. Does the running time change? How about the accuracy?
1. Replace the activation function used in this section with ReLU and repeat the experiments in this section.
1. Prove that the perplexity is the inverse of the harmonic mean of the conditional word probabilities.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2364)

![](../img/qr_rnn-scratch.svg)
