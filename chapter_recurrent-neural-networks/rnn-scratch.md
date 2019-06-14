# Implementation of Recurrent Neural Networks from Scratch
:label:`chapter_rnn_scratch`

In this section we implement a language model from scratch. It is based on a character-level recurrent neural network trained on H. G. Wells' *The Time Machine*. As before, we start by reading the data set first, which is introduced in :numref:`chapter_text_preprocessing`. We only use the first $10,000$ tokens in the data set to make the training easy. 

```{.python .input  n=14}
%matplotlib inline
import d2l
import math
from mxnet import autograd, nd, gluon

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## One-hot Encoding

One-hot encoding vectors provide an easy way to express token indices as vectors in order to process them in a deep network. In a nutshell, we map each index to a different unit vector: assume that the number of different characters in the dictionary is $N$ (the `len(vocab)`) and the character indices range from 0 to $N-1$. If the index of a character is the integer $i$, then we create a vector $\mathbf{e}_i$ of all 0s with a length of $N$ and set the element at position $i$ to 1. This vector is the one-hot vector of the original character. The one-hot vectors with indices 0 and 2 are shown below.

```{.python .input  n=21}
nd.one_hot(nd.array([0, 2]), len(vocab))
```

Note that one-hot encodings are just a convenient way of separating the encoding (e.g. mapping the character `a` to $(1,0,0, \ldots)$ vector) from the embedding (i.e. multiplying the encoded vectors by some weight matrix $\mathbf{W}$). This simplifies the code greatly relative to storing an embedding matrix that the user needs to maintain.

The shape of the mini-batch we sample each time is (batch size, time step). The `one_hot` function transforms such a mini-batch into a 3-D tensor with the last dimension equals to the vocabulary size. We also transpose the input so that we will obtain a (time step, batch size, vocabulary size) output that fits into a sequence model easier. 

```{.python .input  n=18}
X = nd.arange(10).reshape((2, 5))
nd.one_hot(X.T, 28).shape
```

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

## RNN Model 

First, we need an `init_rnn_state` function to return the hidden state at initialization. It returns a tuple consisting of an NDArray with a value of 0 and a shape of (batch size, number of hidden units). Using tuples makes it easier to handle situations where the hidden state contains multiple NDArrays (e.g. when combining multiple layers in an RNN where each layers requires initializing).

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
    # inputs shape: (num_steps, batch_size, vocab_size)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return nd.concat(*outputs, dim=0), (H,)
```

With the state initialization and model forward functions, we now can define a RNN class. 

```{.python .input}
# Save to the d2l package.
class RNNModelScratch(object):
    """A RNN Model based on scratch implementations"""
    def __init__(self, vocab_size, num_hiddens, ctx,
                 get_params, init_state, forward):
        self.vocab_size = vocab_size
        self.params = get_params(vocab_size, num_hiddens, ctx)
        self.init_state = init_state
        self.forward_fn = forward
        self.num_hiddens = num_hiddens

    def __call__(self, X, state):
        X = nd.one_hot(X.T, self.vocab_size)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, ctx):
        return self.init_state(batch_size, self.num_hiddens, ctx)
```

Let's run a simple test to check whether the model makes any sense at all. In particular, let's check whether inputs and outputs have the correct dimensions, e.g. to ensure that the dimensionality of the hidden state hasn't changed.

```{.python .input}
vocab_size, num_hiddens, ctx = len(vocab), 512, d2l.try_gpu()
model = RNNModelScratch(len(vocab), num_hiddens, ctx, get_params, 
                        init_rnn_state, rnn)
model.initialize(ctx)
state = model.begin_state(X.shape[0], ctx)
Y, new_state = model(X.as_in_context(ctx), state)
Y.shape, len(new_state), new_state[0].shape
```

We can see that the output shape is (number steps $\times$ batch size, vocabulary size), while the state shape is unchanged. 

## Prediction

The following function predicts the next `num_predicts` characters based on the `prefix` (a string containing several characters). For the beginning of the sequence, we only update the hidden state. After that we begin generating new characters and emitting them. 

```{.python .input}
# Save to the d2l package.
def predict_ch8(prefix, num_predicts, model, vocab, ctx):
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
predict_ch8('traveller ', 10, model, vocab, ctx)
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
def grad_clipping(model, theta):
    if isinstance(model, gluon.Block):
        params = [p.data() for p in model.collect_params().values()]
    else:
        params = model.params
    norm = math.sqrt(sum((p.grad ** 2).sum().asscalar() for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## Training 

Similar to :numref:`chapter_linear_scratch`, let's first define the function to train the model on one data epoch. It differs to the models training from previous chapters in three places:

1. Different sampling methods for sequential data (independent sampling and
   sequential partitioning) will result in differences in the initialization of
   hidden states. 
1. We clip the gradient before updating the model parameters. This ensures that the model doesn't diverge even when gradients blow up at some point during the training process (effectively it reduces the stepsize automatically).
1. We use perplexity to evaluate the model. This ensures that different tests are comparable.

When the consecutive sampling is used, we initialize the hidden state at the beginning of each epoch. Since the $i$-th example in the next mini-batch is adjacent to the current $i$-th example, so we next mini-batch can use the current hidden state directly, we only detach the gradient so that we only compute the gradients within a mini-batch. When using the random sampling, we need to re-initialize the hidden state for each iteration since each example is sampled with a random position. Same to the `train_epoch_ch3` function (:numref:`chapter_linear_scratch`), we use generalized `updater`, which could be a Gluon trainer or a scratched implementation. 

```{.python .input}
# Save to the d2l package.
def train_epoch_ch8(model, train_iter, loss, updater, ctx, use_random_iter):
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # loss_sum, num_examples
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # Initialize state when either it's the first iteration or 
            # using random sampling.
            state = model.begin_state(batch_size=X.shape[0], ctx=ctx)
        else:            
            for s in state: s.detach()
        y = Y.T.reshape((-1,))
        X, y = X.as_in_context(ctx), y.as_in_context(ctx) 
        with autograd.record():
            py, state = model(X, state)
            l = loss(py, y).mean()
        l.backward()
        grad_clipping(model, 1)
        updater(batch_size=1)  # Since used mean already.
        metric.add((l.asscalar() * y.size, y.size))
    return math.exp(metric[0]/metric[1]), metric[1]/timer.stop()
```

The training function 

```{.python .input  n=11}
# Save to the d2l package.
def train_ch8(model, train_iter, vocab, lr, num_epochs, ctx, 
              use_random_iter=False):
    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity', 
                            legend=['train'], xlim=[1, num_epochs])
    if isinstance(model, gluon.Block):
        model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
        trainer = gluon.Trainer(model.collect_params(), 'sgd', {'learning_rate': lr})
        updater = lambda batch_size : trainer.step(batch_size)
    else:
        updater = lambda batch_size : d2l.sgd(model.params, lr, batch_size)
    
    predict = lambda prefix: predict_ch8(prefix, 50, model, vocab, ctx)
    
    if hasattr(updater, 'step'): updater = updater.step  
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            model, train_iter, loss, updater, ctx, use_random_iter)
        if epoch % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch+1, [ppl])
    print('Perplexity %.1f, %d tokens/sec on %s' % (ppl, speed, ctx))
    print(predict('time traveller'))
    print(predict('traveller'))
```

Train a model.

```{.python .input}
num_epochs, lr = 500, 1
train_ch8(model, train_iter, vocab, lr, num_epochs, ctx, use_random_iter=False)
```

```{.python .input}
train_ch8(model, train_iter, vocab, updater, num_epochs, ctx, use_random_iter=True)
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
