# Implementation of Recurrent Neural Networks from Scratch
:label:`chapter_rnn_scratch`

In this section we implement a language model from scratch. It is based on a character-level recurrent neural network trained on H. G. Wells' 'The Time Machine'. As before, we start by reading the dataset first.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import d2l
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

corpus_indices, vocab = d2l.load_data_time_machine()
```

## One-hot Encoding

One-hot encoding vectors provide an easy way to express words as vectors in order to process them in a deep network. In a nutshell, we map each word to a different unit vector: assume that the number of different characters in the dictionary is $N$ (the `len(vocab)`) and each character has a one-to-one correspondence with a single value in the index of successive integers from 0 to $N-1$. If the index of a character is the integer $i$, then we create a vector $\mathbf{e}_i$ of all 0s with a length of $N$ and set the element at position $i$ to 1. This vector is the one-hot vector of the original character. The one-hot vectors with indices 0 and 2 are shown below (the length of the vector is equal to the dictionary size).

```{.python .input  n=2}
nd.one_hot(nd.array([0, 2]), len(vocab))
```

Note that one-hot encodings are just a convenient way of separating the encoding (e.g. mapping the character `a` to $(1,0,0, \ldots) vector)$ from the embedding (i.e. multiplying the encoded vectors by some weight matrix $\mathbf{W}$). This simplifies the code greatly relative to storing an embedding matrix that the user needs to maintain.

The shape of the mini-batch we sample each time is (batch size, time step). The following function transforms such mini-batches into a number of matrices with the shape of (batch size, dictionary size) that can be entered into the network. The total number of vectors is equal to the number of time steps. That is, the input of time step $t$ is $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$, where $n$ is the batch size and $d$ is the number of inputs. That is the one-hot vector length (the dictionary size).

```{.python .input  n=3}
# This function is saved in the d2l package for future use
def to_onehot(X, size):
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, len(vocab))
len(inputs), inputs[0].shape
```

The code above generates 5 minibatches containing 2 vectors each. Since we have a total of 43 distinct symbols in "The Time Machine" we get 43-dimensional vectors.

## Initializing the Model Parameters

Next, we initialize the model parameters. The number of hidden units `num_hiddens` is a tunable parameter.

```{.python .input  n=4}
num_inputs, num_hiddens, num_outputs = len(vocab), 512, len(vocab)
ctx = d2l.try_gpu()
print('Using', ctx)

# Create the parameters of the model, initialize them and attach gradients
def get_params():
    def _one(shape):
        return nd.random.normal(scale=0.01, shape=shape, ctx=ctx)

    # Hidden layer parameters
    W_xh = _one((num_inputs, num_hiddens))
    W_hh = _one((num_hiddens, num_hiddens))
    b_h = nd.zeros(num_hiddens, ctx=ctx)
    # Output layer parameters
    W_hq = _one((num_hiddens, num_outputs))
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

```{.python .input  n=5}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

The following `rnn` function defines how to compute the hidden state and output
in a time step. The activation function here uses the tanh function. As
described in the multilayer perceptron (:numref:`chapter_mlp`), the
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
        outputs.append(Y)
    return outputs, (H,)
```

Let's run a simple test to check whether the model makes any sense at all. In particular, let's check whether inputs and outputs have the correct dimensions, e.g. to ensure that the dimensionality of the hidden state hasn't changed.

```{.python .input  n=7}
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.as_in_context(ctx), len(vocab))
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape
```

### Prediction Function

The following function predicts the next `num_chars` characters based on the `prefix` (a string containing several characters). This function is a bit more complicated. Whenever the actual sequence is known, i.e. for the beginning of the sequence, we only update the hidden state. After that we begin generating new characters and emitting them. For convenience we use the recurrent neural unit `rnn` as a function parameter, so that this function can be reused in the other recurrent neural networks described in following sections.

```{.python .input  n=8}
# This function is saved in the d2l package for future use
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab, ctx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [vocab[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # The output of the previous time step is taken as the input of the
        # current time step.
        X = to_onehot(nd.array([output[-1]], ctx=ctx), len(vocab))
        # Calculate the output and update the hidden state
        (Y, state) = rnn(X, state, params)
        # The input to the next time step is the character in the prefix or
        # the current best predicted character
        if t < len(prefix) - 1:
            # Read off from the given sequence of characters
            output.append(vocab[prefix[t + 1]])
        else:
            # This is maximum likelihood decoding. Modify this if you want
            # use sampling, beam search or beam sampling for better sequences.
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([vocab.idx_to_token[i] for i in output])
```

We test the `predict_rnn` function first. Given that we didn't train the network it will generate nonsensical predictions. We initialize it with the sequence `traveller ` and have it generate 10 additional characters.

```{.python .input  n=9}
predict_rnn('traveller ', 10, rnn, params, init_rnn_state, num_hiddens,
            vocab, ctx)
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
# This function is saved in the d2l package for future use
def grad_clipping(params, theta, ctx):
    norm = nd.array([0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## Perplexity

One way of measuring how well a sequence model works is to check how surprising the text is. A good language model is able to predict with high accuracy what we will see next. Consider the following continuations of the phrase `It is raining`, as proposed by different language models:

1. `It is raining outside`
1. `It is raining banana tree`
1. `It is raining piouw;kcj pwepoiut`

In terms of quality, example 1 is clearly the best. The words are sensible and logically coherent. While it might not quite so accurately reflect which word follows (`in San Francisco` and `in winter` would have been perfectly reasonable extensions), the model is able to capture which kind of word follows. Example 2 is considerably worse by producing a nonsensical and borderline dysgrammatical extension. Nonetheless, at least the model has learned how to spell words and some degree of correlation between words. Lastly, example 3 indicates a poorly trained model that doesn't fit data.

One way of measuring the quality of the model is to compute $p(w)$, i.e. the likelihood of the sequence. Unfortunately this is a number that is hard to understand and difficult to compare. After all, shorter sequences are *much* more likely than long ones, hence evaluating the model on Tolstoy's magnum opus ['War and Peace'](https://www.gutenberg.org/files/2600/2600-h/2600-h.htm) will inevitably produce a much smaller likelihood than, say, on Saint-Exupery's novella ['The Little Prince'](https://en.wikipedia.org/wiki/The_Little_Prince). What is missing is the equivalent of an average.

Information Theory comes handy here. If we want to compress text we can ask about estimating the next symbol given the current set of symbols. A lower bound on the number of bits is given by $-\log_2 p(w_t|w_{t-1}, \ldots w_1)$. A good language model should allow us to predict the next word quite accurately and thus it should allow us to spend very few bits on compressing the sequence. One way of measuring it is by the average number of bits that we need to spend.

$$\frac{1}{n} \sum_{t=1}^n -\log p(w_t|w_{t-1}, \ldots w_1) = \frac{1}{|w|} -\log p(w)$$

This makes the performance on documents of different lengths comparable. For historical reasons scientists in natural language processing prefer to use a quantity called perplexity rather than bitrate. In a nutshell it is the exponential of the above:

$$\mathrm{PPL} := \exp\left(-\frac{1}{n} \sum_{t=1}^n \log p(w_t|w_{t-1}, \ldots w_1)\right)$$

It can be best understood as the harmonic mean of the number of real choices
that we have when deciding which word to pick next. Note that Perplexity
naturally generalizes the notion of the cross entropy loss defined when we
introduced
the softmax regression (:numref:`chapter_softmax`). That
is, for a single symbol both definitions are identical bar the fact that one is
the exponential of the other. Let's look at a number of cases:

* In the best case scenario, the model always estimates the probability of the next symbol as $1$. In this case the perplexity of the model is $1$.
* In the worst case scenario, the model always predicts the probability of the label category as 0. In this situation, the perplexity is infinite.
* At the baseline, the model predicts a uniform distribution over all tokens. In this case the perplexity equals the size of the dictionary `len(vocab)`. In fact, if we were to store the sequence without any compression this would be the best we could do to encode it. Hence this provides a nontrivial upper bound that any model must satisfy.

## Training the Model

Training a sequence model proceeds quite different from previous codes. In particular we need to take care of the following changes due to the fact that the tokens appear in order:

1. We use perplexity to evaluate the model. This ensures that different tests are comparable.
1. We clip the gradient before updating the model parameters. This ensures that the model doesn't diverge even when gradients blow up at some point during the training process (effectively it reduces the stepsize automatically).
1. Different sampling methods for sequential data (independent sampling and
   sequential partitioning) will result in differences in the initialization of
   hidden states. We discussed these issues in detail when we covered
   :numref:`chapter_lang_model_dataset`.

### Optimization Loop

```{.python .input  n=11}
# This function is saved in the d2l package for future use
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          corpus_indices, vocab, ctx, is_random_iter,
                          num_epochs, num_steps, lr, clipping_theta,
                          batch_size, prefixes):
    if is_random_iter:
        data_iter_fn = d2l.data_iter_random
    else:
        data_iter_fn = d2l.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()
    start = time.time()
    for epoch in range(num_epochs):
        if not is_random_iter:
            # If adjacent sampling is used, the hidden state is initialized
            # at the beginning of the epoch
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        l_sum, n = 0.0, 0
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for X, Y in data_iter:
            if is_random_iter:
                # If random sampling is used, the hidden state is initialized
                # before each mini-batch update
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:
                # Otherwise, the detach function needs to be used to separate
                # the hidden state from the computational graph to avoid
                # backpropagation beyond the current sample
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, len(vocab))
                # outputs is num_steps terms of shape (batch_size, len(vocab))
                (outputs, state) = rnn(inputs, state, params)
                # After stitching it is (num_steps * batch_size, len(vocab))
                outputs = nd.concat(*outputs, dim=0)
                # The shape of Y is (batch_size, num_steps), and then becomes
                # a vector with a length of batch * num_steps after
                # transposition. This gives it a one-to-one correspondence
                # with output rows
                y = Y.T.reshape((-1,))
                # Average classification error via cross entropy loss
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # Clip the gradient
            d2l.sgd(params, lr, 1)
            # Since the error is the mean, no need to average gradients here
            l_sum += l.asscalar() * y.size
            n += y.size
        if (epoch + 1) % 50 == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            start = time.time()
        if (epoch + 1) % 100 == 0:
            for prefix in prefixes:
                print(' -',  predict_rnn(prefix, 50, rnn, params,
                                         init_rnn_state, num_hiddens,
                                         vocab, ctx))
```

### Experiments with a Sequence Model

Now we can train the model. First, we need to set the model hyper-parameters. To allow for some meaningful amount of context we set the sequence length to 64. In particular, we will see how training using the 'separate' and 'sequential' term generation will affect the performance of the model.

```{.python .input  n=12}
num_epochs, num_steps, batch_size, lr, clipping_theta = 500, 64, 32, 1, 1
prefixes = ['traveller', 'time traveller']
```

Let's use random sampling to train the model and produce some text.

```{.python .input  n=13}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      corpus_indices, vocab, ctx, True, num_epochs,
                      num_steps, lr, clipping_theta, batch_size, prefixes)
```

Even though our model was rather primitive, it is nonetheless able to produce text that resembles language. Now let's compare this with sequential partitioning.

```{.python .input  n=19}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      corpus_indices, vocab, ctx, False, num_epochs,
                      num_steps, lr, clipping_theta, batch_size, prefixes)
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
