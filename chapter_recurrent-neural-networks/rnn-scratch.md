# Implementation of a Recurrent Neural Network from Scratch

In this section, we will implement a language model based on a character-level recurrent neural network from scratch and train the model on the Jay Chou album lyrics data set to teach it to write lyrics. First, we read the Jay Chou album lyrics data set.

```{.python .input  n=1}
import gluonbook as gb
import math
from mxnet import autograd, nd
from mxnet.gluon import loss as gloss
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = gb.load_data_jay_lyrics()
```

## One-hot Vector

One-hot vectors provide an easy way to express words as vectors in order to input them in the neural network. Assume the number of different characters in the dictionary is $N$  (the `vocab_size`) and each character has a one-to-one correspondence with a single value in the index of successive integers from 0 to $N-1$. If the index of a character is the integer $i$, then we create a vector of all 0s with a length of $N$ and set the element at position $i$ to 1. This vector is the one-hot vector of the original character. The one-hot vectors with indices 0 and 2 are shown below, and the length of the vector is equal to the dictionary size.

```{.python .input  n=2}
nd.one_hot(nd.array([0, 2]), vocab_size)
```

The shape of the mini-batch we sample each time is (batch size, time step). The following function transforms such mini-batches into a number of matrices with the shape of (batch size, dictionary size) that can be entered into the network. The total number of vectors is equal to the number of time steps. That is, the input of time step $t$ is $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$, where $n$ is the batch size and $d$ is the number of inputs. That is the one-hot vector length (the dictionary size).

```{.python .input  n=3}
def to_onehot(X, size):  # This function is saved in the gluonbook package for future use.
    return [nd.one_hot(x, size) for x in X.T]

X = nd.arange(10).reshape((2, 5))
inputs = to_onehot(X, vocab_size)
len(inputs), inputs[0].shape
```

## Initialize Model Parameters

Next, we initialize the model parameters. The number of hidden units `num_hiddens` is a hyper-parameter.

```{.python .input  n=4}
num_inputs, num_hiddens, num_outputs = vocab_size, 256, vocab_size
ctx = gb.try_gpu()
print('will use', ctx)

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

## Define the Model

We implement this model based on the computational expressions of the recurrent neural network. First, we define the `init_rnn_state` function to return the hidden state at initialization. It returns a tuple consisting of an NDArray with a value of 0 and a shape of (batch size, number of hidden units). Using tuples makes it easier to handle situations where the hidden state contains multiple NDArrays.

```{.python .input  n=5}
def init_rnn_state(batch_size, num_hiddens, ctx):
    return (nd.zeros(shape=(batch_size, num_hiddens), ctx=ctx), )
```

The following `rnn` function defines how to compute the hidden state and output in a time step. The activation function here uses the tanh function. As described in the ["Multilayer Perceptron"](../chapter_deep-learning-basics/mlp.md) section, the mean value of tanh function values is 0 when the elements are evenly distributed over the real number field.

```{.python .input  n=6}
def rnn(inputs, state, params):
    # Both inputs and outputs are composed of num_steps matrices of the shape (batch_size, vocab_size).
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    for X in inputs:
        H = nd.tanh(nd.dot(X, W_xh) + nd.dot(H, W_hh) + b_h)
        Y = nd.dot(H, W_hq) + b_q
        outputs.append(Y)
    return outputs, (H,)
```

Do a simple test to observe the number of output results (number of time steps), as well as the output layer output shape and hidden state shape of the first time step.

```{.python .input  n=7}
state = init_rnn_state(X.shape[0], num_hiddens, ctx)
inputs = to_onehot(X.as_in_context(ctx), vocab_size)
params = get_params()
outputs, state_new = rnn(inputs, state, params)
len(outputs), outputs[0].shape, state_new[0].shape
```

## Define the Prediction Function

The following function predicts the next `num_chars` characters based on the `prefix` (a string containing several characters). This function is a bit more complicated. In it, we set the recurrent neural unit `rnn` as a function parameter, so that this function can be reused in the other recurrent neural networks described in following sections.

```{.python .input  n=8}
# This function is saved in the gluonbook package for future use.
def predict_rnn(prefix, num_chars, rnn, params, init_rnn_state,
                num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx):
    state = init_rnn_state(1, num_hiddens, ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        # The output of the previous time step is taken as the input of the current time step.
        X = to_onehot(nd.array([output[-1]], ctx=ctx), vocab_size)
        # Calculate the output and update the hidden state.
        (Y, state) = rnn(X, state, params)
        # The input to the next time step is the character in the prefix or the current best predicted character.
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y[0].argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

We test the `predict_rnn` function first. We will create a lyric with a length of 10 characters (regardless of the prefix length) based on the prefix "separate". Because the model parameters are random values, the prediction results are also random.

```{.python .input  n=9}
predict_rnn('分开', 10, rnn, params, init_rnn_state, num_hiddens, vocab_size,
            ctx, idx_to_char, char_to_idx)
```

## Clip Gradients

Gradient vanishing or explosion is more likely to occur in recurrent neural networks. We will explain the reason in subsequent sections of this chapter. In order to deal with gradient explosion, we can clip the gradient. Assume we concatenate the elements of all model parameter gradients into a vector $\boldsymbol{g}$ and set the clipping threshold to $\theta$. In the clipped gradient:

$$ \min\left(\frac{\theta}{\|\boldsymbol{g}\|}, 1\right)\boldsymbol{g}$$

the $L_2$ norm does not exceed $\theta$.

```{.python .input  n=10}
# This function is saved in the gluonbook package for future use.
def grad_clipping(params, theta, ctx):
    norm = nd.array([0.0], ctx)
    for param in params:
        norm += (param.grad ** 2).sum()
    norm = norm.sqrt().asscalar()
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm
```

## Perplexity

We generally use perplexity to evaluate the quality of a language model. Recall the definition of the cross entropy loss function in the [“Softmax Regression”](../chapter_deep-learning-basics/softmax-regression.md) section. Perplexity is the value obtained by exponentially computing the cross entropy loss function. In particular:

* In the best case scenario, the model always predicts the probability of the label category as 1. In this situation, the perplexity is 1.
* In the worst case scenario, the model always predicts the probability of the label category as 0. In this situation, the perplexity is positive infinity.
* At the baseline, the model always predicts the same probability for all categories. In this situation, the perplexity is the number of categories.

Obviously, the perplexity of any valid model must be less than the number of categories. In this case, the perplexity must be less than the dictionary size `vocab_size`.

## Define Model Training Functions

Compared with the model training functions of the previous chapters, the model training functions here are different in the following ways:

1. We use perplexity to evaluate the model.
2. We clip the gradient before updating the model parameters.
3. Different sampling methods for timing data will result in differences in the initialization of hidden states. For a discussion of these issues, please refer to the ["Language Model Data Set (Jay Chou Album Lyrics)"](lang-model-dataset.md) section.

In addition, considering the other recurrent neural networks that will be described later, the function implementations here are longer, so as to be more general.

```{.python .input  n=11}
# This function is saved in the gluonbook package for future use.
def train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                          vocab_size, ctx, corpus_indices, idx_to_char,
                          char_to_idx, is_random_iter, num_epochs, num_steps,
                          lr, clipping_theta, batch_size, pred_period,
                          pred_len, prefixes):
    if is_random_iter:
        data_iter_fn = gb.data_iter_random
    else:
        data_iter_fn = gb.data_iter_consecutive
    params = get_params()
    loss = gloss.SoftmaxCrossEntropyLoss()

    for epoch in range(num_epochs):
        if not is_random_iter:  # If adjacent sampling is used, the hidden state is initialized at the beginning of the epoch.
            state = init_rnn_state(batch_size, num_hiddens, ctx)
        loss_sum, start = 0.0, time.time()
        data_iter = data_iter_fn(corpus_indices, batch_size, num_steps, ctx)
        for t, (X, Y) in enumerate(data_iter):
            if is_random_iter:  # If random sampling is used, the hidden state is initialized before each mini-batch update.
                state = init_rnn_state(batch_size, num_hiddens, ctx)
            else:  # Otherwise, the detach function needs to be used to separate the hidden state from the computational graph.
                for s in state:
                    s.detach()
            with autograd.record():
                inputs = to_onehot(X, vocab_size)
                # outputs has num_steps matrices of the shape (batch_size, vocab_size).
                (outputs, state) = rnn(inputs, state, params)
                # The shape after stitching is (num_steps * batch_size, vocab_size).
                outputs = nd.concat(*outputs, dim=0)
                # The shape of Y is (batch_size, num_steps), and then becomes a vector with a length of
                # batch * num_steps after transposition. This gives it a one-to-one correspondence with output rows.
                y = Y.T.reshape((-1,))
                # The average classification error is calculated using cross entropy loss.
                l = loss(outputs, y).mean()
            l.backward()
            grad_clipping(params, clipping_theta, ctx)  # Clip the gradient.
            gb.sgd(params, lr, 1)  # Since the error has already taken the mean, the gradient does not need to be averaged.
            loss_sum += l.asscalar()

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(loss_sum / (t + 1)), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn(
                    prefix, pred_len, rnn, params, init_rnn_state,
                    num_hiddens, vocab_size, ctx, idx_to_char, char_to_idx))
```

## Train the Model and Write Lyrics

Now we can train the model. First, set the model hyper-parameter. We will create a lyrics segment with a length of 50 characters (regardless of the prefix length) respectively based on the prefixes "separate" and "not separated". We create a lyrics segment based on the currently trained model every 50 epochs.

```{.python .input  n=12}
num_epochs, num_steps, batch_size, lr, clipping_theta = 200, 35, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['分开', '不分开']
```

Next, we use random sampling to train the model and write lyrics.

```{.python .input  n=13}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, True, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

Then, we use adjacent sampling to train the model and write lyrics.

```{.python .input  n=19}
train_and_predict_rnn(rnn, get_params, init_rnn_state, num_hiddens,
                      vocab_size, ctx, corpus_indices, idx_to_char,
                      char_to_idx, False, num_epochs, num_steps, lr,
                      clipping_theta, batch_size, pred_period, pred_len,
                      prefixes)
```

## Summary

* We can apply a language model based on a character-level recurrent neural network to generate sequences, such as writing lyrics.
* When training a recurrent neural network, we can clip the gradient to cope with gradient explosion.
* Perplexity is the value obtained by exponentially computing the cross entropy loss function.



## exercise

* Adjust the hyper-parameters and observe and analyze the impact on running time, perplexity, and the written lyrics.
* Run the code in this section without clipping the gradient. What happens?
* Set the `pred_period` variable to 1 to observe how the under-trained model (high perplexity) writes lyrics. What can you learn from this?
* Change adjacent sampling so that it does not separate hidden states from the computational graph. Does the running time change?
* Replace the activation function used in this section with ReLU and repeat the experiments in this section.

## Discuss on our Forum

<div id="discuss" topic_id="23"></div>
