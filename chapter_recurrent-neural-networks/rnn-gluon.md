# Concise Implementation of Recurrent Neural Networks

While the previous section was instructive to see how recurrent neural networks are implemented, this isn't convenient or fast. The current section will show how to implement the same language model more efficiently using functions provided by the deep learning framework. We begin as before by reading the 'Time Machine' corpus. 

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import d2l
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = d2l.load_data_time_machine()
```

## Defining the Model

Gluon's `rnn` module provides a recurrent neural network implementation (beyond many other sequence models). We construct the recurrent neural network layer `rnn_layer` with a single hidden layer and 256 hidden units, and initialize the weights.

```{.python .input  n=26}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

Initializing the state is straightforward. We invoke the member function `rnn_layer.begin_state(batch_size)`. This returns an initial state for each element in the minibatch. That is, it returns an object that is of size (hidden layers, batch size, number of hidden units). The number of hidden layers defaults to 1. In fact, we haven't even discussed yet what it means to have multiple layers - this will happen [later](deep-rnn.md). For now, suffice it to say that multiple layers simply amount to the output of one RNN being used as the input for the next RNN. 

```{.python .input  n=37}
batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)
state[0].shape
```

Unlike the recurrent neural network implemented in the previous section, the input shape of `rnn_layer` is given by (time step, batch size, number of inputs). In the case of a language model the number of inputs would be the one-hot vector length (the dictionary size). In addition, as an `rnn.RNN` instance in Gluon, `rnn_layer` returns the output and hidden state after forward computation. The output refers to the sequence of hidden states that the RNN computes over various time steps. They are used as input for subsequent output layers. Note that the output does not involve any conversion to characters or any other post-processing. This is so, since the RNN itself has no concept of what to do with the vectors that it generates. In short, its shape is given by (time step, batch size, number of hidden units). 

The hidden state returned by the `rnn.RNN` instance in the forward computation is the state of the hidden layer available at the last time step. This can be used to initialize the next time step: when there are multiple layers in the hidden layer, the hidden state of each layer is recorded in this variable. For recurrent neural networks such as [Long Short Term Memory](lstm.md) (LSTM) networks, the variables also contains other state information. We will introduce LSTM and deep RNNs later in this chapter.

```{.python .input  n=38}
num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

Next, define an `RNNModel` block by subclassing the `Block` class to define a complete recurrent neural network. It first uses one-hot vector embeddings to represent input data and enter it into the `rnn_layer`. This is then used by the fully connected layer to obtain the output. For convenience we set the number of outputs to match the dictionary size `vocab_size`.

```{.python .input  n=39}
# This class has been saved in the d2l package for future use
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # Get the one-hot vector representation by transposing the input to
        # (num_steps, batch_size)
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of Y to
        # (num_steps * batch_size, num_hiddens)
        # Its output shape is (num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## Model Training

As before we need a prediction function. The implementation here differs from the previous one in the function interfaces for forward computation and hidden state initialization. The main difference is that the decoding into characters is now clearly separated from the hidden variable model.

```{.python .input  n=41}
# This function is saved in the d2l package for future use
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    # Use the model's member function to initialize the hidden state.
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        # Forward computation does not require incoming model parameters
        (Y, state) = model(X, state)
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

Let's make a prediction with the a model that has random weights.

```{.python .input  n=42}
ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
predict_rnn_gluon('traveller', 10, model, vocab_size, ctx, idx_to_char,
                  char_to_idx)
```

As is quite obvious, this model doesn't work at all (just yet). Next, we implement the training function. Its algorithm is the same as in the previous section. That said, since we observed that random sampling works considerably better for training sequence models, we only use the latter below.

```{.python .input  n=18}
# This function is saved in the d2l package for future use
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        l_sum, n, start = 0.0, 0, time.time()
        data_iter = d2l.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for X, Y in data_iter:
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # Clip the gradient
            params = [p.data() for p in model.collect_params().values()]
            d2l.grad_clipping(params, clipping_theta, ctx)
            # Since the error has already taken the mean, the gradient does
            # not need to be averaged
            trainer.step(1)
            l_sum += l.asscalar() * y.size
            n += y.size

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(l_sum / n), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size, ctx, idx_to_char,
                    char_to_idx))
```

Let's train the model using the same hyper-parameters as in the previous section. The primary difference is that we are now using built-in functions that are considerably faster than when writing code explicitly in Python.

```{.python .input  n=19}
num_epochs, batch_size, lr, clipping_theta = 500, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['traveller', 'time traveller']
train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
```

As we can see, the model achieves comparable perplexity, albeit within a shorter period of time, due to the code being more optimized. 

## Summary

* Gluon's `rnn` module provides an implementation at the recurrent neural network layer.
* Gluon's `nn.RNN` instance returns the output and hidden state after forward computation. This forward computation does not involve output layer computation.
* As before, the compute graph needs to be detached from previous steps for reasons of efficiency.

## Exercises

1. Compare the implementation with the previous section. 
    * Why does Gluon's implementation run faster? 
    * If you observe a significant difference beyond speed, try to find the reason.
1. Can you make the model overfit?
    * Increase the number of hidden units.
    * Increase the number of iterations.
    * What happens if you adjust the clipping parameter? 
1. Implement the autoregressive model of the introduction to the current chapter using an RNN. 
1. Modify the `predict_rnn_gluon` such as to use sampling rather than picking the most likely next character. 
    * What happens?
    * Bias the model towards more likely outputs, e.g. by sampling from $q(w_t|w_{t-1}, \ldots w_1) \propto p^\alpha(w_t|w_{t-1}, \ldots w_1)$ for $\alpha > 1$.
1. What happens if you increase the number of hidden layers in the RNN model? Can you make the model work?
1. How well can you compress the text using this model?
    * How many bits do you need?
    * Why doesn't everyone use this model for text compression? Hint - what about the compressor itself?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2365)

![](../img/qr_rnn-gluon.svg)
