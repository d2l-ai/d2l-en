# Concise Implementation of Recurrent Neural Networks
:label:`chapter_rnn_gluon`

While :numref:`chapter_rnn_scratch` was instructive to see how recurrent neural networks are implemented, this isn't convenient or fast. The current section will show how to implement the same language model more efficiently using functions provided by Gluon. We begin as before by reading the 'Time Machine" corpus.

```{.python .input  n=1}
import d2l
import math
from mxnet import gluon, init, np, npx
from mxnet.gluon import nn, rnn
npx.set_np()

batch_size, num_steps = 32, 35
train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
```

## Defining the Model

Gluon's `rnn` module provides a recurrent neural network implementation (beyond many other sequence models). We construct the recurrent neural network layer `rnn_layer` with a single hidden layer and 256 hidden units, and initialize the weights.

```{.python .input  n=26}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

Initializing the state is straightforward. We invoke the member function `rnn_layer.begin_state(batch_size)`. This returns an initial state for each element in the minibatch. That is, it returns an object that is of size (hidden layers, batch size, number of hidden units). The number of hidden layers defaults to 1. In fact, we haven't even discussed yet what it means to have multiple layers - this will happen in :numref:`chapter_deep_rnn`. For now, suffice it to say that multiple layers simply amount to the output of one RNN being used as the input for the next RNN.

```{.python .input  n=37}
batch_size = 1
state = rnn_layer.begin_state(batch_size=batch_size)
len(state), state[0].shape
```

With a state variable and an input, we can compute the output with the updated state.

```{.python .input  n=38}
num_steps = 1
X = np.random.uniform(size=(num_steps, batch_size, len(vocab)))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

Similar to :numref:`chapter_rnn_scratch`, we define an `RNNModel` block by subclassing the `Block` class for a complete recurrent neural network. Note that `rnn_layer` only contains the hidden recurrent layers, we need to create a separate output layer. While in the previous section, we have the output layer within the `rnn` block.

```{.python .input  n=39}
# Save to the d2l package.
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        X = npx.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of Y to
        # (num_steps * batch_size, num_hiddens)
        # Its output shape is (num_steps * batch_size, vocab_size)
        output = self.dense(Y.reshape(-1, Y.shape[-1]))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## Training

Let's make a prediction with the model that has random weights.

```{.python .input  n=42}
ctx = d2l.try_gpu()
model = RNNModel(rnn_layer, len(vocab))
model.initialize(force_reinit=True, ctx=ctx)
d2l.predict_ch8('time traveller', 10, model, vocab, ctx)
```

As is quite obvious, this model doesn't work at all (just yet). Next, we call just `train_ch8` defined in :numref:`chapter_rnn_scratch` with the same hyper-parameters to train our model.

```{.python .input  n=19}
num_epochs, lr = 500, 1
d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, ctx)
```

The model achieves comparable perplexity, albeit within a shorter period of time, due to the code being more optimized.

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
1. What happens if you increase the number of hidden layers in the RNN model? Can you make the model work?
1. How well can you compress the text using this model?
    * How many bits do you need?
    * Why doesn't everyone use this model for text compression? Hint - what about the compressor itself?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2365)

![](../img/qr_rnn-gluon.svg)
