# Gluon Implementation in Recurrent Neural Networks

@TODO(smolix/astonzhang): the data set was just changed from lyrics to time machine, so descriptions/hyperparameters have to change.

This section will use Gluon to implement a language model based on a recurrent neural network. First, we read the Jay Chou album lyrics data set.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')

import gluonbook as gb
import math
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn, rnn
import time

(corpus_indices, char_to_idx, idx_to_char,
 vocab_size) = gb.load_data_time_machine()
```

## Define the Model

Gluon's `rnn` module provides a recurrent neural network implementation. Next, we construct the recurrent neural network layer `rnn_layer` with a single hidden layer and 256 hidden units, and initialize the weights.

```{.python .input  n=26}
num_hiddens = 256
rnn_layer = rnn.RNN(num_hiddens)
rnn_layer.initialize()
```

Then, we call the `rnn_layer`'s member function `begin_state` to return hidden state list for initialization. It has an element of the shape (number of hidden layers, batch size, number of hidden units).

```{.python .input  n=37}
batch_size = 2
state = rnn_layer.begin_state(batch_size=batch_size)
state[0].shape
```

Unlike the recurrent neural network implemented in the previous section, the input shape of `rnn_layer` here is (time step, batch size, number of inputs). Here, the number of inputs is the one-hot vector length (the dictionary size). In addition, as an `rnn.RNN` instance in Gluon, `rnn_layer` returns the output and hidden state after forward computation. The output refers to the hidden states that the hidden layer computes and outputs at various time steps, which are usually used as input for subsequent output layers. We should emphasize that the "output" itself does not involve the computation of the output layer, and its shape is (time step, batch size, number of hidden units). While the hidden state returned by the `rnn.RNN` instance in the forward computation refers to the hidden state of the hidden layer available at the last time step that can be used to initialize the next time step: when there are multiple layers in the hidden layer, the hidden state of each layer is recorded in this variable. For recurrent neural networks such as long short-term memory networks, the variable also contains other information. We will introduce long short-term memory and deep recurrent neural networks in the later sections of this chapter.

```{.python .input  n=38}
num_steps = 35
X = nd.random.uniform(shape=(num_steps, batch_size, vocab_size))
Y, state_new = rnn_layer(X, state)
Y.shape, len(state_new), state_new[0].shape
```

Next, we inherit the Block class to define a complete recurrent neural network. It first uses one-hot vector to represent input data and enter it into the `rnn_layer`. This, it uses the fully connected output layer to obtain the output. The number of outputs is equal to the dictionary size `vocab_size`.

```{.python .input  n=39}
# This class has been saved in the gluonbook package for future use.
class RNNModel(nn.Block):
    def __init__(self, rnn_layer, vocab_size, **kwargs):
        super(RNNModel, self).__init__(**kwargs)
        self.rnn = rnn_layer
        self.vocab_size = vocab_size
        self.dense = nn.Dense(vocab_size)

    def forward(self, inputs, state):
        # Get the one-hot vector representation by transposing the input to (num_steps, batch_size).
        X = nd.one_hot(inputs.T, self.vocab_size)
        Y, state = self.rnn(X, state)
        # The fully connected layer will first change the shape of Y to (num_steps * batch_size, num_hiddens).
        # Its output shape is (num_steps * batch_size, vocab_size).
        output = self.dense(Y.reshape((-1, Y.shape[-1])))
        return output, state

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

## Model Training

As in the previous section, a prediction function is defined below. The implementation here differs from the previous one in the function interfaces for forward computation and hidden state initialization.

```{.python .input  n=41}
# This function is saved in the gluonbook package for future use.
def predict_rnn_gluon(prefix, num_chars, model, vocab_size, ctx, idx_to_char,
                      char_to_idx):
    # Use model's member function to initialize the hidden state.
    state = model.begin_state(batch_size=1, ctx=ctx)
    output = [char_to_idx[prefix[0]]]
    for t in range(num_chars + len(prefix) - 1):
        X = nd.array([output[-1]], ctx=ctx).reshape((1, 1))
        (Y, state) = model(X, state)  # Forward computation does not require incoming model parameters.
        if t < len(prefix) - 1:
            output.append(char_to_idx[prefix[t + 1]])
        else:
            output.append(int(Y.argmax(axis=1).asscalar()))
    return ''.join([idx_to_char[i] for i in output])
```

Let us make one predication using a model with weights that are random values.

```{.python .input  n=42}
ctx = gb.try_gpu()
model = RNNModel(rnn_layer, vocab_size)
model.initialize(force_reinit=True, ctx=ctx)
predict_rnn_gluon('traveller', 10, model, vocab_size, ctx, idx_to_char,
                  char_to_idx)
```

Next, implement the training function. Its algorithm is the same as in the previous section, but only random sampling is used here to read the data.

```{.python .input  n=18}
# This function is saved in the gluonbook package for future use.
def train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                                corpus_indices, idx_to_char, char_to_idx,
                                num_epochs, num_steps, lr, clipping_theta,
                                batch_size, pred_period, pred_len, prefixes):
    loss = gloss.SoftmaxCrossEntropyLoss()
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(0.01))
    trainer = gluon.Trainer(model.collect_params(), 'sgd',
                            {'learning_rate': lr, 'momentum': 0, 'wd': 0})

    for epoch in range(num_epochs):
        loss_sum, start = 0.0, time.time()
        data_iter = gb.data_iter_consecutive(
            corpus_indices, batch_size, num_steps, ctx)
        state = model.begin_state(batch_size=batch_size, ctx=ctx)
        for t, (X, Y) in enumerate(data_iter):
            for s in state:
                s.detach()
            with autograd.record():
                (output, state) = model(X, state)
                y = Y.T.reshape((-1,))
                l = loss(output, y).mean()
            l.backward()
            # Clip the gradient.
            params = [p.data() for p in model.collect_params().values()]
            gb.grad_clipping(params, clipping_theta, ctx)
            trainer.step(1)  # Since the error has already taken the mean, the gradient does not need to be averaged.
            loss_sum += l.asscalar()

        if (epoch + 1) % pred_period == 0:
            print('epoch %d, perplexity %f, time %.2f sec' % (
                epoch + 1, math.exp(loss_sum / (t + 1)), time.time() - start))
            for prefix in prefixes:
                print(' -', predict_rnn_gluon(
                    prefix, pred_len, model, vocab_size,
                    ctx, idx_to_char, char_to_idx))
```

Train the model using the same hyper-parameters as in the previous experiments.

```{.python .input  n=19}
num_epochs, batch_size, lr, clipping_theta = 200, 32, 1e2, 1e-2
pred_period, pred_len, prefixes = 50, 50, ['traveller', 'time traveller']
train_and_predict_rnn_gluon(model, num_hiddens, vocab_size, ctx,
                            corpus_indices, idx_to_char, char_to_idx,
                            num_epochs, num_steps, lr, clipping_theta,
                            batch_size, pred_period, pred_len, prefixes)
```

## Summary

* Gluon's `rnn` module provides an implementation at the recurrent neural network layer.
* Gluon's `nn.RNN` instance returns the output and hidden state after forward computation. This forward computation does not involve output layer computation.

## Problems

* Compare the implementation with the previous section. Does Gluon's implementation run faster? If you observe a significant difference, try to find the reason.

## Discuss on our Forum

<div id="discuss" topic_id="2365"></div>
