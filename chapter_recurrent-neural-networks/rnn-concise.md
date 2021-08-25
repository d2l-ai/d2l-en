# Concise Implementation of Recurrent Neural Networks
:label:`sec_rnn-concise`

While :numref:`sec_rnn_scratch` was instructive to see how RNNs are implemented,
this is not convenient or fast.
This section will show how to implement the same language model more efficiently
using functions provided by high-level APIs
of a deep learning framework.
We begin as before by reading *The Time Machine* dataset.

```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## [**Defining the Model**]

High-level APIs provide implementations of recurrent neural networks.
We construct the recurrent neural network layer `rnn_layer` with a single hidden layer and 256 hidden units.
In fact, we have not even discussed yet what it means to have multiple layers---this will happen in :numref:`sec_deep_rnn`.
For now, suffice it to say that multiple layers simply amount to the output of one layer of RNN being used as the input for the next layer of RNN.

```{.python .input}
%%tab mxnet
class RNN(d2l.RNNScratch):
    def init_params(self):
        self.rnn = rnn.RNN(self.num_hiddens)
        self.linear = nn.Dense(self.num_outputs, flatten=False)
        self.initialize()
        
    def forward(self, X, state=None):
        if state is None: state = self.init_state(X)
        embs = npx.one_hot(X.T, self.num_inputs)        
        Y, state = self.rnn(embs, state)
        return self.linear(Y), state

    def init_state(self, X):        
        return [d2l.to(s, X.context) 
                for s in self.rnn.begin_state(batch_size=X.shape[0])]
```

```{.python .input  n=10}
%%tab pytorch
class RNN(d2l.RNNScratch):
    def init_params(self):
        self.rnn =  nn.RNN(self.num_inputs, self.num_hiddens)
        self.linear = nn.Linear(self.num_hiddens, self.num_outputs)

    def forward(self, X, state=None):
        if state is None: state = self.init_state(X)
        embs = F.one_hot(X.T, self.num_inputs).type(torch.float32)
        Y, state = self.rnn(embs, state)        
        return self.linear(Y), state

    def init_state(self, X):
        return torch.zeros((1, X.shape[0], self.num_hiddens)).to(X.device)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.RNNScratch):
    def init_params(self):
        rnn_cell = tf.keras.layers.SimpleRNNCell(self.num_hiddens)
        self.rnn = tf.keras.layers.RNN(rnn_cell, time_major=True, 
                                       return_sequences=True, return_state=True)
        self.linear = tf.keras.layers.Dense(self.num_outputs)

    def forward(self, X, state=None):
        if state is None: state = self.init_state(X)
        X = tf.one_hot(tf.transpose(X), self.num_inputs)        
        Y, *state = self.rnn(X, state)
        return self.linear(Y), state

    def init_state(self, X):
        return self.rnn.cell.get_initial_state(
            batch_size=X.shape[0], dtype=d2l.float32)
```

```{.python .input  n=1}
%%tab all
data = d2l.TimeMachine(batch_size=32, num_steps=35)
model = RNN(num_inputs=len(data.vocab), num_outputs=len(data.vocab), num_hiddens=32, lr=1)
trainer = d2l.Trainer(max_epochs=500, gradient_clip_val=1)
trainer.fit(model, data)
```

```{.json .output n=1}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "UsageError: Cell magic `%%tab` not found.\n"
 }
]
```

## Training and Predicting

Before training the model, let's [**make a prediction with the a model that has random weights.**]

As is quite obvious, this model does not work at all. Next, we call `train_ch8` with the same hyperparameters defined in :numref:`sec_rnn_scratch` and [**train our model with high-level APIs**].

Compared with the last section, this model achieves comparable perplexity,
albeit within a shorter period of time, due to the code being more optimized by
high-level APIs of the deep learning framework.


## Summary

* High-level APIs of the deep learning framework provides an implementation of the RNN layer.
* The RNN layer of high-level APIs returns an output and an updated hidden state, where the output does not involve output layer computation.
* Using high-level APIs leads to faster RNN training than using its implementation from scratch.

## Exercises

1. Can you make the RNN model overfit using the high-level APIs?
1. What happens if you increase the number of hidden layers in the RNN model? Can you make the model work?
1. Implement the autoregressive model of :numref:`sec_sequence` using an RNN.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/335)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1053)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/2211)
:end_tab:
