# Concise Implementation of Recurrent Neural Networks
:label:`sec_rnn-concise`

Like most of our from-scratch implementations,
:numref:`sec_rnn-scratch` was designed 
to provide insight into how each component works.
But when you're using RNNs every day 
or writing production code,
you'll want to rely more on libraries
that cut down on both implementation time 
(by supplying library code for common models and functions)
and computation time 
(by optimizing the heck out of these library implementations).
This section will show you how to implement 
the same language model more efficiently
using the high-level API provided 
by your deep learning framework.
We begin, as before, by loading 
the *The Time Machine* dataset.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.gluon import nn, rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

## [**Defining the Model**]

We define the following class
using the RNN implemented
by high-level APIs.

:begin_tab:`mxnet`
Specifically, to initialize the hidden state,
we invoke the member method `begin_state`.
This returns a list that contains
an initial hidden state
for each example in the minibatch,
whose shape is
(number of hidden layers, batch size, number of hidden units).
For some models to be introduced later
(e.g., long short-term memory),
this list will also contains other information.
:end_tab:

```{.python .input}
%%tab mxnet
class RNN(d2l.Module):  #@save
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()        
        self.rnn = rnn.RNN(num_hiddens)
        
    def forward(self, inputs, H=None):
        if H is None:
            H, = self.rnn.begin_state(inputs.shape[1], ctx=inputs.ctx)
        outputs, (H, ) = self.rnn(inputs, (H, ))
        return outputs, H
```

```{.python .input}
%%tab pytorch
class RNN(d2l.Module):  #@save
    def __init__(self, num_inputs, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = nn.RNN(num_inputs, num_hiddens)
        
    def forward(self, inputs, H=None):
        return self.rnn(inputs, H)
```

```{.python .input}
%%tab tensorflow
class RNN(d2l.Module):  #@save
    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()            
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True,
            time_major=True)
        
    def forward(self, inputs, H=None):
        outputs, H = self.rnn(inputs, H)
        return outputs, H
```

Inheriting from the `RNNLMScratch` class in :numref:`sec_rnn-scratch`, 
the following `RNNLM` class defines a complete RNN-based language model.
Note that we need to create a separate fully connected output layer.

```{.python .input}
%%tab all
class RNNLM(d2l.RNNLMScratch):  #@save
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('pytorch'):
            self.linear = nn.LazyLinear(self.vocab_size)
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet', 'pytorch'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

## Training and Predicting

Before training the model, let's [**make a prediction 
with a model initialized with random weights.**]
Given that we have not trained the network, 
it will generate nonsensical predictions.

```{.python .input}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn = RNN(num_inputs=len(data.vocab), num_hiddens=32)
model = RNNLM(rnn, vocab_size=len(data.vocab), lr=1)
model.predict('it has', 20, data.vocab)
```

Next, we [**train our model, leveraging the high-level API**].

```{.python .input}
%%tab all
if tab.selected('mxnet', 'pytorch'):
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

Compared with :numref:`sec_rnn-scratch`,
this model achieves comparable perplexity,
but runs faster due to the optimized implementations.
As before, we can generate predicted tokens 
following the specified prefix string.

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
```

## Summary

* High-level APIs in deep learning frameworks provide implementations of standard RNNs.
* These libraries help you to avoid wasting time reimplementing standard models.
* Framework implementations are often highly optimized, 
  leading to significant (computational) performance gains 
  as compared to implementations from scratch.

## Exercises

1. Can you make the RNN model overfit using the high-level APIs?
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
