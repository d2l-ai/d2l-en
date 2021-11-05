# Concise Implementation of Recurrent Neural Networks
:label:`sec_rnn-concise`

While :numref:`sec_rnn_scratch` was instructive to see how RNNs are implemented,
this is not convenient or fast.
This section will show how to implement the same language model more efficiently
using functions provided by high-level APIs
of a deep learning framework.
We begin as before by reading *The Time Machine* dataset.

```{.python .input  n=40}
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

```{.python .input  n=41}
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

```{.python .input}
%%tab all
class RNNLM(d2l.RNNLMScratch):  #@save
    def init_params(self):
        if tab.selected('mxnet'):
            self.linear = nn.Dense(self.vocab_size, flatten=False)
            self.initialize()
        if tab.selected('pytorch'):
            self.linear = nn.Linear(self.rnn.num_hiddens, self.vocab_size)
        if tab.selected('tensorflow'):
            self.linear = tf.keras.layers.Dense(self.vocab_size)
        
    def output_layer(self, hiddens):
        if tab.selected('mxnet', 'pytorch'):
            return d2l.swapaxes(self.linear(hiddens), 0, 1)        
        if tab.selected('tensorflow'):
            return d2l.transpose(self.linear(hiddens), (1, 0, 2))
```

```{.python .input  n=1}
%%tab all
data = d2l.TimeMachine(batch_size=1024, num_steps=32)
if tab.selected('mxnet', 'tensorflow'):
    rnn_layer = RNN(num_hiddens=32)
if tab.selected('pytorch'):
    rnn_layer = RNN(num_inputs=len(data.vocab), num_hiddens=32)
if tab.selected('mxnet', 'pytorch'):
    model = RNNLM(rnn_layer, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        model = RNNLM(rnn_layer, vocab_size=len(data.vocab), lr=1)
    trainer = d2l.Trainer(max_epochs=100, gradient_clip_val=1)
trainer.fit(model, data)
```

```{.python .input}
%%tab mxnet, pytorch
model.predict('it has', 20, data.vocab, d2l.try_gpu())
```

```{.python .input}
%%tab tensorflow
model.predict('it has', 20, data.vocab)
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
