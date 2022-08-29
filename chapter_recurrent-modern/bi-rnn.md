# Bidirectional Recurrent Neural Networks
:label:`sec_bi_rnn`

So far, our working example of a sequence learning task has been language modeling,
where we aim to predict the next token given all previous tokens in a sequence. 
In this scenario, we wish only to condition upon the leftward context,
and thus the unidirectional chaining of a standard RNN seems appropriate. 
However, there are many other sequence learning tasks contexts 
where it's perfectly fine to condition the prediction at every time step
on both the leftward and the rightward context. 
Consider, for example, part of speech detection. 
Why shouldn't we take the context in both directions into account
when assessing the part of speech associated with a given word?

Another common task---often useful as a pretraining exercise
prior to fine-tuning a model on an actual task of interest---is
to mask out random tokens in a text document and then to train 
a sequence model to predict the values of the missing tokens.
Note that depending on what comes after the blank,
the likely value of the missing token changes dramatically:

* I am `___`.
* I am `___` hungry.
* I am `___` hungry, and I can eat half a pig.

In the first sentence "happy" seems to be a likely candidate.
The words "not" and "very" seem plausible in the second sentence, 
but "not" seems incompatible with the third sentences. 


Fortunately, a simple technique transforms any unidirectional RNN 
into a bidrectional RNN :cite:`Schuster.Paliwal.1997`.
We simply implement two unidirectional RNN layers
chained together in opposite directions 
and acting on the same input (:numref:`fig_birnn`).
For the first RNN layer,
the first input is $\mathbf{x}_1$
and the last input is $\mathbf{x}_T$,
but for the second RNN layer, 
the first input is $\mathbf{x}_T$
and the last input is $\mathbf{x}_1$.
To produce the output of this bidirectional RNN layer,
we simply concatenate together the corresponding outputs
of the two underlying unidirectional RNN layers. 


![Architecture of a bidirectional RNN.](../img/birnn.svg)
:label:`fig_birnn`


Formally for any time step $t$,
we consider a minibatch input $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ 
(number of examples: $n$, number of inputs in each example: $d$) 
and let the hidden layer activation function be $\phi$.
In the bidirectional architecture,
the forward and backward hidden states for this time step 
are $\overrightarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$ 
and $\overleftarrow{\mathbf{H}}_t  \in \mathbb{R}^{n \times h}$, respectively,
where $h$ is the number of hidden units.
The forward and backward hidden state updates are as follows:


$$
\begin{aligned}
\overrightarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1} \mathbf{W}_{hh}^{(f)}  + \mathbf{b}_h^{(f)}),\\
\overleftarrow{\mathbf{H}}_t &= \phi(\mathbf{X}_t \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1} \mathbf{W}_{hh}^{(b)}  + \mathbf{b}_h^{(b)}),
\end{aligned}
$$

where the weights $\mathbf{W}_{xh}^{(f)} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh}^{(f)} \in \mathbb{R}^{h \times h}, \mathbf{W}_{xh}^{(b)} \in \mathbb{R}^{d \times h}, \text{ and } \mathbf{W}_{hh}^{(b)} \in \mathbb{R}^{h \times h}$, and biases $\mathbf{b}_h^{(f)} \in \mathbb{R}^{1 \times h}$ and $\mathbf{b}_h^{(b)} \in \mathbb{R}^{1 \times h}$ are all the model parameters.

Next, we concatenate the forward and backward hidden states
$\overrightarrow{\mathbf{H}}_t$ and $\overleftarrow{\mathbf{H}}_t$
to obtain the hidden state $\mathbf{H}_t \in \mathbb{R}^{n \times 2h}$ to be fed into the output layer.
In deep bidirectional RNNs with multiple hidden layers,
such information is passed on as *input* to the next bidirectional layer. 
Last, the output layer computes the output 
$\mathbf{O}_t \in \mathbb{R}^{n \times q}$ (number of outputs: $q$):

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q.$$

Here, the weight matrix $\mathbf{W}_{hq} \in \mathbb{R}^{2h \times q}$ 
and the bias $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ 
are the model parameters of the output layer. 
While technically, the two directions can have different numbers of hidden units,
this design choice is seldom made in practice. 
We now demonstrate a simple implementation of a bidirectional RNN.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import npx, np
from mxnet.gluon import rnn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

### Implementation from Scratch

To implement a bidirectional RNN from scratch, we can
include two unidirectional `RNNScratch` instances
with separate learnable parameters.

```{.python .input}
%%tab all
class BiRNNScratch(d2l.Module):
    def __init__(self, num_inputs, num_hiddens, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.f_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.b_rnn = d2l.RNNScratch(num_inputs, num_hiddens, sigma)
        self.num_hiddens *= 2  # The output dimension will be doubled
```

States of forward and backward RNNs
are updated separately,
while outputs of these two RNNs are concatenated.

```{.python .input}
%%tab all
@d2l.add_to_class(BiRNNScratch)
def forward(self, inputs, Hs=None):
    f_H, b_H = Hs if Hs is not None else (None, None)
    f_outputs, f_H = self.f_rnn(inputs, f_H)
    b_outputs, b_H = self.b_rnn(reversed(inputs), b_H)
    outputs = [d2l.concat((f, b), -1) for f, b in zip(f_outputs, b_outputs)]
    return outputs, (f_H, b_H)
```

### Concise Implementation

Using the high-level APIs,
we can implement bidirectional RNNs more concisely.
Here we take a GRU model as an example.

```{.python .input}
%%tab mxnet, pytorch
class BiGRU(d2l.RNN):
    def __init__(self, num_inputs, num_hiddens):
        d2l.Module.__init__(self)
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.rnn = rnn.GRU(num_hiddens, bidirectional=True)
        if tab.selected('pytorch'):
            self.rnn = nn.GRU(num_inputs, num_hiddens, bidirectional=True)
        self.num_hiddens *= 2
```

## Summary

In bidirectional RNNs, the hidden state for each time step is simultaneously determined by the data prior to and after the current time step. Bidirectional RNNs are mostly useful for sequence encoding and the estimation of observations given bidirectional context. Bidirectional RNNs are very costly to train due to long gradient chains.

## Exercises

1. If the different directions use a different number of hidden units, how will the shape of $\mathbf{H}_t$ change?
1. Design a bidirectional RNN with multiple hidden layers.
1. Polysemy is common in natural languages. For example, the word "bank" has different meanings in contexts “i went to the bank to deposit cash” and “i went to the bank to sit down”. How can we design a neural network model such that given a context sequence and a word, a vector representation of the word in the context will be returned? What type of neural architectures is preferred for handling polysemy?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/339)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1059)
:end_tab:
