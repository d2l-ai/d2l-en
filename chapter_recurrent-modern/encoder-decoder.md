```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Encoder-Decoder Architecture
:label:`sec_encoder-decoder`

In general seq2seq problems 
like machine translation 
(:numref:`sec_machine_translation`),
inputs and outputs are of varying lengths
that are unaligned. 
The standard approach to handling this sort of data
is to design an *encoder-decoder* architecture (:numref:`fig_encoder_decoder`)
consisting of two major components:
an *encoder* that takes a variable-length sequence as input,
and a *decoder* that acts as a conditional language model,
taking in the encoded input 
and the leftwards context of the target sequence 
and predicting the subsequent token in the target sequence. 


![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Let's take machine translation from English to French as an example.
Given an input sequence in English:
"They", "are", "watching", ".",
this encoder-decoder architecture
first encodes the variable-length input into a state,
then decodes the state 
to generate the translated sequence,
token by token, as output:
"Ils", "regardent", ".".
Since the encoder-decoder architecture
forms the basis of different seq2seq models
in subsequent sections,
this section will convert this architecture
into an interface that will be implemented later.

## (**Encoder**)

In the encoder interface,
we just specify that
the encoder takes variable-length sequences as input `X`.
The implementation will be provided 
by any model that inherits this base `Encoder` class.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

#@save
class Encoder(tf.keras.layers.Layer):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def call(self, X, *args):
        raise NotImplementedError
```

## [**Decoder**]

In the following decoder interface,
we add an additional `init_state` function
to convert the encoder output (`enc_outputs`)
into the encoded state.
Note that this step
may require extra inputs,
such as the valid length of the input,
which was explained
in :numref:`sec_machine_translation`.
To generate a variable-length sequence token by token,
every time the decoder may map an input 
(e.g., the generated token at the previous time step)
and the encoded state 
into an output token at the current time step.

```{.python .input}
%%tab mxnet
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
#@save
class Decoder(tf.keras.layers.Layer):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

## [**Putting the Encoder and Decoder Together**]

In the forward propagation,
the output of the encoder
is used to produce the encoded state,
and this state will be further used
by the decoder as one of its input.

```{.python .input}
%%tab mxnet, pytorch
#@save
class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
#@save
class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]
```

In the next section, 
we will see how to apply RNNs to design 
seq2seq models based on 
this encoder-decoder architecture.


## Summary

Encoder-decoder architectures
can handle inputs and outputs 
that both consist of variable-length sequences
and thus are suitable for seq2seq problems 
such as machine translation.
The encoder takes a variable-length sequence as input 
and transforms it into a state with a fixed shape.
The decoder maps the encoded state of a fixed shape
to a variable-length sequence.


## Exercises

1. Suppose that we use neural networks to implement the encoder-decoder architecture. Do the encoder and the decoder have to be the same type of neural network?  
1. Besides machine translation, can you think of another application where the encoder-decoder architecture can be applied?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3864)
:end_tab:
