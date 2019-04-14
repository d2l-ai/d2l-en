# Encoder-Decoder Architecture

The encoder-decoder architecture is a neural network design pattern. In this architecture, the network is partitioned into two parts, the encoder and the decoder. The encoder's role is encoding the inputs into state, which often contains several tensors. Then the state is passed into the decoder to generate the outputs. In machine translation, the encoder transforms a source sentence, e.g. "Hello world.", into state, e.g. a vector, that captures its semantic information. The decoder then uses this state to generate the translated target sentence, e.g. "Bonjour le monde.". 

![The encoder-decoder architecture.](../img/encoder-decoder.svg)

In this section, we will show an interface to implement this encoder-decoder architecture.

```{.python .input  n=1}
from mxnet.gluon import nn
```

## Encoder

The encoder is a normal neural network that takes inputs, e.g. a source sentence, to return outputs.

```{.python .input  n=2}
class Encoder(nn.Block):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X):
        raise NotImplementedError
```

## Decoder

The decoder has an additional method `init_state` to parse the outputs of the encoder with possible additional information, e.g. the valid lengths of inputs, to return the state it needs. In the forward method, the decoder takes both inputs, e.g. a target sentence, and the state. It returns outputs, with potentially modified state if the encoder contains RNN layers.

```{.python .input  n=3}
class Decoder(nn.Block):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

## Model

The encoder-decoder model contains both an encoder an decoder. We implement its forward method for training. It takes both encoder inputs and decoder inputs, with optional additional information. During computation, it first compute encoder outputs to initialize the decoder state, and then returns the decoder outputs.

```{.python .input  n=4}
class EncoderDecoder(nn.Block):
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
```

## Summary
