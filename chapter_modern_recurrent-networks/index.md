# Modern Recurrent Networks
:label:`chap_modern_rnn`

Although we have learned the basics of recurrent neural networks,
they are not sufficient for a practitioner to solve today's sequence learning problems.
For instance, given the numerical unstability during gradient calculation,
gated recurrent neural networks much more common in practice.
We will begin by introducing two of such widely-used networks,
namely gated recurrent units (GRUs) and long short term memory (LSTM),
with illustrations using the same language modeling problem as introduced in :numref:`chap_rnn`.

Furthermore, we will modify recurrent neural networks
with a single undirectional hidden layer.
We will describe deep architectures,
and discuss the bidirectional design with both forward and backward recursion.
They are frequently adopted in modern recurrent networks.


In fact, a large portion of sequence learning problems 
such as automatic speech recognition, text to speech, and machine translation,
consider both inputs and outputs to be sequences of arbitrary length.
Finally, we will take machine translation as an example,
and introduce the encoder-decoder architecture based on
recurrent neural networks and modern practices for such sequence to sequence learning problems.

```toc
:maxdepth: 2

gru
lstm
deep-rnn
bi-rnn
machine-translation
encoder-decoder
seq2seq
beam-search
```

