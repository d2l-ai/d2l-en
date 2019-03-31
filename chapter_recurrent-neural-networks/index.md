# Recurrent Neural Networks

So far we encountered two types of data: generic vectors and
images. For the latter we designed specialized layers to take
advantage of the regularity properties in them. In other words, if we
were to permute the pixels in an image, it would be much more
difficult to reason about its content of something that would look
much like the background of a test pattern in the times of Analog TV.

Most importantly, so far we tacitly assumed that our data is generated
iid, i.e. independently and identically distributed, all drawn from some
distribution. Unfortunately, this isn't true for most data. For
instance, the words in this paragraph are written in sequence, and it
would be quite difficult to decipher its meaning if they were
permuted randomly. Likewise, image frames in a video, the audio signal
in a conversation, or the browsing behavior on a website, all follow
sequential order. It is thus only reasonable to assume that
specialized models for such data will do better at describing it and
at solving estimation problems.

Another issue arises from the fact that we might not only receive a
sequence as an input but rather might be expected to continue the
sequence. For instance, the task could be to continue the series 2,
4, 6, 8, 10, ... This is quite common in time series analysis, to
predict the stock market, the fever curve of a patient or the
acceleration needed for a race car. Again we want to have models that
can handle such data.

In short, while convolutional neural networks can efficiently process
spatial information, recurrent neural networks are designed to better
handle sequential information. These networks introduces state
variables to store past information and, together with the current
input, determine the current output.

Many of the examples for using recurrent networks are based on text
data. Hence, we will emphasize language models in this chapter. After
a more formal review of sequence data we discuss basic concepts of a
language model and use this discussion as the inspiration for the
design of recurrent neural networks. Next, we describe the gradient
calculation method in recurrent neural networks to explore problems
that may be encountered in recurrent neural network training. For some
of these problems, we can use gated recurrent neural networks, such as
LSTMs and GRUs, described later in this chapter.

bptt
gru
lstm
deep-rnn
bi-rnn

```eval_rst

.. toctree::
   :maxdepth: 2

   sequence
   lang-model
   rnn
   lang-model-dataset
   rnn-scratch
   rnn-gluon
```
