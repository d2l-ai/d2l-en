# Recurrent Neural Networks

Unlike the multilayer perceptrons and convolutional neural networks that can efficiently process spatial information previously described, recurrent neural networks are designed to better handle timing information. These networks introduces state variables to store past information and, together with the current input, determine the current output.

Recurrent neural networks are often used to process sequence data, such as a segment of text or audio, the order of shopping or viewing behavior, or even a row or column of pixels in an image. Therefore, recurrent neural networks have a wide range of applications in practice, such as language models, text classification, machine translation, speech recognition, image analysis, handwriting recognition, and recommendation systems.

Since the application in this chapter is based on a language model, we will first introduce the basic concepts of the language model and use this discussion as the inspiration for the design of a recurrent neural network. Next, we will describe the gradient calculation method in recurrent neural networks to explore problems that may be encountered in recurrent neural network training. For some of these problems, we can use gated recurrent neural networks, described later in this chapter, to resolve them. Finally, we will expand the architecture of the recurrent neural network.

```eval_rst

.. toctree::
   :maxdepth: 2

   lang-model
   rnn
   lang-model-dataset
   rnn-scratch
   rnn-gluon
   bptt
   gru
   lstm
   deep-rnn
   bi-rnn
```
