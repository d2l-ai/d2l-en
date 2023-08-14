# Convolutional Neural Networks
:label:`chap_cnn`

Image data is represented as a two-dimensional grid of pixels, be the image
monochromatic or in color. Accordingly each pixel corresponds to one
or multiple numerical values respectively. So far we have ignored this rich
structure and treated images as vectors of numbers by *flattening* them, irrespective of the spatial relation between pixels. This
deeply unsatisfying approach was necessary in order to feed the
resulting one-dimensional vectors through a fully connected MLP.

Because these networks are invariant to the order of the features, we
could get similar results regardless of whether we preserve an order
corresponding to the spatial structure of the pixels or if we permute
the columns of our design matrix before fitting the MLP's parameters.
Ideally, we would leverage our prior knowledge that nearby pixels
are typically related to each other, to build efficient models for
learning from image data.

This chapter introduces *convolutional neural networks* (CNNs)
:cite:`LeCun.Jackel.Bottou.ea.1995`, a powerful family of neural networks that
are designed for precisely this purpose.
CNN-based architectures are
now ubiquitous in the field of computer vision.
For instance, on the Imagnet collection
:cite:`Deng.Dong.Socher.ea.2009` it was only the use of convolutional neural
networks, in short Convnets, that provided significant performance
improvements :cite:`Krizhevsky.Sutskever.Hinton.2012`.

Modern CNNs, as they are called colloquially, owe their design to
inspirations from biology, group theory, and a healthy dose of
experimental tinkering.  In addition to their sample efficiency in
achieving accurate models, CNNs tend to be computationally efficient,
both because they require fewer parameters than fully connected
architectures and because convolutions are easy to parallelize across
GPU cores :cite:`Chetlur.Woolley.Vandermersch.ea.2014`.  Consequently, practitioners often
apply CNNs whenever possible, and increasingly they have emerged as
credible competitors even on tasks with a one-dimensional sequence
structure, such as audio :cite:`Abdel-Hamid.Mohamed.Jiang.ea.2014`, text
:cite:`Kalchbrenner.Grefenstette.Blunsom.2014`, and time series analysis
:cite:`LeCun.Bengio.ea.1995`, where recurrent neural networks are
conventionally used.  Some clever adaptations of CNNs have also
brought them to bear on graph-structured data :cite:`Kipf.Welling.2016` and
in recommender systems.

First, we will dive more deeply into the motivation for convolutional
neural networks. This is followed by a walk through the basic operations
that comprise the backbone of all convolutional networks.
These include the convolutional layers themselves,
nitty-gritty details including padding and stride,
the pooling layers used to aggregate information
across adjacent spatial regions,
the use of multiple channels  at each layer,
and a careful discussion of the structure of modern architectures.
We will conclude the chapter with a full working example of LeNet,
the first convolutional network successfully deployed,
long before the rise of modern deep learning.
In the next chapter, we will dive into full implementations
of some popular and comparatively recent CNN architectures
whose designs represent most of the techniques
commonly used by modern practitioners.

```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```

