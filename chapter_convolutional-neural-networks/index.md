# Convolutional Neural Networks
:label:`sec_cnn`

In several of our previous examples, we have already come up
against image data, which consist of pixels arranged in a 2D grid.
Depending on whether we are looking at a black and white or color image,
we might have either one or multiple numerical values
corresponding to each pixel location.
Until now, we have dealt with this rich structure
in the least satisfying possible way.
We simply threw away this spatial structure
by flattening each image into a 1D vector,
and fed it into a fully-connected network.
These networks are invariant to the order of their inputs.
We will get qualitatively identical results
out of a multilayer perceptron
whether we preserve the original order of our features or
if we permute the columns of our design matrix before learning the parameters.
Ideally, we would find a way to leverage our prior knowledge
that nearby pixels are more related to each other.

In this chapter, we introduce convolutional neural networks (CNNs),
a powerful family of neural networks
that were designed for precisely this purpose.
CNN-based network *architecures*
now dominate the field of computer vision to such an extent
that hardly anyone these days would develop
a commerical application or enter a competition
related to image recognition, object detection,
or semantic segmentation,
without basing their approach on them.

Modern 'convnets', as they are often called owe their design
to inspirations from biology, group theory,
and a healthy dose of experimental tinkering.
In addition to their strong predictive performance,
convolutional neural networks tend to be computationally efficient,
both because they tend to require fewer parameters
than dense architectures
and also because convolutions are easy to parralelize across GPU cores.
As a result, researchers have sought to apply convnets whenever possible,
and increasingly they have emerged as credible competitors
even on tasks with 1D sequence structure,
such as audio, text, and time series analysis,
where recurrent neural networks (introduced in the next chapter)
are conventionally used.
Some clever adaptations of CNNs have also brought them to bear
on graph-structured data and in recommender systems.

First, we will walk through the basic operations
that comprise the backbone of all modern convolutional networks.
These include the convolutional layers themselves,
nitty-gritty details including padding and stride,
the pooling layers used to aggregate information
across adjacent spatial regions,
the use of multiple *channels* (also called *filters*) at each layer,
and a careful discussion of the structure of modern architectures.
We will conclude the chapter with a full working example of LeNet,
the first convolutional network successfully deployed,
long before the rise of modern deep learning.
In the next chapter we'll dive into full implementations
of some of the recent popular neural networks
whose designs are representative of most of the techniques
commonly used to design modern convolutional neural networks.


```toc
:maxdepth: 2

why-conv
conv-layer
padding-and-strides
channels
pooling
lenet
```
