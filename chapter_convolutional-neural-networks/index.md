# Convolutional Neural Networks

In this chapter we introduce convolutional neural networks. They are
the first nontrivial *architecture* beyond the humble multilayer
perceptron. In their design scientists used inspiration from biology,
group theory, and lots of experimentation to achieve stunning results
in object reognition, segmentation, image synthesis and related
computer vision tasks. 'Convnets', as they are often called, have
become a cornerstone for deep learning research. Their applications
reach beyond images to audio, text, video, time series analysis,
graphs and recommender systems.

We will first describe the operating principles of the convolutional
layer and pooling layer in a convolutional neural network, and then
explain padding, stride, input channels, and output channels. Next we
will explore the design concepts of several representative deep
convolutional neural networks. These models include the AlexNet, the
first such network proposed, and later networks that use repeating
elements (VGG), network in network (NiN), networks with parallel
concatenations (GoogLeNet), residual networks (ResNet), and densely
connected networks (DenseNet).  Many of these networks have led to
significant progress in the ImageNet competition (a famous computer
vision contest) within the past few years.

Over time the networks have increased in depth significantly,
exceeding hundreds of layers. To train on them efficiently tools for
capacity control, reparametrization and training acceleration are
needed. Batch normalization and residual networks are both used to
address these probems. We will describe them in this chapter.

```eval_rst

.. toctree::
   :maxdepth: 2

   why-conv
   conv-layer
   padding-and-strides
   channels
   pooling
   lenet
   alexnet
   vgg
   nin
   googlenet
   batch-norm
   resnet
   densenet
```
