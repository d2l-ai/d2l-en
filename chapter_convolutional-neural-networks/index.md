# Convolutional Neural Networks

This chapter will introduce the convolutional neural network. This is the cornerstone on which deep learning has achieved breakthrough results in computer vision in recent years. It is also being used extensively in other fields, such as natural language processing, recommendation systems, and speech recognition. We will first describe the operating principles of the convolutional layer and pooling layer in a convolutional neural network, and then explain the meaning of padding, stride, input channels, and output channels. After mastering this basic knowledge, we will explore the design concepts of several representative deep convolutional neural networks. These models include the AlexNet, the first such network proposed, and later networks that use repeating elements (VGG), network in network (NiN), networks with parallel concatenations (GoogLeNet), residual networks (ResNet), and densely connected networks (DenseNet).  Many of these networks have yielded unusually brilliant results in the ImageNet competition (a famous computer vision competition) within the past few years. Although the deep model seems to be just a neural network with many layers, it is not easy to obtain an effective deep model. Fortunately, the batch normalization and residual networks described in this chapter provide two important concepts used for training and designing deep models.

```eval_rst

.. toctree::
   :maxdepth: 2

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
