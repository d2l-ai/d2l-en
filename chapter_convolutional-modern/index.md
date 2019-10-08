# Modern Convolutional Networks
:label:`chap_modern_cnn`

Now that we understand the basics of wiring together convolutional neural networks, we will take you through a tour of modern deep learning.
In this chapter, each section will correspond to a significant neural network architecture that was at some point (or currently) the base model upon which an enormous amount of research and projects were built.
Each of these networks was at briefly
a dominant architecture and many were
at one point winners or runners-up in the famous ImageNet competition,
which has served as a barometer of progress
on supervised learning in computer vision since 2010.

These models include AlexNet, the first large-scale network deployed to beat conventional computer vision methods on a large-scale vision challenge;
the VGG network, which makes use of a number of repeating blocks of elements; the network in network (NiN) which convolves whole neural networks patch-wise over inputs; the GoogLeNet, which makes use of networks with parallel
concatenations (GoogLeNet); residual networks (ResNet) which are currently the most popular go-to architecture today, and densely connected networks (DenseNet), which are expensive to compute but have set some recent benchmarks.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
```
