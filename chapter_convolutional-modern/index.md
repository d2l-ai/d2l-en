# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

Now that we understand the basics of wiring together CNNs, we will
take you through a tour of modern CNN architectures. This tour is, by
necessity, incomplete, thanks to the plethora of exciting new designs
being added. Their importance derives from the fact that not only can
they be used directly for vision tasks, but they also serve as basic
feature generators for more advanced tasks such as tracking
:cite:`bytetrack.2021`, segmentation :cite:`long2015fully`, object
detection :cite:`redmon2018yolov3`, or style transformation
:cite:`Gatys.Ecker.Bethge.2016`.  In this chapter, most sections
correspond to a significant CNN architecture that was at some point
(or currently) the base model upon which many research projects and
deployed systems were built.  Each of these networks was briefly a
dominant architecture and many were winners or runners-up in the
[ImageNet competition](https://www.image-net.org/challenges/LSVRC/)
which has served as a barometer of progress on supervised learning in
computer vision since 2010.

These models include AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`,
the first large-scale network deployed to beat conventional computer
vision methods on a large-scale vision challenge; the VGG network
:cite:`Simonyan.Zisserman.2014`, which makes use of a number of
repeating blocks of elements; the network in network (NiN) which
convolves whole neural networks patch-wise over inputs
:cite:`Lin.Chen.Yan.2013`; GoogLeNet, which uses networks with
parallel concatenations :cite:`Szegedy.Liu.Jia.ea.2015`; residual
networks (ResNet) :cite:`He.Zhang.Ren.ea.2016`, which remain some of
the most popular off-the-shelf architectures in computer vision;
MobileNet, which uses network learning to achieve high accuracy in
resource-constrained settings :cite:`howard2019searching`, and the
structured network search strategy leading to RegNetX/Y
:cite:`radosavovic2020designing`. In addition to that, we cover key
advances such as ResNeXt :cite:`xie2017aggregated` which partitions
channels for significant computational savings, DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017` for a generalization of the
residual architecture, and the Squeeze-and-excitation networks to
allow for efficient information transfer between channels
:cite:`Hu.Shen.Sun.2018`.

While the idea of *deep* neural networks is quite simple (stack
together a bunch of layers), performance can vary wildly across
architectures and hyperparameter choices.  The neural networks
described in this chapter are the product of intuition, a few
mathematical insights, and a lot of trial and error.  We present these
models in chronological order, partly to convey a sense of the history
so that you can form your own intuitions about where the field is
heading and perhaps develop your own architectures.  For instance,
batch normalization and residual connections described in this chapter
have offered two popular ideas for training and designing deep models,
both of which have since been applied to architectures beyond computer
vision, too.

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

