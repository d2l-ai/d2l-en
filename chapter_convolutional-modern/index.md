# Modern Convolutional Neural Networks
:label:`chap_modern_cnn`

Now that we understand the basics of wiring together CNNs, let's take
a tour of modern CNN architectures. This tour is, by
necessity, incomplete, thanks to the plethora of exciting new designs
being added. Their importance derives from the fact that not only can
they be used directly for vision tasks, but they also serve as basic
feature generators for more advanced tasks such as tracking
:cite:`Zhang.Sun.Jiang.ea.2021`, segmentation :cite:`Long.Shelhamer.Darrell.2015`, object
detection :cite:`Redmon.Farhadi.2018`, or style transformation
:cite:`Gatys.Ecker.Bethge.2016`.  In this chapter, most sections
correspond to a significant CNN architecture that was at some point
(or currently) the base model upon which many research projects and
deployed systems were built.  Each of these networks was briefly a
dominant architecture and many were winners or runners-up in the
[ImageNet competition](https://www.image-net.org/challenges/LSVRC/)
which has served as a barometer of progress on supervised learning in
computer vision since 2010. It is only recently that Transformers have begun
to displace CNNs, starting with :citet:`Dosovitskiy.Beyer.Kolesnikov.ea.2021` and 
followed by the Swin Transformer :cite:`liu2021swin`. We will cover this development later 
in :numref:`chap_attention-and-transformers`. 

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
both of which have since also been applied to architectures beyond computer
vision.

We begin our tour of modern CNNs with AlexNet :cite:`Krizhevsky.Sutskever.Hinton.2012`,
the first large-scale network deployed to beat conventional computer
vision methods on a large-scale vision challenge; the VGG network
:cite:`Simonyan.Zisserman.2014`, which makes use of a number of
repeating blocks of elements; the network in network (NiN) that
convolves whole neural networks patch-wise over inputs
:cite:`Lin.Chen.Yan.2013`; GoogLeNet that uses networks with
multi-branch convolutions :cite:`Szegedy.Liu.Jia.ea.2015`; the residual
network (ResNet) :cite:`He.Zhang.Ren.ea.2016`, which remains one of
the most popular off-the-shelf architectures in computer vision;
ResNeXt blocks :cite:`Xie.Girshick.Dollar.ea.2017`
for sparser connections;
and DenseNet
:cite:`Huang.Liu.Van-Der-Maaten.ea.2017` for a generalization of the
residual architecture. Over time many special optimizations for efficient 
networks have been developed, such as coordinate shifts (ShiftNet) :cite:`wu2018shift`. This 
culminated in the automatic search for efficient architectures such as 
MobileNet v3 :cite:`Howard.Sandler.Chu.ea.2019`. It also includes the 
semi-automatic design exploration of :citet:`Radosavovic.Kosaraju.Girshick.ea.2020`
that led to the RegNetX/Y which we will discuss later in this chapter. 
The work is instructive insofar as it offers a path for marrying brute force computation with 
the ingenuity of an experimenter in the search for efficient design spaces. Of note is
also the work of :citet:`liu2022convnet` as it shows that training techniques (e.g., optimizers, data augmentation, and regularization)
play a pivotal role in improving accuracy. It also shows that long-held assumptions, such as 
the size of a convolution window, may need to be revisited, given the increase in 
computation and data. We will cover this and many more questions in due course throughout this chapter.

```toc
:maxdepth: 2

alexnet
vgg
nin
googlenet
batch-norm
resnet
densenet
cnn-design
```

