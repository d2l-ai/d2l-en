# Multilayer Perceptrons
:label:`chapter_perceptrons`

In this chapter, we will introduce your first truly *deep* networks.
The simplest deep networks are called multilayer perceptrons,
and they consist of many layers of neurons
each fully connected to those in the layer below
(from which they receive input)
and those above (which they, in turn, influence).
When we train high-capacity models we run the risk of overfitting.
Thus, we will need to provide your first rigorous introduction
to the notions of overfitting, underfitting, and capacity control.
To help you combat these problems,
we will introduce regularization techniques such as dropout and weight decay.
We will also discuss issues relating to numerical stability and parameter initialization that are key to successfully training deep networks.
Throughout, we focus on applying models to real data,
aiming to give the reader a firm grasp not just of the concepts
but also of the practice of using deep networks.
We punt matters relating to the computational performance,
scalability and efficiency of our models to subsequent chapters.

```toc
:maxdepth: 2

mlp
mlp-scratch
mlp-gluon
underfit-overfit
weight-decay
dropout
backprop
numerical-stability-and-init
environment
kaggle-house-price
```
