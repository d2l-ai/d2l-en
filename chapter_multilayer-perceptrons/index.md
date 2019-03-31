# Multilayer Perceptrons

In this chapter we will introduce your first truly *deep* networks. The simplest deep networks are called multilayer perceptrons, and they consist of many layers of neurons each fully connected to those in the layer below (from which they receive input) and those above (which they, in turn, influence).
When we train high-capacity models we run the risk of overfitting. Thus, we will need to introduce the notions of overfitting, underfitting, and capacity control. Regularization techniques such as dropout, numerical stability and initialization round out the presentation. Throughout, we focus on applying the models to real data, to give the reader a firm grasp not just of the concepts but also of the practice of using deep networks. Issues of performance, scalability and efficiency are relegated to the next chapters.

```eval_rst

.. toctree::
   :maxdepth: 2

   linear-regression
   linear-regression-scratch
   linear-regression-gluon
   softmax-regression
   fashion-mnist
   softmax-regression-scratch
   softmax-regression-gluon
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
