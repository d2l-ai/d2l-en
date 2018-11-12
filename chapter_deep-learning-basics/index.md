# Deep Learning Basics 

This chapter introduces the basics of deep learning. This includes network architectures, data, loss functions, optimization, and capacity control. In order to make things easier to grasp, we begin with very simple concepts, such as linear functions, linear regression, and stochastic gradient descent. This forms the basis for slightly more complex techniques such as the softmax (statisticians refer to it as multinomial regression) and multilayer perceptrons. At this point we are already able to design fairly powerful networks, albeit not with the requisite control and finesse. For this we need to understand the notion of capacity control, overfitting and underfitting. Regularization techniques such as dropout, numerical stability and initialization round out the presentation. Throughout, we focus on applying the models to real data, such as to give the reader a firm grasp not just of the concepts but also of the practice of using deep networks. Issues of performance, scalability and efficiency are relegated to the next chapters. 

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
   kaggle-house-price
   environment

```
