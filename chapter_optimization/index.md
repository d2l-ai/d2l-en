# Optimization Algorithms

If you have read this book in order to this point, then you have already used optimization algorithms to train deep learning models. Specifically, when training models, we use optimization algorithms to continue updating the model parameters to reduce the value of the model loss function. When iteration ends, model training ends along with it. The model parameters we get here are the parameters that the model learned through training.

Optimization algorithms are important for deep learning. On the one hand, training a complex deep learning model can take hours, days, or even weeks. The performance of the optimization algorithm directly affects the model's training efficiency. On the other hand, understanding the principles of different optimization algorithms and the meanings of their hyperparameters will enable us to tune the hyperparameters in a targeted manner to improve the performance of deep learning models.

In this chapter, we explore common deep learning optimization algorithms in depth.

```eval_rst

.. toctree::
   :maxdepth: 2

   optimization-intro
   gd-sgd
   minibatch-sgd
   momentum
   adagrad
   rmsprop
   adadelta
   adam
```
