# Overview of Hyperparameter Optimization



- Hyperparameter optimization gain a lot of interest recently

- several paper showed that one can beat sota performance with simpler models + HPO
- 

## What are Hyperparameters?

- Exlain the difference between hyperparameters and parameters
- Why do we need hyperparameters
- Types of hyperparameters that control the training process or the capacity of the model



## Problem Definition

We treat the performance of our machine learning algorithm as a function $f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from our hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the validation performance. Everytime we evaluate $f(\mathbf{x})$ we have to train and validate our machine learning algorithm. Especially, with contemporary deep neural networks this can consume quite a bit of time. But will later show how we can obtain cheaper approximations of $f$.  Also, due to the intrinsic randomness of the most machine learning methods, we usually cannot observe $f$ directly, but instead $y \sim f(\mathbf{x}) + \epsilon$ where we assume that $\epsilon \sim N(0, \sigma)$.

Now, given our objective function $f$, hyperparameter optimization aims to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. Note that, $f$ here models the validation performance, which means that there is no straight-forward way to compute gradients with respect to $\mathbf{x}$.

## Metrics 

Most common way to estimate the validation performance is to compute the loss on a hold out validation set. We cannot use the training loss to optimize the hyperparameters since this would lead to overfitting, for example, if we optimize regularization parameters.
In case of small datasets we often do not have access to a sufficient large validation dataset. In this case we can apply $k$ fold cross validation and use the average validation loss across all folds as metric to optimize. However, at least in the standard form of HPO, this makes the optimization process $k$ times slower.

In practice, hyperparameter configuratons $\mathbf{x}$ come with different costs $c: \mathcal{X} \rightarrow \mathbf{R}$, such as wall-clock time. For example, if we optimize the number of units of neural networks, larger networks are more expensive to train the smaller networks. Usually, we are not so much interested in how often we have to evaluate $f$, but rather try to find $\mathbf{x}_{\star}$ as quickly as possible.


## Constraint Optimization

In many scenarios we are not just interested in finding $\mathbf{x}_{\star}$, but a hyperparameter configuration that additionally full fills certain constraints. More formally, we seek to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x}) s.t c_1(\mathbf{x}) > 0, ...,c_m(\mathbf{x}) > 0 $. Typical constrains could be, for example, the memory consumption of $\mathbf{x}$ or fairness constraints.

## Search Spaces

Before we can optimize our machine learning algorithm, we first need to define for each hyperparameter the type, e.g float, integer, categoriacl and the domain with all possible values. This leads to our search space $\mathcal{X}$.

Below we show an example search space for optimize the learning rate and weight decay parameter of a neural network. 

TODO: Maybe we should show this example later when we introduce the CNN example?

```{.python .input  n=4}
from sagemaker_tune.search_space import loguniform

search_space = {
    'learning_rate': loguniform(1e-6, 1e-4),
    'weight_decay': loguniform(1e-6, 1e-4)
}

```



## Introduction SageMakerTune

- How to install the package
- Describe the general idea of the package
- 



## Summary

