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

- validation performance
- cross validation performance
- 



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



- What is the problem? Why is it useful?
- Fixing concepts: search space, types of hyperparameters, dmetrics
- Concepts of sequential HPO (i.e., BO): Could also be done further down
- Early stopping and multi-fidelity in a nutshell: Could also be done further down
- Mention AutoGluon (but details more below). But if this needs to be installed, this has to be detailed here. It would be easiest if AutoGluon is installed already by default in the D2L setup

