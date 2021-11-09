# Overview of Hyperparameter Optimization

While weight parameters of a neural network model are automatically determined during training, e.g. by stochastic gradient descent, **hyperparameters** cannot be learned in this way. Without a different form of automation, the user has to set them manually by trial and error, in what amounts to a time-consuming and difficult part of machine learning workflows. For example, to train a neural networks with stochastic gradient descent we need to choose a learning rate and a batch size. Broadly, we distinguish between hyperparameters that control the learning process (e.g., learning rate, batch size, optimizer choice) and hyperparameters that define model shape and capacity (e.g., type of activation function, number of units per layer). The choice of hyperparameters in general directly affects the final performance of our machine learning algorithm. For example, previous work beat the performance of advanced state-of-the-art models by optimizing the hyperparameter of much simpler models (TODO: add ref). 

The need to manually tune the training process of a deep neural network constitutes a significant gap towards the promise of end to end learning and artificial intelligence. If we are willing to spend sufficient computational resources, our methods should be able to configure themselves. Hyperparameter optimization aims to automatically find a performant hyperparameter configuration of any machine learning method. The main idea is to cast the search for the right hyperparameters as an optimization problem, to maximize the validation performance of the algorithm.

In this chapter, we provide an overview of the basics of hyperparameter optimization and look at several state-of-the-art methods from the literature. As a running example, we will show how to automatically tune the hyperparameters of a convolutional neural network. Any successful HPO method needs to provide solutions for two decision-making primitives, **scheduling** and **search**, and we will highlight the most prominent current solutions for either. Scheduling amounts to decisions of how much resources to spend on a hyperparameter configuration, e.g. when to stop, pause, or resume training, while search is about which configurations to evaluate in the first place. A specific focus in this chapter will lie on model-based approaches to search, which in practice are often more sample efficient than their random-search based counterparts. Since hyperparameter optimization requires us to train and validate several machine learning models, we will also see how we can distribute these methods. To avoid distracting boiler-plate code, we will use the Python framework **SageMaker Tune**, providing us with an simple interface for distributed hyperparameter optimization.



## Problem Definition

The performance of our machine learning algorithm can be seen as a function $f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from our hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the validation performance. For every evaluation of $f(\mathbf{x})$, we have to train and validate our machine learning algorithm, which can take a long time. We will see below how much cheaper surrogates can help with the optimization of $f$. Training is stochastic in general (e.g., weights are randomly initialized), so that our observations will be noisy: $y \sim f(\mathbf{x}) + \epsilon$, where we assume that $\epsilon \sim N(0, \sigma)$.

Now, given our objective function $f$, hyperparameter optimization aims to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. Note that $f$ models the validation performance after training, and there is no efficient way to compute gradients with respect to $\mathbf{x}$. While there is recent work to drive HPO by approximate "hypergradients", none of the existing approaches are competitive with the state-of-the-art yet, and we will not discuss them here.


## Metrics 

Arguably the most common way to estimate the validation performance is to compute the loss on a hold out validation set. We cannot use the training loss to optimize the hyperparameters, as this would lead to overfitting. In case of small datasets we often do not have access to a sufficient large validation dataset. In this case we can apply $k$-fold cross validation and use the average validation loss across all folds as metric to optimize. However, at least in the standard form of HPO, this makes the optimization process $k$ times slower.

In practice, hyperparameter configuratons $\mathbf{x}$ come with different costs $c: \mathcal{X} \rightarrow \mathbf{R}$, such as wall-clock time. For example, if we optimize the number of units of neural networks, larger networks are more expensive to train the smaller networks. Usually, we are not so much interested in how often we have to evaluate $f$, but rather try to find $\mathbf{x}_{\star}$ as quickly as possible.



## Constraint Hyperparameter Optimization

In many scenarios we are not just interested in finding $\mathbf{x}_{\star}$, but a hyperparameter configuration that additionally full fills certain constraints. More formally, we seek to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ s.t $c_1(\mathbf{x}) > 0, ..., c_m(\mathbf{x}) > 0$. Typical constrains could be, for example, the memory consumption of $\mathbf{x}$ or fairness constraints.



## Search Spaces

Before we can optimize our machine learning algorithm, we first need to define for each hyperparameter the type, e.g float, integer, categorical and the domain with all possible values. This leads to our search space $\mathcal{X}$.

Below we show an typical search space for a neural network. Important for the search space definition is to first determine if a hyperparameter changes on a logarithmic or a linear scale. Learning rates for example typically live on a logarithmic scale, whereas, for example, momentum are usually on a linear scale. For continuous parameters we mostly define a uniform distribution, to not induce any bias.

TODO: Maybe we should show this example in the next section when we introduce the CNN example?

```{.python .input  n=4}
import scipy.stats as stats

search_space = {
  'learning_rate': stats.loguniform(1e-6, 1),
  'weight_decay': stats.loguniform(1e-9, 1e-1),
  'momentum': stats.uniform(0, 1),
  'batch_size': stats.randint(8, 128)
}
```

## How to Evaluate Hyperparameter Optimization Methods?

In the next section we will look at different hyperparameter optimization methods. To understand their difference better, we will described here how we can evaluate them. In practice, we usually run the hyperparameter optimization once and use the best found hyperparameters to train our final model. However, since the most hyperparameter optimization method come with an intrinsic randomness, we have to run them multiple times with a different seed for the random number generator and average the results to assess their performance.

Key requirement for any hyperparameter optimization methods is not just to return a well peforming configuration, but to find it as quickly as possible. Therefor, we will look at the anytime performance of an algorithm. We define the anytime performance as the validation performance of the best found configuration, i.e incumbent, at the current time step. Furthermore, we usually care less about the number of function evaluations, but rather the time (i.e wall-clock time) we have to wait until our optimizer returns a hyperparameter configuration. Hence, we will compare hyperparameter optimization algorithms by plotting the anytime performance over wall-clock time. 



## Introduction SageMakerTune

- How to install the package
- Describe the general idea of the package
- 



## Summary

