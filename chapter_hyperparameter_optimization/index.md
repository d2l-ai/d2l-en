# Hyperparameter Optimization
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), and **Cedric Archambeau** (*Amazon*)

In practice the performance of every machine learning model depends on its hyperparameters. Hyperparameters control the learning algorithm or the structure of the underlying statistical model. However, there is no general way to choose hyperparameters in practice. Instead hyperparameters are often set in a trial-and-error manner or sometimes left to their default values by practioners, leading to suboptimal generalization.

Hyperparameter optimization provides a systematic approach to this problem, by casting it as an optimization problem: a good set of hyperparameters should (at least) minimize a validation error. Compared to most other optimization problems arising in machine learning, hyperparameter optimization is a nested one, where each iteration requires to traine and validate a machine learning model.

In this chapter, we will first introduce the basics of hyperparameter optimization. We will also present some recent advancements that improve the overall efficiency of hyperparameter optimization by exploiting cheap-to-evaluate proxies of the original objective function. At the end of this chapter, you should be able to apply state-of-the-art hyperparameter opimtization techniques to optimize the hyperparameter of your own machine learning algorithm.

*CA: it is not clear to me how the text above relates to the intro later*



<!-- In this chapter, we provide an overview of the basics of hyperparameter optimization (HPO) and look at several state-of-the-art methods from the literature. As a running example, we will show how to automatically tune hyperparameters of a convolutional neural network. Any successful HPO method needs to provide solutions for two decision-making primitives, **scheduling** and **search**, and we will highlight the most prominent current solutions for either. Scheduling amounts to decisions of how much resources to spend on a hyperparameter configuration, e.g. when to stop, pause, or resume training, while search is about which configurations to evaluate in the first place. A specific focus in this chapter will lie on model-based approaches to search, which in practice are more sample efficient than their random search-based counterparts. Since hyperparameter optimization requires us to train and validate several neural networks, we will also see how we can distribute these methods. To avoid distracting boiler-plate code, we will use the Python framework **Syne Tune**, providing us with an simple interface for distributed hyperparameter optimization. You can install it via:
 -->

```toc
:maxdepth: 2

hyperopt-intro
hyperopt-api
rs-intro
rs-async
hyperband-intro
hyperband-async
```

