# Hyperparameter Optimization
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), and **Cedric Archambeau** (*Amazon*)

In practice the performance of every machine learning models depends on its hyperparameters. Hyperparameters control the training process or the structure of the underlying statistical model. However, there is no general way to choose hyperparameters in practice. Instead hyperparameters are often set in a trial-and-error manner or sometimes left to their default values.

Hyperparameter optimization provides a systematic approach to this problem, by casting it as an optimization problem: a good set of hyperparameters should minimize the validation loss. Compared to most other optimization problems arising in machine learning, hyperparameter optimization is a nested one, where each iteration requires the trained and the validation of a machine learning model.

In this chapter, we will first introduce the basics of hyperparameter optimization. We will also present some recent advancements that improve the overall efficiency of hyperparameter optimization by exploiting cheap-to-evaluate proxies of the original objective function. At the end of this chapter, you should be able to apply state-of-the-art hyperparameter opimtization techniques to optimize the hyperparameter of your own machine learning algorithm.

```toc
:maxdepth: 2

hyperopt-intro
rs-intro
rs-async
hyperband-intro
hyperband-async
```

