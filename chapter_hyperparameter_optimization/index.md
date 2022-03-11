# Hyperparameter Optimization
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), and **Cedric Archambeau** (*Amazon*)

In practice the performance of every machine learning algorithm depends on its hyperparameters that control the training process or the capacity of the underlying statistical model. However, for the most algorithms, there is no general way to set these hyperparameter. Instead hyperparameters are often found in a trial-and-error manner or left to their default values.

Hyperparameter optimization provides a more systematic approach, by casting it as an optimization problem. We aim to find the right hyperparameters that maximize the validation performance. Compared to the most other optimization problems in machine learning, hyperparameter optimization is a nested optimization problem, where in each iteration we train and validate a machine learning model.

In this chapter, we will first introduce the basics of hyperparameter optimization. We will also present some recent advancements that improve the overall efficiency of hyperparameter optimization by exploiting cheap-to-evaluate proxies of the original objective function. At the end of this chapter, you should be able to apply state-of-the-art hyperparameter opimtization techniques to optimize the hyperparameter of your own machine learning algorithm.

```toc
:maxdepth: 2

hyperopt-intro
rs-intro
rs-async
hyperband-intro
hyperband-async
```
