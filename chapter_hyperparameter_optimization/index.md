# Hyperparameter Optimization
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), and **Cedric Archambeau** (*Amazon*)

In practice the performance of every machine learning algorithm depends on its hyperparameters that control the training process or the capacity of the underlyin statistical model. However, for the most algorithm, there is no general way to set these hyperparameter. Instead hyperparameters are often found by trial-and-error search or left to their default values.

Hyperparameter optimization provides a more systematic approach to the problem, by casting it as a optimization problem. We aim to find the right hyperparameters that maximize the validation performance. Compared to the most other optimization problems, hyperparameter optimization is a nested optimization problem, where in each iteration we train and validate a machine learning model.

In this chapter, we will first introduce the basics of hyperparameter optimization. We will also present some recent advancements that improve the overall efficiency of hyperparameter optimization by exploiting cheap-to-evaluate proxies of the original objective function. At the end of this chapter, you should be able to apply state-of-the-art hyperparameter opimtization techniques to optimize the hyperparameter of your own machine learning models.   

```toc
:maxdepth: 2

hyperopt-intro
rs
bo
advanced-hpo
```

