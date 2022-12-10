# Hyperparameter Optimization
:label:`chap_hyperopt`

**Aaron Klein** (*Amazon*), **Matthias Seeger** (*Amazon*), and **Cedric Archambeau** (*Amazon*)

The performance of every machine learning model depends on its hyperparameters.
They control the learning algorithm or the structure of the underlying
statistical model. However, there is no general way to choose hyperparameters
in practice. Instead, hyperparameters are often set in a trial-and-error manner
or sometimes left to their default values by practitioners, leading to
suboptimal generalization.

Hyperparameter optimization provides a systematic approach to this problem, by
casting it as an optimization problem: a good set of hyperparameters should (at
least) minimize a validation error. Compared to most other optimization problems
arising in machine learning, hyperparameter optimization is a nested one, where
each iteration requires training and validating a machine learning model.

In this chapter, we will first introduce the basics of hyperparameter
optimization. We will also present some recent advancements that improve the
overall efficiency of hyperparameter optimization by exploiting cheap-to-evaluate
proxies of the original objective function. At the end of this chapter, you
should be able to apply state-of-the-art hyperparameter optimization techniques
to optimize the hyperparameter of your own machine learning algorithm.

```toc
:maxdepth: 2

hyperopt-intro
hyperopt-api
rs-async.md
sh-intro
sh-async
```

