# Machine Learning Fundamentals
:label:`chap_ml-fundamentals`

As illustrated in :numref:`chap_introduction`,
deep learning is just one among many popular methods for solving machine learning problems.
As we have encountered when training
linear regressions, softmax regressions,
and multilayer perceptrons,
optimization algorithms
reduce loss function values
by iteratively updating model parameters.
However,
when we train high-capacity models,
such as deep neural networks, we run the risk of overfitting.
Thus, we will need to provide your first rigorous introduction
to the notions of overfitting, underfitting, and model selection.
To help you combat these problems,
we will introduce regularization techniques such as weight decay and dropout.
In view of many failed machine learning *deployments*,
it is necessary to
expose some common concerns
and stimulate the critical thinking required to detect these situations early, mitigate damage, and use machine learning responsibly.
Throughout, we aim to give you a firm grasp not just of the concepts
but also of the practice of using machine learning models.
At the end of this chapter,
we apply what we have introduced so far to a real case: house price prediction.
We punt matters relating to the computational performance,
scalability, and efficiency of our models to subsequent chapters.

```toc
:maxdepth: 2

model-selection
underfit-overfit
```

