# Linear Neural Networks for Regression
:label:`chap_regression`

Before we worry about making our neural networks deep,
it will be helpful to implement some shallow neural networks,
for which the inputs connect directly to the outputs.
This will prove important for a few reasons.
First, rather than getting distracted by complicated architectures,
we can focus on the basics of neural network training,
including parameterizing the output layer, handling data,
specifying a loss function, and training the model.
Second, this class of shallow networks happens
to comprise the set of linear models,
which subsumes many classical methods for statistical prediction,
including linear and softmax regression.
Understanding these classical tools is pivotal
because they are widely used in many contexts
and we will often need to use them as baselines
when justifying the use of fancier architectures.
This chapter will focus narrowly on linear regression
and the subsequent chapter will extend our modeling repertoire
by developing linear neural networks for classification.

```toc
:maxdepth: 2

linear-regression
oo-design
synthetic-regression-data
linear-regression-scratch
linear-regression-concise
generalization
weight-decay
```

