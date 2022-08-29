```{.python .input  n=2}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=2}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "afe7cbb4e15c441290b51d5debc66594",
    "version_major": 2,
    "version_minor": 0
   },
   "text/plain": "interactive(children=(Dropdown(description='tab', options=('mxnet', 'pytorch', 'tensorflow'), value=None), Out\u2026"
  },
  "metadata": {},
  "output_type": "display_data"
 }
]
```

# What is Hyperparameter Optimization?
:label:`sec_what_is_hpo`

A deep neural network comes with a large number of weight and bias parameters. These parameters are learned (that is, estimated) during training. As we have seen in Chapter :ref:`sec_sgd`, stochastic gradient is a widely adopted algorithm suitable for this task.
On top of these, every neural networks has additional **hyperparameters** that need to be configured by the user.
For example, to ensure that stochastic gradient descent converges to a local optimum of the training loss :ref:`chap_optimization`, we have to adjust the learning rate and batch size. To avoid overfitting on the training dataset :ref:`sec_polynomial`, we might have to set regularization parameters, such as weight decay :ref:`sec_weight_decay` or dropout :ref:`sec_dropout`. We can define the capacity and inductive bias of the model by setting the number of layers and number of units or filters per layer (i.e., the effective number
of weights).

Unfortunately, we cannot simply adjust these hyperparameters by minimizing the training loss, because this would lead to overfitting on the training data. For example, setting regularization parameters, such as dropout :ref:`sec_dropout` or weight decay :ref:`sec_weight_decay` to zero leads to a small training loss, but might hurt the generalization performance.

Without a different form of automation, hyperparameters have to be set manually in a
trial-and-error fashion, in what amounts to a time-consuming and difficult part of machine
learning workflows :cite:`hpo`. For example, consider training a ResNet :ref:`sec_resnet` on CIFAR-10, which takes on an Amazon Elastic Cloud Compute (EC2) g4dn.xlarge instance more than 2 hours. Even if we manually try only a small set of hyperparameters configurations (~10) in sequence, this would already take us roughly one day. To make matters worse, hyperparameters are usually not directly transferable
across architectures and datasets :cite:`feurer-arxiv22`,`wistuba-ml18`,`bardenet-icml13a`,
and need to be re-optimized for every new task. Also, for most hyperparameters,
there are no rule-of-thumbs, and expert knowledge is required to find sensible values.

Hyperparameter optimization (HPO) algorithms are designed to tackle this problem in 
a principled and automated fashion :cite:`feurer-automlbook18a`, by framing it as a global optimization problem.
The default objective is the error on a hold-out validation dataset, but could
in principle be any other business metric. It can be combined with or constrained by
secondary objectives (see :ref:'sec_hpo_advanced'), such as training time, inference time, or model complexity. 


Recently, hyperparameter optimization has been extended to **neural architecture
search (NAS)** :cite:`elsken-arxiv18a`,`wistuba-arxiv19`, where the goal is to find
entirely new neural network architectures. Compared to classical HPO, NAS is even more
expensive in terms of computation and requires additional efforts to remain feasible in
practice. Both, HPO and NAS can be considered as sub-fields of 
AutoML :cite:`hutter-book19a`, which aims to automated the entire ML pipeline.

In this section we will introduce HPO and show how we can automatically find the best hyperparameters of the logistic regression example introduced in :ref:`sec_softmax_concise`. 

##  The Optimization Problem
:label:`sec_definition_hpo`

We will start with a simple toy problem: searching for the learning rate of the multi-class logistic regression model from :ref:sec_sgd to minimize the validation error on the Fashion MNIST dataset. While other hyperparameters
like batch size or number of epochs are also worth tuning, we focus on learning
rate alone for simplicity.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l

import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
```

```{.python .input  n=9}
%%tab pytorch
class SoftmaxClassification(d2l.Classification): #@save
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(nn.Flatten(),
                                 nn.LazyLinear(num_outputs))
    def forward(self, X):
        return self.net(X)
```

```{.json .output n=9}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

Before we can run HPO, we first need to define two ingredients: the objective function and the configuration space.

### The Objective Function


The performance of a learning algorithm can be seen as a function $f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from the hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the validation loss. For every evaluation of $f(\mathbf{x})$, we have to train and validate our machine learning model, which can be time and compute intensive in the case of deep neural networks trained on large datasets. Now, given our criterion $f(\mathbf{x})$ our goal is to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. 

There is no simple way to compute gradients of $f$ with respect to $\mathbf{x}$, because it would require to propagate the gradient through the entire training process. While there is recent work :cite:`maclaurin-icml15`,`franceschi-icml17a` to drive HPO by approximate "hypergradients", none of the existing approaches are competitive with the state-of-the-art yet, and we will not discuss them here. Furthermore, the computational burden of evaluating $f$ requires HPO algorithms to approach the global optimum with as few samples as possible.

The training of neural networks is stochastic (e.g., weights are randomly initialized, mini-batches are randomly sampled), so that our observations will be noisy: $y \sim f(\mathbf{x}) + \epsilon$, where we assume that $\epsilon \sim N(0, \sigma)$.

Faced with all these challenges, we usually try to identify a small set of well performing hyperparameter configurations quickly, instead of hitting the global optima exactly. However, due to large computational demands of most neural networks models, even this can take days or weeks of compute. We will explore in further sections :ref:`sec_mf_hpo`, how we can speed-up the optimization process by either distributing the search or using cheaper-to-evaluate approximations of the objective function.


Now, since we would like to optimize the validation error, we need to add a function computing this quantity.

```{.python .input  n=8}
%%tab pytorch

@d2l.add_to_class(d2l.Trainer) #@save
def validate(self, model):
    model.eval()
    accuracy = 0
    val_batch_idx = 0
    
    for batch in self.val_dataloader:
        with torch.no_grad():
            x, y = self.prepare_batch(batch)
            y_hat = model(x)
            accuracy += model.accuracy(y_hat, y)
        val_batch_idx += 1
    
    return 1 -  accuracy / val_batch_idx
```

```{.json .output n=8}
[
 {
  "ename": "NameError",
  "evalue": "name 'Trainer' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m/var/folders/ld/vzcn3j2d7yg493b1c6m0ypprdqgxkm/T/ipykernel_94597/2202539186.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0;34m@\u001b[0m\u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0madd_to_class\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mTrainer\u001b[0m\u001b[0;34m)\u001b[0m \u001b[0;31m#@save\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      2\u001b[0m \u001b[0;32mdef\u001b[0m \u001b[0mvalidate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mself\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0mmodel\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0meval\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      4\u001b[0m     \u001b[0maccuracy\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mval_batch_idx\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0;36m0\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mNameError\u001b[0m: name 'Trainer' is not defined"
  ]
 }
]
```

We optimize validation error with respect to the hyperparameter configuration `config`, consisting of the `learning_rate`. For each evaluation, we train our model
for `max_epochs` epochs, then compute and return its validation error:

```{.python .input  n=5}
%%tab all

def hpo_objective_softmax_classification(config, max_epochs=10):  #@save 
    learning_rate = config['learning_rate']
    trainer = d2l.Trainer(max_epochs=max_epochs)
    data = d2l.FashionMNIST(batch_size=16)
    model = d2l.SoftmaxClassification(num_outputs=10, lr=learning_rate)
    trainer.fit(model=model, data=data)
    validation_error = trainer.validate(model=model)
    return validation_error.numpy()
```

### The Configuration Space

:label:`sec_intro_config_spaces`

Along with the objective function $f(\mathbf{x})$, we also need to define the feasible set
$\mathbf{x} \in \mathcal{X}$ to optimize over, known as *configuration space* or *search
space*. For our logistic regression example, we will use:

```{.python .input  n=6}
from scipy import stats

config_space = {
   "learning_rate": stats.loguniform(1e-4, 1)
} 
```

Each hyperparameter has a data type, such as `float` for `learning_rate`, as well as a closed bounded range
(i.e., lower and upper bounds). We usually also assign a prior distribution (e.g uniform or log-uniform) to each hyperparameter. Some positive parameters, such as `learning_rate`, are best represented on a logarithmic scale as optimal values can differ by several orders of magnitude, while others come with linear scale.

Below we show a simple example of a configuration space consisting of typical hyperparameters of feed-forward neural networks including their type and standard ranges.

![Example configuration space for a simple neural network architecture](img/example_search_space.png)
:width:`40px`
:label:`example_search_space`


In general, the structure of the configuration space $\mathcal{X}$ can be complex and it can be quite different from $\mathbb{R}^d$. In practice, some hyperparameters may depend on the value of others. For example, if we try to tune both the number of layers and widths per layer for a multi-layer perceptron,
the width of the $l$-th layer is relevant only if the network has at least $l+1$ layers. These advanced HPO problems are beyond the scope of this chapter. We refer the interested reader to :cite:`hutter-lion11a`,`jenatton-icml17a`, `baptista-icml18a`.

The configuration spaces plays an important role for hyperparameter optimization, since no algorithms can find something that is not included in the configuration space. On the other hand, if the ranges are too large, the time to find a well performing configurations might become infeasible.

## Random Search

Now, we look at the first algorithm to solve our hyperparameter optimization problem: random search. 
The main idea of random search is to independently sample from the configuration space until a predefined budget (e.g maximum number of iterations) is exhausted and to return the best observed configuration. The evaluation can be executed sequentially (as we do here) or in parallel (see Section :ref:`sec_rs`).

We sample the learning rate from log uniform distribution, since it usually changes on a logarithmic scale.

```{.python .input  n=7}
import numpy as np

errors, values = [], []
num_iterations = 10

for iteration in range(num_iterations):
    learning_rate = config_space['learning_rate'].rvs()
    y = hpo_objective_softmax_classification({'learning_rate': learning_rate})
    values.append(learning_rate)
    errors.append(y)
```

```{.json .output n=7}
[
 {
  "ename": "AttributeError",
  "evalue": "module 'd2l.torch' has no attribute 'SoftmaxClassification'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
   "\u001b[0;32m/var/folders/ld/vzcn3j2d7yg493b1c6m0ypprdqgxkm/T/ipykernel_94597/2308979264.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      6\u001b[0m \u001b[0;32mfor\u001b[0m \u001b[0miteration\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mrange\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnum_iterations\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      7\u001b[0m     \u001b[0mlearning_rate\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0msearch_space\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0;34m'learning_rate'\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrvs\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 8\u001b[0;31m     \u001b[0my\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mhpo_objective_softmax_classification\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m{\u001b[0m\u001b[0;34m'learning_rate'\u001b[0m\u001b[0;34m:\u001b[0m \u001b[0mlearning_rate\u001b[0m\u001b[0;34m}\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      9\u001b[0m     \u001b[0mvalues\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mlearning_rate\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m     \u001b[0merrors\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mappend\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0my\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;32m/var/folders/ld/vzcn3j2d7yg493b1c6m0ypprdqgxkm/T/ipykernel_94597/3098096573.py\u001b[0m in \u001b[0;36mhpo_objective_softmax_classification\u001b[0;34m(config, max_epochs)\u001b[0m\n\u001b[1;32m      5\u001b[0m     \u001b[0mtrainer\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mTrainer\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmax_epochs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmax_epochs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m     \u001b[0mdata\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mFashionMNIST\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbatch_size\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m16\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 7\u001b[0;31m     \u001b[0mmodel\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mSoftmaxClassification\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mnum_outputs\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m10\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mlr\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mlearning_rate\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      8\u001b[0m     \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdata\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mdata\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      9\u001b[0m     \u001b[0mvalidation_error\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mtrainer\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mvalidate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mmodel\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mAttributeError\u001b[0m: module 'd2l.torch' has no attribute 'SoftmaxClassification'"
  ]
 }
]
```

```{.python .input  n=7}
import numpy as np
best_idx = np.argmin(errors)
print(f'optimal learning rate = {values[best_idx]}')
```

Arguably because of its simplicity, random search is one of the most frequently used HPO algorithms. It doesn't require any sophisticated implementation and can be applied to any hyperparameter type.

Below we plot the validation error of each hyperparameter configuration we just evaluated.

```{.python .input  n=8}
import matplotlib.pyplot as plt

plt.figure(dpi=200)
plt.scatter(values, errors)
plt.xscale('log')
plt.ylabel('validation error')
plt.xlabel('learning rate')
plt.show()
```

Unfortunately random search comes with a few shortcomings. First, it does not adapt the sampling distribution based on the previous observations it collected. Hence, it is equally likely to sample a poorly performing configurations  than a better performing configuration. Second, the same amount of resources are spend for all configurations, even though they are less likely to outperform previously seen configurations.

In the next sections we will look at more sample efficient hyperparameter optimization algorithms that overcome the shortcomings of random search by using a model to guide the search. We will also look at algorithms that automatically stop the evaluation process of poorly performing configurations.

# Summary

# Exercise
