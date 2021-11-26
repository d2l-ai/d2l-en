```{.python .input  n=1}
# the pypi version seems outdated, load the latest github version directly
import sys
sys.path.append('/Users/kleiaaro/git/d2l-en/')
```

# What is Hyperparameter Optimization?


While weight parameters of a neural network model are automatically determined during training, e.g. by stochastic gradient descent, **hyperparameters** cannot be learned in this way. Without a different form of automation, the user has to set them manually by trial and error, in what amounts to a time-consuming and difficult part of machine learning workflows. We distinguish between hyperparameters that control the learning process (e.g., learning rate, batch size, momentum, optimizer choice) and hyperparameters that define model shape and capacity (e.g., type of activation function, number of units per layer). The choice of hyperparameters in general directly affects the final performance of our machine learning algorithm. For example, previous work beat the performance of advanced state-of-the-art models by optimizing the hyperparameter of much simpler models :cite:`snoek-nips12a`. 

Hyperparameters play a critical role in machine learning. Not only do they determine the generalization capabilities of trained models, they can even be critical for what constitutes the state-of-the-art. Indeed, results reported in an empirical study might look very differently for another choice of 
hyperparameters and so would be the conclusions drawn. Unfortunately, it is not uncommon that published results do not report the specific hyperparameters that were used to run experiments, for instance, to demonstrate that a proposed method  is superior to previously published ones, and are thus not reproducible, putting in 
question what actually is the state-of-the-art in machine learning today :cite:`haibe-kains:2020:transparency`.

The need to manually tune the training process and structure of a deep neural network constitutes a significant gap towards the promise of end to end learning and artificial intelligence. If we are willing to spend sufficient computational resources, our methods should be able to configure themselves. Hyperparameter optimization aims to automatically find a performant hyperparameter configuration of any machine learning method. The main idea is to cast the search for the right hyperparameters as an optimization problem, to maximize the validation performance of the algorithm.


In this chapter, we provide an overview of the basics of hyperparameter optimization and look at several state-of-the-art methods from the literature. As a running example, we will show how to automatically tune hyperparameters of a convolutional neural network. Any successful HPO method needs to provide solutions for two decision-making primitives, **scheduling** and **search**, and we will highlight the most prominent current solutions for either. Scheduling amounts to decisions of how much resources to spend on a hyperparameter configuration, e.g. when to stop, pause, or resume training, while search is about which configurations to evaluate in the first place. A specific focus in this chapter will lie on model-based approaches to search, which in practice are often more sample efficient than their random-search based counterparts. Since hyperparameter optimization requires us to train and validate several machine learning models, we will also see how we can distribute these methods. To avoid distracting boiler-plate code, we will use the Python framework **Syne Tune**, providing us with an simple interface for distributed hyperparameter optimization. You can install it via:

```{.python .input  n=23}
!pip install syne-tune
```

```{.json .output n=23}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Cloning into 'sagemaker_tune'...\nremote: Enumerating objects: 1488, done.\u001b[K\nremote: Counting objects: 100% (1488/1488), done.\u001b[K\nremote: Compressing objects: 100% (750/750), done.\u001b[K\nremote: Total 1488 (delta 955), reused 1173 (delta 672), pack-reused 0\u001b[K\nReceiving objects: 100% (1488/1488), 1016.16 KiB | 2.98 MiB/s, done.\nResolving deltas: 100% (955/955), done.\n"
 }
]
```

```{.python .input  n=16}
# TODO: can we import this from the d2l package?

from d2l import torch as d2l

import torch
from torch import nn

class AlexNet(d2l.Classification):
    def __init__(self, lr=0.1, momentum=0.9, dropout=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(224, 9, kernel_size=11, stride=4, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(9, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.Flatten(),
            nn.Linear(16, 20), nn.ReLU(), nn.Dropout(p=dropout),
            nn.Linear(20, 20), nn.ReLU(),nn.Dropout(p=dropout),
            nn.Linear(20, 10))
```

```{.python .input}

```

## How do we define Hyperparameter Optimization?
:label:`sec_definition_hpo`

The performance of our machine learning algorithm can be seen as a function $f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from our hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the validation performance. For every evaluation of $f(\mathbf{x})$, we have to train and validate our machine learning algorithm, which can take a long time. We will see below how much cheaper surrogates can help with the optimization of $f$. Training is stochastic in general (e.g., weights are randomly initialized), so that our observations will be noisy: $y \sim f(\mathbf{x}) + \epsilon$, where we assume that $\epsilon \sim N(0, \sigma)$.

Note how `objective` is parameterized by the hyperparameter configuration `config`,
consisting of `batch_size`, `learning_rate`, `momentum`, `weight_decay`, and it returns the
validation error after training for `epochs` epochs.


```{.python .input  n=22}
# TODO: are functions ok or should we use classes?


def objective(x, max_epochs = 10):
    batch_size = x['batch_size']
    lr = x['learning_rate']
    momentum = x['momentum']
    dropout = x['dropout']

    model = AlexNet(lr=lr, momentum=momentum, dropout=dropout)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=0)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
    
    # TODO: get validation error
        
    return validation_error
```

Now, given our objective function $f$, hyperparameter optimization aims to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. Since $f$ is the validation performance after training, there is no efficient way to compute gradients with respect to $\mathbf{x}$. While there is recent work to drive HPO by approximate "hypergradients", none of the existing approaches are competitive with the state-of-the-art yet, and we will not discuss them here.

## Which metrics are important?


MS: I think this is good for now, but we may want to check whether some basics here are already well explained in
early chapters, and if so, just refer to them.

### Objective Function

Arguably the most common way to estimate the validation performance of a machine learning algorithm is to compute its error (e.g classification error) on a hold out validation set. We cannot use the training loss to optimize the hyperparameters, as this would lead to overfitting. Unfortunately, in case of small datasets we often do not have access to a sufficient large validation dataset. In this case we can apply $k$-fold cross validation and use the average validation loss across all folds as metric to optimize. However, at least in the standard form of HPO, this makes the optimization process $k$ times slower. 

We can generalize the definition of HPO in order to deal with multiple objectives $f_0, ... f_k$ at the same time. For example, we might not only interested in optimize the validation performance, but also the cost or the latency of the model. However, this means we will not have a single $\mathbf{x}$ anymore that optimizes all objective functions at the same time.
We can resolve this situation by optimizing one of the objectives, subject to constraints formulated in terms of the others. In our example,
we could minimize validation error, subject to a bound on latency dictated by service level agreements. More ambitiously, we can aim
to sample the Pareto front of such configurations not strictly dominated by any other points.

```{.python .input  n=20}
def multi_objective(config, max_epochs = 10):
    batch_size = config['batch_size']
    lr = config['learning_rate']
    momentum = config['momentum']
    dropout = config['dropout']

    model = AlexNet(lr=lr, momentum=momentum, dropout=dropout)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=0)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
            
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return validation_error, num_params
```

MS: Is this data from the example above? Fine if not, but eventually should be consistent.

![Pareto front of two objectives](img/pareto_front.png)
:width:`400px`
:label:`pareto_front`

### Cost

Another relevant metric is the cost of evaluating $f(\mathbf{x})$ at a configuration $\mathbf{x}$. Different to validation error or prediction latency, this metric is not a function of the final trained model, but a measure of training wall-clock time. For example, if we tune the number of layers or units per layer, larger networks are slower
to train than smaller ones. In our runnning example, training time does not depend on `learning_rate`, `momentum`, `weight_decay`, but varies in general on `batch_size`, due to how GPUs
work. Counting cost in terms of wall-clock time is more relevant in practice than counting the number of evaluations. Some HPO algorithms explicitly model training cost and take it into
account for making decisions.

```{.python .input  n=21}
import time

def objective_function_with_cost(config, max_epochs=10):
    start_time = time.time()
    validation_error = objective(config, max_epochs)
    return validation_error, time.time() - start_time
```

### Constraints

MS: Could be mentioned together with multi-objective.

In many scenarios we are not just interested in finding $\mathbf{x}_{\star}$, but a hyperparameter configuration that additionally full fills certain constraints. More formally, we seek to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$ s.t $c_1(\mathbf{x}) > 0, ..., c_m(\mathbf{x}) > 0$. Typical constraints could be, for example, the memory consumption of $\mathbf{x}$ or fairness constraints.

## Search Spaces
:label:`sec_intro_search_spaces`

Along with the objective function $f(\mathbf{x})$, we also need to define the feasible set $\mathcal{X}$ to optimize over, the *search space* or *configuration space*.
In this chapter, we restrict ourselves to search spaces which decompose as product over the individual hyperparameters.
Here is a possible search space for our running example:

```{.python .input  n=31}
from syne_tune.syne_tune.search_space import loguniform, uniform, randint

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "momentum": uniform(0.0, 0.99),
   "weight_decay": loguniform(1e-9, 1e-2),
   "batch_size": randint(8, 128)
}
```

Each parameter has a data type, such as `float` (for `learning_rate`, `momentum`, `weight_decay`) or `int` (for `batch_size`), as well as a closed bounded range
(lower and upper bounds). Some positive parameters (such as `learning_rate` or `weight_decay`) are best represented on a logarithmic scale
(optimal values can differ by orders of magnitude), while others (like `momentum`) come with linear scale. As suggested by the
naming in Syne Tune, another way to define hyperparameter types is as bounded distributions, typically uniform or loguniform.
Methods driven by random search sample independent values from these distributions for every search decision.

One important data type missing from our running example is `categorical`. For example, we could extend it by
`activation` of type `categorical(['ReLU', 'LeakyReLU', 'Softplus'])`, in order to specify the
non-linear activation functions. This data type is for finite parameters, whose values have no ordering or distance
relations with each other.

It is tempting to try and "simplify" an HPO problem by turning numerical into categorical parameters. For example, why not specify
`batch_size` as `categorical([8, 32, 128])`, i.e. 3 instead of 121 possible values? However, not only does this constitute another "choice by hand" we want to avoid,
for most competitive HPO methods, it either does not matter or makes things worse. Uniform random sampling just as effectively covers a
bounded range than a finite set. As we will see, many model-based HPO methods relax `int` to `float` and use
one-hot encoding for `categorical`, so turning `int` or `float` into `categorical` increases the dimension of the encoding
space. If you still insist on finite ranges for `float` variables, Syne Tune provides the `finrange` and `logfinrange` types.

In general, the structure of the search space $\mathcal{X}$ can be complex and it can be quite different from $\mathbb{R}^d$. Some hyperparameters are integers (like `batch_size`) or
discrete choices (like the hypothetical `activation` above). Beyond the scope of this chapter, some hyperparameters may
depend on the value of others. For example, if we try to tune both the number of layers and widths per layer for a multi-layer perceptron,
the width of the $l$-th layer is relevant only if the network has at least $l+1$ layers. Hence, hyperparameter optimization consists in determining a small set of good hyperparameters to search
over and probing the underlying machine learning algorithm at these values in the hope that 
one of them will be close to the best hyperparameters $\mathbf{x}_*$. 


## How can we evaluate hyperparameter optimization methods?

In the next section we will look at different hyperparameter optimization methods.
Just as with training algorithms or model architectures, it is important to understand how to best
best compare different HPO techniques. First, each HPO experiment depends on substantial sources of randomness,
from the seed for random configuration choices and non-convex optimization of surrogate models to random
effects in each training run (such as random weight initialization). When comparing different methods, it is
therefore crucial to run each experiment several times and present average or quantile statistics.

The key requirement for an HPO method is that it finds well performing configurations as rapidly as possible, given the
allocated compute budget. At least in the current practice, HPO is used as part of an iterative data
acquisition and model building process overseen by human experts. The faster each individual decision is taken,
the faster the overall job is done, or the more high level alternatives can be considered.

Therefore, we will score an HPO algorithm by its anytime performance, defined as the validation performance of the best found
configuration at the current wall-clock time step. Compared to the practice of plotting best validation performance against number of
criterion function evaluations, this metric is not only more relevant in practice, but also allows to compare different
scheduling techniques (e.g., synchronous versus asynchronous; early stopping versus full training). It also captures the decision making
time of the HPO method itself, which for some model-based techniques can be significant.

## Summary

## Exercise
