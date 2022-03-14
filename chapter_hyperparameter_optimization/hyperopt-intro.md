```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "cb2addeb39c842a7bcbad0de2e416207",
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

Hyperparameters play an important role in deep learning, and machine learning in general. Not only do they determine the generalization capabilities of trained models, they can even be critical for what constitutes the state-of-the-art. Indeed, results reported in an empirical study might look very differently for another choice of 
hyperparameters and so would be the conclusions drawn. Unfortunately, it is not uncommon that publications do not report the specific hyperparameters that were used in experiments, for instance, to demonstrate that a proposed method is superior to previously published ones. Such studies are not reproducible, and their impact on the state-of-the-art in machine learning should be questioned :cite:`haibe-kains:2020:transparency`.

Consider the weight parameters of a deep neural network. They are automatically determined during training, e.g. by stochastic gradient descent. The **hyperparameters** of the neural network, however, cannot be learned in this way in general as there training error might not be differentiable with respect to them. Without a different form of automation, the user has to set them manually by trial-and-error, in what amounts to a time-consuming and difficult part of machine learning workflows. We distinguish between hyperparameters that control the learning algorithm (e.g., learning rate, batch size, momentum, optimizer choice) and hyperparameters that define model shape and capacity (e.g., type of activation function, number of units per layer). The choice of hyperparameters will directly affect the performance of our neural network once trained. For example, previous work beat the performance of advanced state-of-the-art machine leanrning models by optimizing the hyperparameter of much simpler ones :cite:`snoek-nips12a`. 

The current need to manually tune the optimization algorithm and the structure of a deep neural network constitutes a significant gap towards the promise of end-to-end learning and artificial intelligence. If we are willing to spend sufficient computational resources, algorithms could be used to configure our learning algorithms. Hyperparameter optimization algorithms are designed to tackle this problem. The main idea is to cast the search for the right hyperparameter configuration as a global optimization problem, to minimize the validation loss

In this chapter, we provide an overview of the basics of hyperparameter optimization and look at several state-of-the-art methods from the literature. As a running example, we will show how to automatically tune hyperparameters of a convolutional neural network. Any successful HPO method needs to provide solutions for two decision-making primitives, **scheduling** and **search**, and we will highlight the most prominent current solutions for either. Scheduling amounts to decisions of how much resources to spend on a hyperparameter configuration, e.g. when to stop, pause, or resume training, while search is about which configurations to evaluate in the first place. A specific focus in this chapter will lie on model-based approaches to search, which in practice are more sample efficient than their random search-based counterparts. Since hyperparameter optimization requires us to train and validate several neural networks, we will also see how we can distribute these methods. To avoid distracting boiler-plate code, we will use the Python framework **Syne Tune**, providing us with an simple interface for distributed hyperparameter optimization. You can install it via:

```{.python .input  n=2}
!pip install syne-tune
```

## Ingredients for Hyperparameter Optimization
:label:`sec_definition_hpo`

The performance of our machine learning algorithm can be seen as a function $f: \mathcal{X} \rightarrow \mathbb{R}$ that maps from our hyperparameter space $\mathbf{x} \in \mathcal{X}$ to the validation performance. For every evaluation of $f(\mathbf{x})$, we have to train and validate our machine learning algorithm, which can take a long time. We will see later how much cheaper approximations can help with the optimization of $f$. Training is stochastic in general (e.g., weights are randomly initialized, mini-batches are randomly sampled), so that our observations will be noisy: $y \sim f(\mathbf{x}) + \epsilon$, where we assume that $\epsilon \sim N(0, \sigma)$.

As a running example, we will train the model from Chapter :numref:`sec_alexnet`
on the FashionMNIST dataset. As we would like to optimize the validation error,
we need to add a method computing this metric.

```{.python .input  n=3}
# %%tab pytorch, mxnet, tensorflow

from d2l import torch as d2l
import torch
from torch import nn

@d2l.add_to_class(d2l.Trainer) #@save
def evaluate(self):
    self.model.eval()
    error = 0
    for batch in self.val_dataloader:
        with torch.no_grad():
            x, y = self.prepare_batch(batch)
            y_hat = self.model(x)
            l = self.model.loss(y_hat, y)
        error += l
        self.val_batch_idx += 1
    return error / self.val_batch_idx
```

We optimize validation error with respect to the hyperparameter configuration `config`,
consisting of `batch_size` and `learning_rate`. For each evaluation, we train our model
for `max_epochs` epochs, then compute and return its validation error:

```{.python .input  n=4}
%%tab pytorch, mxnet, tensorflow

def objective(config, max_epochs=16): #@save
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    model = d2l.AlexNet(lr=learning_rate)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
    validation_error = trainer.evaluate()
    return validation_error    
```

```{.json .output n=4}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

Given our criterion $f(\mathbf{x})$ in terms of `objective(config)`, where $\mathbf{x}$
corresponds to `config`, we would like to find $\mathbf{x}_{\star} \in argmin_{\mathbf{x} \in \mathcal{X}} f(\mathbf{x})$. Since $f$ is the validation performance after training, there is no efficient way to compute gradients with respect to $\mathbf{x}$. While there is recent work :cite:`maclaurin-icml15`,`franceschi-icml17a` to drive HPO by approximate "hypergradients", none of the existing approaches are competitive with the state-of-the-art yet, and we will not discuss them here.


### Search Spaces
:label:`sec_intro_search_spaces`

Along with the objective function $f(\mathbf{x})$, we also need to define the feasible set $\mathcal{X}$ to optimize over, the *search space* or *configuration space*.
Here is a possible search space for our running example:

```{.python .input  n=1}
from syne_tune.search_space import loguniform, uniform, randint

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128)
} 
```

Each parameter has a data type, such as `float` (for `learning_rate`) or `int` (for `batch_size`), as well as a closed bounded range
(lower and upper bounds). Some positive parameters (such as `learning_rate`) are best represented on a logarithmic scale
(optimal values can differ by orders of magnitude), while others (such as  `batch size`) come with linear scale.
Another way to define hyperparameter types is as bounded distributions, typically uniform or loguniform.
Methods driven by random search sample independent values from these distributions for every search decision.

One data type missing from our running example is `categorical`. For example, we could extend it by
`"activation": choice(['ReLU', 'LeakyReLU', 'Softplus'])`, in order to specify the
non-linear activation functions. This data type is for finite parameters, whose values have no ordering or distance
relations with each other.

It is tempting to try and "simplify" an HPO problem by turning numerical into categorical parameters. For example, why not specify
`batch_size` as `choice([8, 32, 64, 128])`, i.e. 4 instead of 121 possible values? However, not only does this constitute another "choice by hand" we want to avoid,
for most competitive HPO methods, it either does not matter or makes things worse. Uniform random sampling just as effectively covers a
bounded range than a finite set. As we will see, many model-based HPO methods relax `int` to `float` and use
one-hot encoding for `categorical`, so turning `int` or `float` into `categorical` increases the dimension of the encoding
space.

In general, the structure of the search space $\mathcal{X}$ can be complex and it can be quite different from $\mathbb{R}^d$. Some hyperparameters are integers (like `batch_size`) or
discrete choices (like the hypothetical `activation` above). Beyond the scope of this chapter, some hyperparameters may
depend on the value of others. For example, if we try to tune both the number of layers and widths per layer for a multi-layer perceptron,
the width of the $l$-th layer is relevant only if the network has at least $l+1$ layers. Hence, hyperparameter optimization consists in determining a small set of good hyperparameters to search
over and probing the underlying machine learning algorithm at these values in the hope that 
one of them will be close to the best hyperparameters $\mathbf{x}_*$. 

## An API for Hyperparameter Optimization
:label:`sec_api_hpo`

Before we dive into details, let us get the basic code structure in place. All HPO
methods considered here need to implement two decision making primitives. First, they
need to sample new configurations to be trained and evaluated, given that resources
are available, which often involves some kind of search over the configuration space.
Once a configuration is marked for execution, we will refer to it as a
**trial**. Second, they need to schedule trials, which means deciding for how long to
run them, or when to stop, pause or resume them. We map these to two classes,
`Searcher` and `Scheduler`.

### Searcher

Below we define a base class for searchers, which provides a new candidate
configuration through the `sample_configuration` method. Many algorithms will make
these decisions based on observed performances of previously run trials. Such an
observation can be passed via the `update` method.

```{.python .input  n=6}
%%tab pytorch, mxnet, tensorflow

class Searcher(d2l.HyperParameters): #@save
    def sample_configuration():
        raise NotImplementedError
    
    def update(self, config, error, additional_info=None):
        pass
        
```

```{.json .output n=6}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

We will mostly be interested in asynchronous methods in this chapter, for which
`searcher.sample_configuration` is called whenever some resource for training
becomes available, and the `searcher.update` callback is invoked whenever an
evaluation produces a new metric value.

### Scheduler

*MS: This API does not support stopping a trial as result of a call to `update`.
I see this is not needed to implement synchronous SH and HB, so it's probably
OK. But it runs a bit contrary to saying that scheduling is about stopping a
trial. If we want that, we'd have to allow `update` to return a flag for
(continue, stop).
AK: We would only need this for ASHA stopping not for ASHA promotion right?
If so I would leave it, to avoid making it overly complicated.
MS: OK, we should not overdo this "from scratch" anyway.*

Beyond sampling configurations for new trials, we also need to decide how long to
run a trial for, or whether to stop it early. In practice, all these decisions are
done by the `Scheduler`, who delegates the choice of new configurations to a
`Searcher`. The `suggest` method is called whenever some resource for training
becomes available. Apart from invoking `sample_configuration` of a searcher, it
may also decide upon parameters like `max_epochs` (i.e., how long to train the
model for). The `update` method is called whenever a trial produces a new
metric value.

```{.python .input  n=9}
%%tab pytorch, mxnet, tensorflow

class Scheduler(d2l.HyperParameters): #@save
    def suggest(self):
        raise NotImplementedError
    
    def update(self, config, error, info=None):
        raise NotImplementedError
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

Below we define a basic first-in first-out scheduler which simply schedules the next configuration once resources become available.

*AK: I know we use FIFOScheduler in SyneTune but I am not sure if it's the
right name, since we do not have a fifo queue.
MS: That name is from Ray Tune.*

```{.python .input  n=11}
%%tab pytorch, mxnet, tensorflow

class FIFOScheduler(d2l.Scheduler): #@save
    def __init__(self, searcher):
        self.save_hyperparameters()
        
    def suggest(self):
        return self.searcher.sample_configuration()

    def update(self, config, error, info=None):
        searcher.update(config, error, additional_info=info)
```

### Tuner

Finally, we need a component running the optimizer and doing some book-keeping
of the results. The following code implements a sequential execution of the HPO process (one training
job after the next) and will serve as a basic example. We will later use **Syne Tune** for more sophisticated distributed HPO cases.

```{.python .input  n=12}
%%tab pytorch, mxnet, tensorflow

class Tuner(d2l.HyperParameters): #@save
    def __init__(self, scheduler, objective):
        self.save_hyperparameters()
        
        # for bookeeping
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_time = 0
        
    def run(self, max_wallclock_time):
        while self.current_time < max_wallclock_time:
            start_time = time.time()
            config = self.scheduler.suggest()
        
            error = self.objective(config)
        
            self.scheduler.update(config, error)
        
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)        
```

```{.json .output n=12}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

With any HPO method, we are mostly interested in the best performing
configuration (called **incumbent**) and its error after a given 
wall-clock time. This is why we track `runtime` per iteration, which includes
both the time to run an evaluation (call of `self.objective`) and the time to
make a decision (call of `self.scheduler.suggest`). In the sequel, we will plot
`cumulative_runtime` against `incumbent_trajectory` in  order to visualize the
**any-time performance** of the HPO method defined in  terms of `scheduler`
(and `searcher`).

```{.python .input  n=1}
%%tab pytorch, mxnet, tensorflow

@d2l.add_to_class(Tuner) #@save
def bookkeeping(self, config, error, runtime): 
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
        
    self.incumbent_trajectory.append(self.incumbent_error)
    
    self.current_time += runtime
    self.cumulative_runtime.append(self.current_time)
```

Just as with training algorithms or model architectures, it is important to understand how to best
compare different HPO techniques. Each run of an HPO experiment depends on substantial sources of randomness,
from the seed for random configuration choices and non-convex optimization of surrogate models to random
effects in each training run (such as random weight initialization or mini-batch ordering). When comparing 
different methods, it is  therefore crucial to run each experiment several times and present average or
quantile statistics.


## Summary

## Exercise
