```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Multi-fidelity Hyperparameter Optimization
:label:`sec_mf_hpo`


Training neural network models can be expensive even for moderate size datasets. For example, training a ResNet-50 on a rather small dataset set, such as CIFAR10 might take 2 hours on a Amazon Elastic Cloud Compute (EC2) g4dn.xlarge instance. Hence, without a massively parallel compute budget,
random search will often be too slow to be of real use in practice for modest to large size neural networks trained on massive data sets.

The question that springs to mind is the following: do we really have to train every hyperparameter configuration for the same amount of time until we know if the configuration is a good one? Indeed, we could just evaluate configurations after a few epochs only,
and stop the worst ones early. In this section, we will develop some competitive
algorithms based on this **early stopping** idea.

## Early Stopping Hyperparameter Configurations

The Figure below depicts learning curves of a set of neural networks with different hyperparameter configurations trained for the same number of epochs. After a few epochs we are already
able to visually distinguish between the well-performing and the poorly performing
ones. However, the ordering is not perfect, and we might still require
the full amount of 100 epochs to identify the best performing configuration.

<!-- ![Learning curves of random hyperparameter configurations](../../img/samples_lc.pdf) -->
![Learning curves of random hyperparameter configurations](img/samples_lc.png)
:width:`400px`
:label:`img_samples_lc`

The idea of early-stopping based HPO methods is to free up compute resources by early stopping
the evaluation of poorly performing configurations and allocate them to
more promising ones. This will eventually speed up the optimization
process, since we have a higher throughput of configurations that we can try.

More formally, we expand our definition in Section :ref:sec_definition_hpo,
such that our objective function $f(\mathbf{x}, r)$ gets an additional input
$r \in [r_{min}, r_{max}]$ that specifies the amount of resource that we are
willing to spend for the evaluation of $\mathbf{x}$. We assume that both the
correlation to $f(\mathbf{x}) = f(\mathbf{x}, r_{max})$ and the computational
cost $c(\mathbf{x}, r)$ increases with $r$ (the latter should be affine linear
in $r$). Typically, $r$ represents the number of epochs for training the neural
network. But also other resources are possible, such as the training dataset
size or the number of cross-validation folds.

## Successive Halving

One of the simplest ways to combine random search with early stopping is
**successive halving** (SH) :cite:`jamieson-aistats16`,`karnin-icml13`. The basic idea is to start with $N$ configurations, for example randomly sampled from the search space, and train each of them for only $r_{min}$ epochs (e.g., $r_{min} = 1$). We then
discard a fraction of the worst performing trials and train the remaining ones
for longer. Iterating this process, less and less trials run for longer and
longer, until at least one trial reaches $r_{max}$ epochs.

More formally, consider a minimum budget $r_{min}$, for example 1 epoch, a maximum budget $r_{max}$, equal to `max_epochs` in our previous example, and a halving constant
$\eta\in\{2, 3, \dots\}$. For simplicity, assume that $r_{max} = r_{min} \eta^K$,
with $K \in \mathbb{I}$ . Moreover, set the initial number of configurations to $N = \eta^K$. Let us
define *rung levels* $\mathcal{R} = \{ r_{min}, r_{min}\eta, r_{min}\eta^2,
\dots, r_{max} \}$. In general, a trial is trained until reaching a rung level,
then evaluated there, and the validation errors of all trials at a rung level
are used to decide which of them to discard.
We start with running $N$ trials
until rung level $r_{min}$. Sorting the validation errors, we keep the top
$1 / \eta$ fraction ($\eta^{K-1}$ configurations) and discard all the rest.
The surviving
trials are trained for $r_{min}\eta$ epochs (next rung level), and the process
is repeated. In each round, a $1 / \eta$ fraction of trials
survives and finds it budget to be multiplied by $\eta$. With this particular
choice of $N$, only a single trial will be trained to the full budget $r_{max}$. Finally, once we finished one round of SH, we continue with a new round with a new set of initial configurations until the total budget is spent.

To implement SH, we use the HPOScheduler base class from the last Section. As we will discuss in Section :ref:`sec_mf_bo`, we can combine SH with Bayesian optimization, because of that, we will use a generic searcher object to sample configurations. Additionally, we assume the minimum resource $r_{min}$, the maximum resource $r_{max}$ and $\eta$ as input. We will also add an optional parameter $s$ that is multiplied to the number of configurations, which we need later to implement Hyperband.

Inside our scheduler we maintain a queue of configurations that need to be evaluated for the current rung level $r_i$. We update the queue every time we jump to the next rung level.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
```

```{.python .input  n=2}
%%tab all
import numpy as np
from collections import defaultdict

class SuccessiveHalvingScheduler(d2l.HPOScheduler):#@save
    def __init__(self, searcher, eta, r_min, r_max, s=1):
        self.save_hyperparameters()
        # only used for Hyperband later
        self.s = s 
        # compute K, which is later used to determine the number of configurations
        self.K = int(np.log(r_max / r_min) / np.log(eta))
        # define the rung levels
        self.rung_levels = [r_min * eta ** k for k in range(self.K + 1)]
        # bookkeeping
        self.observed_error_at_rungs = defaultdict(list)
        # our processing queue
        self.queue = []
```

In the beginning our queue is empty and we fill it with $N = s * \eta^{K}$ configurations, which are first evaluated on the smallest rung level $r_{min}$. The effect of $s$ will become important later if we look at Hyperband, for now let's just assume $s=1$. Now, every time resources become available and the HPOTuner object queries the suggest function, we return an element from the queue. Once we finish one round of SH - which means that we evaluated all surviving configuration on the highest resource level $r_{max}$ and our queue is empty - we start the entire process again with a new set of configurations.

```{.python .input  n=12}
%%tab all
@d2l.add_to_class(SuccessiveHalvingScheduler) #@save
def suggest(self):
    if len(self.queue) == 0:  # our queue is empty, which means we start a new round of SH
        # we track the error of all configuration at the rung levels, 
        # to sort them later for the promotion on the next rung level
        self.observed_error_at_rungs = defaultdict(list)  
        N = int(self.s * self.eta ** self.K)  # the number of configurations for the first rung level
        for i in range(N):
            config = searcher.sample_configuration()
            config['max_epochs'] = self.r_min  # set r = r_min
            self.queue.append(config)
    # return an element from the queue
    c = self.queue.pop()
    return c
```

When we collected a new data point, we first update the searcher module. Afterwards we check if we already collect enough data points $n_i = \eta^{K - i}$ on the current rung level. If so, we sort all configurations based on their performance and estimate the top $\frac{1}{\eta}$ configuration and append them to the queue, which should be empty at this point.

```{.python .input  n=4}
%%tab all
@d2l.add_to_class(SuccessiveHalvingScheduler) #@save
def update(self, config, error, info=None):
    # determine the rung level r_i
    ri = config['max_epochs']
    # bookkeeping
    self.observed_error_at_rungs[ri].append((config, error))
    # update our searcher, e.g if we use Bayesian optimization later
    self.searcher.update(config, error, additional_info=info)     
    # determine how many configurations should be evaluated on this rung level
    ki = self.K - self.rung_levels.index(ri)
    n = int(self.s * self.eta ** ki)
    # if we observed all configuration on this rung level r_i, we estimate the top 1 / eta configuration and 
    # add them to queue and promote them for the next rung level r_i+1
    if len(self.observed_error_at_rungs[ri]) == n and ri < self.r_max:
        best_performing_configurations = self.get_top_n_configurations(ri, n // self.eta)
        for config in best_performing_configurations:
            config['max_epochs'] = ri * self.eta
            self.queue.append(config)
        
@d2l.add_to_class(SuccessiveHalvingScheduler) #@save
def get_top_n_configurations(self, rung_level, n):
    rung = self.observed_error_at_rungs[rung_level]
    if not rung:
        return []
    configs, errors = zip(*rung)
    indices = [np.argsort(errors)[i] for i in range(n)]
    return [configs[i] for i in indices]        
```

Let us see how this is doing on our example.

```{.python .input  n=5}
from scipy import stats

search_space = {
   "learning_rate": stats.loguniform(1e-4, 1),
   "batch_size": stats.randint(8, 128),
} 
```

```{.python .input  n=14}
searcher = d2l.RandomSearcher(search_space)
scheduler = SuccessiveHalvingScheduler(searcher=searcher, eta=2, r_min=1, r_max=16)
tuner = d2l.HPOTuner(scheduler=scheduler, objective=d2l.objective)
tuner.run(number_of_trials=31)
```

We can visualize the learning curves of all configuration that we evaluated. Most of the configurations are stopped early and only the better performing configurations survive until $r_{max}$. Compare this to vanilla random search which would allocate $r_{max}$ to every configuration.

```{.python .input  n=19}
import matplotlib.pyplot as plt
for rung_index, rung in scheduler.observed_error_at_rungs.items():
    errors = [xi[1] for xi in rung]   
    plt.scatter([rung_index] * len(errors), errors)
    plt.xlim(0, 17)
plt.ylabel('validation error')
plt.xlabel('epochs')        
```

## Hyperband

While SH can greatly improve upon random search, the choice of $r_{min}$
can have a large impact on its performance. If $r_{min}$ is too small, our network might have not enough time to learn anything,
and even the best configurations may be filtered out at random. If $r_{min}$ is too large on the other hand,
the benefits of early stopping may be greatly diminished.

Hyperband :cite:`li-iclr17` is an extension of SH that mitigates the risk of setting
$r_{min}$ too small. It runs SH as subroutine, where each round of SH, called a bracket, balances between $r_{min}$ and the number of initial configurations $N$, such that the same total amount of resources is used.

Let's define $s_{max} = \lfloor log_{\eta} \frac{r_{max}}{r_{min}} \rfloor$.
Now for each bracket $s \in \{s_{max}, ..., 0\}$, we call SH with $r_{min} = \eta^{-s} * r_{max}$ and the number of configuration $N = \lceil \frac{s_{max} + 1}{s+1} * \eta^s \rceil$. Note that the last bracket where $s=0$ evaluates all configurations on $r_{min} = r_{max}$, which means that we effectively run random search. In practice we execute brackets in an round robin fashion, which means we start with $s=s_{max}$ again once we finished the loop.

We implement a new scheduler, that maintains a SuccessiveHalvingScheduler object.

```{.python .input  n=8}
%%tab all
import numpy as np
import copy

class HyperbandScheduler(d2l.HPOScheduler): #@save
    def __init__(self, searcher, eta, r_min, r_max):
        self.save_hyperparameters()
        self.s_max = int(np.ceil((np.log(r_max) - np.log(r_min)) / np.log(eta)))
        self.s = self.s_max
        self.successive_halving = SuccessiveHalvingScheduler(
            searcher=self.searcher, eta=self.eta, r_min=self.r_min,
            r_max=self.r_max, s=(self.s_max +1 ) / (self.s + 1))
        self.brackets = defaultdict(list)

    def suggest(self):
        return self.successive_halving.suggest()        
```

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(HyperbandScheduler) #@save
def update(self, config, error, info=None):
    self.brackets[self.s].append((config['max_epochs'], error))
    self.successive_halving.update(config, error, info=info)
    # if the queue of successive halving is empty, than we finished this round and start with
    # a new round with different r_min and N
    if len(self.successive_halving.queue) == 0:
        self.s -= 1
        if self.s < 0:
            self.s = self.s_max
        self.successive_halving = SuccessiveHalvingScheduler(
            searcher=self.searcher, eta=self.eta, r_min=self.r_max * self.eta ** (-self.s),
            r_max=self.r_max, s=(self.s_max +1 ) / (self.s + 1))
```

Let us see how this is doing.

```{.python .input  n=21}
searcher = d2l.RandomSearcher(search_space)
scheduler = HyperbandScheduler(searcher=searcher, eta=2, r_min=1, r_max=16)
tuner = d2l.HPOTuner(scheduler=scheduler, objective=objective_with_resource)
tuner.run(number_of_trials=100)
```

```{.python .input  n=24}
import matplotlib.pyplot as plt

for bi, bracket in scheduler.brackets.items():
    rung_levels = [xi[0] for xi in bracket]
    errors = [xi[1] for xi in bracket]
    plt.scatter(rung_levels, errors)

    plt.xlim(0, 17)
    plt.title(f'bracket s={bi}')
    plt.ylabel('objective function')
    plt.xlabel('epochs')        
    plt.show()

```

## Summary


## Exercises
