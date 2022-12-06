```{.python .input  n=2}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Hyperparameter Optimization API
:label:`sec_api_hpo`


Before we dive into the methodology, we will first discuss a basic code structure that allows us to efficiently implement various HPO algorithms. In general, all HPO
algorithms considered here need to implement two decision making primitives, *searching* and *scheduling*.
First, they need to sample new hyperparameter configurations, which often involves some kind of search over
the configuration space. Second, for each configuration, an HPO algorithm needs to schedule its evaluation
and decide how many resources to allocate for it.
Once we start to evaluate a configuration, we will refer to it as a *trial*. We map these decisions to two classes,
`HPOSearcher` and `HPOScheduler`. On top of that, we also provide a `HPOTuner` class that executes the optimization process.

This concept of scheduler and searcher is also implemented in popular HPO libraries, such as Syne Tune :cite:`salinas-automl22`, Ray Tune :cite:`liaw-arxiv18` or Optuna :cite:`akiba-sigkdd19`.

### Searcher

Below we define a base class for searchers, which provides a new candidate
configuration through the `sample_configuration` function. A simple way to
implement this function would be to sample configurations uniformly at random,
as we did for random search in the previous :numref:`sec_what_is_hpo`.
More sophisticated algorithms, such as Bayesian optimization (:numref:`sec_bo`) will
make these decisions based on the performance of previous trials. As a result,
these algorithms are able to sample more promising candidates over time. We add
the `update` function in order to update the history of previous trials, which can
then be exploited to improve our sampling distribution.

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
from scipy import stats

```

```{.python .input  n=2}
%%tab all

class HPOSearcher(d2l.HyperParameters):  #@save
    def sample_configuration():
        raise NotImplementedError
    def update(self, config, error, additional_info=None):
        pass
```

The following code shows how to implement our random search optimizer from the previous section in this API:

```{.python .input  n=3}
%%tab all

class RandomSearcher(HPOSearcher):  #@save
    def __init__(self, config_space):
        self.save_hyperparameters()

    def sample_configuration(self):
        return {
            name: domain.rvs()
            for name, domain in self.config_space.items()
        }
```

### Scheduler

Beyond sampling configurations for new trials, we also need to decide when and for how long to
run a trial. In practice, all these decisions are
done by the `HPOScheduler`, which delegates the choice of new configurations to a
`HPOSearcher`. The `suggest` method is called whenever some resource for training
becomes available. Apart from invoking `sample_configuration` of a searcher, it
may also decide upon parameters like `max_epochs` (i.e., how long to train the
model for). The `update` method is called whenever a trial returns a new
observation.

```{.python .input  n=4}
%%tab all

class HPOScheduler(d2l.HyperParameters):  #@save
    def suggest(self):
        raise NotImplementedError
    
    def update(self, config, error, info=None):
        raise NotImplementedError
```

To implement random search, but also other HPO algorithms, we only need a basic scheduler that schedules a new configuration every time new resources become available.

```{.python .input  n=5}
%%tab all

class BasicScheduler(HPOScheduler):  #@save
    def __init__(self, searcher):
        self.save_hyperparameters()
        
    def suggest(self):
        return self.searcher.sample_configuration()

    def update(self, config, error, info=None):
        searcher.update(config, error, additional_info=info)
```

### Tuner

Finally, we need a component that runs the scheduler/searcher and does some book-keeping
of the results. The following code implements a sequential execution of the HPO trials that
evaluates one training job after the next and will serve as a basic example. We will later
use *Syne Tune* for more scalable distributed HPO cases.

```{.python .input  n=7}
%%tab pytorch

class HPOTuner(d2l.HyperParameters):  #@save
    def __init__(self, scheduler, objective):
        self.save_hyperparameters()
        
        # Bookeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        self.current_runtime = 0
        
    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
            error = self.objective(**config)
            self.scheduler.update(config, error)
            runtime = time.time() - start_time
            self.bookkeeping(config, d2l.numpy(error.cpu()), runtime)        
```

### Bookkeeping the Performance of HPO Algorithms

With any HPO algorithm, we are mostly interested in the best performing
configuration (called *incumbent*) and its validation error after a given 
wall-clock time. This is why we track `runtime` per iteration, which includes
both the time to run an evaluation (call of `objective`) and the time to
make a decision (call of `scheduler.suggest`). In the sequel, we will plot
`cumulative_runtime` against `incumbent_trajectory` in  order to visualize the
*any-time performance* of the HPO algorithm defined in  terms of `scheduler`
(and `searcher`). This allows us to quantify not only how well the configuration
found by an optimizer works, but also how quickly an optimizer is able to find it.

```{.python .input  n=7}
%%tab pytorch

@d2l.add_to_class(HPOTuner) #@save
def bookkeeping(self, config, error, runtime): 
    # Check if the last hyperparameter configuration performs better 
    # than the incumbent
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
        
    # Add current best observed performance to the optimization trajectory
    self.incumbent_trajectory.append(self.incumbent_error)
    
    # Update runtime
    self.current_runtime += runtime
    self.cumulative_runtime.append(self.current_runtime)
```

### Example: Optimizing the Hyperparameters of a Convolutional Neural Network

We now use our new implementation of random search to optimize the *batch size*
and *learning rate* of a convolutional neural networks from :numref:`sec_alexnet`.
For that, we first have to define the objective function.

```{.python .input  n=6}
%%tab all

def objective(batch_size, learning_rate, max_epochs=8):  #@save
    model = d2l.AlexNet(lr=learning_rate)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
    validation_error = trainer.validate(model=model)
    return validation_error    
```

We also need to define the configuration space.

```{.python .input  n=15}
config_space = {
   "learning_rate": stats.loguniform(1e-4, 1),
   "batch_size": stats.randint(8, 128),
} 
```

Now we can start our random search:

```{.python .input}
searcher = RandomSearcher(config_space)
scheduler = BasicScheduler(searcher=searcher)
tuner = HPOTuner(scheduler=scheduler, objective=objective)

tuner.run(number_of_trials=5)
```

Below we plot the optimization trajectory of the incumbent to get the any-time
performance of random search:

```{.python .input  n=11}
board = d2l.ProgressBoard(xlabel='time', ylabel='error')
for time_stamp, error in zip(tuner.cumulative_runtime, tuner.incumbent_trajectory):
    board.draw(time_stamp, error, 'random search', every_n=1)
```

## Comparing HPO Algorithms

Just as with training algorithms or model architectures, it is important to understand how to best
compare different HPO algorithms. Each HPO run depends on two major sources of randomness:
the random effects of the training process, such as random weight initialization or mini-batch
ordering, and the intrinsic randomness of the HPO algorithm itself, such as the random sampling
of random search. Hence, when comparing different algorithms, it is crucial to run each experiment
several times and report statistics, such as mean or median, across a population of multiple
repetitions of an algorithm based on different seeds of the random number generator.

To illustrate this, we compare random search (see :numref:`sec_rs`) and Bayesian optimization,
which we will introduce in :numref:`sec_bo`, for optimizing the hyperparameters of a
feed-forward neural network. Each algorithm was evaluated $50$ times with a different random
seed. The solid line indicates the average performance of the incumbent across these $50$
repetitions and the dashed line the standard deviation. We can see that random search and
Bayesian optimization perform roughly the same up to ~1000 seconds, but Bayesian optimization
can make use of the past observation to identify better configurations and thus quickly
outperforms random search afterwards.


![Example any-time performance plot to compare two algorithms A and B.](img/example_anytime_performance.svg)
:label:`example_anytime_performance`


## Summary

This section laid out a simple, yet flexible interface to implement various HPO algorithms that we will look at in this chapter. Similar interfaces can be found in popular open-source HPO frameworks.
We also looked at how we can compare HPO algorithms, and potential pitfall one needs to be aware. 


## Exercise
