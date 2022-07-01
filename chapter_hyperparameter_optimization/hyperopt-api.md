```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "8a3ac93a6904412a99c512e005e5aa3a",
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

# An API for Hyperparameter Optimization
:label:`sec_api_hpo`



Before we dive into different techniques, we will first discuss a basic code structure that allows us to efficiently implement various HPO algorithms. In general, all HPO
algorithms considered here need to implement two decision making primitives, **searching** and **scheduling**. First, they need to sample new hyperparameter configurations, which often involves some kind of search over the configuration space. Additional, for each configuration an HPO algorithm needs to schedule its evaluation and how much resources should be allocated to it.
Once we started to evaluate a configurations, we will refer to it as a **trial**. We map these decisions to two classes,
`HPOSearcher` and `HPOScheduler`.

This concept of scheduler and searcher is also implemented in popular HPO libraries, such as Syne Tune, Ray Tune or Optuna.

### Searcher

Below we define a base class for searchers, which provides a new candidate
configuration through the `sample_configuration` function. A trivial way to to implement this function, would be to sample configuraiton uniformly at random, as we did for Random Search in the previous Section :ref:`sec_what_is_hpo`. More sophisticated algorithms, such as Bayesian optimization :ref:`sec_bo` will make
these decisions based on observed performances of previously trials to sample more promising candidates over time. To update the history of previous trials, such that we can exploit these observation to adjust our sample distribution, we add the `update` function.

```{.python .input  n=2}
%%tab all

from d2l import torch as d2l


class HPOSearcher(d2l.HyperParameters): #@save
    def sample_configuration():
        raise NotImplementedError
    def update(self, config, error, additional_info=None):
        pass
```

The following code shows how to implement our random search optimizer from the previous section in this API:

```{.python .input  n=3}
%%tab pytorch, mxnet, tensorflow

from d2l import torch as d2l

class RandomSearcher(d2l.HPOSearcher): #@save
    def __init__(self, search_space):
        self.save_hyperparameters()

    def sample_configuration(self):
        return {
            name: domain.rvs()()
            for name, domain in self.search_space.items()
        }
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "Ignored to run as it is not marked as a \"None\" cell."
 }
]
```

### Scheduler

Beyond sampling configurations for new trials, we also need to decide how long to
run a trial for, or whether to stop it early. In practice, all these decisions are
done by the `HPOScheduler`, who delegates the choice of new configurations to a
`HPOSearcher`. The `suggest` method is called whenever some resource for training
becomes available. Apart from invoking `sample_configuration` of a searcher, it
may also decide upon parameters like `max_epochs` (i.e., how long to train the
model for). The `update` method is called whenever a trial produces a new
metric value.

```{.python .input  n=4}
%%tab all

class HPOScheduler(d2l.HyperParameters): #@save
    def suggest(self):
        raise NotImplementedError
    
    def update(self, config, error, info=None):
        raise NotImplementedError
```

To implement Random Search, but also other HPO algorithms, we only need a basic scheduler that schedules a new configuration every time new resources become available.

```{.python .input  n=5}
%%tab all

class BasicScheduler(HPOScheduler): #@save
    def __init__(self, searcher):
        self.save_hyperparameters()
        
    def suggest(self):
        return self.searcher.sample_configuration()

    def update(self, config, error, info=None):
        searcher.update(config, error, additional_info=info)
```

### Tuner

Finally, we need a component that runs the scheduler / searcher and does some book-keeping
of the results. The following code implements a sequential execution of the HPO trials that evaluates one training job after the next and will serve as a basic example. We will later use **Syne Tune** for more sophisticated distributed HPO cases.

```{.python .input  n=6}
%%tab all

class HPOTuner(d2l.HyperParameters): #@save
    def __init__(self, scheduler, objective):
        self.save_hyperparameters()
        
        # bookeeping results for plotting
        self.incumbent = None
        self.incumbent_error = None
        self.incumbent_trajectory = []
        self.cumulative_runtime = []
        
    def run(self, number_of_trials):
        for i in range(number_of_trials):
            start_time = time.time()
            config = self.scheduler.suggest()
        
            error = self.objective(config)
        
            self.scheduler.update(config, error)
        
            runtime = time.time() - start_time
            self.bookkeeping(config, error, runtime)        
```

### Bookkeeping the Performance of HPO Algorithms

With any HPO algorithm, we are mostly interested in the best performing
configuration (called **incumbent**) and its validation error after a given 
wall-clock time. This is why we track `runtime` per iteration, which includes
both the time to run an evaluation (call of `objective`) and the time to
make a decision (call of `scheduler.suggest`). In the sequel, we will plot
`cumulative_runtime` against `incumbent_trajectory` in  order to visualize the
**any-time performance** of the HPO algorithm defined in  terms of `scheduler`
(and `searcher`). This allows us to quantify not only how well the configuration
found by an optimizer works, but also how quickly an optimizer is able to find it.

```{.python .input  n=7}
%%tab all

@d2l.add_to_class(HPOTuner) #@save
def bookkeeping(self, config, error, runtime): 
    # check if the last hyperparameter configuration performs better 
    # than the incumbent
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
        
    # add current best observed performance to the optimization trajectory
    self.incumbent_trajectory.append(self.incumbent_error)
    
    # update runtime
    self.current_time += runtime
    self.cumulative_runtime.append(self.current_time)
```

```{.python .input  n=8}
from d2l import torch as d2l

from scipy import stats

search_space = {
   "learning_rate": stats.loguniform(1e-4, 1)
} 

searcher = d2l.RandomSearcher(search_space)
scheduler = BasicScheduler(searcher=searcher)
tuner = d2l.HPOTuner(scheduler=scheduler, objective=d2l.hpo_objective_softmax_classification)

tuner.run(max_wallclock_time=600)
```

```{.json .output n=8}
[
 {
  "ename": "AttributeError",
  "evalue": "module 'd2l.torch' has no attribute 'hpo_objective_softmax_classification'",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mAttributeError\u001b[0m                            Traceback (most recent call last)",
   "\u001b[0;32m/var/folders/ld/vzcn3j2d7yg493b1c6m0ypprdqgxkm/T/ipykernel_46053/3783712813.py\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[1;32m      9\u001b[0m \u001b[0msearcher\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mRandomSearcher\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msearch_space\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     10\u001b[0m \u001b[0mscheduler\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mBasicScheduler\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msearcher\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0msearcher\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m---> 11\u001b[0;31m \u001b[0mtuner\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mHPOTuner\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mscheduler\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0mscheduler\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobjective\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0md2l\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mhpo_objective_softmax_classification\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m     12\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m     13\u001b[0m \u001b[0mtuner\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mrun\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mmax_wallclock_time\u001b[0m\u001b[0;34m=\u001b[0m\u001b[0;36m600\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
   "\u001b[0;31mAttributeError\u001b[0m: module 'd2l.torch' has no attribute 'hpo_objective_softmax_classification'"
  ]
 }
]
```

Now we can plot the optimization trajectory of the incumbent to get the any-time
performance of random search:

```{.python .input}
board = d2l.ProgressBoard(xlabel='time', ylabel='error')
for time_stamp, error in zip(tuner.cumulative_runtime, tuner.incumbent_trajectory):
    board.draw(time_stamp, error, 'random search', every_n=1)
```

## Comparing HPO Algorithms

Just as with training algorithms or model architectures, it is important to understand how to best
compare different HPO algorithms. Each HPO run depends on mostly two sources of randomness:
the random effects of the training process, such as random weight initialization or mini-batch ordering, and, the intrinsic randomness of the HPO algorithm itself, e.g the random sampling of random search. Hence, when comparing different algorithms, it is crucial to run each experiment several times and report statistics, such as mean or median, across a population of multiple repetitions of an algorithm based on different seeds of the random number generator.

To illustrate this, we compare random search (see :ref:'sec_rs') and Bayesian optimization, which we will discuss in Section :ref:'sec_bo', for optimizing the hyperparameters of a feed forward neural network. Each algorithm was evaluated $50$ times with a different random seed. The solid line indicates the average performance of the incumbent across these $50$ repetitions and the dashed line the standard deviation. We can see that random search and Bayesian optimization perform roughly the same up to ~1000 seconds, but Bayesian optimization quickly outperforms random search afterwards.


![Example any-time performance plot to compare two algorithms A and B](img/example_anytime_performance.png)
:width:`400px`
:label:`example_anytime_performance`


## Summary

## Exercise

```{.python .input}

```
