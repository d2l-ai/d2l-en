## How can we evaluate hyperparameter optimization methods?

MS: This should be placed where the first evaluation is actually done, and it should
be an extra small chapter (just after random search?).

In the next sections we will analyse and compare different hyperparameter optimization methods. All optimzers that we will look at are iterative algorithms, where in each iteration we first sample a new hyperparameter configuration, evaluate the objective function with this configuration and then update all internal states.
The following module is a generic wrapper that, provided with a scheduler and a searcher, helps us to run our optimization algorithm. 

```{.python .input  n=13}
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
```

```{.python .input  n=19}
%%tab pytorch, mxnet, tensorflow

@d2l.add_to_class(Tuner) #@save
def run(self, num_iterations):
    for i in range(num_iterations):
        start_time = time.time()
        config = self.scheduler.suggest()
        
        error = self.objective(config)
        
        self.scheduler.update(config, error)
        
        runtime = time.time() - start_time
        
        self.bookkeeping(config, error, runtime)
```

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

Below we add a function to our `Tuner` that keeps track of the optimization process by book keeping the incumbent performance and the runtime, which allows us to later compare different optimizers.

```{.python .input  n=17}
%%tab pytorch, mxnet, tensorflow

@d2l.add_to_class(Tuner) #@save
def bookkeeping(self, config, error, runtime): 
    if self.incumbent is None or self.incumbent_error > error:
        self.incumbent = config
        self.incumbent_error = error
        
    self.incumbent_trajectory.append(self.incumbent_error)
    
    self.current_time =+ runtime
    self.cumulative_runtime.append(self.current_time)
```
