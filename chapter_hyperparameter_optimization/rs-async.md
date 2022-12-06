```{.python .input  n=17}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Asynchronous Random Search

:label:`sec_rs_async`

As we have seen in the previous :numref:`sec_api_hpo`, we might have to wait hours or even days before random search returns a good hyperparameter configuration, because of the expensive evaluation of hyperparameter configurations. In practice, we have often access to a pool of resources such as multiple GPUs on the same machine or multiple machines with a single GPU. This begs the question: *How do we  efficiently distribute random search?*

In general, we distinguish between synchronous and asynchronous parallel hyperparameter
optimization (see :numref:`distributed_scheduling`). In the synchronous setting,
we wait for all concurrently running trials to finish, before we start the next
batch. Consider search spaces that contain hyperparameters such as the number of filters
or number of layers of a deep neural network. Hyperparameter configurations that contain
a larger number of layers of filters will naturally take more time to finish, and all
other trials in the same batch will have to wait at synchronisation points (grey area
in :numref:`distributed_scheduling`) before we can continue the optimization process.

In the asynchronous setting we immediately schedule a new trial as soon as resources
become available. This will optimally exploit our resources, since we can avoid any
synchronisation overhead. For random search, each new hyperparameter configuration
is chosen independently of all others, and in particular without exploiting
observations from any prior evaluation. This means we can trivially parallelize random
search asynchronously. This is not straight-forward with more sophisticated methods
that make decision based on previous observations (see :numref:`sec_sh_async`).
While we need access to more resources than in the sequential setting, asynchronous
random search exhibits a linear speed-up, in that a certain performance is reached
$K$ times faster if $K$ trials can be run in parallel. 


![Distributing the hyperparameter optimization process either synchronously or asynchronously. Compared to the sequential setting, we can reduce the overall wall-clock time while keep the total compute constant. Synchronous scheduling might lead to idling workers in the case of stragglers.](img/distributed_scheduling.svg)
:label:`distributed_scheduling.svg`

In this notebook, we will look at asynchronous random search that, where trials are executed in multiple python processes on the same machine. Distributed job scheduling
and execution is difficult to implement from scratch. We will use *Syne Tune*
:cite:`salinas-automl22`, which provides us with a simple interface for asynchronous
HPO. Syne Tune is designed to be run with different execution back-ends, and the
interested reader is invited to study its simple APIs in order to learn more about
distributed HPO. Syne Tune can be installed via:

## Objective Function

First, we have to define a new objective function such that it now returns the performance back
to Syne Tune via the `report` function.

```{.python .input  n=34}
import matplotlib.pyplot as plt

from syne_tune.config_space import loguniform, randint
from syne_tune.backend.python_backend import PythonBackend
from syne_tune.optimizer.baselines import RandomSearch
from syne_tune import Tuner, StoppingCriterion
from syne_tune.experiments import load_experiment
from syne_tune.report import Reporter


def objective(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l    
    from syne_tune import Reporter
    model = d2l.AlexNet(lr=learning_rate)
    trainer = d2l.Trainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    report = Reporter() 
    for epoch in range(1, max_epochs + 1):
        if epoch == 1:
            trainer.fit(model=model, data=data)  # Initialize the state of Trainer
        else:
            trainer.fit_epoch()
        validation_error = d2l.numpy(trainer.validate().cpu())
        report(epoch=epoch, validation_error=float(validation_error))
```

## Asynchronous Scheduler

First, we define the number of workers that evaluate trials concurrently. We also need to specify
how long we want to run random search, by defining an upper limit on the total wall-clock time.

```{.python .input  n=37}
n_workers = 2  # Needs to be lower equal to the number of available GPUs
max_wallclock_time = 12 * 60
```

Next, we state which metric we want to optimize and whether we want to minimize or
maximize this metric. Namely, `metric` needs to correspond to the argument name
passed to the `report` callback.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

We use the configuration space from our previous example. In Syne Tune, this
dictionary can also be used to pass constant attributes to the training script.
We make use of this feature in order to pass `max_epochs`.

```{.python .input  n=39}
config_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128),
   "max_epochs": 4,
}
```

Next, we need to specify the back-end for job executions. Here we just consider the distribution on a local machine where parallel jobs are executed as sub-processes. However, for large scale HPO, we could run this also on a cluster or cloud environment, where each trial consumes a full instance.

```{.python .input  n=40}
trial_backend = PythonBackend(tune_function=objective, config_space=config_space)
```

We can now create the scheduler for asynchronous random search, which is similar in
behaviour to our `BasicScheduler` from the previous Section.

```{.python .input  n=41}
scheduler = RandomSearch(config_space, metric=metric, mode=mode,)
```

Syne Tune also features a `Tuner`, where the main experiment loop and bookkeeping is
centralized, and interactions between scheduler and back-end are mediated.

```{.python .input  n=42}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(trial_backend=trial_backend, scheduler=scheduler, 
              stop_criterion=stop_criterion, n_workers=n_workers)
```

Let us run our distributed HPO experiment. According to our stopping criterion,
it will run for about 15 minutes.

```{.python .input  n=43}
tuner.run()
```

The logs of all evaluated hyperparameter configurations are stored for further
analysis. At any time during the tuning job, we can easily get the results
obtained so far and plot the incumbent trajectory.

```{.python .input  n=46}
tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Visualize the Asynchronous Optimization Process

Below we visualize how the learning curves of every trial evolve during the
asynchronous optimization process. At any point in time, there are as many trials
running concurrently as we have workers. Once a trial finishes, we immediately
start the next trial, without waiting for the other trials to finish. Idle time
of workers is reduced to a minimum with asynchronous scheduling.

```{.python .input  n=45}
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results['trial_id'] == trial_id]
    plt.plot(df['st_tuner_time'], df['validation_error'], marker='o', label=f'trial {trial_id}')
    
plt.xlabel('wall-clock time')
plt.ylabel('objective function')
plt.legend()
```

## Summary

We can reduce the waiting time for random search substantially by distribution trials across parallel resources. In general, we distinguish between synchronous scheduling and asynchronous scheduling. Synchronous scheduling means that we sample a new batch of hyperparameter configurations once the previous batch finished. If we have a stragglers - trials that takes more time to finish than other trials - our workers need to wait at synchronization points. Asynchronous scheduling evaluates a new hyperparameter configurations as soon as resources become available, and, hence, ensures that all workers are busy at any point in time. While random search is easy to distribute asynchronously and does not require any  change of the actual algorithm, other methods require some additional modifications.

## Exercise
