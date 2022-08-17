```{.python .input  n=17}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Distributed Random Search

:label:`sec_rs_async`


For random search, each new configuration is chosen independently of all others, and
in particular without exploiting observations from any prior evaluation. This means we can
easily parallelize random search by running trials concurrently, either by using multiple GPUs on the same machine, or across
multiple machines. Random search exhibits a linear speed-up, in that a certain
performance is reached $K$ times faster if $K$ trials can be run in parallel. Also,
there is no need to synchronize trials. Instead we can immediately sample a new configuration once an evaluation finished, without waiting on pending configurations. This is called asynchronous scheduling.

Instead of implementing the complex machinery of distributed job execution, we will use **Syne Tune** which provides us with a simple interface for asynchronous HPO.

You can install it via:

```{.python .input}
!pip install syne-tune
```

## Objective Function

First, we have to modify our objective function slightly and return the performance back to Syne Tune via the `report(...)` function.

```{.python .input  n=34}
def objective(learning_rate, batch_size, max_epochs):
    from d2l import torch as d2l    
    from syne_tune import Reporter
    model = d2l.AlexNet(lr=learning_rate)
    trainer = d2l.Trainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    report = Reporter() 
    for epoch in range(1, max_epochs + 1):
        trainer.fit(model=model, data=data)
        validation_error = trainer.validate(model=model).cpu().numpy()
        report(epoch=epoch, validation_error=float(validation_error))
```

## Asynchronous Scheduler

First, we define the number of workers that evaluate trials concurrently. We also need to specify how long we want to run Random Search, by defining a limited on the total wall-clock time.

```{.python .input  n=37}
n_workers = 4 # We have to set this number equal to the number of GPUs that are in the machine to run this notebook
max_wallclock_time = 900
```

Next, we define specify the path to our training script, which metric we want to optimize and whether we want to minimize of maximize this metric. Namely, `metric` needs
to correspond to the argument name passed to the `report` callback.

```{.python .input  n=38}
mode = "min"
metric = "validation_error"
```

We also use the same
the search space from our previous example. Note that in **Syne Tune**, the search space
dictionary can also be used to pass constant attributes to the training script.
We make use of this feature in order to pass `max_epochs`.

```{.python .input  n=39}
from syne_tune.config_space import loguniform, randint

config_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128),
   "max_epochs": 4,
}
```

Next, we need to specify the back-end for job executions. The simplest choice in Syne
Tune is the local back-end, which runs on the given instance and executes parallel jobs
as sub-processes. For larger workloads, Syne Tune also provides a SageMaker back-end,
which can execute trials on separate instances, but this feature will not be used here.

```{.python .input  n=40}
from syne_tune.backend.python_backend import PythonBackend

backend = PythonBackend(tune_function=objective, config_space=config_space)
```

Now we define how we want to schedule hyperparameter configurations. For random search,
we will use the `FIFOScheduler`, which is similar in behaviour to our *BasicScheduler* from the previous Section.

```{.python .input  n=41}
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

scheduler = FIFOScheduler(
    config_space,
    searcher='random',
    metric=metric,
    mode=mode)
```

Syne Tune also features a `Tuner`, where the main experiment loop and book-keeping is
centralized, and interactions between scheduler and back-end are mediated.

```{.python .input  n=42}
from syne_tune import Tuner, StoppingCriterion

stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    trial_backend=backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
)
```

We can now
run our distributed HPO experiment.

```{.python .input  n=43}
tuner.run()
```

The logs of all evaluated hyperparameter configuratons are stored for further
analysis. At any time during the tuning job, we can easily get the results
obtained so far and plot the incumbent trajectory.

```{.python .input  n=46}
from syne_tune.experiments import load_experiment

tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Visualize the Optimization Process

Below we visualize how the learning curves of every trial evolves during the optimization process. At any point in time there are as many trials running concurrently as we have workers. Once a trial finishes, we immediately start the next trial without waiting for the other trials to finish in order to synchronize workers, which would cause idling times for workers.

```{.python .input  n=45}
import matplotlib.pyplot as plt
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results['trial_id'] == trial_id]
    plt.plot(df['st_tuner_time'], df['validation_error'], marker='o', label=f'trial {trial_id}')
    
plt.xlabel('wall-clock time')
plt.ylabel('objective function')
plt.legend()
```

## Summary

## Excercise
