```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=1}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "43027478a42d4f15867fffc50c113612",
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

# Distributed Random Search

For random search, each new configuration is chosen independently of all others, and
in particular without exploiting observations from any prior evaluation. This means we can
easily parallelize random search by running trials concurrently, either by using multiple GPUs on the same machine, or across
multiple machines. Random search exhibits a linear speed-up, in that a certain
performance is reached $K$ times faster if $K$ trials can be run in parallel. Also,
there is no need to synchronize trials. Instead we can immediately sample a new configuration once an evaluation finished, without waiting on pending configurations. This is called asynchronous scheduling.

Instead of implementing the complex machinery of distributed job execution, we will use **Syne Tune** which provides us with a simple interface for asynchronous HPO.

You can install it via:

```
pip install syne-tune

```


## Prepare Training Script

Syne Tune requires the code for training and evaluation as a python script,
where the hyperparameters are passed as input arguments, and the validation metric
is reported back via a callback function `report`. Here is our example from the previous Section, which we
store as `train_script.py`.

```{.python .input}
from argparse import ArgumentParser

from d2l import torch as d2l
import torch
from torch import nn

from syne_tune.report import Reporter

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)

    args, _ = parser.parse_known_args()
    report = Reporter()

    config = vars(args)
    validation_error = d2l.objective(config)
    report(validation_error=validation_error)
```

## Asynchronous Scheduler

First, we define the number of workers that evaluate trials concurrently. We also need to specify how long we want to run Random Search, by defining a limited on the total wall-clock time.

```{.python .input  n=6}
n_workers = 4
max_wallclock_time = 600
```

Next, we define specify the path to our training script, which metric we want to optimize and whether we want to minimize of maximize this metric. Namely, `metric` needs
to correspond to the argument name passed to the `report` callback.

```{.python .input  n=6}
entry_point = "train_script.py"
mode = "min"
metric = "validation_error"
```

We also use the same
the search space from our previous example. Note that in **Syne Tune**, the search space
dictionary can also be used to pass constant attributes to the training script.
We make use of this feature in order to pass `max_epochs`.

```{.python .input  n=1}
from syne_tune.search_space import loguniform, randint

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128),
   "max_epochs": 16,
}
```

Next, we need to specify the back-end for job executions. The simplest choice in Syne
Tune is the local back-end, which runs on the given instance and executes parallel jobs
as sub-processes. For larger workloads, Syne Tune also provides a SageMaker back-end,
which can execute trials on separate instances, but this feature will not be used here.

```{.python .input  n=7}
from syne_tune.backend.local_backend import LocalBackend

backend = LocalBackend(entry_point=entry_point)
```

Now we define how we want to schedule hyperparameter configurations. For random search,
we will use the `FIFOScheduler`, which is similar in behaviour to our *BasicScheduler* from the previous Section.

```{.python .input  n=4}
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

scheduler = FIFOScheduler(
    search_space,
    searcher='random',
    metric=metric,
    mode=mode)
```

Syne Tune also features a `Tuner`, where the main experiment loop and book-keeping is
centralized, and interactions between scheduler and back-end are mediated. 

```{.python .input  n=4}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    backend=backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
)
```

We can now
run our distributed HPO experiment.

```{.python .input  n=4}
tuner.run()
```

The logs of all evaluated hyperparameter configuratons are stored for further
analysis. At any time during the tuning job, we can easily get the results
obtained so far and plot the incumbent trajectory.

```{.python .input}
from syne_tune.experiments import load_experiment

tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Summary

## Excercise
