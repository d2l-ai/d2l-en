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

# Asynchronously Parallel Random Search

In random search, each new configuration is chosen independent of all others, and
in particular without exploiting observations from any prior evaluation. We can trivally parallize random search by running trials in parallel, either by using multiple GPUs on the same instance, or even
multiple instances. Random search exhibits a linear speed-up, in that a certain
performance is reached K times faster if K trials can be run in parallel. Also,
there is no need to synchronize job executions: random search is best run
asynchronously distributed.

Unfortunately, our basic `Tuner` implementation does not cater for distributed
scheduling, and we will use **Syne Tune** for that.

## Prepare Training Script

Syne Tune requires the code for training and evaluation to be given as a script,
where the hyperparameters are passed as input arguments, and the validation metric
is reported via a callback function. Here is our example from above, which we
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

First, we define the number of workers that evaluate trials concurrently. Since
we will use the local back-end of Syne Tune, we need to make sure to choose an
instance on which (at least) this number of training processes can run in
parallel. We also need to select a stopping criterion for our experiment and point
to our training script.

```{.python .input  n=6}
n_workers = 4
max_wallclock_time = 600
max_epochs = 16

entry_point = "train_script.py"
mode = "min"
metric = "validation_error"

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128),
   "max_epochs": max_epochs,
}
```

In this example, we use 4 workers, allow the experiment to run for 600 seconds (or
10 minutes), and specify that `validation_error` should be minimized. Namely, `metric` needs
to correspond to the argument name passed to the `report` callback. We also reuse
the search space from our example. Note that in **Syne Tune**, the search space
dictionary can also be used to pass constant attributes to the training script.
We make use of this feature in order to pass `max_epochs`.

Next, we need to specify the back-end for job executions. The simplest choice in Syne
Tune is the local back-end, which runs on the given instance and executes parallel jobs
as sub-processes. For larger workloads, Syne Tune also provides a SageMaker back-end,
which can execute trials on separate instances, but this feature will not be used here.

```{.python .input  n=7}
from syne_tune.backend.local_backend import LocalBackend

backend = LocalBackend(entry_point=entry_point)
```

Next we define how we want to schedule hyperparameter configurations. For random search,
we will use the `FIFOScheduler`, which is similar in behaviour to our code above.

```{.python .input  n=4}
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler

scheduler = FIFOScheduler(
    search_space,
    searcher='random',
    metric=metric,
    mode=mode)
```

Syne Tune also features a `Tuner`, where the main experiment loop and book-keeping is
centralized, and interactions between scheduler and back-end are mediated. We can now
run our distributed HPO experiment.

```{.python .input  n=4}
stop_criterion = StoppingCriterion(max_wallclock_time=max_wallclock_time)

tuner = Tuner(
    backend=backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=n_workers,
)

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

MS: Following our recommendation above, should we not present results averaged
over many runs? We could say these are easy to obtain with Syne Tune, without
going into details.

## Summary

## Excercise
