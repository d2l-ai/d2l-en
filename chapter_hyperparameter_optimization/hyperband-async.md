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

# Asynchronous Successive Halving

Our basic implementations of successive halving (SH) and Hyperband have a number of
shortcomings, especially when it comes to tuning neural network models. First,
as in random search above, we would like to use parallel training, making
efficient use of multiple cores or multiple GPUs. In other words, training jobs
should be executed in parallel, and asynchronously.

Second, and more important, the algorithms developed above also synchronize
their decision making. Each rung has an a priori fixed number of slots.
For example, the lowest rung in SH, at $r_{min}$, has size $N = \eta^K$. Now,
*all* these slots have to be populated with trials trained for $r_{min}$ epochs,
before *any* of them can continue towards the next rung. Do we really need to
evaluate all $N$ trials before we can identify the best and the worst ones?

At least for tuning of neural networks, synchronous SH and Hyperband can often be
improved dramatically by adopting asynchronous decision making. This is done
in *asynchronous successive halving (ASHA)*. TODO: REFERENCE? Roughly speaking,
ASHA employs the same rung levels and the same rule to decide between stop and
continue than synchronous SH, but the decision is made whenever a trial reaches
a rung level, *based on the data available until then*.

Our basic `Tuner` implementation neither caters for distributed scheduling, nor
for stopping, pausing, or resuming trials. In this section, we will use
**Syne Tune** once more.

## Prepare Training Script

First, we will need to change our training script. The variant of ASHA we are
interested in here, implements scheduling by early stopping. Namely, a trial
evaluation is started with a fixed `max_epochs` parameter, but the scheduler
may terminate the training job early. To this end, we cannot report the
validation error at the end, but need to do that after every epoch. This needs
just a small modification of what we developed above, to be stored as
`train_script_report_eachepoch.py`.

```{.python .input}
from argparse import ArgumentParser

from d2l import torch as d2l
import torch
from torch import nn

from syne_tune.report import Reporter

def objective(config):
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    max_epochs = config['max_epochs']
    model = d2l.AlexNet(lr=learning_rate)
    trainer = d2l.Trainer(max_epochs=1, num_gpus=1)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    report = Reporter()

    for epoch in range(1, max_epochs + 1):
        trainer.fit(model=model, data=data)
        validation_error = trainer.evaluate()
        report(epoch=epoch, validation_error=validation_error)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--max_epochs', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=16)

    args, _ = parser.parse_known_args()

    objective(vars(args))
```

Note how this script reports a validation metric after every epoch, passing
both the value and the epoch number to the `report` callback.

## Asynchronous Scheduler

With our new training script in place, the code for running ASHA is a simple
variation of what we did for asynchronous random search above.

```{.python .input  n=6}
from syne_tune.backend.local_backend import LocalBackend

n_workers = 4
max_wallclock_time = 600
max_epochs = 16

entry_point = "train_script_report_eachepoch.py"
mode = "min"
metric = "validation_error"
resource_attr = "epoch"
max_resource_attr = "max_epochs"

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128),
   max_resource_attr: max_epochs,
}

backend = LocalBackend(entry_point=entry_point)
```

At this point, we choose the `HyperbandScheduler` provided by **Syne Tune**, which
implements several variants of ASHA and an extension to multiple brackets called
asynchronous Hyperband (note that **Syne Tune** also implements synchronous
Hyperband as `SynchronousHyperbandScheduler`, but we will not use this here).

```{.python .input  n=4}
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler

scheduler = HyperbandScheduler(
    search_space,
    searcher='random',
    metric=metric,
    mode=mode,
    resource_attr=resource_attr,
    max_resource_attr=max_resource_attr,
    grace_period=1,
    reduction_factor=2)
```

Here, `metric` and `resource_attr` specify the key names used with the `report`
callback, and `max_resource_attr` allows to infer $r_{max}$ from `search_space`.
Moreover, `grace_period` provides $r_{min}$, and `reduction_factor` is $\eta$.
The remainder is the same as for asynchronous random search.

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

After the experiment has finished, we can retrieve and plot results.

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
