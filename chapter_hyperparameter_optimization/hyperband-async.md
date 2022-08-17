```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Asynchronous Successive Halving


As we have seen in Section :ref:`sec_rs_async`, we can accelerate the HPO process by distributing the evaluation of hyperparameter configurations across either multiple instances or multiples CPUs / GPUs on a single instance.
However, compared to random search, it is not straight forward to run SH asynchronously in a distributed setting.
Before we can decide which configuration to run next, we first have to collect all observations on the current rung level. This leads to synchronization points at each rung level.
For example, for the lowest rung level $r_{min}$, we first have to evaluate all $N = \eta^K$ configurations, 
before we can promote the $\frac{1}{\eta}$ of them to the next rung level.

If every trial would consume the same amount of wall-clock time, synchronization wouldn't be a problem, since all results come in at the same time. However, in practice, we often have a high variations in training time across hyperparameter configurations. For example, assuming the number of filter per layer as a hyperparameter, networks with smaller filter sizes require less time to train for a fix amount of epochs than networks with larger filter sizes. Especially in the case of stragglers, this might lead to large idling times of workers. 

Asynchronous successive halving (ASHA) :cite:`li-arxiv18` adapts SH to the asynchronous parallel scenario. The main idea of ASHA is to promote configurations to the next rung level as soon as we collected at least $\eta$ observations on the current rung level. While this potential leads to incorrect promotions, i.e configurations are promoted to the next rung level, that would have been promoted if we evaluated all configurations on the current rung level, in removes any synchronization points. If a resources become free and we have less than $\eta$ configurations, we start a new configuration on the first rung level $r_0$. 

As for asynchronous random search, we will use **Syne Tune** once more, to use asynchronous scheduling.

## Objective Function

As in Section :ref:`sec_rs_async`, we use the following objective function.

```{.python .input  n=54}
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

We also use the same search space as before

```{.python .input  n=55}
from syne_tune.config_space import randint, loguniform

max_epochs = 4

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "batch_size": randint(8, 128),
   "max_epochs": max_epochs,
}
```

```{.python .input}
n_workers = 4 # We have to set this number equal to the number of GPUs that are in the machine to run this notebook
max_wallclock_time = 900
```

## Asynchronous Scheduler

With our new objective function in place, the code for running ASHA is a simple
variation of what we did for asynchronous random search.

```{.python .input  n=56}
from syne_tune.optimizer.baselines import ASHA

scheduler = ASHA(
    search_space,
    metric="validation_error",
    resource_attr="epoch",
    max_t=max_epochs,
    mode="min",
    type='promotion',
    grace_period=1,  # this corresponds to r_min 
    reduction_factor=2  # this corresponds to eta
    max_resource_attr='max_epochs'
)  
```

Here, `metric` and `resource_attr` specify the key names used with the `report`
callback, and `max_resource_attr` denotes which input to the objective function corresponds to $r_i$.
Moreover, `grace_period` provides $r_{min}$, and `reduction_factor` is $\eta$.

Now, we can run Syne Tune as before:

```{.python .input  n=57}
import logging

logging.basicConfig(level=logging.INFO)

from syne_tune import Tuner, StoppingCriterion
from syne_tune.backend.python_backend import PythonBackend

trial_backend = PythonBackend(tune_function=objective, config_space=search_space)

stop_criterion = StoppingCriterion(
    max_wallclock_time=max_wallclock_time
)
tuner = Tuner(
    trial_backend=trial_backend,
    scheduler=scheduler,
    stop_criterion=stop_criterion,
    n_workers=num_workers,
)
tuner.run()
```

After the experiment has finished, we can retrieve and plot results.

```{.python .input  n=59}
from syne_tune.experiments import load_experiment
e = load_experiment(tuner.name)
e.plot()
```

## Visualize the Optimization Process

We visualize again the learning curves of every trial. Compares this to asynchronous random search in Section :ref:`sec_rs_async`. As we have seen for successive halving in Section :ref:`sec_mf_hpo`, most of the trials are stopped at 1 or 2 epochs, i.e $r_{min}$ or $\eta * r_{min}$. However, trials do not stop at the same point, because they require different amount of time per epoch. If we would run standard successive halving instead of ASHA, we would need to synchronize our workers, before we can promote configurations to the next rung level. 

```{.python .input  n=60}
import matplotlib.pyplot as plt

results = e.results
for trial_id in results.trial_id.unique():
    df = results[results['trial_id'] == trial_id]
    plt.plot(df['st_tuner_time'], df['validation_error'], marker='o', label=f'trial {trial_id}')
plt.xlabel('wall-clock time')
plt.ylabel('objective function')
plt.legend()
```

## Summary

## Excercise
