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
in particular without using metric values from any prior evaluations. We might as
well run trials in parallel, using multiple GPUs on the same instance, or even
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

@d2l.add_to_class(d2l.Trainer) #@save
def evaluate(self):
    self.model.eval()
    error = 0
    for batch in self.val_dataloader:
        with torch.no_grad():
            x, y = self.prepare_batch(batch)
            y_hat = self.model(x)
            l = self.model.loss(y_hat, y)
        error += l
        self.val_batch_idx += 1
    return error / self.val_batch_idx

def objective(config, max_epochs = 10): #@save
    batch_size = config['batch_size']
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    model = d2l.AlexNet(lr=learning_rate, momentum=momentum)
    trainer = d2l.Trainer(max_epochs=max_epochs, num_gpus=0)
    data = d2l.FashionMNIST(batch_size=batch_size, resize=(224, 224))
    trainer.fit(model=model, data=data)
    validation_error = trainer.evaluate()
    return validation_error    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--learning_rate', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--batch_size', type=float)

    args, _ = parser.parse_known_args()
    report = Reporter()

    accuracy = objective(vars(args))
    report(accuracy=accuracy)
```

MS: We suddenly have momentum here. If so, we should also have it above

## Asynchronous Scheduler

First, we define the number of workers that evaluate trials concurrently. Since
we will use the local back-end of Syne Tune, we need to make sure to choose an
instance on which (at least) this number of training processes can run in
parallel. We also need to select a stopping criterion for our experiment and point
to our training script.

```{.python .input  n=6}
n_workers = 4
max_wallclock_time = 600
entry_point = "train_script.py"
mode = "max"
metric = "accuracy"

search_space = {
   "learning_rate": loguniform(1e-5, 1e-1),
   "momentum" : ???,
   "batch_size": randint(8, 128)
}
```

In this example, we use 4 workers, allow the experiment to run for 600 seconds (or
10 minutes), and specify that `accuracy` should be maximized. Namely, `metric` needs
to correspond to the argument name passed to the `report` callback.

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

The logs of all evaluated hyperparameter configuratons are stored for further analysis. At any time during the tuning job, we can easily get the results obtained so far and plotting the incumbent trajectory

```{.python .input}
from syne_tune.experiments import load_experiment

tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

## Summary

## Excercise
