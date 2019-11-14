# Learning Rate Scheduler
:label:`sec_scheduler`

Until now, we have discussed a number of advanced optimization algorithms to train deep learning models, such as "SGD", "Momentum", and "Adam". As we shown in :numref:`sec_momentum`, a learning rate of $0.4$ will lead to a convergence of both $x_1$ and $x_2$, while a learning rate of $0.6$ will result in a divergence of $x_2$. So is there a way to "schedule" the learning rate to decrease regularly rather than being constant all the time?

In this section, we will explore how to set up a *learning rate scheduler*

```{.python .input}
%matplotlib inline
import d2l
from matplotlib import pyplot as plt
from IPython import display
from mxnet import np, npx, lr_scheduler, optimizer
npx.set_np()
```

The `plot` function will visualize several commonly used schedulers. We first define a function to plot the learning rate for the first $10$ epochs.

```{.python .input}
## for plotting th scheduler
def plot(scheduler, num_epochs=10):
    epochs = [i+1 for i in range(num_epochs)]
    lrs = [scheduler(i) for i in epochs]
    display.set_matplotlib_formats('svg')
    plt.scatter(epochs, lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.show()
```

## The Schedulers

### Factor Scheduler

A scheduler returns a learning rate for a given epoch count, which starts at base_lr.

```{.python .input}
class FactorSchedulerScratch(object):
    """
        base_lr * pow(factor, floor(num_update/step))
    """
    def __init__(self, step, factor=1, stop_factor_lr=1e-8, base_lr=0.01):
        self.step = step
        self.base_lr = base_lr
        self.factor = factor
        self.stop_factor_lr = stop_factor_lr
        self.count = 0

    def __call__(self, num_update):
        while num_update > self.count + self.step:
            self.count += self.step
            self.base_lr *= self.factor
            if self.base_lr < self.stop_factor_lr:
                self.base_lr = self.stop_factor_lr
        return self.base_lr
```

In the following example, we create a scheduler that returns the initial learning rate $0.1$, then then halve its value for every $2$ epochs.

```{.python .input}
factor_scheduler_scratch= FactorSchedulerScratch(base_lr=0.1, step=2, factor=0.5)
plot(factor_scheduler_scratch)
```

We can also directly import the built-in class `FactorScheduler` from MXNet. If we are using the same parameter settings, as you can see from the plot below, we will obtain the same learning rate scheduler.

```{.python .input}
factor_scheduler = lr_scheduler.FactorScheduler(base_lr=0.1, step=2, factor=0.5)
plot(factor_scheduler)
```

### MultiFactorScheduler

We can define non-uniform intervals with `MultiFactorSchedulerScratch`.

```{.python .input}
class MultiFactorSchedulerScratch(object):
    """
       base_lr * pow(factor, k+1)
    """
    def __init__(self, step, factor=1, base_lr=0.01):
        self.step = step
        self.base_lr = base_lr
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        while self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= self.factor
            else:
                return self.base_lr
        return self.base_lr
```

In the example below we halve the learning rate after the 2nd, 4th, and 6th epochs. As before, the learning rate of the 2nd epoch will be 1 and the 3rd epoch will be 0.5.

```{.python .input}
multi_factor_scheduler_scratch = MultiFactorSchedulerScratch(base_lr=0.1, step=[2, 4, 6], factor=0.5)
plot(multi_factor_scheduler_scratch)
```

We can also directly import the built-in class `MultiFactorScheduler` from MXNet. If we are using the same parameter settings, as you can see from the plot below, we will obtain the same learning rate scheduler.

```{.python .input}
multi_factor_scheduler = lr_scheduler.MultiFactorScheduler(base_lr=0.1, step=[2, 4, 6], factor=0.5)
plot(multi_factor_scheduler)
```

### Polynomial Scheduler

`PolyScheduler` gives a smooth decay using a polynomial function and reaches a learning rate of final_lr after max_update iterations.

```{.python .input}
class PolySchedulerScratch(object):
    """ 
    Reduce the learning rate according to a polynomial of given power.
   
       final_lr + (start_lr - final_lr) * (1-nup/max_nup)^pwr
       if nup < max_nup, 0 otherwise.
    """
    
    def __init__(self, max_update, base_lr=0.01, pwr=2, final_lr=0):
        self.power = pwr
        self.base_lr = base_lr
        self.base_lr_orig = self.base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.max_steps = self.max_update

    def __call__(self, num_update):
        if num_update <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                pow(1 - float(num_update) / float(self.max_steps), self.power)
        return self.base_lr
```

In the example below, we have a quadratic function (`pwr`$=100$) that falls from $0.1$ to $0.01$ at epoch $5$. After this the learning rate stays at $0.01$.

```{.python .input}
poly_scheduler_scratch = PolySchedulerScratch(base_lr=0.1, final_lr=0.01, max_update=100, pwr=100)
plot(poly_scheduler_scratch)
```

`PolyScheduler` is a built-in function in MXNet, which we can also directly import from `lr_scheduler`. As we use see, it will return the same plot as before.

```{.python .input}
poly_scheduler_mx = lr_scheduler.PolyScheduler(base_lr=0.1, final_lr=0.01, max_update=100, pwr=100)
plot(poly_scheduler_mx)
```

## Applications

Continuously from previously sections, we call the same function `train_gluon_ch10` and see how the schedulers make the difference.


Before we test the learning rate schedulers, we set up a baseline for comparison.

```{.python .input}
data_iter, feature_dim = d2l.get_data_ch10(batch_size=10)
d2l.train_gluon_ch10('sgd',  {'learning_rate': 0.1}, data_iter, feature_dim)
```

Let us try the `poly_scheduler_mx` and `multi_factor_scheduler` as we defined above.

```{.python .input}
d2l.train_gluon_ch10('sgd', {'lr_scheduler': poly_scheduler_mx}, data_iter, feature_dim)
```

```{.python .input}
d2l.train_gluon_ch10('sgd', {'lr_scheduler': multi_factor_scheduler}, data_iter, feature_dim)
```

```{.python .input}

```
