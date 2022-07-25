```{.python .input  n=17}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

```{.json .output n=17}
[
 {
  "data": {
   "application/vnd.jupyter.widget-view+json": {
    "model_id": "4e3e744ae2da4c4991719cf242eb04fa",
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

:label:`sec_rs_async`


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
    import numpy as np
    validation_error = np.e ** - max_epochs
    for epoch in range(1, max_epochs + 1):
#         trainer.fit(model=model, data=data)
#         validation_error = trainer.evaluate()
        validation_error = np.e ** - max_epochs
        report(epoch=epoch, validation_error=validation_error)
```

## Asynchronous Scheduler

First, we define the number of workers that evaluate trials concurrently. We also need to specify how long we want to run Random Search, by defining a limited on the total wall-clock time.

```{.python .input  n=37}
n_workers = 4
max_wallclock_time = 60
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
   "max_epochs": 16,
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

```{.json .output n=43}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "--------------------\nResource summary (last result is reported):\n trial_id     status  iter  learning_rate  batch_size  max_epochs  epoch  validation_error  worker-time\n        0  Completed    16       0.001000          68          16   16.0      1.125352e-07     0.001662\n        1  Completed    16       0.000432          55          16   16.0      1.125352e-07     0.002148\n        2  Completed    16       0.029438          58          16   16.0      1.125352e-07     0.001898\n        3  Completed    16       0.000033          23          16   16.0      1.125352e-07     0.002259\n        4  Completed    16       0.009267         104          16   16.0      1.125352e-07     0.000809\n        5  Completed    16       0.000028          92          16   16.0      1.125352e-07     0.001141\n        6  Completed    16       0.018922         102          16   16.0      1.125352e-07     0.000776\n        7  Completed    16       0.000841          49          16   16.0      1.125352e-07     0.000851\n        8  Completed    16       0.000106          19          16   16.0      1.125352e-07     0.000978\n        9  Completed    16       0.000095          55          16   16.0      1.125352e-07     0.001085\n       10  Completed    16       0.000075          30          16   16.0      1.125352e-07     0.001275\n       11  Completed    16       0.000202          66          16   16.0      1.125352e-07     0.000873\n       12  Completed    16       0.042135         117          16   16.0      1.125352e-07     0.004768\n       13  Completed    16       0.000076         122          16   16.0      1.125352e-07     0.001552\n       14  Completed    16       0.001623          14          16   16.0      1.125352e-07     0.001701\n       15  Completed    16       0.009344          92          16   16.0      1.125352e-07     0.000820\n       16  Completed    16       0.000087          92          16   16.0      1.125352e-07     0.000848\n       17  Completed    16       0.000053          56          16   16.0      1.125352e-07     0.001234\n       18  Completed    16       0.002184          44          16   16.0      1.125352e-07     0.000862\n       19  Completed    16       0.000068          79          16   16.0      1.125352e-07     0.000852\n       20  Completed    16       0.078976         102          16   16.0      1.125352e-07     0.000929\n       21  Completed    16       0.051054         110          16   16.0      1.125352e-07     0.000914\n       22  Completed    16       0.002295          24          16   16.0      1.125352e-07     0.000983\n       23  Completed    16       0.020732         125          16   16.0      1.125352e-07     0.001019\n       24  Completed    16       0.006241          60          16   16.0      1.125352e-07     0.000919\n       25  Completed    16       0.000116          60          16   16.0      1.125352e-07     0.000969\n       26  Completed    16       0.052987          15          16   16.0      1.125352e-07     0.000814\n       27  Completed    16       0.000627          63          16   16.0      1.125352e-07     0.000929\n       28  Completed    16       0.000013         123          16   16.0      1.125352e-07     0.000844\n       29  Completed    16       0.001884          18          16   16.0      1.125352e-07     0.000648\n       30  Completed    16       0.000034          83          16   16.0      1.125352e-07     0.001345\n       31  Completed    16       0.000026          92          16   16.0      1.125352e-07     0.000866\n       32  Completed    16       0.000151          36          16   16.0      1.125352e-07     0.000953\n       33  Completed    16       0.000111          54          16   16.0      1.125352e-07     0.000875\n       34  Completed    16       0.036009          74          16   16.0      1.125352e-07     0.002312\n       35  Completed    16       0.000022          48          16   16.0      1.125352e-07     0.000833\n       36 InProgress     0       0.053670          85          16      -                 -            -\n       37 InProgress     0       0.001895          40          16      -                 -            -\n       38 InProgress     0       0.000127          14          16      -                 -            -\n       39 InProgress     0       0.008380          45          16      -                 -            -\n4 trials running, 36 finished (36 until the end), 60.81s wallclock-time\n\nvalidation_error: best 1.1253517471925921e-07 for trial-id 0\n--------------------\n"
 }
]
```

The logs of all evaluated hyperparameter configuratons are stored for further
analysis. At any time during the tuning job, we can easily get the results
obtained so far and plot the incumbent trajectory.

```{.python .input  n=46}
from syne_tune.experiments import load_experiment

tuning_experiment = load_experiment(tuner.name)
tuning_experiment.plot()
```

```{.json .output n=46}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "WARNING:matplotlib.legend:No artists with labels found to put in legend.  Note that artists whose label start with an underscore are ignored when legend() is called with no argument.\n"
 },
 {
  "data": {
   "image/png": "iVBORw0KGgoAAAANSUhEUgAAAaEAAAEWCAYAAADPZygPAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjUuMCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8/fFQqAAAACXBIWXMAAAsTAAALEwEAmpwYAAAjZElEQVR4nO3deZgdVZ3/8fcHEggQwpYgS4hBQTBqiBhHEPgZ0VFwUHBURgYUEI3bCIjoyMjPII674zKjIw+jMSAYWRRFR9wwkcWwJKyBuIBsDYQ0CRC2YCDf+eOcayqXe/tWd7q6On0/r+e5T9+qU8u3TlfVt5ZzqxQRmJmZ1WGjugMwM7Pu5SRkZma1cRIyM7PaOAmZmVltnITMzKw2TkJmZlabDT4JSTpG0hV1x7EhkDRJ0mOSNq47llYkhaTd6o5jOJN0pKRf1R2H2WDZoJKQpMl5RzWq7lgGStKdkl5bx7wj4u6IGBsRz3Qatuq6ljRf0rurmHaV6l4HI+LciHhdmWHLHKBJ2kfSryWtkNQr6QJJOxbKJekLkpbnzxckKZe9QNJP8ngrJP1S0h6FcY+WtEjSSkk9kr7Yqd4kfVjS0jzObEmb5v6NA6jiJyR9pM10tpc0V9J9kh6RdKWkVxTKd5R0cS4PSZM7xFVqeEnb5vroVO/bSrpI0uOS7pL0z22Gm93p4EzSi3PdPyjpWT/8lHSOpPtznf6p03bXV2x5ffiEpLvz9H4gaVyh/HBJv5f0hKT5fc2nYYNKQt1gQ06wlmxg/8NtgDOBycBzgUeB7xbKZwKHAXsBU4E3Au/NZVsDFwN7AM8BrgF+Uhh3c+BEYDzwCuA1wMntApH0euDjebjnAs8DPgXrHECNjYixwEuANcAP20xuLHAt8DJgW+As4H8ljc3la4BfAG9pF0+TssN/AVhSYnrfBP5KqrcjgW9JelFxAEn7A88vMa3VwPnAcW3KPwdMjohxwJuAf5f0sgHG9k7gHcB+wE7AZsB/FcZdAXwN+HyJuJOIGPIPcCdwCnAr8BBppR+TyxYDbywMOxp4EHgpcDcQwGP5sy9wDHAF8OU8rTuAgwvj70TaUFYAtwHvKZSdRvrnnU3a+G4BppeIfyfSyt+b53d8mWkC3yOtzE/m+D9G2viDtALdDVwG/C/woaZ53gS8OX8P4HjgL7luvgRslMs2Ak4F7gKW5Ti2ymWNeY3K3fOBTwNX5lh/BYzPZc+q6xb1cBpwIXBeHv86YK9c9lHgh03D/yfwdeAzwDPAqjztbxSW633An4GHSRuD+rFcR+e4HwQ+0eF/uA/w+zyfG4EZhbJ+1QtpHbwS+CqwHPgsaX17SWGa2wNPABOAGUAP8G851juBIwvDbpWXrzcv76mF/+8xwBWFYVvWGfDCXL/P5DgfLrlt7g08Wuj+PTCz0H0ccFWbcbfN8WzXpvwk4Kd9zPv7wGcL3a8BlrYZdhYwr5/7nZXAy5r6jcoxTy45jbbDA68EFgDHFv9HLYbbgrSTf0Gh3/eAzzfN53pS4g9gtxKx7QZEh2H2AO4HDh9IbKTt/aNNy7wK2LxpOu8G5peq0/78Ewfrkze6xcAuecW9Evj3XPYx4LzCsIcCN+fvkynsRHO/Y0hHAu8BNgbeD9zH2p3XZcB/A2OAaaQN+8BcdlquwDfkcT/XbgMrzG8jYBHwSWAT0tHaX4DXl5lmXvbXFroby3R2XgE2Aw4Hri4Msxdp57ZJ7g5gXq67ScCfgHfnsneRku3zSEeDPwK+16r+SDvb24EX5PnOL6xsz6rrFnVxWq77t5IOFk4mJeXRwI7A48DWhY1qGXknkOf17qbpBfAz0hH2pPy/Oqgfy/U/eTn2Ap4CXtgm7p1zfb4h/z//PndPGEi9kNbBp4EP5eXcjLTOfaEwzAnkHTApCT0NfAXYFHhVrqs9cvnZpDOKLfP8/gQcV5hXcxJqV2frDFty2zyRddfXR4BXFLqnU0hSTeMeBtzfx7R/TGFH26L8RuCfCt3jaZHUSEn2duCYfizXNNJ2uVVT/0FJQqRt/TrSmVef9U46oH6iqd/JFBI06SDu64X/8Xolobw+PpGndR0wdiCxkZLQxwpl++Vp7tU0zgaRhN5X6H4DcHv+vhPp6HNc80LTfgdwW6F78zzMg8AfSEeCWxbKPwfMyd9PA35TKJsCPNkU66uBGwqfp4DepmFOAb5bZpq0T0LPK/QbQzqr2z13fxn470J5kHc0ufsDwKX5+6XABwple5ASxajm+iPtXE9tms4v2tV1i//jaay7w9qIdJR1QO6+hHzmCRwC3FoYdj6tk9D+he7zgY/3Y7kmFsqvAd7eJu5/JSewQr9fAkcPpF5I6+DdTdN7BemsqXEwtJB89MnaJLRF07L+f9LO7K/AlELZe8kbNK2TULs6W2fYEtvlVNIZ3AGFfs8Aexa6d8/zVNO4E4F7gSPaTPtdpLO/8X3M/3bWXa9H03qHfwDp7K7ljrTFdMcBNwOntCgbrCT0YeBbZeo9x7+0qd97Cv/jXUgHXFsV/sfrfSaU1639SWfWowcY27tJB0WTSWfsF+f49m0ap3QSqvOe0D2F73eRkg8RcR/pzOgtkrYGDgbO7TCtpY0vEfFE/vou0kq8IiIebZrXzq3GJR0pjCle04+IeRExLSKmAQeSdnzbSHq48SFdVnlO2Wm28bf6iIhVpEtcR0naCDiCdErccngK9Zf/3tVUNqopvqLmWMe2Ga5M3GtIO5pGLGcBR+XvR/HsZehPPGWWq+W4TTezJ5HuN7yt6X+4P+nsrVMc7RT/H0TE1Xm8GZL2JO0gLi4M8lBEPN60PDuRjv5Ht1jW4jrbrFSszTf3m8p2Ix00nBARlxeKHiPtxBvGAY9F3tPkcSeQLln+d0TMbTHfw0gHfwdHxIO535GFWC7pY16QDkqLjiZd6v3bMki6pTC9Awr9NwN+SjpY+lyremkR7wGFad1SYvidSJfHP9Gm/JLC9I5ssZzk7sZyfg04PSIeaTGtVvVWSkQ8ExFXkA4Y3j/A2GYDc0kHareQrshA2u4HpM4bqLsUvk8iXUJrOIuUSUcBCyLi3tw/KO8RUsLYVtKWpGvy3yRdqtko7xj6662kBPm8iNh9AOND+2Vo7n8Waad9Ben0eEFT+S6klQDWrb/7SDtZCmVPAw+QVr71jbPZ3/6POWFOLMTyY9JNzReTzoQ+NoDpNwx4uSLdyP4bSfeQzoTe088YoPz/D9Ym4aXAhfngomEbSVsUEtEk0iXqB0nr7XNJ90wbZffSf+vEFBF30yJBSXou8Bvg0xHRfKBwC2mbuSZ378Xa9Q5J25AS0MUR8ZkW0z6IdJn0HyLi5kIs5/Lsg8vGvM4vzOuBiFhemN5mwNuANzct2zo39fOwm5LWwR7WNqboKCfh/hyM/R3pAOZWpYaDmwGbSVoK7BwRBzfFtQUwStLuEfHn3LtYr68B9pf0xcJoCySd0Kbe+msUucFDf2PLB5qz8gdJryOtmwNZP4F6W8d9UNJESduSjiDOK5T9mHSD9ATS9fGGXtKN/eeVnMfTpBurnwO+DXyDtMyfJl0j7a+3kxLZo5L+VdJmkjbOTSRfXnIaD1Ai/px01gD/QesziI9K2kbSLqR6atTfXODDknbNLYE+S7rH9nTJ+BrK1vXLJP1jPtM7kXS58qq8DKtIl1O/D1yTd4INpeqhYLCWC+Ac4I2SXp//f2MkzZBUJkn3Zx08h7SzPIp11+OGT0naJB+5HwJcEKn5/PnAZyRtmRPESXla/fUAMFHSJu0GkLQz8FtS45AzWgxyNnCSpJ3zEf9HgDl53HGky5hXRsTHW0z7QNIO8y0RcU1zeZt5HSdpSr4KcmpjXgVvJl2qnkcfJI0mrXtPki6zrmkxzBjSPTmATXN3X9NsN/wlpMtT0/Lnk6RGBdOixc8h8oHHj4DTJW0haT/Sve/Gdv4C0o6/MT1IrRIvahOXciybNOLU2qbt20t6u6SxeV1/PenKyqWtptUpNqXm28/P85xCuq95eqN+G9sTKdFtlGMZ3bJCCzMd8g/rto57mHTE2Ny64tukm7Vjm/qfTtoRPExq4XQMTddfSUeAryIdWU4krSRB2kHeQ7q3s4R0T+N3ebjFwB/zcIuBXzZNc8c839GkyyZzSUe4D5F2uq/Nw50GnFMYbzLr3oc5lHSv4GHSDb91ypvmeSpN94sKy9doHbeclKg2zmUbkTaCe3K85wDbtIllPoX7Ms112VzXLeI7jXVbx10P7N00zP55nsc29d+XdG35IeA/C8u1W2GYOaxtsFJ6uVotW4vYX5H/9yvy9P4XmDSQemkub5rPb0jruwr9ZpCOzj9BOvO5G3hHoXybvHy9eXk/Sd+t49rV2SZ5uVYAD7aJbxbrtvZ7jHS5rVEu4It5Givy98Z9rqPzuI83jd+ox3mkA8Fi2SUd9g0nkZLnSlKr2U2byn9JOmPrtI95VY7tiab5H9BUd+t8Okyz1PB9rQ+FYbYlHWw/nv///9xhvm3vCbF2/S9+7sxlE0jr+cO5Tm+m0EK4v7GREuQfc73eBZzUYtmbY5nT1/waK9OQknQnaSP/TR/DfJLUTPCodsN0mMdk4GcR8eJ8xPbHiNixw2h9Te8E4EURMXOg0xjAPN9Jah67f1P/IDVauG2oYmlF0mmkjaPt/0jpHswfgB0iYuVQxTZcSJoN3BcRpxb6zSAdqPTn8qjZiDQsf6yaL9EdR/oR3XrLO787JL0tT1+S9urnZI4gnf0MCUmbk1plDUod1CHfIzoJ+EGXJqDJwD8C36k5FLNha9glIUnvIV2CuCQiLhvgNOaSfjS2h9LjQo4j/fL3OEk3km6yHdrH+M2PCHmcdPPxp/nIvlL5um0v6bLE96ueXxXyDc6VpN/gzKo5nCEn6dOky7pfiog76o7HbLiq5XKcmZkZDMMzITMz6x4b0oMWOxo/fnxMnjy57jDMzDYoixYtejAiJtQx7xGVhCZPnszChQvrDsPMbIMi6a7OQ1XDl+PMzKw2TkJmZlYbJyEzM6vNiLonZGbW7VavXk1PTw+rVq16VtmYMWOYOHEio0f3/Ti3oeQkZGY2gvT09LDlllsyefJk8lO9gfSc0OXLl9PT08Ouu+5aY4Tr8uU4M7MRZNWqVWy33XbrJCAASWy33XYtz5Dq5CRkZjbCNCegTv3r5CRkZma1qTQJSZotaZmkxW3K95S0QNJTkk5uKvuw0it7F0ua2+mFU2ZmtuGp+kxoDnBQH+UrSC9n+3KxZ37b4/HA9Ih4MbAx6a2mZmbWQbsHUw/HB1ZXmoTyqxhW9FG+LCKuBVa3KB5Fek/7KGBz4L5qojQzGznGjBnD8uXLn5VwGq3jxowZXheVhmUT7Yi4V9KXSa+WfRL4VUT8qtWwkmYCMwEmTar8VT9mZsPaxIkT6enpobe391lljd8JDSfDMglJ2ob00rldSe9Gv0DSURFxTvOwEXEm+e2j06dPH37nmmZmQ2j06NHD6ndAnQzX1nGvBe6IiN6IWA38CHhlzTGZmdkgG65J6G5gH0mbKzVsfw2wpOaYzMxskFV6OU7SXGAGMF5SDzALGA0QEWdI2gFYCIwD1kg6EZgSEVdLuhC4DngauJ58yc3MzEaOSpNQRBzRoXwp0PIuWUTMIiUtMzMboYbr5TgzM+sCTkJmZlYbJyEzM6uNk5CZmdXGScjMzGrjJGRmZrVxEjIzs9o4CZmZWW2chMzMrDZOQmZmVhsnITMzq42TkJmZ1cZJyMzMauMkZGZmtXESMjOz2jgJmZlZbZyEzMysNk5CZmZWGychMzOrjZOQmZnVxknIzMxq4yRkZma1cRIyM7PaOAmZmVltnITMzKw2TkJmZlabSpOQpNmSlkla3KZ8T0kLJD0l6eSmsq0lXSjpD5KWSNq3yljNzGzoVX0mNAc4qI/yFcDxwJdblH0d+EVE7AnsBSwZ9OjMzKxWlSahiLiMlGjalS+LiGuB1cX+krYC/h/wnTzcXyPi4QpDNTOzGgzXe0K7Ar3AdyVdL+nbkrZoNaCkmZIWSlrY29s7tFGamdl6Ga5JaBSwN/CtiHgp8Djw8VYDRsSZETE9IqZPmDBhKGM0M7P1NFyTUA/QExFX5+4LSUnJzMxGkGGZhCJiKXCPpD1yr9cAt9YYkpmZVWBUlROXNBeYAYyX1APMAkYDRMQZknYAFgLjgDWSTgSmRMRK4EPAuZI2Af4CHFtlrGZmNvQqTUIRcUSH8qXAxDZlNwDTKwjLzMyGiWF5Oc7MzLqDk5CZmdXGScjMzGrjJGRmZrVxEjIzs9o4CZmZWW2chMzMrDZOQmZmVhsnITMzq42TkJmZ1cZJyMzMauMkZGZmtXESMjOz2jgJmZlZbZyEzMysNk5CZmZWGychMzOrjZOQmZnVpmMSkrSxpA8PRTBmZtZdOiahiHgGOGIIYjEzsy4zquRwV0r6BnAe8HijZ0RcV0lUZmbWFcomoWn57+mFfgEcOKjRmJlZVymVhCLi1VUHYmZm3adU6zhJW0n6iqSF+fMfkraqOjgzMxvZyjbRng08ChyePyuB71YVlJmZdYey94SeHxFvKXR/StINFcRjZmZdpOyZ0JOS9m90SNoPeLLTSJJmS1omaXGb8j0lLZD0lKSTW5RvLOl6ST8rGaeZmW1Ayp4JvQ84u3Af6CHg6BLjzQG+AZzdpnwFcDxwWJvyE4AlwLiScZqZ2Qak1BMTgHdExF7AVGBqRLw0Im7qNG5EXEZKNO3Kl0XEtcDqFvOdCPwD8O1O8zEzsw1T2Scm7J+/r4yIlZVHlXwN+Biwpq+BJM1stNrr7e0dksDMzGxwlL0cd72ki4ELWPeJCT+qIihJhwDLImKRpBl9DRsRZwJnAkyfPj2qiMfMzKpRNgmNAZaz7hMSAqgkCQH7AW+S9IY873GSzomIoyqan5mZ1aBjEsr3hJZHxLNar1UlIk4BTsnznwGc7ARkZjbydExCEfFMbpLdb5LmAjOA8ZJ6gFnA6DzdMyTtACwktX5bI+lEYMoQ3ncyM7Malb0cd8NA7glFRJ+vgIiIpcDEDsPMB+aXjNPMzDYgw/WekJmZdYGyT9E+tupAzMys+5R9ivYLJF3aePyOpKmSTq02NDMzG+nKPjvuf0it1VYD5KclvL2qoMzMrDuUTUKbR8Q1Tf2eHuxgzMysu5RNQg9Kej6pMQKS3grcX1lUZmbWFcq2jvsg6dE4e0q6F7gDOLKyqMzMrCuUbR33F+C1krYANoqIR4vlko6OiLOqCNDMzEauspfjAIiIx5sTUHbCIMVjZmZdpF9JqA8apOmYmVkXGawk5FcomJlZv/lMyMzMajNYSejKQZqOmZl1kVKt4yRtCrwFmFwcJyJOz3//pYrgzMxsZCv7O6GfAI8Ai4CnqgvHzMy6SdkkNDEiDqo0EjMz6zpl7wn9XtJLKo3EzMy6Ttkzof2BYyTdQbocJyAiYmplkZmZ2YhXNgkdXGkUZmbWlco+O+4uSXsBB+Rel0fEjdWFNfQ+9dNbuPW+lXWHYWY2IFN2GsesN76o7jD6reybVU8AzgW2z59zJH2oysDMzGzkU0TnJ+5IugnYNyIez91bAAuG2z2h6dOnx8KFC+sOw8xsgyJpUURMr2PeZVvHCXim0P0MflSPmZmtp7INE74LXC3potx9GPCdSiIyM7OuUbZhwlckzSc11QY4NiKurywqMzPrCn0mIUnjImKlpG2BO/OnUbZtRKyoNjwzMxvJOt0T+n7+uwhYWPg0uvskabakZZIWtynfU9ICSU9JOrnQfxdJ8yTdKumW3DrPzMxGmD7PhCLikPx31wFOfw7wDeDsNuUrgONJ95iKngY+EhHXSdoSWCTp1xFx6wDjMDOzYajs74QuLdOvWURcRko07cqXRcS1wOqm/vdHxHX5+6PAEmDnMrGamdmGo9M9oTHA5sB4Sduwtln2OIYoKUiaDLwUuLpN+UxgJsCkSZOGIiQzMxsknVrHvRc4EdiJdB+okYRWki6zVUrSWOCHwIkR0fKZOhFxJnAmpB+rVh2TmZkNnk73hL4OfF3ShyLiv4YoJgAkjSYloHMj4kdDOW8zMxsaZX8n9F+SXgxMAcYU+rdrcLBeJIn0Y9glEfGVKuZhZmb1K5WEJM0CZpCS0M9Jr3a4gvat3hrjzc3jjZfUA8wCRgNExBmSdiA19R4HrJF0Yp7HVOAdwM2SbsiT+7eI+Hn5RTMzs+Gu7GN73grsBVwfEcdKeg5wTqeRIuKIDuVLgYktiq7Az6YzMxvxyj7A9MmIWAM8LWkcsAzYpbqwzMysG5Q9E1ooaWvgf0it5B4DFlQVlJmZdYeyDRM+kL+eIekXwLiIuKm6sMzMrBt0+rHq3n2VNZ5qYGZmNhCdzoT+I/8dA0wHbiQ1GJhKatW2b3WhmZnZSNdnw4SIeHVEvBq4H9g7IqZHxMtIj9G5dygCNDOzkats67g9IuLmRkdELAZeWE1IZmbWLcq2jrtJ0rdZ+9ugIwE3TDAzs/VSNgkdC7wfaLxc7jLgW5VEZGZmXaNsE+1VwFfzx8zMbFB0aqJ9fkQcLulm4FmvSYiIqZVFZmZmI16nM6HG5bdDqg7EzMy6T6f3Cd2f/941NOGYmVk36XQ57lFaXIYj/WA1ImJcJVGZmVlX6HQmtOVQBWJmZt2nbBNtACRtz7pvVr170CMyM7OuUeqJCZLeJOnPwB3A74A7gUsqjMvMzLpA2cf2fBrYB/hTROwKvAa4qrKozMysK5RNQqsjYjmwkaSNImIe6anaZmZmA1b2ntDDksYClwPnSloGPF5dWGZm1g3KngnNA7Yi/Xj1F8DtwBurCsrMzLpD2SQ0CvgVMB/YEjgvX54zMzMbsFJJKCI+FREvAj4I7Aj8TtJvKo3MzMxGvLJnQg3LgKXAcmD7wQ/HzMy6SdnfCX1A0nzgUmA74D1+graZma2vsq3jdgFOjIgbKozFzMy6TNl7QqcMJAFJmi1pmaTFbcr3lLRA0lOSTm4qO0jSHyXdJunj/Z23mZkNf/29J9Rfc4CD+ihfARwPfLnYU9LGwDeBg4EpwBGSplQUo5mZ1aTSJBQRl5ESTbvyZRFxLbC6qejvgNsi4i8R8VfgB8Ch1UVqZmZ1qPpMaKB2Bu4pdPfkfmZmNoIM1yRUmqSZkhZKWtjb21t3OGZm1g/DNQndS2qR1zAx93uWiDgzIqZHxPQJEyYMSXBmZjY4hmsSuhbYXdKukjYB3g5cXHNMZmY2yPr1ZtX+kjQXmAGMl9QDzAJGA0TEGZJ2ABYC44A1kk4EpkTESkn/AvwS2BiYHRG3VBmrmZkNvUqTUEQc0aF8KelSW6uynwM/ryIuMzMbHobr5TgzM+sCTkJmZlYbJyEzM6uNk5CZmdXGScjMzGrjJGRmZrVxEjIzs9o4CZmZWW2chMzMrDZOQmZmVhsnITMzq42TkJmZ1cZJyMzMauMkZGZmtXESMjOz2jgJmZlZbZyEzMysNk5CZmZWGychMzOrjZOQmZnVxknIzMxq4yRkZma1cRIyM7PaOAmZmVltnITMzKw2TkJmZlYbJyEzM6tNpUlI0mxJyyQtblMuSf8p6TZJN0nau1D2RUm3SFqSh1GVsZqZ2dCr+kxoDnBQH+UHA7vnz0zgWwCSXgnsB0wFXgy8HHhVlYGamdnQqzQJRcRlwIo+BjkUODuSq4CtJe0IBDAG2ATYFBgNPFBlrGZmNvTqvie0M3BPobsH2DkiFgDzgPvz55cRsaTVBCTNlLRQ0sLe3t7KAzYzs8FTdxJqSdJuwAuBiaREdaCkA1oNGxFnRsT0iJg+YcKEoQzTzMzWU91J6F5gl0L3xNzvzcBVEfFYRDwGXALsW0N8ZmZWobqT0MXAO3MruX2ARyLifuBu4FWSRkkaTWqU0PJynJmZbbhGVTlxSXOBGcB4ST3ALFIjAyLiDODnwBuA24AngGPzqBcCBwI3kxop/CIiflplrGZmNvQqTUIRcUSH8gA+2KL/M8B7q4rLzMyGh7ovx5mZWRdzEjIzs9o4CZmZWW2chMzMrDZOQmZmVhsnITMzq42TkJmZ1cZJyMzMauMkZGZmtXESMjOz2jgJmZlZbZyEzMysNk5CZmZWGychMzOrjZOQmZnVxknIzMxq4yRkZma1cRIyM7PaOAmZmVltnITMzKw2TkJmZlYbJyEzM6uNk5CZmdVGEVF3DINGUi9w1wBHHw88OIjhbMhcF2u5LhLXw1ojsS6eGxET6pjxiEpC60PSwoiYXnccw4HrYi3XReJ6WMt1Mbh8Oc7MzGrjJGRmZrVxElrrzLoDGEZcF2u5LhLXw1qui0Hke0JmZlYbnwmZmVltnITMzKw2XZmEJM2WtEzS4kK/bSX9WtKf899t6oxxKEjaRdI8SbdKukXSCbl/N9bFGEnXSLox18Wncv9dJV0t6TZJ50napO5Yh4KkjSVdL+lnubtb6+FOSTdLukHSwtyv67aPKnVlEgLmAAc19fs4cGlE7A5cmrtHuqeBj0TEFGAf4IOSptCddfEUcGBE7AVMAw6StA/wBeCrEbEb8BBwXH0hDqkTgCWF7m6tB4BXR8S0wm+DunH7qExXJqGIuAxY0dT7UOCs/P0s4LChjKkOEXF/RFyXvz9K2unsTHfWRUTEY7lzdP4EcCBwYe7fFXUhaSLwD8C3c7fownroQ9dtH1XqyiTUxnMi4v78fSnwnDqDGWqSJgMvBa6mS+siX4K6AVgG/Bq4HXg4Ip7Og/SQkvRI9zXgY8Ca3L0d3VkPkA5EfiVpkaSZuV9Xbh9VGVV3AMNRRISkrmm7Lmks8EPgxIhYmQ58k26qi4h4BpgmaWvgImDPeiMaepIOAZZFxCJJM2oOZzjYPyLulbQ98GtJfygWdtP2URWfCa31gKQdAfLfZTXHMyQkjSYloHMj4ke5d1fWRUNEPAzMA/YFtpbUOFibCNxbV1xDZD/gTZLuBH5Augz3dbqvHgCIiHvz32WkA5O/o8u3j8HmJLTWxcDR+fvRwE9qjGVI5Gv93wGWRMRXCkXdWBcT8hkQkjYD/p50j2we8NY82Iivi4g4JSImRsRk4O3AbyPiSLqsHgAkbSFpy8Z34HXAYrpw+6hSVz4xQdJcYAbpkewPALOAHwPnA5NIr4M4PCKaGy+MKJL2By4Hbmbt9f9/I90X6ra6mEq6ybwx6eDs/Ig4XdLzSGcE2wLXA0dFxFP1RTp08uW4kyPikG6sh7zMF+XOUcD3I+Izkrajy7aPKnVlEjIzs+HBl+PMzKw2TkJmZlYbJyEzM6uNk5CZmdXGScjMzGrjJGTWgqT5kqbn73dKGt/HsKdJOnkA85jReEp1H8NMk/SGQvebJPmBmTZiOAmZDW/TgL8loYi4OCI+X184ZoPLSchGLEkflXR8/v5VSb/N3w+UdG7+/i1JC4vvEOowzXdKuim/d+h7LcqnSboqD3NR410zknaT9Js83nWSnt803svz+3ueX+i3CXA68E/5fTb/JOkYSd/I5XNy/FdJ+ks+s5otaYmkOYXpvE7SgjzfC/KzAs2GBSchG8kuBw7I36cDY/Oz8g4ALsv9P5HfEzMVeFV+ckJLkl4EnMra9w6d0GKws4F/jYippCdRzMr9zwW+mcd7JdB4CjOSXgmcARwaEbc3+kfEX4FPAufl99mc12J+25Cecfdh0uNkvgq8CHhJTojjc8yvjYi9gYXASe2W0Wyo+SnaNpItAl4maRzppXXXkZLRAcDxeZjD8yP6RwE7AlOAm9pM70Dggoh4EKD5US2StgK2jojf5V5nARfk54/tHBEX5fFW5eEBXgicCbwuIu4bwDL+ND/J+WbggYi4OU/7FmAy6WGjU4Ar8/w2ARYMYD5mlXASshErIlZLugM4Bvg9Kbm8GtgNWCJpV+Bk4OUR8VC+hDVmiMO8P8/zpcBAklDj+W1rCt8b3aOAZ4BfR8QR6xOkWVV8Oc5GustJieay/P19wPWRHpo4DngceETSc4CDO0zrt8Db8gMskbRtsTAiHgEektS4BPgO4Hf5rbU9kg7L420qafM8zMOkt5h+rs37ex4Ftiy7sC1cBewnabc87y0kvWA9pmc2qJyEbKS7nHSZbUFEPACsyv2IiBtJT4T+A/B94Mq+JhQRtwCfAX4n6UbgKy0GOxr4kqSbSC3bTs/93wEcn/v/HtihMN0HgEOAb0p6RdP05gFTGg0Tyi50Ydq9pDPBuXneC+jCl/XZ8OWnaJuZWW18JmRmZrVxEjIzs9o4CZmZWW2chMzMrDZOQmZmVhsnITMzq42TkJmZ1eb/AF22BeGKTHDlAAAAAElFTkSuQmCC\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

```{.python .input  n=45}
import matplotlib.pyplot as plt
results = tuning_experiment.results

for trial_id in results.trial_id.unique():
    df = results[results['trial_id'] == trial_id]
    plt.plot(df['st_tuner_time'], df['validation_error'], marker='o')
```

```{.json .output n=45}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## Summary

## Excercise
