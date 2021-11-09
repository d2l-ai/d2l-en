# Random Search

The first method we will look at is called random search. The idea is to first define a probability distribution for each hyperparameter individually. We then iteratively sample hyperparameter configurations from these probability distributions. Compared to grid search, which evaluate a user defined grid of hyperparameter configurations, random search does not suffer from the curse of dimensionality.

If we use probability distributions that assign non-zero probability to each point in the search space, random search will asymptotically converge to the global optimum. However, it does not maintain a history of the previous observed points, such as model-based approaches that we will describe in later chapters, which renders it less sample efficient in practice.

## Sampling Configurations from the Search Space

First, we implement a method that allows us to sample random configurations from our search space. Each hyperparameter will be sampled independently from the other hyperparameters.

```{.python .input  n=4}
import scipy.stats as stats

search_space = {
  'learning_rate': stats.loguniform(1e-6, 1),
  'weight_decay': stats.loguniform(1e-9, 1e-1)
}


def sample(search_space):
  config = {}
  for hyperparameter in search_space:
      config[hyperparameter] = search_space[hyperparameter].rvs()
  return config
```

## The HPO Loop

Now, we can implement the main optimization loop of random search, that iterates until we reach the final number of iterations specified by the users. In each iteration, we first sample a hyperparameter configuration from the subroutine that we implemented above and then train and validate the model with the new candidate. We also maintain the current incumbent, i.e the best configuration we have found so far. This will be the configuration we will later return as the final configuration.

```{.python .input  n=4}
def train_and_validate(config):
    # TODO: this function will be defined in the previous section
    return 0.1

num_iterations = 100

incumbent = None
incumbent_performance = None
incumbent_trajectory = []

for i in range(num_iterations):
    config = sample(search_space)
    performance = train_and_validate(config)

    # bookkeeping
    if incumbent is None or incumbent_performance < performance:
        incumbent = config
        incumbent_performance = performance
    incumbent_trajectory.append(incumbent_performance)
```



Now we can plot the optimization trajectory of the incumbent to get the anytime performance of random search:

```{.python .input  n=4}
%matplotlib inline
from IPython import display
from matplotlib import pyplot as plt

display.set_matplotlib_formats('svg')

plt.plot(incumbent_trajectory);

```



## Parallelizing Random Search

- Random search can be trivially paralleled, by just simple sampling each configuration independently. This implies that we can a perfect linear scaling with the number of workers.



## Using Pseudo Random Sequences





## Summary



## Excercise

