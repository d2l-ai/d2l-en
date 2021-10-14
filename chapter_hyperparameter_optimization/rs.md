# Random Search

The first method we will look at is called random search. The idea is to define a probability distribution for each hyperparameter individually. We then iteratively sample hyperparameter configurations from these probability distributions. Compared to grid search, random search does not suffer from the curse of dimensionality.

If we use proability distribution that assign non-zero probability to each point in the search space,  random search will  asympototically converge to the global optimum. However, it does not maintain a history of the previous observed potins, such as model-based approaches, which we will describe in later chapters, which renders it less sample efficient in practice.

## Sampling Configurations from the Search Space

First, we implement a method that allows us to sample random configurations from our search space. Each hyperparameter will sampled independently from the other hyperparameters.

```{.python .input  n=2}
def sample(search_space):
  config = dict()
  for hyperparameter in search_space:
    config[hyperparameter.name] = hyperparameter.domain.sample()
  return config

```

## The HPO Loop

Now, we can implement the main optimization loop of random search, that iterates until we reach the final number of iterations specified by the users. In each iteration, we first sample a hyperparameter configurations from the subroutine that we implemented above and then train and validate the model with the new candidate. We also maintain the current incumbent, i.e the best configuration we have found so far. This will be the configuration we will later return as the final configuration.

```{.python .input  n=2}
  num_iterations = 100

  incumbent = None
  incumbent_performance = None

  for i in range(num_iterations):
    config = sample()
    performance = train_and_validate(config)

    # bookkeeping
    if incumbent is None or incumbent_performance < performance:
      incumbent = config
      incumbent_performance = performance
  ```


## Parallelizing Random Search

- Random search can be trivially paralleled, by just simple sampling each configuration independently. This implies that we can a perfect linear scaling with the number of workers.



## Using Pseudo Random Sequences





## Summary



## Excercise

