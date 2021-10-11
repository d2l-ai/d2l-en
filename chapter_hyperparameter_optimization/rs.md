# Random Search

- Sampling configurations randomly from the search space.
- Advantage to grid search
- If we have non-zero probability distributions we are asympototically converge to the global optimum
- Intrinsic dimensionality

## Sampling Configurations from the Search Space

```{.python .input  n=2}
def sample():
  return config
```

## The HPO Loop

- Implement a simple loop that iterates until the final number of iterations

- sample configuration

- train and validate the model

- Given the trajectory we estimate the incumbent as the best observed value so far

- ```{.python .input  n=2}
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

- ```{.python .input  n=2}
  
  ```

## Parallelizing Random Search



## Summary



## Excercise

