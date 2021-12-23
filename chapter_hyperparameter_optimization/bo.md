# Bayesian Optimization

MS: It will be hard to implement anything from scratch here. The only thing I
can think of would be UCB acquisition function, maximized on a finite set of
candidates. But this still needs the model, and optimization of its
hyperparameters, which is just not possible to do from scratch.

We could stick the ST scheduler into the home-made Tuner, which would give us
sequential BO (you also get that with ST and 1 worker).

## Modelling the Objective Function


## Sequential Bayesian Optimization

## Parallel Bayesian Optimization

## Combining Bayesian Optimization with Successive Halving



