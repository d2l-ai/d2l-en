# Adam
:label:`sec_adam`

Created on the basis of RMSProp, Adam also uses EWMA on the minibatch stochastic gradient[1]. Here, we are going to introduce this algorithm.

## The Algorithm

Adam :cite:`Kingma.Ba.2014` uses the momentum variable $\mathbf{v}_t$ and variable $\mathbf{s}_t$, which is an EWMA on the squares of elements in the minibatch stochastic gradient from RMSProp, and initializes each element of the variables to 0 at time step 0. Given the hyperparameter $0 \leq \beta_1 < 1$ (the author of the algorithm suggests a value of 0.9), the momentum variable $\mathbf{v}_t$ at time step $t$ is the EWMA of the minibatch stochastic gradient $\mathbf{g}_t$:

$$\mathbf{v}_t \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t. $$

Just as in RMSProp, given the hyperparameter $0 \leq \beta_2 < 1$ (the author of the algorithm suggests a value of 0.999),
After taken the squares of elements in the minibatch stochastic gradient, find $\mathbf{g}_t \odot \mathbf{g}_t$ and perform EWMA on it to obtain $\mathbf{s}_t$:

$$\mathbf{s}_t \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t \odot \mathbf{g}_t. $$

Since we initialized elements in $\mathbf{v}_0$ and $\mathbf{s}_0$ to 0,
we get $\mathbf{v}_t =  (1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} \mathbf{g}_i$ at time step $t$. Sum the minibatch stochastic gradient weights from each previous time step to get $(1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} = 1 - \beta_1^t$. Notice that when $t$ is small, the sum of the minibatch stochastic gradient weights from each previous time step will be small. For example, when $\beta_1 = 0.9$, $\mathbf{v}_1 = 0.1\mathbf{g}_1$. To eliminate this effect, for any time step $t$, we can divide $\mathbf{v}_t$ by $1 - \beta_1^t$, so that the sum of the minibatch stochastic gradient weights from each previous time step is 1. This is also called bias correction. In the Adam algorithm, we perform bias corrections for variables $\mathbf{v}_t$ and $\mathbf{s}_t$:

$$\hat{\mathbf{v}}_t \leftarrow \frac{\mathbf{v}_t}{1 - \beta_1^t}, $$

$$\hat{\mathbf{s}}_t \leftarrow \frac{\mathbf{s}_t}{1 - \beta_2^t}. $$


Next, the Adam algorithm will use the bias-corrected variables $\hat{\mathbf{v}}_t$ and $\hat{\mathbf{s}}_t$ from above to re-adjust the learning rate of each element in the model parameters using element operations.

$$\mathbf{g}_t' \leftarrow \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon},$$

Here, $\eta$ is the learning rate while $\epsilon$ is a constant added to maintain numerical stability, such as $10^{-8}$. Just as for Adagrad, RMSProp, and Adadelta, each element in the independent variable of the objective function has its own learning rate. Finally, use $\mathbf{g}_t'$ to iterate the independent variable:

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'. $$

## Implementation from Scratch

We use the formula from the algorithm to implement Adam. Here, time step $t$ uses `hyperparams` to input parameters to the `adam` function.

```{.python .input  n=2}
%matplotlib inline
import d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = np.zeros((feature_dim, 1)), np.zeros(1)
    s_w, s_b = np.zeros((feature_dim, 1)), np.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

Use Adam to train the model with a learning rate of $0.01$.

```{.python .input  n=5}
data_iter, feature_dim = d2l.get_data_ch10(batch_size=10)
d2l.train_ch10(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Concise Implementation

From the `Trainer` instance of the algorithm named "adam", we can implement Adam with Gluon.

```{.python .input  n=11}
d2l.train_gluon_ch10('adam', {'learning_rate': 0.01}, data_iter)
```

## Summary

* Created on the basis of RMSProp, Adam also uses EWMA on the minibatch stochastic gradient
* Adam uses bias correction.

## Exercises

* Adjust the learning rate and observe and analyze the experimental results.
* Some people say that Adam is a combination of RMSProp and momentum. Why do you think they say this?



## [Discussions](https://discuss.mxnet.io/t/2378)

![](../img/qr_adam.svg)
