# Adam
:label:`sec_adam`

In the discussions leading up to this chapter we encountered a number of techniquest for efficient optimization. Let's recap them in detail here:

* We saw that :ref:`sec_sgd` is more effective than Gradient Descent when solving optimization problems, e.g. due to its inherent resilience to redundant data. 
* We saw that :ref:`sec_minibatch_sgd` affords significant additional efficiency arising from vectorization, using larger sets of observations in one minibatch. This is the key to efficient multi-machine, multi-GPU and overall parallel processing. 
* :ref:`sec_momentum` added a mechanism for aggregating a history of past gradients to accelerate convergence.
* :ref:`sec_adagrad` used per-coordinate scaling to allow for a computationally efficient preconditioner. 
* :ref:`sec_rmsprop` decoupled per-coordinate scaling from a learning rate adjustment. 

Adam :cite:`Kingma.Ba.2014` combines all these techniques into one efficient learning algorithm. As expected, this is an algorithm that has become rather popular as one of the more robust and effective optimization algorithms to use in deep learning. It is not without issues, though. In particular, :cite:`Reddi.Kale.Kumar.2019` show that there are situations where Adam can diverge due to poor variance control. In a followup work :cite:`Zaheer.Reddi.Sachan.ea.2018` proposed a hotfix to Adam, called Yogi which addresses these issues. More on this later. For now let's review the Adam algorithm. 

## The Algorithm

One of the key components of Adam is that it uses exponential weighted moving averages (aka leaky averaging) to obtain an estimate of both the momentum and also the second moment of the gradient. That is, it uses the state variables

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2
\end{aligned}$$

Here $\beta_1$ and $\beta_2$ are nonnegative weighting parameters. Common choices for them are $\beta_1 = 0.9$ and $\beta_2 = 0.999$. That is, the variance estimate moves *much more slowly* than the momentum term. Note that if we initialize $\mathbf{v}_0 = \mathbf{s}_0 = 0$ we have a significant amount of bias initially towards smaller values. This can be addressed by using the fact that $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ to re-normalize terms. Correspondingly the normalized state variables are given by 

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$

Armed with the proper estimates we can now write out the update equations. Firstly we rescale the gradient in a manner very much akin to that of RMSProp to obtain

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$$

Unlike RMSProp our update uses the momentum $\hat{\mathbf{v}}_t$ rather than the gradient itself. Moreover, there's a slight cosmetic difference as the rescaling happens using $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ instead of $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. The former works arguably slightly better in practice, hence the deviation from RMSProp. Typically we pick $\epsilon = 10^{-6}$ for a good trade-off between numerical stability and fidelity. 

Now we have all the pieces in place to compute updates. This is slightly anticlimactic and we have a simple update of the form

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$

Reviewing the design of Adam its inspiration is clear. Momentum and scale are clearly visible in the state variables. Their rather peculiar definition forces us to debias terms (this could be fixed by a slightly different initialization and update condition). Secondly, the combination of both terms is pretty straightforward, given RMSProp. Lastly, the explicit learning rate $\eta$ allows us to control the step length to address issues of convergence. 

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

## Implementation 

Implementing Adam from scratch isn't very daunting. For convenience we store the time step counter $t$ in the `hyperparams` dictionary. Beyond that all is straightforward.

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

We are ready to use Adam to train the model. We use a learning rate of $\eta = 0.01$.

```{.python .input  n=5}
data_iter, feature_dim = d2l.get_data_ch10(batch_size=10)
d2l.train_ch10(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

A more concise implementation is straightforward since `adam` is one of the algorithms provided as part of the Gluon `trainer` optimization library. Hence we only need to pass configuration parameters for an implementation in Gluon.

```{.python .input  n=11}
d2l.train_gluon_ch10('adam', {'learning_rate': 0.01}, data_iter)
```

## TODO - Yogi


## Summary

* Created on the basis of RMSProp, Adam also uses EWMA on the minibatch stochastic gradient
* Adam uses bias correction.

## Exercises

* Adjust the learning rate and observe and analyze the experimental results.
* Some people say that Adam is a combination of RMSProp and momentum. Why do you think they say this?



## [Discussions](https://discuss.mxnet.io/t/2378)

![](../img/qr_adam.svg)
