# Adadelta
:label:`sec_adadelta`

In addition to RMSProp, Adadelta is another common optimization algorithm that
helps improve the chances of finding useful solutions at later stages of
iteration, which is difficult to do when using the Adagrad algorithm for the
same purpose :cite:`Zeiler.2012`. The interesting thing is that there is no learning rate
hyperparameter in the Adadelta algorithm.

## The Algorithm

Like RMSProp, the Adadelta algorithm uses the variable $\mathbf{s}_t$, which is an EWMA on the squares of elements in minibatch stochastic gradient $\mathbf{g}_t$. At time step 0, all the elements are initialized to 0.
Given the hyperparameter $0 \leq \rho < 1$ (counterpart of $\gamma$ in RMSProp), at time step $t>0$, compute using the same method as RMSProp:

$$\mathbf{s}_t \leftarrow \rho \mathbf{s}_{t-1} + (1 - \rho) \mathbf{g}_t \odot \mathbf{g}_t. $$

Unlike RMSProp, Adadelta maintains an additional state variable, $\Delta\mathbf{x}_t$ the elements of which are also initialized to 0 at time step 0. We use $\Delta\mathbf{x}_{t-1}$ to compute the variation of the independent variable:

$$ \mathbf{g}_t' \leftarrow \sqrt{\frac{\Delta\mathbf{x}_{t-1} + \epsilon}{\mathbf{s}_t + \epsilon}}   \odot \mathbf{g}_t, $$

Here, $\epsilon$ is a constant added to maintain the numerical stability, such as $10^{-5}$. Next, we update the independent variable:

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}'_t. $$

Finally, we use $\Delta\mathbf{x}$ to record the EWMA on the squares of elements in $\mathbf{g}'$, which is the variation of the independent variable.

$$\Delta\mathbf{x}_t \leftarrow \rho \Delta\mathbf{x}_{t-1} + (1 - \rho) \mathbf{g}'_t \odot \mathbf{g}'_t. $$

As we can see, if the impact of $\epsilon$ is not considered here, Adadelta differs from RMSProp in its replacement of the hyperparameter $\eta$ with $\sqrt{\Delta\mathbf{x}_{t-1}}$.


## Implementation from Scratch

Adadelta needs to maintain two state variables for each independent variable, $\mathbf{s}_t$ and $\Delta\mathbf{x}_t$. We use the formula from the algorithm to implement Adadelta.

```{.python .input}

```

```{.python .input  n=11}
%matplotlib inline
import d2l
from mxnet import np, npx
npx.set_np()

def init_adadelta_states(feature_dim):
    s_w, s_b = np.zeros((feature_dim, 1)), np.zeros(1)
    delta_w, delta_b = np.zeros((feature_dim, 1)), np.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * np.square(p.grad)
        g = (np.sqrt(delta + eps) / np.sqrt(s + eps)) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

Then, we train the model with the hyperparameter $\rho=0.9$.

```{.python .input  n=12}
data_iter, feature_dim = d2l.get_data_ch10(batch_size=10)
d2l.train_ch10(adadelta, init_adadelta_states(feature_dim),
               {'rho': 0.9}, data_iter, feature_dim);
```

## Concise Implementation

From the `Trainer` instance for the algorithm named "adadelta", we can implement Adadelta in Gluon. Its hyperparameters can be specified by `rho`.

```{.python .input  n=9}
d2l.train_gluon_ch10('adadelta', {'rho': 0.9}, data_iter)

```

## Summary

* Adadelta has no learning rate hyperparameter, it uses an EWMA on the squares of elements in the variation of the independent variable to replace the learning rate.

## Exercises

* Adjust the value of $\rho$ and observe the experimental results.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2377)

![](../img/qr_adadelta.svg)
