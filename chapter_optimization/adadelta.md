# Adadelta

In addition to RMSProp, Adadelta is another common optimization algorithm that helps improve the chances of finding useful solutions at later stages of iteration, which is difficult to do when using the Adagrad algorithm for the same purpose[1]. The interesting thing is that there is no learning rate hyperparameter in the Adadelta algorithm.

## The Algorithm

Like RMSProp, the Adadelta algorithm uses the variable $\boldsymbol{s}_t$, which is an EWMA on the squares of elements in mini-batch stochastic gradient $\boldsymbol{g}_t$. At time step 0, all the elements are initialized to 0.
Given the hyperparameter $0 \leq \rho < 1$ (counterpart of $\gamma$ in RMSProp), at time step $t>0$, compute using the same method as RMSProp:

$$\boldsymbol{s}_t \leftarrow \rho \boldsymbol{s}_{t-1} + (1 - \rho) \boldsymbol{g}_t \odot \boldsymbol{g}_t. $$

Unlike RMSProp, Adadelta maintains an additional state variable, $\Delta\boldsymbol{x}_t$ the elements of which are also initialized to 0 at time step 0. We use $\Delta\boldsymbol{x}_{t-1}$ to compute the variation of the independent variable:

$$ \boldsymbol{g}_t' \leftarrow \sqrt{\frac{\Delta\boldsymbol{x}_{t-1} + \epsilon}{\boldsymbol{s}_t + \epsilon}}   \odot \boldsymbol{g}_t, $$

Here, $\epsilon$ is a constant added to maintain the numerical stability, such as $10^{-5}$. Next, we update the independent variable:

$$\boldsymbol{x}_t \leftarrow \boldsymbol{x}_{t-1} - \boldsymbol{g}'_t. $$

Finally, we use $\Delta\boldsymbol{x}$ to record the EWMA on the squares of elements in $\boldsymbol{g}'$, which is the variation of the independent variable.

$$\Delta\boldsymbol{x}_t \leftarrow \rho \Delta\boldsymbol{x}_{t-1} + (1 - \rho) \boldsymbol{g}'_t \odot \boldsymbol{g}'_t. $$

As we can see, if the impact of $\epsilon$ is not considered here, Adadelta differs from RMSProp in its replacement of the hyperparameter $\eta$ with $\sqrt{\Delta\boldsymbol{x}_{t-1}}$.


## Implementation from Scratch

Adadelta needs to maintain two state variables for each independent variable, $\boldsymbol{s}_t$ and $\Delta\boldsymbol{x}_t$. We use the formula from the algorithm to implement Adadelta.

```{.python .input  n=11}
import sys
sys.path.insert(0, '..')

%matplotlib inline
import d2l
from mxnet import nd

features, labels = d2l.get_data_ch7()

def init_adadelta_states():
    s_w, s_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    delta_w, delta_b = nd.zeros((features.shape[1], 1)), nd.zeros(1)
    return ((s_w, delta_w), (s_b, delta_b))

def adadelta(params, states, hyperparams):
    rho, eps = hyperparams['rho'], 1e-5
    for p, (s, delta) in zip(params, states):
        s[:] = rho * s + (1 - rho) * p.grad.square()
        g = ((delta + eps).sqrt() / (s + eps).sqrt()) * p.grad
        p[:] -= g
        delta[:] = rho * delta + (1 - rho) * g * g
```

Then, we train the model with the hyperparameter $\rho=0.9$.

```{.python .input  n=12}
d2l.train_ch7(adadelta, init_adadelta_states(), {'rho': 0.9}, features,
              labels)
```

## Concise Implementation

From the `Trainer` instance for the algorithm named "adadelta", we can implement Adadelta in Gluon. Its hyperparameters can be specified by `rho`.

```{.python .input  n=9}
d2l.train_gluon_ch7('adadelta', {'rho': 0.9}, features, labels)
```

## Summary

* Adadelta has no learning rate hyperparameter, it uses an EWMA on the squares of elements in the variation of the independent variable to replace the learning rate.

## Problems

* Adjust the value of $\rho$ and observe the experimental results.

## Reference

[1] Zeiler, M. D. (2012). ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2377)

![](../img/qr_adadelta.svg)
