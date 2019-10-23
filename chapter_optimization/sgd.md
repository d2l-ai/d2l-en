# Stochastic Gradient Descent
:label:`sec_sgd`

In this section, we are going to introduce the basic principles of stochastic gradient descent.

```{.python .input  n=2}
%matplotlib inline
import d2l
import math
from mxnet import np, npx 
npx.set_np()
```

## Stochastic Gradient Updates

In deep learning, the objective function is usually the average of the loss functions for each example in the training data set. We assume that $f_i(\boldsymbol{x})$ is the loss function of the training data instance with $n$ examples, an index of $i$, and parameter vector of $\boldsymbol{x}$, then we have the objective function

$$f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n f_i(\boldsymbol{x}).$$

The gradient of the objective function at $\boldsymbol{x}$ is computed as

$$\nabla f(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}).$$

If gradient descent is used, the computing cost for each independent variable iteration is $\mathcal{O}(n)$, which grows linearly with $n$. Therefore, when the model training data instance is large, the cost of gradient descent for each iteration will be very high.

Stochastic gradient descent (SGD) reduces computational cost at each iteration. At each iteration of stochastic gradient descent, we uniformly sample an index $i\in\{1,\ldots,n\}$ for data instances at random, and compute the gradient $\nabla f_i(\boldsymbol{x})$ to update $\boldsymbol{x}$:

$$\boldsymbol{x} \leftarrow \boldsymbol{x} - \eta \nabla f_i(\boldsymbol{x}).$$

Here, $\eta$ is the learning rate. We can see that the computing cost for each iteration drops from $\mathcal{O}(n)$ of the gradient descent to the constant $\mathcal{O}(1)$. We should mention that the stochastic gradient $\nabla f_i(\boldsymbol{x})$ is the unbiased estimate of gradient $\nabla f(\boldsymbol{x})$.

$$\mathbb{E}_i \nabla f_i(\boldsymbol{x}) = \frac{1}{n} \sum_{i = 1}^n \nabla f_i(\boldsymbol{x}) = \nabla f(\boldsymbol{x}).$$

This means that, on average, the stochastic gradient is a good estimate of the gradient.

Now, we will compare it to gradient descent by adding random noise with a mean of 0 to the gradient to simulate a SGD.

```{.python .input  n=3}
def f(x1, x2): return x1 ** 2 + 2 * x2 ** 2   # objective
def gradf(x1, x2): return (2 * x1, 4 * x2)    # gradient
def sgd(x1, x2, s1, s2):                      # simulate noisy gradient
    global lr                                 # learning rate scheduler
    (g1, g2) = gradf(x1, x2)                  # compute gradient
    (g1, g2) = (g1 + np.random.normal(0.1), g2 + np.random.normal(0.1))
    eta_t = eta * lr()                        # learning rate at time t
    return (x1 - eta_t * g1, x2 - eta_t * g2, 0, 0) # update variables

eta = 0.1
lr = (lambda: 1)                              # constant learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```

As we can see, the trajectory of the variables in the SGD is much more noisy than the one we observed in gradient descent in the previous section. This is due to the stocastic nature of the gradient. That is, even when we arrive near the minimum, we are still subject to the uncertainty injected by the instantaneous gradient via $\eta \nabla f_i(\boldsymbol{x})$. Even after 50 steps the quality is still not so good. Even worse, it will not improve after additional steps (we encourage the reader to experiment with a larger number of steps to confirm this on his own). This leaves us with the only alterative - change the learning rate $\eta$. However, if we pick this too small, we will not make any meaningful progress initially. On the other hand, if we pick it too large, we will not get a good solution, as seen above. The only way to resolve these conflicting goals is to reduce the learning rate *dynamically* as optimization progresses.

This is also the reason for adding a learning rate function `lr` into the `sgd` step function. In the example above any functionality for learning rate scheduling lies dormant as we set the associated `lr` function to be constant, i.e., `lr = (lambda: 1)`.

## Dynamic Learning Rate

Replacing $\eta$ with a time-dependent learning rate $\eta(t)$ adds to the complexity of controlling convergence of an optimization algorithm. In particular, need to figure out how rapidly $\eta$ should decay. If it is too quick, we will stop optimizing prematurely. If we decrease it too slowly, we waste too much time on optimization. There are a few basic strategies that are used in adjusting $\eta$ over time (we will discuss more advanced strategies in a later chapter):

$$
\begin{aligned}
    \eta(t) & = \eta_i \text{ if } t_i \leq t \leq t_{i+1}  && \mathrm{piecewise~constant} \\
    \eta(t) & = \eta_0 \cdot e^{-\lambda t} && \mathrm{exponential} \\
    \eta(t) & = \eta_0 \cdot (\beta t + 1)^{-\alpha} && \mathrm{polynomial}
\end{aligned}
$$

In the first scenario we decrease the learning rate, e.g., whenever progress in optimization has stalled. This is a common strategy for training deep networks. Alternatively we could decrease it much more aggressively by an exponential decay. Unfortunately this leads to premature stopping before the algorithm has converged. A popular choice is polynomial decay with $\alpha = 0.5$. In the case of convex optimization there are a number of proofs which show that this rate is well behaved. Let us see what this looks like in practice.

```{.python .input  n=4}
def exponential():
    global ctr; ctr += 1
    return math.exp(-0.1 * ctr)

ctr = 1; lr = exponential   # set up learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=1000))
```

As expected, the variance in the parameters is significantly reduced. However, this comes at the expense of failing to converge to the optimal solution $\mathbf{x} = (0, 0)$. Even after 1000 steps are we are still very far away from the optimal solution. Indeed, the algorithm fails to converge at all. On the other hand, if we use a polynomial decay where the learning rate decays with the inverse square root of the number of steps convergence is good.

```{.python .input  n=5}
def polynomial():
    global ctr; ctr += 1
    return (1 + 0.1 * ctr)**(-0.5)

ctr = 1; lr = polynomial    # set up learning rate
d2l.show_trace_2d(f, d2l.train_2d(sgd, steps=50))
```

There exist many more choices for how to set the learning rate. For instance, we could start with a small rate, then rapidly ramp up and then decrease it again, albeit more slowly. We could even alternate between smaller and larger learning rates. There exists a large variety of such schedules. For now let us focus on learning rate schedules for which a comprehensive theoretical analysis is possible, i.e., on learning rates in a convex setting. For general nonconvex problems it is very difficult to obtain meaningful convergence guarantees, since in general minimizing nonlinear nonconvex problems is NP hard. For a survey see e.g., the excellent [lecture notes](https://www.stat.cmu.edu/~ryantibs/convexopt-F15/lectures/26-nonconvex.pdf) of Tibshirani 2015.

## Convergence Analysis for Convex Objectives

The following is optional and primarily serves to convey more intuition about the problem. We limit ourselves to one of the simplest proofs, as described by :cite:`Nesterov.Vial.2000`. Significantly more advanced proof techniques exist, e.g., whenever the objective function is particularly well behaved. :cite:`Bartlett.Hazan.Rakhlin.2008` show that for strongly convex functions, i.e., for functions that can be bounded from below by $\mathbf{x}^\top Q \mathbf{x}$, it is possible to minimize them in a small number of steps while decreasing the learning rate like $\eta(t) = \eta_0/(\beta t + 1)$. Unfortunately this case never really occurs in deep learning and we are left with a much more slowly decreasing rate in practice.

Consider the case where
$$w_{t+1} = w_{t} - \eta_t \partial_w l(x_t, w)$$
In particular, assume that $x_t$ is drawn from some distribution $p(x)$ and that $l(x,w)$ is a convex function in $w$ for all $x$. Lastly denote by
$$R(w) = \mathbf{E}_{x \sim p}[l(x,w)]$$
the expected risk and by $R^*$ its minimum with regard to $w$. Lastly let $w^*$ be the minimizer (we assume that it exists within the domain within wich $w$ is defined). In this case we can track the distance between the current parameter $w_t$ and the risk minimizer $w^*$ and see whether it improves over time.

$$\begin{aligned}
    \|w_{t+1} - w^*\|^2 & = \|w_{t} - \eta_t \partial_w l(x_t, w) - w^*\|^2 \\
    & = \|w_{t} - w^*\|^2 + \eta_t^2 \|\partial_w l(x_t, w)\|^2 - 2 \eta_t
    \left\langle w_t - w^*, \partial_w l(x_t, w)\right\rangle
   \end{aligned}
$$
The gradient $\partial_w l(x_t, w)$ can be bounded from above by some Lipschitz constant $L$, hence we have that
$$\eta_t^2 \|\partial_w l(x_t, w)\|^2 \leq \eta_t^2 L^2.$$
We are mostly interested in how the distance between $w_t$ and $w^*$ changes *in expectation*. In fact, for any specific sequence of steps the distance might well increase, depending on whichever $x_t$ we encounter. Hence we need to bound the inner product. By convexity we have that
$$
l(x_t, w^*) \geq l(x_t, w_t) + \left\langle w^* - w_t, \partial_{w} l(x_t, w_t) \right\rangle.
$$
Using both inequalities and plugging it into the above we obtain a bound on the distance between parameters at time $t+1$ as follows:
$$\|w_{t} - w^*\|^2 - \|w_{t+1} - w^*\|^2 \geq 2 \eta_t (l(x_t, w_t) - l(x_t, w^*)) - \eta_t^2 L^2.$$
This means that we make progress as long as the expected difference between current loss and the optimal loss outweights $\eta_t L^2$. Since the former is bound to converge to $0$ it follows that the learning rate $\eta_t$ also needs to vanish.

Next we take expectations over this expression. This yields
$$\mathbf{E}_{w_t}\left[\|w_{t} - w^*\|^2\right] - \mathbf{E}_{w_{t+1}|w_t}\left[\|w_{t+1} - w^*\|^2\right] \geq 2 \eta_t [\mathbf{E}[R[w_t]] - R^*] -  \eta_t^2 L^2.$$
The last step involves summing over the inequalities for $t \in \{t, \ldots, T\}$. Since the sum telescopes and by dropping the lower term we obtain
$$\|w_{0} - w^*\|^2 \geq 2 \sum_{t=1}^T \eta_t [\mathbf{E}[R[w_t]] - R^*] - L^2 \sum_{t=1}^T \eta_t^2.$$
Note that we exploited that $w_0$ is given and thus the expectation can be dropped. Lastly define
$$\bar{w} := \frac{\sum_{t=1}^T \eta_t w_t}{\sum_{t=1}^T \eta_t}.$$
Then by convexity it follows that
$$\sum_t \eta_t \mathbf{E}[R[w_t]] \geq \sum \eta_t \cdot \left[\mathbf{E}[\bar{w}]\right].$$
Plugging this into the above inequality yields the bound
$$
\left[\mathbf{E}[\bar{w}]\right] - R^* \leq \frac{r^2 + L^2 \sum_{t=1}^T \eta_t^2}{2 \sum_{t=1}^T \eta_t}
$$
Here $r^2 := \|w_0 - w^*\|^2$ is a bound on the distance between the initial choice of parameters and the final outcome. In short, the speed of convergence depends on how rapidly the loss function changes via the Lipschitz constant $L$ and how far away from optimality the initial value is $r$. Note that the bound is in terms of $\bar{w}$ rather than $w_T$. This is the case since $\bar{w}$ is a smoothed version of the optimization path. Now let us analyze some choices for $\eta_t$.

* **Known Time Horizon** - whenever $r, L$ and $T$ are known we can pick $\eta = r/L \sqrt{T}$. This yields as upper bound $r L (1 + 1/T)/2\sqrt{T} < rL/\sqrt{T}$. That is, we converge with rate $O(1/\sqrt{T})$ to the optimal solution.
* **Unknown Time Horizon** - whenever we want to have a good solution for *any* time $T$ we can pick $\eta = O(1/\sqrt{T})$. This costs us an extra logarithmic factor and it leads to an upper bound of the form $O(\log T / \sqrt{T})$.

Note that for strongly convex losses $l(x,w') \geq l(x,w) + \langle w'-w, \partial_w l(x,w) \rangle + \frac{\lambda}{2} \|w-w'\|^2$ we can design even more rapidly converging optimization schedules. In fact, an exponential decay in $\eta$ leads to a bound of the form $O(\log T / T)$.

## Stochastic Gradients and Finite Samples

So far we've played a bit fast and loose when it comes to talking about stochastic gradient descent. We posited that we draw instances $x_i$, typically with labels $y_i$ from some distribution $p(x,y)$ and that we use this to update the weights $w$ in some manner. In particular, for a finite sample size we simply argued that the discrete distribution $p(x,y) = \frac{1}{n} \sum_{i=1}^n \delta_{x_i}(x) \delta_{y_i}(y)$ allows us to perform SGD over it.

However, this is not really what we did. In the toy examples in the current section we simply added noise to an otherwise non-stochastic gradient, i.e., we pretended to have pairs $(x_i, y_i)$. It turns out that this is justified here (see the exercises for a detailed discussion). More troubling is that in all previous discussions we clearly didn't do this. Instead we iterated over all instances exactly once. To see why this is preferable consider the converse, namely that we are sampling $n$ observations from the discrete distribution with replacement. The probability of choosing an element $i$ at random is $N^{-1}$. Thus to choose it at least once is

$$\Pr(\mathrm{choose~} i) = 1 - \Pr(\mathrm{omit~} i) = 1 - (1-N^{-1})^N \approx 1-e^{-1} \approx 0.63.$$

A similar reasoning shows that the probability of picking a sample exactly once is given by ${N \choose 1} N^{-1} (1-N^{-1})^{N-1} = \frac{N-1}{N} (1-N^{-1})^{N} \approx e^{-1} \approx 0.37$. This leads to an increased variance and decreased data efficiency relative to sampling without replacement. Hence, in practice we perform the latter (and this is the default choice throughout this book). Lastly note that repeated passes through the dataset traverse it in a *different* random order.


## Summary

* For convex problems we can prove that for a wide choice of learning rates Stochastic Gradient Descent will converge to the optimal solution.
* For deep learning this is generally not the case. However, the analysis of convex problems gives us useful insight into how to approach optimization, namely to reduce the learning rate progressively, albeit not too quickly.
* Problems occur when the learning rate is too small or too large. In practice  a suitable learning rate is often found only after multiple experiments.
* When there are more examples in the training data set, it costs more to compute each iteration for gradient descent, so SGD is preferred in these cases.
* Optimality guarantees for SGD are in general not available in nonconvex cases since the number of local minima that require checking might well be exponential.


## Exercises

1. Experiment with different learning rate schedules for SGD and with different numbers of iterations. In particular, plot the distance from the optimal solution $(0,0)$ as a function of the number of iterations.
1. Prove that for the function $f(x_1, x_2) = x_1^2 + 2 x_2^2$ adding normal noise to the gradient is equivalent to minimizing a loss function $l(\mathbf{x}, \mathbf{w}) = (x_1 - w_1)^2 + 2 (x_2 - w_2)^2$ where $x$ is drawn from a normal distribution.
    * Derive mean and variance of the distribution for $\mathbf{x}$.
    * Show that this property holds in general for objective functions $f(\mathbf{x}) = \frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top Q (\mathbf{x} - \mathbf{\mu})$ for $Q \succeq 0$.
1. Compare convergence of SGD when you sample from $\{(x_1, y_1), \ldots, (x_m, y_m)\}$ with replacement and when you sample without replacement.
1. How would you change the SGD solver if some gradient (or rather some coordinate associated with it) was consistently larger than all other gradients?
1. Assume that $f(x) = x^2 (1 + \sin x)$. How many local minima does $f$ have? Can you change $f$ in such a way that to minimize it one needs to evaluate all local minima?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2372)

![](../img/qr_gd-sgd.svg)
