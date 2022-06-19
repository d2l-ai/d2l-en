In this section, we will show how to perform posterior inference and make predictions using the GP priors we introduced in the last section. We will start with regression, where we can perform inference in _closed form_. We will then consider situations where approximate inference is required --- classification, point processes, or any non-Gaussian likelihoods. This is a "GPs in a nutshell" section to quickly get up and running with Gaussian processes in practice. We'll start coding all the basic operations from scratch, and then introduce _GPyTorch_, which will make working with state-of-the-art Gaussian processes and integration with deep neural networks much more convenient. 

## Posterior Inference for Regression

An _observation_ model relates the function we want to learn, $f(x)$, to our observations $y(x)$, both indexed by some input $x$. In classification, $x$ could be the pixels of an image, and $y$ could be the associated class label. In regression, $y$ typically represents a continuous output, such as a land surface temperature, a sea-level, a $CO_2$ concentration, etc.  

In regression, we often assume the outputs are given by a latent noise-free function $f(x)$ plus i.i.d. Gaussian noise $\epsilon(x)$: 

$y(x) = f(x) + \epsilon(x)$, (1) 

with $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$. Let $\mathbf{y} = y(X) = (y(x_1),\dots,y(x_n))^{\top}$ be a vector of our training observations, and $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ be a vector of the latent noise-free function values, queried at the training inputs $X = {x_1, \dots, x_n}$.

We will assume $f(x) \sim \mathcal{GP}(m,k)$, which means that any collection fo function values \textbf{f} has a joint multivariate Gaussian distribution, with mean vector $\mu_i = m(x_i)$ and covariance matrix $K_{ij} = k(x_i,x_j)$. The RBF kernel $k(x_i,x_j) = a^2 \exp\left(-\frac{1}{2\ell^2}||x_i-x_j||^2\right)$ would be a standard choice of covariance function. For notational simplicity, we'll assume the mean function $m(x)=0$; our derivations can easily be generalized later on.

Suppose we want to make predictions at a set of inputs $$X_* = x_{*1},x_{*2},\dots,x_{*m}$$. Then we want to find $x^2$ and $p(\mathbf{f}_* | \mathbf{y}, X)$. In the regression setting, we can conveniently find this distribution by using Gaussian identities, after finding the joint distribution over $\mathbf{f}_* = f(X_*)$ and $\mathbf{y}$. 

If we evaluate equation (1) at the training inputs $X$, we have $\mathbf{y} = \mathbf{f} + \mathbf{\epsilon}$. By the definition of a Gaussian process (see last section), $\mathbf{f} \sim \mathcal{N}(0,K(X,X))$ where $K(X,X)$ is an $n \times n$ matrix formed by evaluating our covariance function (aka _kernel_) at all possible pairs of inputs $x_i, x_j \in X$. $\mathbf{\epsilon}$ is simply a vector comprised of iid samples from $\mathcal{N}(0,\sigma^2)$ and thus has distribution $\mathcal{N}(0,\sigma^2I)$. $\mathbf{y}$ is therefore a sum of two independent multivariate Gaussian variables, and thus has distribution $\mathcal{N}(0, K(X,X) + \sigma^2I)$. One can also show that $\text{cov}(\mathbf{f}_*, \mathbf{y}) = \text{cov}(\mathbf{y},\mathbf{f}_*)^{\top} = K(X_*,X)$ where $K(X_*,X)$ is an $m \times n$ matrix formed by evaluating the kernel at all pairs of test and training inputs. 

$$
\begin{bmatrix}
\mathbf{y} \\
\mathbf{f}_*
\end{bmatrix}
\sim
\mathcal{N}\left(0, 
\mathbf{A} = \begin{bmatrix}
K(X,X)+\sigma^2I & K(X,X_\*) \\
K(X_\*,X) & K(X_\*,X_\*)
\end{bmatrix}
\right)
$$

We can then use standard Gaussian identities to find the conditional distribution from the joint distribution (see, e.g., Bishop Chapter 2), 
$\mathbf{f}_* | \mathbf{y}, X, X_* \sim \mathcal{N}(m_*,S_*)$, where $m_* = K(X_*,X)[K(X,X)+\sigma^2I]^{-1}\textbf{y}$, and $S = K(X_*,X_*) - K(X_*,X)[K(X,X)+\sigma^2I]^{-1}K(X,X_*)$.

Typically, we do not need to make use of the full predictive covariance matrix $S$, and instead use the diagonal of $S$ for uncertainty about each prediction. Often for this reason we write the predictive distribution for a single test point $x_*$, rather than a collection of test points. 

The kernel matrix has parameters $\theta$ that we also wish to estimate, such the amplitude $a$ and lengthscale $\ell$ of the RBF kernel above. For these purposes we use the _marginal likelihood_, $p(\textbf{y} | \theta, X)$, which we already derived in working out the marginal distributions to find the joint distribution over $\textbf{y},\textbf{f}_*$. As we will see, the marginal likelihood compartmentalizes into model fit and model complexity terms, and automatically encodes a notion of Occam's razor for learning hyperparameters. For a full discussion, see MacKay Ch. 28, and Rasmussen and Williams Ch. 5 (! Add references).

## Equations for Making Predictions and Learning Kernel Hyperparameters in GP Regression

We list here the equations you will use for learning hyperparameters and making predictions in Gaussian process regression. Again, we assume a vector of regression targets $\textbf{y}$, indexed by inputs $X = \{x_1,\dots,x_n\}$, and we wish to make a prediction at a test input $x_*$. We assume i.i.d. additive zero-mean Gaussian noise with variance $\sigma^2$. We use a Gaussian process prior $f(x) \sim \mathcal{GP}(m,k)$ for the latent noise-free function, with mean function $m$ and kernel function $k$. The kernel itself has parameters $\theta$ that we want to learn. For example, if we use an RBF kernel, $k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$, we want to learn $\theta = \{a^2, \ell^2\}$. Let $K(X,X)$ represent an $n \times n$ matrix corresponding to evaluating the kernel for all possible pairs of $n$ training inputs. Let $K(x_*,X)$ represent a $1 \times n$ vector formed by evaluating $k(x_*, x_i)$, $i=1,\dots,n$. Let $\mu$ be a mean vector formed by evaluating the mean function $m(x)$ at every training points $x$.

Typically in working with Gaussian processes, we follow a two-step procedure. 
1. Learn kernel hyperparameters $\hat{\theta}$ by maximizing the marginal likelihood with respect to these hyperparameters.
2. Use the predictive mean as a point predictor, and 2 times the predictive standard deviation to form a 95\% credible set, conditioning on these learned hyperparameters $\hat{\theta}$.

The log marginal likelihood is simply a log Gaussian density, which has the form:
$$\log p(\textbf{y} | \theta, X) = -\frac{1}{2}\textbf{y}^{\top}[K_{\theta}(X,X) + \sigma^2I]^{-1}\textbf{y} - \frac{1}{2}\log|K_{\theta}(X,X)| + c$$

The predictive distribution has the form:
$$p(y_* | x_*, \textbf{y}, \theta) = \mathcal{N}(a_*,v_*)$$
$$a_* = k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k_{\theta}(x_*,x_*) - K_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}k_{\theta}(X,x_*)$$

## Interpreting Equations for Learning and Predictions

There are some key points to note about the predictive distributions for Gaussian processes:
- Despite the flexibility of the model class, it is possible to do _exact_ Bayesian inference for GP regression in _closed form_. Aside from learning the kernel hyperparameters, there is no _training_. We can write down exactly what equations we want to use to make predictions. Gaussian processes are relatively exceptional in this respect, and it has greatly contributed to their convenience, versatility, and continued popularity. 
- The predictive mean $a_*$ is a linear combination of the training targets $\textbf{y}$, weighted by the kernel $k_{\theta}(x_*,X)[K_{\theta}(X,X)+\sigma^2I]^{-1}$. As we will see, the kernel (and its hyperparameters) thus plays a crucial role in the generalization properties of the model.
- The predictive mean explicitly depends on the target values $\textbf{y}$ but the predictive variance does not. The predictive uncertainty instead grows as the test input $x_*$ moves away from the target locations $X$, as governed by the kernel function. However, uncertainty will implicitly depend on the values of the targets $\textbf{y}$ through the kernel hyperparameters $\theta$, which are learned from the data.
- The marginal likelihood compartmentalizes into model fit and model complexity (log determinant) terms. The marginal likelihood tends to select for hyperparameters that provide the simplest fits that are still consistent with the data. 
- The key computational bottlenecks come from solving a linear system and computing a log determinant over an $n \times n$ symmetric positive definite matrix $K(X,X)$ for $n$ training points. Naively, these operations each incur $\mathcal{O}(n^3)$ computations, as well as $\mathcal{O}(n^2)$ storage for each entry of the kernel (covariance) matrix, often starting with a Cholesky decomposition. Historically, these bottlenecks have limited GPs to problems with fewer than about 10,000 training points, and have given GPs a reputation for ``being slow'' that has been inaccurate now for almost a decade. In advanced topics, we will discuss how GPs can be scaled to problems with millions of points.
- For popular choices of kernel functions, $K(X,X)$ is often close to singular, which can cause numerical issues when performing Cholesky decompositions or other operations intended to solve linear systems. Fortunately, in regression we are often working with $K_{\theta}(X,X)+\sigma^2I$, such that the noise variance $\sigma^2$ gets added to the diagonal of $K(X,X)$, significantly improving its conditioning. If the noise variance is small, or we are doing noise free regression, it is common practice to add a small amount of "jitter" to the diagonal, on the order of $10^{-6}$, to improve conditioning.


## Worked Example from Scratch

Let's create some regression data, and then fit the data with a GP, implementing every step from scratch. 
We'll sample data from 
$$y(x) = sin(x) + \frac{1}{2}sin(4x) + \epsilon$$, with \epsilon \sim \mathcal{N}(0,\sigma^2). The noise free function we wish to find is $f(x) = sin(x) + \frac{1}{2}sin(4x)$. We'll start by using a noise standard deviation $\sigma$ = 0.25. 


```{.python .input}
import numpy as np
import matplotlib.pyplot as plt

def data_maker1(x, sig):
    return np.sin(x) + 0.5 * np.sin(4*x) + np.random.randn(x.shape[0])*sig

sig = 0.25
train_x = np.linspace(0, 5, 50)
test_x = np.linspace(0, 5, 500)

train_y = data_maker1(train_x, sig=sig)
test_y = data_maker1(test_x, sig=0.)

plt.scatter(train_x, train_y)
plt.plot(test_x, test_y)
plt.xlabel("x",fontsize=20)
plt.ylabel("Observations y",fontsize=20)
plt.show()

```

![datafit](https://user-images.githubusercontent.com/6753639/174501277-f0b9ee48-cc29-4ba9-b325-3e08844c7ed3.png)

Here we see the noisy observations as circles, and the noise-free function in blue that we wish to find. 

Now, let's specify a GP prior over the latent noise-free function, $f(x)\sim \mathcal{GP}(m,k)$. We'll use a mean function $m(x) = 0$, and an RBF covariance function (kernel)
$$k(x_i,x_j) = a^2\exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right)$$.

```{.python .input}
from scipy.spatial import distance_matrix
mean = np.zeros(test_x.shape[0])
cov = kernel(test_x, test_x, ls=0.2)
```

We've started with a length-scale of 0.2. Before we fit the data, it's important to consider whether we've specified a reasonable prior. Let's visualize some sample functions from this prior, as well as the 95\% credible set (we believe there's a 95\% chance that the true function is within this region).

```{.python .input}
prior_samples = np.random.multivariate_normal(mean=mean, cov=cov, size=5)
plt.plot(test_x, prior_samples.T, color='black', alpha=0.5)
plt.plot(test_x, mean, linewidth=2.)
plt.fill_between(test_x, mean - 2*np.diag(cov), mean + 2*np.diag(cov), alpha=0.25)
plt.show()
```

![priorsamples](https://user-images.githubusercontent.com/6753639/174501312-7d8bbdc7-16fb-44f8-9458-05115b673b74.png)


Do these samples look reasonable? Are the high-level properties of the functions aligned with the type of data we are trying to model?

Now let's form the mean and variance of the posterior predictive distribution at any arbitrary test point $x_*$.

$$
\bar{f}_{*} = K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}y
$$

$$
V(f_{*}) = K(x_*, x_*) - K(x, x_*)^T (K(x, x) + \sigma^2 I)^{-1}K(x, x_*)
$$

Before we make predictions, we should learn our kernel hyperparameters $\theta$ and noise variance $\sigma^2$. Let's initialize our length-scale at 0.75, as our prior functions looked too quickly varying compared to the data we are fitting. We'll also guess a noise standard deviation $\sigma$ of 0.75. 

In order to learn these parameters, we'll maximize the marginal likelihood with respect to these parameters.

$$
\log p(y | X) = \log \int p(y | f, X)p(f | X)df
$$
$$
\log p(y | X) = -\frac{1}{2}y^T(K(x, x) + \sigma^2 I)^{-1}y - \frac{1}{2}\log |K(x, x) + \sigma^2 I| - \frac{n}{2}\log 2\pi
$$


Perhaps our prior functions were too quickly varying. Let's guess a length-scale of 0.4. We'll also guess a noise standard deviation of 0.75. These are simply hyperparameter initializations --- we'll learn these parameters from the marginal likelihood.

```{.python .input}
ell_est = 0.4
post_sig_est = 0.5

def neg_MLL(pars):
    K = kernel(train_x, train_x, ls=pars[0])
    kernel_term = -0.5 * train_y @ np.linalg.inv(K + pars[1]**2 * np.eye(train_x.shape[0])) @ train_y
    logdet = -0.5 * np.log(np.linalg.det(K + pars[1]**2*np.eye(train_x.shape[0])))
    const = -train_x.shape[0]/2. * np.log(2 * np.pi)
    
    return -(kernel_term + logdet + const)

from scipy import optimize
learned_hypers = optimize.minimize(neg_MLL, x0 = np.array([ell_est,post_sig_est]), bounds=((0.01, 10.), (0.01, 10.)))
```

In this instance, we learn a length-scale of 0.299, and a noise standard deviation of 0.24. Note that the learned noise is extremely close to the true noise, which helps indicate that our GP is a very well-specified to this problem. 

In general, it is crucial to put careful thought into selecting the kernel and initializing the hyperparameters. However, marginal likelihood learning of some hyperparameters such as length-scale and noise variance can be relatively robust to initialization. If we instead try the (very poor) initialization of $\ell = 4$ and $\sigma = 4$, we still converge to the same hyperparameters. 

```{.python .input}
learned_hypers = optimize.minimize(neg_MLL, x0 = np.array([4,4]), bounds=((0.01, 10.), (0.01, 10.)))
ell = learned_hypers.x[0]
post_sig_est = learned_hypers.x[1] 
```

Now, let's make predictions with these learned hypers.

```{.python .input}
K_x_xstar = kernel(train_x, test_x, ls=ell)
K_x_x = kernel(train_x, train_x, ls=ell)
K_xstar_xstar = kernel(test_x, test_x, ls=ell)

post_mean = K_x_xstar.T @ np.linalg.inv((K_x_x + post_sig_est**2 * np.eye(train_x.shape[0]))) @ train_y
post_cov = K_xstar_xstar - K_x_xstar.T @ np.linalg.inv((K_x_x + post_sig_est**2 * np.eye(train_x.shape[0]))) @ K_x_xstar

lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov))
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov))

plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.plot(test_x, post_mean, linewidth=2.)
plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.show()
```

![gpfit](https://user-images.githubusercontent.com/6753639/174501329-df29e79b-c342-4efd-8f9b-1739afdfbafc.png)

We see the posterior mean in orange almost perfectly matches the true noise free function! Note that the 95\% credible set we are showing is for the latent _noise free_ function, and not the data points. We see that this credible set entirely contains the true function, and does not seem overly wide or narrow. We would not want nor expect it to contain the data points. If we wish to have a credible set for the observations, we should compute 

```{.python .input}
lw_bd = post_mean - 2 * np.sqrt(np.diag(post_cov) + post_sig_est**2)
up_bd = post_mean + 2 * np.sqrt(np.diag(post_cov) + post_sig_est**2)
```

There are two sources of uncertainty, _epistemic_ uncertainty, representing _reducible_ uncertainty, and _aleatoric_ or _irreducible_ uncertainty. The _epistemic_ uncertainty here represents uncertainty about the true values of the noise free function. This uncertainty should grow as we move away from the data points, as away from the data there are a greater variety of function values consistent with our data. As we observe more and more data, our beliefs about the true function become more confident, and the epistemic uncertainty disappears. The _aleatoric_ uncertainty in this instance is the observation noise, since the data are given to us with this noise, and it cannot be reduced.

The _epistemic_ uncertainty in the data is captured by variance of the latent noise free function np.diag(post\_cov). The _aleatoric_ uncertainty is captured by the noise variance post_sig_est**2. 

Unfortunately, people are often careless about how they represent uncertainty, with many papers showing error bars that are completely undefined, no clear sense of whether we are visualizing epistemic or aleatoric uncertainty or both, and confusing noise variances with noise standard deviations, standard deviations with standard errors, confidence intervals with credible sets, and so on. Without being precise about what the uncertainty represents, it is essentially meaningless. 

In the spirit of playing close attention to what our uncertainty represents, it's crucial to note that we are taking _two times_ the _square root_ of our variance estimate for the noise free function. Since our predictive distribution is Gaussian, this quantity enables us to form a 95\% credible set, representing our beliefs about the interval which is 95\% likely to contain the ground truth function. The noise _variance_ is living on a completely different scale, and is much less interpretable. 

Finally, let's take a look at 20 posterior samples. These samples tell us what types of functions we believe might fit our data, a posteriori. 

```{.python .input}
post_samples = np.random.multivariate_normal(post_mean, post_cov, size=20)
plt.scatter(train_x, train_y)
plt.plot(test_x, test_y, linewidth=2.)
plt.plot(test_x, post_mean, linewidth=2.)
plt.plot(test_x, post_samples.T, color='gray', alpha=0.25)
plt.fill_between(test_x, lw_bd, up_bd, alpha=0.25)
plt.show()

```

![posteriorsamples](https://user-images.githubusercontent.com/6753639/174501353-4c2a8c44-2c6e-44c7-a92f-4ece68fa784d.png)

In basic regression applications, it's most common to use the posterior predictive mean and standard deviation as a point predictor and metric for uncertainty, respectively. In more advanced applications, such as Bayesian optimization with Monte Carlo acquisition functions, or Gaussian processes for model-based RL, it often necessary to take posterior samples. However, even if not strictly required in the basic applications, these samples give us more intuition about the fit we have for the data, and are often useful to include in visualizations. 

## Making Life Easy with GPyTorch

As we have seen, it is actually pretty easy to implement basic Gaussian process regression entirely from scratch. However, as soon as we want to explore a variety of kernel choices, consider approximate inference (which is needed even for classification), combine GPs with neural networks, or even have a dataset larger than about 10,000 points, then an implementation from scratch becomes unwieldy and cumbersome. Some of the most effective methods for scalable GP inference, such as SKI (also known as KISS-GP), can require hundreds of lines of code implementing advanced numerical linear algebra routines. 

In these cases, the _GPyTorch_ library will make our lives a lot easier. We'll be discussing GPyTorch much more in the next section on advanced methods. But to get a feel for the package, let's reproduce our results above using GPyTorch.



## Exercises

Making predictions without the learned hypers
Marginal likelihood is not a convex objective... 
Epistemic vs aleatoric uncertainty



