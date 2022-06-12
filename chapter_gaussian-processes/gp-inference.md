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
$$a_* = k(x_*,X)[K(X,X)+\sigma^2I]^{-1}(\textbf{y}-\mu) + \mu$$
$$v_* = k(x_*,x_*) - K(x_*,X)[K(X,X)+\sigma^2I]^{-1}k(X,x_*)$$

## Interpreting Equations for Learning and Predictions



## Worked Example from Scratch


## Making Life Easy with GPyTorch







