# Exponential Family
:label:`sec_exp_family`

Exponential family plays a crucial role in deep learning optimization. Without
carefully constructing the distribution of exponential family, there won't be 
any objective function constructing by maximum likelihood estimation 
:numref:`sec_maximum_likelihood`, nor optimizing the objection. The objection functions of linear 
regression in :numref:`sec_linear_regression` and softmax regression in 
:numref:`sec_softmax_regression` are two commonly used examples for  exponential family. 
In this section, we will dive into the fundamental of exponential family.


The *exponential family* is a set of distributions which can be expressed in 
the below form:

$$\mathbf{x} \sim exp \big{(} \mathbf{\theta}, T(\cdot) \big{)}$$

where 
- $\mathbf{x} = (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ is a random vector;
- $\mathbf{\theta} = (\theta_1, \theta_2, ..., \theta_l) \in \mathbb{R}^l$ 
is called as *natural parameters* (or *canonical parameters*).


To be more concrete, the probability density function of the exponential 
family can be written in as follows:

$$f_{\theta}(\mathbf{x}) = h(\mathbf{x}) \cdot g(\theta) \cdot exp \big{(} 
\theta \cdot T\mathbf(x) \big{)}$$
:eqlabel:`eq_exp_pdf`



where the three functions of $\mathbf{x}$ and $\mathbf{\theta}$ in the 
$f_{\theta}(\mathbf{x})$ are known:
- $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ 
is called as the *sufficient statistics* for $\theta$. That is, the 
information represented by $T(\mathbf{x})$ is sufficient to calculate the 
parameter $\theta$, no other information from the sample $\mathbf{x}$'s is 
needed for estimating
$\theta$. For example, the sample mean is the sufficient statistics for the 
true mean parameter ($\theta$) of a normal distribution with known variance;
- $h(\mathbf{x})$ is known as the *underlying measure* or the *base measure*;
- $g(\theta)$ is referred to as the *log normalizer*, which ensures that the 
above distribution $f_{\theta}(\mathbf{x})$ integrates to one.


## The Concaveness

You may curious that why exponential family is so important to optimization 
in deep learning? Let's unveil the black box through a math trick -- by taking 
the logarithm of the exponential family:

$$log \big{(} f_{\theta}(\mathbf{x}) \big{)} = log \big{(} h(\mathbf{x})  
\big{)} \cdot \theta \cdot T\mathbf(x) .$$

If you dig into the math and calculate the gradients of the above function, 
you may suddenly realize that $log \big{(} f_{\theta}(\mathbf{x}) \big{)}$ is 
a concave function! Why the concaveness helps here? Because it enable us to 
maximize the likelihood function numerically over all the possible values of 
the parameter $\theta$ and ultimately reach to the unique global maximum 
(or global minimum if we reverse the sign of the loss function). As a result, 
the neural network is able to optimize towards the best sets of weights
using stochastic gradient descent.


## Examples of Exponential Family

Now, let's take a look of some distributions in :numref:`sec_distributions`
and demonstrate that their ties of blood to the exponential family. Supposed we have a random variable $X$ for the before sections.

### Bernoulli

As we have seen in :numref:`sec_distributions`, if $X$ follows a Bernoulli distribution with a probability $p$ (i.e., $X \sim \mathrm{Bernoulli}(p)$), then the probability density function of $X$ is:

$$
\begin{aligned}
p(x | p) &= p^x \cdot (1-p)^{1-x} \\
&= exp \big{\{} \log \big{(} p^x \cdot (1-p)^{1-x} \big{)} \\
&= exp \big{\{} x \log p + (1-x) \log (1-p) \big{\}}\\
&= (1-p) \cdot exp \big{\{} x \log \frac{p}{1-p} \big{\}}.\\
\end{aligned}
$$

Hence, the parameters of the exponential family format in :eqref:`eq_exp_pdf` can be explained as:

* $\theta = \log \frac{p}{1-p}$, 
* $T(x) = x$,
* $h(x) = 1$,
* and $g(\theta) = 1-p$.

### Poisson

If $X$ follows a Poisson distribution (in :numref:`sec_distributions`) with a rate or shape parameter $\lambda$ (i.e., $X \sim \mathrm{Poisson}(\lambda)$), then the probability density function of $X$ is:

$$
\begin{aligned}
p(x | \lambda) &= \frac{\lambda^x e^{-\lambda}}{x!} \\
&= \frac{1}{x!} e^{-\lambda} exp\big{\{} x \log \lambda \big{\}}.\\
\end{aligned}
$$


Hence, the parameters in :eqref:`eq_exp_pdf` can be explained as:

* $\theta = \log \lambda$, 
* $T(x) = x$,
* $h(x) = \frac{1}{x!}$,
* and $g(\theta) = e^{-\lambda} = e^{-e^\theta}$.


### Gaussian

From :numref:`sec_distributions`, if $X$ follows a Gaussian distribution with mean $\mu$ and standard deviation $\sigma$ (i.e., $X \sim \mathcal{N}(\mu, \sigma^2)$), then the probability density function of $X$ is:

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} exp \big{\{} \frac{-(x-\mu)^2}{2 \sigma^2} \big{\}} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \frac{1}{\sigma } e^{- \frac{1}{2 \sigma^2} \mu^2 } \cdot exp \big{\{} \frac{\mu}{\sigma^2}x - \frac{1}{2 \sigma^2} x^2 \big{\}} .
\end{aligned}
$$

Hence, the parameters in :eqref:`eq_exp_pdf` can be explained as:

* $\theta = \big{[} \frac{\mu}{\sigma^2} , -\frac{1}{2 \sigma^2} \big{]}$, 
* $T(x) = \begin{bmatrix}x\\x^2\end{bmatrix}$,
* $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* and $g(\theta) = \frac{1}{\sigma } e^{- \frac{1}{2 \sigma^2} \mu^2 }$.


Besides the above distributions, Binomial, Multinomial, Exponential and other commonly seen distributions are members of exponential family as well.



## Generalized Linear Model

One modeling technique that laid its foundation on exponential family is the generalized linear model. Fundamentally, the *generalized linear model (GLM)* is a generalization from linear regression to more general exponential family. As we explained in :numref:`sec_linear_regression`, the response variable of linear regression, $\mathbf{\hat{y}} = \mathbf{w}^T \mathbf{x}$, was assumed to vary linearly as its inputs $X$ varies. However, this implementation seems to be inappropriate for lots of real-life applications. For example, in :numref:`sec_kaggle_house`, a pure linear regression model may predict a negative price, while the realistic house price cannot be a negative number. To model the response variable $\mathbf{\hat{Y}}$ with more widely used distributions, generalized linear model formulates the problem with the following assumptions:


1. The conditional mean of response variable $\mathbf{\hat{y}}$, is denoted as a function of the linear regression of inputs $\mathbf{x}$:
$$E[y|x]= \mu = g^{-1}(\mathbf{w}^T \mathbf{x}).$$ Here the function $g$ is referred as a *link function*.

1. The response variable $\mathbf{\hat{y}}$ is drawn from an exponential family distribution with conditional mean $\mu$.











1. A *link function* $g$ which connects the parameter $\mathbf{\theta}$ with a linear predictor $\mathbf{X}\mathbf{W}$, i.e., $g(\mathbf{\theta}) = \mathbf{X}\mathbf{W}$;
1. An exponential family distribution, so that $\mathbf{Y} \sim exp \big{(} \mathbf{\theta}, T(\cdot) \big{)}$



Figure 3 illustrates the graphical model representation of a generalized linear model. The model is based on the following assumptions:




## Summary

* Exponential family plays a crucial role in constructing objective functions of deep learning.
* Linear regression and softmax regression both belong to exponential family.
* The log-likelihood of exponential family distribution is concave.
* 


## Exercises

1. Can you try to list other examples of exponential family from its mathematical definition?
1.

```{.python .input}

```
