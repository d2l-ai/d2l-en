# Exponential Family
:label:`sec_exp_family`

Exponential family plays a crucial role in deep learning optimization. Without
carefully constructing the distribution of exponential family, there won't be 
any objective function approximation using maximum likelihood estimation 
(:numref:sec_maximum_likelihood), nor optimizing the objection. Linear 
regression in (:numref:sec_linear_regression) and softmax regression in 
(:numref:sec_softmax_regression) are two commonly used examples which 
objection functions can be classified into the exponential family. 
In this section, we will dive into the fundamental of exponential family.


An *exponential family* is a set of distributions which can be expressed in 
the below form:

$$\mathbf{x} \sim exp \big{(} \mathbf{\theta}, T(\cdot) \big{)}$$

where 
- $\mathbf{x} = (x_1, x_2, ..., x_n) \in \mathbb{R}^n$ is a random vector;
- $\mathbf{\theta} = (\theta_1, \theta_2, ..., \theta_l) \in \mathbb{R}^l$ is 
a vector of natural parameters (or canonical parameters).


To be more concrete, the p.d.f. of the exponential family can be written in as 
                            follows:

$$f_{\theta}(\mathbf{x}) = h(\mathbf{x}) \cdot g(\theta) \cdot exp \big{(} 
\theta \cdot T\mathbf(x) \big{)}$$



where the four functions of $\mathbf{x}$ and $\mathbf{\theta}$ in the 
$f_{\theta}(\mathbf{x})$ are known:
- $T(\mathbf{x})= (T_1(\mathbf{x}), T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ 
is called as the *sufficient statistics* for $\theta$. More concretely,  the 
information 
represented by $T(\mathbf{x})$ is sufficient to calculate the parameter 
$\theta$, no other information from the sample $\mathbf{x}$'s is needed. 
For example, the sample mean is the sufficient statistics for the true mean 
parameter ($\theta$) of a normal distribution with known variance;
- $h(\mathbf{x})$ is known as the *underlying measure* or the *base measure*;
- $g(\theta)$ is referred to as the *log normalizer*, which ensures that the 
above distribution $f_{\theta}(\mathbf{x})$ integrates to one.

```{.python .input}

```

```{.python .input}

```

![](./exponential_family.png)

```{.python .input}

```
