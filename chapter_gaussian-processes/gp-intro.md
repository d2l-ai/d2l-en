# Introduction to Gaussian Processes

In many cases, machine learning amounts to estimating parameters from data. These parameters are often numerous and  relatively uninterpretable --- such as the weights of a neural network. Gaussian processes, by contrast, provide a mechanism for directly reasoning about the high-level properties of functions that could fit our data. For example, we may have a sense of whether the functions that fit our data are quickly varying, periodic, involve conditional independencies, or translation invariance. Gaussian processes enable us to easily incorporate these properties into our model, by directly specifying a Gaussian distribution over the function values that could fit our data. 

Let's get a feel for how Gaussian processes operate, by starting with some examples.

Suppose we observe the following dataset, of regression targets (outputs), y, indexed by inputs, x. As an example, the targets could be changes in carbon dioxide concentrations, and the inputs could be the times at which these targets have been recorded. What are some features of the data? How quickly does it seem to varying? Do we have data points collected at regular intervals, or are there missing inputs? How would you imagine filling in the missing regions, or forecasting up until x=25?

![data](https://user-images.githubusercontent.com/6753639/178247765-650772fb-2622-42d0-8eff-316dc835816f.png)

In order to fit the data with a Gaussian process, we start by specifying a prior distribution over what types of functions we might believe to be reasonable. Here we show several sample functions from a Gaussian process. Does this prior look reasonable? Note here we are not looking for functions that fit our dataset, but instead for specifying reasonable high-level properties of the solutions, such as how quickly they vary with inputs. 

![priorsamp](https://user-images.githubusercontent.com/6753639/178247905-ca6d5812-92eb-45d2-9004-a435da917e78.png)

Once we condition on data, we can use this prior to infer a posterior distribution over functions that could fit the data. Here we show sample posterior functions.

![postsampnomean](https://user-images.githubusercontent.com/6753639/178248696-bb31053e-68c9-4679-b09b-59a319d6479b.png)

We see that each of these functions are entirely consistent with our data, perfectly running through each observation. In order to use these posterior samples to make predictions, we can average the values of every possible sample function from the posterior, to create the curve below, in thick blue. Note that we don't actually have to take an infinite number of samples to compute this expection; as we will see later, we can compute the expectation in closed form.  

![postsamp](https://user-images.githubusercontent.com/6753639/178248173-9d13e613-85f3-4414-ab13-8eb763580225.png)

We may also want a representation of uncertainty, so we know how confident we should be in our predictions. Intuitively, we should have more uncertainty where there is more variability in the sample posterior functions, as this tells us there are many more possible values the true function could take. This type of uncertainty is called _epistemic uncertainty_, which is the _reducible uncertainty_ associated with lack of information. As we acquire more data, this type of uncertainty disappears, as there will be increasingly fewer solutions consistent with what we observe. Like with the posterior mean, we can compute the posterior variance (the variability of these functions in the posterior) in closed form. With shade, we show two times the posterior standard deviation on either side of the mean, creating a _credible interval_ that has a 95% probability of containing the true value of the function for any input $x$.

![postsampcredibleset](https://user-images.githubusercontent.com/6753639/178248952-b14e3d72-e65f-41ed-9577-8d3363c8cf11.png)

The plot looks somewhat cleaner if we remove the posterior samples, simply visualizing the data, posterior mean, and 95% credible set. Notice how the uncertainty grows away from the data, a property of epistemic uncertainty. 

![datafitcredibleset](https://user-images.githubusercontent.com/6753639/178249137-23af70e9-0753-4491-9215-7a757ff60652.png)

The properties of the Gaussian process that we used to fit the data are strongly controlled by what's called a _covariance function_, also known as a _kernel_. The covariance function we used is called the _RBF (Radial Basis Function) kernel_, which has the form
$$ k_{\text{RBF}}(x,x') = \text{cov}(f(x),f(x')) = a^2 \exp\left(-\frac{1}{2\ell^2}||x-x'||^2\right) $$

The _hyperparameters_ of this kernel are interpretable. The _amplitude_ parameter $a$ controls the vertical scale over which the function is varying, and the _length-scale_ parameter
$\ell$
controls the rate of variation (the wiggliness) of the function. Larger $a$ means larger function values, and larger 
$\ell$ 
means more slowly varying functions. Let's see what happens to our sample prior and posterior functions as we vary $a$ and 
$\ell$. 

The _length-scale_ has a particularly pronounced effect on the predictions and uncertainty of a GP. At 
$||x-x'|| = \ell$
, the covariance between a pair of function values is $a^2\exp(-0.5)$. At larger distances than 
$\ell$
, the values of the function values becomes nearly uncorrelated. This means that if we want to make a prediction at a point $x_*$, then function values with inputs $x$ such that 
$||x-x'||>\ell$
will not have a strong effect on our predictions. 

Let's see how changing the lengthscale affects sample prior and posterior functions, and credible sets. The above fits use a length-scale of $2$. Let's now consider 
$\ell = 0.1, 0.5, 2, 5, 10$
. A length-scale of $0.1$ is very small relative to the range of the input domain we are considering, $25$. For example, the values of the function at $x=5$ and $x=10$ will have essentially no correlation at such a length-scale. On the other hand, for a length-scale of $10$, the function values at these inputs will be highly correlated. Note that the vertical scale changes in the following figures.

![priorpoint1](https://user-images.githubusercontent.com/6753639/178250594-d2032bcd-f5bc-4938-8cfa-aa1658c18425.png)
![postpoint1](https://user-images.githubusercontent.com/6753639/178250619-121ad67f-45f4-47ae-9637-c5f367afd211.png)

![priorpoint5](https://user-images.githubusercontent.com/6753639/178250705-1f0ec480-235e-4ad7-a3c6-a282d8d4e60b.png)
![postpoint5](https://user-images.githubusercontent.com/6753639/178250716-9238a419-e43e-405e-b1e3-857790ce52c3.png)

![prior2](https://user-images.githubusercontent.com/6753639/178250738-dd0708de-c008-4708-9a3c-5466b0ac6504.png)
![post2](https://user-images.githubusercontent.com/6753639/178250763-066698cc-4b93-496f-8a01-c2b1f1d6815c.png)

![prior5](https://user-images.githubusercontent.com/6753639/178250780-e5c522b7-f9c7-416c-8017-3cb921ff14b2.png)
![post5](https://user-images.githubusercontent.com/6753639/178250794-89470592-cdb3-4e63-b0d8-d66f002fc593.png)

![prior10](https://user-images.githubusercontent.com/6753639/178250805-080a5c66-69ec-456d-ade5-e0664874782f.png)
![post10](https://user-images.githubusercontent.com/6753639/178250815-588fccee-bfcd-4d46-87af-7429596ddc6e.png)

Notice as the length-scale increases the 'wiggliness' of the functions decrease, and our uncertainty decreases. If the length-scale is small, the uncertainty will quickly increase as we move away from the data, as the datapoints become less informative about the function values. 

Now, let's vary the amplitude parameter, holding the length-scale fixed at $2$. Note the vertical scale is held fixed for the prior samples, and varies for the posterior samples, so you can clearly see both the increasing scale of the function, and the fits to the data.

![priorap1](https://user-images.githubusercontent.com/6753639/178252126-8a984a0c-56f8-409c-b817-68b21af98582.png)
![postapoint1](https://user-images.githubusercontent.com/6753639/178252136-868dd45a-b21e-4311-8164-a60ea41c221c.png)

![priora2](https://user-images.githubusercontent.com/6753639/178252163-c9ac2360-6bee-44fe-985c-731101d8c575.png)
![posta2](https://user-images.githubusercontent.com/6753639/178252195-c325e446-4c61-4851-a841-b547bbab2e2d.png)

![priora4](https://user-images.githubusercontent.com/6753639/178252214-eca6fe2a-0af0-4a13-a71f-4851c02dc4d7.png)
![posta4](https://user-images.githubusercontent.com/6753639/178252232-38e229a9-bf48-4a67-9883-3cf494f8ff6a.png)

![priora8](https://user-images.githubusercontent.com/6753639/178252271-ccabde74-8ec3-44d1-9842-6309444c4ab5.png)
![posta8](https://user-images.githubusercontent.com/6753639/178252284-b59daae3-2648-4ef6-bc09-7c0b4d9a4f02.png)

![priora16](https://user-images.githubusercontent.com/6753639/178252311-b9a5c51f-a0f8-4d65-ba5e-1f8b917c0d7c.png)
![posta16](https://user-images.githubusercontent.com/6753639/178252339-db99413b-78f7-41f7-8c6b-92fe10d634a4.png)

We see the amplitude parameter affects the scale of the function, but not the rate of variation. At this point, we also have the sense that the generalization performance of our procedure will depend on having reasonable values for these hyperparameters. Values of $\ell=2$ and $a=1$ appeared to provide reasonable fits, while some of the other values did not. Fortunately, there is a robust and automatic way to specify these hyperparameters, using what is called the _marginal likelihood_, which we will return to in the notebook on inference. 

So what is a GP, really? As we started, a GP simply says that any collection of function values 
$f(x_1),\dots,f(x_n)$, 
indexed by any collection of inputs 
$x_1,\dots,x_n$ 
has a joint multivariate Gaussian distribution. The mean vector $\mu$ of this distribution is given by a _mean function_, which is typically taken to be a constant or zero. The covariance matrix of this distribution is given by the _kernel_ evaluated at all pairs of the inputs $x$. 

$$
\begin{bmatrix}
f(x) \\ 
f(x_1) \\
\vdots \\ 
f(x_n)
\end{bmatrix}
\sim
\mathcal{N}\left(\mu, 
\begin{bmatrix}
k(x,x) & k(x, x_1) & \dots & k(x,x_n) \\
k(x_1,x) & k(x_1,x_1) & \dots & k(x_1,x_n) \\
\vdots & \vdots & \ddots & \vdots \\
k(x_n, x) & k(x_n, x_1) & \dots & k(x_n,x_n)
\end{bmatrix}
\right)
$$

The above equation specifies a GP prior. We can compute the conditional distribution of $f(x)$ for any $x$ given $f(x_1), \dots, f(x_n)$, the function values we have observed. This conditional distribution is called the _posterior_, and it is what we use to make predictions.

In particular, 
$$f(x) | f(x_1), \dots, f(x_n) \sim \mathcal{N}(m,s^2)$$  where
$$m = k(x,x_{1:n}) k(x_{1:n},x_{1:n})^{-1} f(x_{1:n})$$ 
$$s^2 = k(x,x) - k(x,x_{1:n})k(x_{1:n},x_{1:n})^{-1}k(x,x_{1:n})$$ 
$k(x,x_{1:n})$ is a $1 \times n$ vector formed by evaluating $k(x,x_{i})$ for $i=1,\dots,n$ and $k(x_{1:n},x_{1:n})$ is an $n \times n$ matrix formed by evaluating $k(x_i,x_j)$ for $i,j = 1,\dots,n$. $m$ is what we can use as a point predictor for any $x$, and $s^2$ is what we use for uncertainty: if we want to create an interval with a 95% probability that $f(x)$ is in the interval, we would use $m \pm 2s$. The predictive means and uncertainties for all the above figures were created using these equations. The observed data points were given by 
$f(x_1), \dots, f(x_n)$
and chose a fine grained set of $x$ points to make predictions.

Let's suppose we observe a single datapoint, $f(x_1)$, and we want to determine the value of $f(x)$ at some $x$. Because $f(x)$ is described by a Gaussian process, we know the joint distribution over 
$(f(x), f(x_1))$ 
is Gaussian: 

$$
\begin{bmatrix}
f(x) \\ 
f(x_1) \\
\end{bmatrix}
\sim
\mathcal{N}\left(\mu, 
\begin{bmatrix}
k(x,x) & k(x, x_1) \\
k(x_1,x) & k(x_1,x_1)
\end{bmatrix}
\right)
$$

The off-diagonal expression $k(x,x_1) = k(x_1,x)$ 
tells us how correlated the function values will be --- how strongly determined $f(x)$
will be from $f(x_1)$. 
We've seen already that if we use a large length-scale, relative to the distance between $x$ and $x'$, 
$||x-x'||$, then the function values will be highly correlated. We can visualize the process of determining $f(x')$ from $f(x_1)$ both in the space of functions, and in the joint distribution over $f(x_1, x)$. Let's initially consider an $x$ such that $k(x,x_1) = 0.7$, and $k(x,x)=1$, meaning that the value of $f(x)$ is moderately correlated with the value of $f(x_1)$. In the joint distribution, the contours of constant probability will be relatively narrow ellipses.

Suppose we observe $f(x_1) = 1.2$. 
To condition on this value of $f(x_1)$, 
we can draw a horizontal line at $1.2$ on our plot of the density, and see that the value of $f(x)$ 
is mostly constrained to $[0.8,1.4]$. We have also drawn this plot in function space. 

Now suppose we have a stronger correlation, $k(x,x_1) = 0.9$. 
Now the ellipses have narrowed further, and the value of $f(x)$ 
is even more strongly determined by $f(x_1)$. Drawing a horizontal line at $1.2$, we see the contours for $f(x)$
support values mostly within $[1.15, 1.25]$. 

This procedure can give us a posterior on $f(x)$ for any $x$, for any number of points we have observed. Suppose we observe $f(x_1), f(x_2)$. We now visualize the posterior for $f(x)$ at a particular $x=x'$ in function space. The exact distribution for $f(x)$ is given by the above equations. $f(x)$ is Gaussian distributed, with mean 
$$m = k(x,x_{1:3}) k(x_{1:3},x_{1:3})^{-1} f(x_{1:3})$$
and variance 
$$s^2 = k(x,x) - k(x,x_{1:3})k(x_{1:3},x_{1:3})^{-1}k(x,x_{1:3})$$

(Contour density and function-space plots for the above example are in progress).

In this introductory notebook, we have been considering _noise free_ observations. As we will see, it easy to include observation noise. If we assume that the data are generated from a latent noise free function $f(x)$ plus iid Gaussian noise 
$\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$
with variance $\sigma^2$, then our covariance function simply becomes 
$k(x_i,x_j) \to k(x_i,x_j) + \delta_{ij}\sigma^2$,
where $\delta_{ij} = 1$ if $i=j$ and $0$ otherwise.

We've already started getting some intuition about how we can use a Gaussian process to specify a prior and posterior over solutions, and how the kernel function affects the properties of these solutions. In the following notebooks, we'll precisely show how to specify a Gaussian process prior, introduce and derive various kernel functions, and then go through the mechanics of how to automatically learn kernel hyperparameters, and form a Gaussian process posterior to make predictions. While it takes time and practice to get used to concepts such as a "distributions over functions", the actual mechanics of finding the GP predictive equations is actually quite simple --- making it easy to get practice to form an intuitive understanding of these concepts.
