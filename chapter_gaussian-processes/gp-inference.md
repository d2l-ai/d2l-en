In this section, we will show how to perform posterior inference and make predictions using the GP priors we introduced in the last section. We will start with regression, where we can perform inference in _closed form_. We will then consider situations where approximate inference is required --- classification, point processes, or any non-Gaussian likelihoods. This is a "GPs in a nutshell" section to quickly get up and running with Gaussian processes in practice. We'll start coding all the basic operations from scratch, and then introduce _GPyTorch_, which will make working with state-of-the-art Gaussian processes and integration with deep neural networks much more convenient. 

## Posterior Inference for Regression

An _observation_ model relates the function we want to learn, $f(x)$, to our observations $y(x)$, both indexed by some input $x$. In classification, $x$ could be the pixels of an image, and $y$ could be the associated class label. In regression, $y$ typically represents a continuous output, such as a land surface temperature, a sea-level, a $CO_2$ concentration, etc.  

In regression, we often assume the outputs are given by a latent noise-free function $f(x)$ plus i.i.d. Gaussian noise $\epsilon(x)$: $y(x) = f(x) + \epsilon(x)$, with $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$. Let $\mathbf{y} = (y(x_1),\dots,y(x_n))^{\top}$ be a vector of our training observations, and $\textbf{f} = (f(x_1),\dots,f(x_n))^{\top}$ be a vector of the latent noise-free function values queried at the training inputs $X = {x_1, \dots, x_n}$. Suppose we want to make predictions at a set of inputs $X_* = x_{*1}, x_{*2},\dots, x_{*m}$. 



