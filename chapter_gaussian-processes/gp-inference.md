In this section, we will show how to perform posterior inference and make predictions using the GP priors we introduced in the last section. We will start with regression, where we can perform inference in _closed form_. We will then consider situations where approximate inference is required --- classification, point processes, or any non-Gaussian likelihoods. This is the section to read to quickly get "up and running" with Gaussian processes.

## Posterior Inference for Regression

An _observation_ model relates the function we want to learn, $f(x)$, to our observations $y(x)$, both indexed by some input $x$. In classification, $x$ could be the pixels of an image, and $y$ could be the associated class label. In regression, $y$ typically represents a continuous output, such as a land surface temperature, a sea-level, a $CO_2$ concentration, etc.  

In regression, we often assume the outputs are given by a latent noise-free function $f(x)$ plus i.i.d. Gaussian noise $\epsilon(x)$: $y(x) = f(x) + \epsilon(x)$, with $\epsilon(x) \sim \mathcal{N}(0,\sigma^2)$. 



