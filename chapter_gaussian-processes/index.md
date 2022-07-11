# Gaussian Processes
:label:`chap_gp`

Written by Andrew Gordon Wilson, 2022

Gaussian processes (GPs) are ubitiquous. You have already encountered many examples of GPs without realizing it. Any model that is linear in its parameters with a Gaussian distribution over the parameters is a Gaussian process. This class spans discrete models, including random walks, and autoregressive processes, as well as continuous models, including Bayesian linear regression models, as well as polynomials, Fourier series, radial basis functions, and even neural networks with an infinite number of hidden units. There is a running joke that "everything is a special case of a Gaussian process".

Learning about Gaussian processes is important for three reasons: (1) they provide a _function space_ perspective of modelling, which makes understanding a variety of model classes, including deep neural networks, much more approachable; (2) they have an extraordinary range of applications where they are state-of-the-art, including active learning, hyperparameter learning, auto-ML, and spatiotemporal regression; (3) over the last few years, algorithmic advances have made Gaussian processes increasingly scalable and relevant, harmonizing with deep learning through frameworks such as [GPyTorch](https://gpytorch.ai). Indeed, GPs and and deep neural networks are not competing approaches, but highly complementary, and can be combined to great effect.

In this chapter, we introduce Gaussian processes, starting with a familiar weight-space perspective, and then moving directly into function space, which will help us reason about constructing and understanding machine learning models, including deep neural networks. We will then introduce popular covariance functions, also known as _kernels_, which control the generalization properties of a Gaussian process. A GP with a given kernel defines a prior over functions. We will show how to use data to infer a _posterior_, in order to make predictions. We will show to make those predictions scale to large datasets, and how to use Gaussian processes in a number of practical setting, including hyperparameter tuning in deep learning. Our examples will make use of GPyTorch, which makes Gaussian processes scale, and is closely integrated with deep learning functionality and PyTorch.

```toc
:maxdepth: 2

gp-priors
gp-inference
```

