# Documentation

Due to constraints on the length of this book, we cannot possibly introduce every single MXNet function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the MXNet API.

## Finding All the Functions and Classes in a Module

In order to know which functions and classes can be called in a module, we
invoke the `dir` function. For instance, we can query all properties in the
module for generating random numbers:

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_mx_nd_np', 'absolute_import', 'choice', 'multinomial', 'normal', 'rand', 'randint', 'shuffle', 'uniform']\n"
 }
]
```

```{.python .input  n=2}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['AbsTransform', 'AffineTransform', 'Bernoulli', 'Beta', 'Binomial', 'CatTransform', 'Categorical', 'Cauchy', 'Chi2', 'ComposeTransform', 'ContinuousBernoulli', 'Dirichlet', 'Distribution', 'ExpTransform', 'Exponential', 'ExponentialFamily', 'FisherSnedecor', 'Gamma', 'Geometric', 'Gumbel', 'HalfCauchy', 'HalfNormal', 'Independent', 'Laplace', 'LogNormal', 'LogisticNormal', 'LowRankMultivariateNormal', 'LowerCholeskyTransform', 'MixtureSameFamily', 'Multinomial', 'MultivariateNormal', 'NegativeBinomial', 'Normal', 'OneHotCategorical', 'Pareto', 'Poisson', 'PowerTransform', 'RelaxedBernoulli', 'RelaxedOneHotCategorical', 'SigmoidTransform', 'SoftmaxTransform', 'StackTransform', 'StickBreakingTransform', 'StudentT', 'TanhTransform', 'Transform', 'TransformedDistribution', 'Uniform', 'VonMises', 'Weibull', '__all__', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', 'bernoulli', 'beta', 'biject_to', 'binomial', 'categorical', 'cauchy', 'chi2', 'constraint_registry', 'constraints', 'continuous_bernoulli', 'dirichlet', 'distribution', 'exp_family', 'exponential', 'fishersnedecor', 'gamma', 'geometric', 'gumbel', 'half_cauchy', 'half_normal', 'identity_transform', 'independent', 'kl', 'kl_divergence', 'laplace', 'log_normal', 'logistic_normal', 'lowrank_multivariate_normal', 'mixture_same_family', 'multinomial', 'multivariate_normal', 'negative_binomial', 'normal', 'one_hot_categorical', 'pareto', 'poisson', 'register_kl', 'relaxed_bernoulli', 'relaxed_categorical', 'studentT', 'transform_to', 'transformed_distribution', 'transforms', 'uniform', 'utils', 'von_mises', 'weibull']\n"
 }
]
```

```{.python .input  n=3}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['Algorithm', 'Generator', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__path__', '__spec__', '_sys', 'all_candidate_sampler', 'categorical', 'create_rng_state', 'experimental', 'fixed_unigram_candidate_sampler', 'gamma', 'get_global_generator', 'learned_unigram_candidate_sampler', 'log_uniform_candidate_sampler', 'normal', 'poisson', 'set_global_generator', 'set_seed', 'shuffle', 'stateless_binomial', 'stateless_categorical', 'stateless_gamma', 'stateless_normal', 'stateless_poisson', 'stateless_truncated_normal', 'stateless_uniform', 'truncated_normal', 'uniform', 'uniform_candidate_sampler']\n"
 }
]
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).

## Finding the Usage of Specific Functions and Classes

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let us explore the usage instructions for tensor's `ones` function.

```{.python .input  n=4}
help(np.ones)
```

```{.json .output n=4}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on function ones in module mxnet.numpy:\n\nones(shape, dtype=<class 'numpy.float32'>, order='C', ctx=None)\n    Return a new array of given shape and type, filled with ones.\n    This function currently only supports storing multi-dimensional data\n    in row-major (C-style).\n    \n    Parameters\n    ----------\n    shape : int or tuple of int\n        The shape of the empty array.\n    dtype : str or numpy.dtype, optional\n        An optional value type. Default is `numpy.float32`. Note that this\n        behavior is different from NumPy's `ones` function where `float64`\n        is the default value, because `float32` is considered as the default\n        data type in deep learning.\n    order : {'C'}, optional, default: 'C'\n        How to store multi-dimensional data in memory, currently only row-major\n        (C-style) is supported.\n    ctx : Context, optional\n        An optional device context (default is the current default context).\n    \n    Returns\n    -------\n    out : ndarray\n        Array of ones with the given shape, dtype, and ctx.\n    \n    Examples\n    --------\n    >>> np.ones(5)\n    array([1., 1., 1., 1., 1.])\n    \n    >>> np.ones((5,), dtype=int)\n    array([1, 1, 1, 1, 1], dtype=int64)\n    \n    >>> np.ones((2, 1))\n    array([[1.],\n           [1.]])\n    \n    >>> s = (2,2)\n    >>> np.ones(s)\n    array([[1., 1.],\n           [1., 1.]])\n\n"
 }
]
```

```{.python .input  n=5}
#@tab pytorch
help(torch.ones)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on built-in function ones:\n\nones(...)\n    ones(*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) -> Tensor\n    \n    Returns a tensor filled with the scalar value `1`, with the shape defined\n    by the variable argument :attr:`size`.\n    \n    Args:\n        size (int...): a sequence of integers defining the shape of the output tensor.\n            Can be a variable number of arguments or a collection like a list or tuple.\n        out (Tensor, optional): the output tensor.\n        dtype (:class:`torch.dtype`, optional): the desired data type of returned tensor.\n            Default: if ``None``, uses a global default (see :func:`torch.set_default_tensor_type`).\n        layout (:class:`torch.layout`, optional): the desired layout of returned Tensor.\n            Default: ``torch.strided``.\n        device (:class:`torch.device`, optional): the desired device of returned tensor.\n            Default: if ``None``, uses the current device for the default tensor type\n            (see :func:`torch.set_default_tensor_type`). :attr:`device` will be the CPU\n            for CPU tensor types and the current CUDA device for CUDA tensor types.\n        requires_grad (bool, optional): If autograd should record operations on the\n            returned tensor. Default: ``False``.\n    \n    Example::\n    \n        >>> torch.ones(2, 3)\n        tensor([[ 1.,  1.,  1.],\n                [ 1.,  1.,  1.]])\n    \n        >>> torch.ones(5)\n        tensor([ 1.,  1.,  1.,  1.,  1.])\n\n"
 }
]
```

```{.python .input  n=6}
#@tab tensorflow
help(tf.ones)
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on function ones in module tensorflow.python.ops.array_ops:\n\nones(shape, dtype=tf.float32, name=None)\n    Creates a tensor with all elements set to one (1).\n    \n    See also `tf.ones_like`.\n    \n    This operation returns a tensor of type `dtype` with shape `shape` and\n    all elements set to one.\n    \n    >>> tf.ones([3, 4], tf.int32)\n    <tf.Tensor: shape=(3, 4), dtype=int32, numpy=\n    array([[1, 1, 1, 1],\n           [1, 1, 1, 1],\n           [1, 1, 1, 1]], dtype=int32)>\n    \n    Args:\n      shape: A `list` of integers, a `tuple` of integers, or\n        a 1-D `Tensor` of type `int32`.\n      dtype: Optional DType of an element in the resulting `Tensor`. Default is\n        `tf.float32`.\n      name: Optional string. A name for the operation.\n    \n    Returns:\n      A `Tensor` with all elements set to one (1).\n\n"
 }
]
```

From the documentation, we can see that the `ones` function creates a new tensor with specified shape and sets all the elements to the value of 1. Whenever possible, you should run a quick test to confirm your interpretation:

```{.python .input  n=7}
np.ones(4)
```

```{.json .output n=7}
[
 {
  "data": {
   "text/plain": "array([1., 1., 1., 1.])"
  },
  "execution_count": 7,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=8}
#@tab pytorch
torch.ones(4)
```

```{.json .output n=8}
[
 {
  "data": {
   "text/plain": "tensor([1., 1., 1., 1.])"
  },
  "execution_count": 8,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=9}
#@tab tensorflow
tf.ones(4)
```

```{.json .output n=9}
[
 {
  "data": {
   "text/plain": "<tf.Tensor: shape=(4,), dtype=float32, numpy=array([1., 1., 1., 1.], dtype=float32)>"
  },
  "execution_count": 9,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `list?` will create content that is almost
identical to `help(list)`, displaying it in a new browser
window. In addition, if we use two question marks, such as
`list??`, the Python code implementing the function will also be
displayed.


## Summary

* The official documentation provides plenty of descriptions and examples that are beyond this book.
* We can look up documentation for the usage of an API by calling the `dir` and `help` functions, or `?` and `??` in Jupyter notebook.


## Exercises

1. Find a function and a class in the deep learning framework and then look up their documents, also check the documents on the official website. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

