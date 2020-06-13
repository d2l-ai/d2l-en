# Documentation

Due to constraints on the length of this book, we cannot possibly introduce every single MXNet function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the MXNet API.

## Finding All the Functions and Classes in a Module

In order to know which functions and classes can be called in a module, we
invoke the `dir` function. For instance, we can query all properties in the
module for generating random numbers:

```{.python .input}
from mxnet import np
print(dir(np.random))
```

```{.python .input}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

```{.python .input}
#@tab jax
import jax
print(dir(jax.random))
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "['Optional', 'PRNGKey', 'Sequence', 'Union', '_UINT_DTYPES', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__', '__name__', '__package__', '__spec__', '_bernoulli', '_beta', '_bit_stats', '_bivariate_coef', '_cauchy', '_check_shape', '_constant_like', '_dirichlet', '_exponential', '_fold_in', '_gamma', '_gamma_batching_rule', '_gamma_grad', '_gamma_grad_one', '_gamma_impl', '_gamma_one', '_gumbel', '_is_prng_key', '_laplace', '_logistic', '_make_rotate_left', '_multivariate_normal', '_normal', '_pareto', '_poisson', '_poisson_knuth', '_poisson_rejection', '_randint', '_random_bits', '_shuffle', '_split', '_t', '_threefry2x32_abstract_eval', '_threefry2x32_gpu_translation_rule', '_threefry2x32_lowering', '_truncated_normal', '_uniform', 'abstract_arrays', 'ad', 'apply_round', 'asarray', 'batching', 'bernoulli', 'beta', 'categorical', 'cauchy', 'cholesky', 'core', 'cuda_prng', 'dirichlet', 'dtypes', 'exponential', 'fold_in', 'gamma', 'gumbel', 'jit', 'jnp', 'laplace', 'lax', 'logistic', 'multivariate_normal', 'normal', 'np', 'pareto', 'partial', 'permutation', 'poisson', 'prod', 'randint', 'random_gamma_p', 'rolled_loop_step', 'rotate_left', 'rotate_list', 'shuffle', 'split', 't', 'threefry2x32_p', 'threefry_2x32', 'truncated_normal', 'uniform', 'vmap', 'warnings', 'xla', 'xla_bridge', 'xla_client']\n"
 }
]
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).

## Finding the Usage of Specific Functions and Classes

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let us explore the usage instructions for tensors' `ones` function.

```{.python .input}
help(np.ones)
```

```{.python .input}
#@tab pytorch
help(torch.ones)
```

```{.python .input}
#@tab tensorflow
help(tf.ones)
```

```{.python .input}
#@tab jax
help(jax.numpy.ones)
```

```{.json .output n=2}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Help on function ones in module jax.numpy.lax_numpy:\n\nones(shape, dtype=None)\n    Return a new array of given shape and type, filled with ones.\n    \n    LAX-backend implementation of :func:`ones`.\n    Original docstring below.\n    \n    Parameters\n    ----------\n    shape : int or sequence of ints\n        Shape of the new array, e.g., ``(2, 3)`` or ``2``.\n    dtype : data-type, optional\n        The desired data-type for the array, e.g., `numpy.int8`.  Default is\n        `numpy.float64`.\n    \n    Returns\n    -------\n    out : ndarray\n        Array of ones with the given shape, dtype, and order.\n    \n    See Also\n    --------\n    ones_like : Return an array of ones with shape and type of input.\n    empty : Return a new uninitialized array.\n    zeros : Return a new array setting values to zero.\n    full : Return a new array of given shape filled with value.\n    \n    \n    Examples\n    --------\n    >>> np.ones(5)\n    array([1., 1., 1., 1., 1.])\n    \n    >>> np.ones((5,), dtype=int)\n    array([1, 1, 1, 1, 1])\n    \n    >>> np.ones((2, 1))\n    array([[1.],\n           [1.]])\n    \n    >>> s = (2,2)\n    >>> np.ones(s)\n    array([[1.,  1.],\n           [1.,  1.]])\n\n"
 }
]
```

From the documentation, we can see that the `ones` function creates a new tensor with the specified shape and sets all the elements to the value of 1. Whenever possible, you should run a quick test to confirm your interpretation:

```{.python .input}
np.ones(4)
```

```{.python .input}
#@tab pytorch
torch.ones(4)
```

```{.python .input}
#@tab tensorflow
tf.ones(4)
```

```{.python .input}
#@tab jax
jax.numpy.ones(4)
```

```{.json .output n=3}
[
 {
  "name": "stderr",
  "output_type": "stream",
  "text": "/Users/georgioskaissis/opt/miniconda3/envs/d2l/lib/python3.7/site-packages/jax/lib/xla_bridge.py:125: UserWarning: No GPU/TPU found, falling back to CPU.\n  warnings.warn('No GPU/TPU found, falling back to CPU.')\n"
 },
 {
  "data": {
   "text/plain": "DeviceArray([1., 1., 1., 1.], dtype=float32)"
  },
  "execution_count": 3,
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
* We can look up documentation for the usage of an API by calling the `dir` and `help` functions, or `?` and `??` in Jupyter notebooks.


## Exercises

1. Look up the documentation for any function or class in the deep learning framework. Can you also find the documentation on the official website of the framework?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:

```{.python .input}

```
