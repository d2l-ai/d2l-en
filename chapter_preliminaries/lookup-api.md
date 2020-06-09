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

```{.python .input  n=1}
#@tab pytorch
import torch
print(dir(torch.distributions))
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).

## Finding the Usage of Specific Functions and Classes

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let us explore the usage instructions for tensor's `ones_like` function.

```{.python .input}
help(np.ones_like)
```

```{.python .input}
#@tab pytorch
help(torch.ones_like)
```

From the documentation, we can see that the `ones_like` function creates a new tensor with the same shape as the supplied tensor and sets all the elements to the value of 1. Whenever possible, you should run a quick test to confirm your interpretation:

```{.python .input}
x = np.array([[0, 0, 0], [2, 2, 2]])
np.ones_like(x)
```

```{.python .input}
#@tab pytorch
x = torch.tensor([[0., 0., 0.], [2., 2., 2.]])
torch.ones_like(x)
```

:begin_tab:`mxnet`

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `np.ones_like?` will create content that is almost
identical to `help(np.ones_like)`, displaying it in a new browser
window. In addition, if we use two question marks, such as
`np.ones_like??`, the code implementing the function will also be
displayed.

:end_tab:

:begin_tab:`pytorch`

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `torch.ones_like?` will create content that is almost
identical to `torch.ones_like`, displaying it in a new browser
window. In addition, if we use two question marks, such as
`torch.ones_like??`, the code implementing the function will also be
displayed.

:end_tab:

## API Documentation

:begin_tab:`mxnet`

For further details on the API details check the MXNet website at
[http://mxnet.apache.org/](http://mxnet.apache.org/). You can find the details
under the appropriate headings (also for programming languages other than
Python).

:end_tab:

:begin_tab:`pytorch`

For further details on the API details check the PyTorch website at
[https://pytorch.org/](https://pytorch.org/docs/stable/index.html).

:end_tab:

## Summary

* The official documentation provides plenty of descriptions and examples that are beyond this book.
* We can look up documentation for the usage of an API by calling the `dir` and `help` functions, or checking the website.


## Exercises

:begin_tab:`mxnet`
1. Look up `ones_like` and `autograd` on the MXNet website.
2. What are all the possible outputs after running `np.random.choice(4, 2)`?
3. Can you rewrite `np.random.choice(4, 2)` by using the `np.random.randint` function?

[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
1. Look up `ones_like` on the MXNet website.

[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:
