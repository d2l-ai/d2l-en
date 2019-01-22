# Documentation

Due to the length of this book, it is impossible for us to introduce all  MXNet functions and classes. The API documentation and additional tutorials and examples provide plenty of documentation beyond the book.

## Finding all the functions and classes in the module

In order to know which functions and classes can be called in a module, we use the `dir` function. For instance we can query all the members or properties in the `nd.random` module.

```{.python .input  n=1}
from mxnet import nd
print(dir(nd.random))
```

Generally speaking, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). According to the remaining member names, we can then hazard a  guess that this module offers a generation method for various random numbers, including uniform distribution sampling (`uniform`), normal distribution sampling (`normal`), and Poisson sampling  (`poisson`).

## Finding the usage of specific functions and classes

For specific function or class usage, we can use the  `help` function. Let's take a look at the usage of the `ones_like` function of an NDArray as an example.

```{.python .input}
help(nd.ones_like)
```

From the documentation, we learned that the `ones_like` function creates a new one with the same shape as the NDArray and an element of 1. Let's verify it:

```{.python .input}
x = nd.array([[0, 0, 0], [2, 2, 2]])
y = x.ones_like()
y
```

In the Jupyter notebook, we can use `?` to display the document in another window. For example, `nd.random.uniform?` will create content that is almost identical to `help(nd.random.uniform)`, but will be displayed in an extra window. In addition, if we use two `nd.random.uniform??`, the function implementation code will also be displayed.

## API Documentation

For further details on the API details check the MXNet website at  [http://mxnet.apache.org/](http://mxnet.apache.org/). You can find the details under the appropriate headings (also for programming languages other than Python).

## Problem

Look up `ones_like` and `autograd` in the API documentation.

## Discuss on our Forum

<div id="discuss" topic_id="2322"></div>
