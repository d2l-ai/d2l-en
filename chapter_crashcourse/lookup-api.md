# Documentation

Due to constraints on the length of this book, we cannot possibly introduce every single MXNet function and class (and you probably would no want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you some guidance to exploring the MXNet API.

## Finding all the functions and classes in the module

In order to know which functions and classes can be called in a module, we invoke the `dir` function. For instance, we can query all properties in the `nd.random` module as follows:

```{.python .input  n=1}
from mxnet import nd
print(dir(nd.random))
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function/attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and Poisson distribution  (`poisson`).

## Finding the usage of specific functions and classes

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let's explore the usage instructions for NDArray's `ones_like` function.

```{.python .input}
help(nd.ones_like)
```

From the documentation, we can see that the `ones_like` function creates a new array with the same shape as the supplied NDArray and all elements set to `1`. Whenever possible, you should run a quick test to confirm your interpretation:

```{.python .input}
x = nd.array([[0, 0, 0], [2, 2, 2]])
y = x.ones_like()
y
```

In the Jupyter notebook, we can use `?` to display the document in another window. For example, `nd.random.uniform?` will create content that is almost identical to `help(nd.random.uniform)`, displaying it in a new browser window. In addition, if we use two question marks, e.g. `nd.random.uniform??`, the code implementing the function will also be displayed.

## API Documentation

For further details on the API details check the MXNet website at  [http://mxnet.apache.org/](http://mxnet.apache.org/). You can find the details under the appropriate headings (also for programming languages other than Python).

## Exercise

Look up `ones_like` and `autograd` in the API documentation.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2322)

![](../img/qr_lookup-api.svg)
