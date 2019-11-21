# Documentation

Due to constraints on the length of this book, we cannot possibly introduce every single MXNet function and class (and you probably would not want us to). The API documentation and additional tutorials and examples provide plenty of documentation beyond the book. In this section we provide you with some guidance to exploring the MXNet API.

## Finding All the Functions and Classes in a Module

In order to know which functions and classes can be called in a module, we invoke the `dir` function. For instance, we can query all properties in the `np.random` module as follows:

```{.python .input  n=1}
from mxnet import np
print(dir(np.random))
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). Based on the remaining function or attribute names, we might hazard a guess that this module offers various methods for generating random numbers, including sampling from the uniform distribution (`uniform`), normal distribution (`normal`), and multinomial distribution  (`multinomial`).

## Finding the Usage of Specific Functions and Classes

For more specific instructions on how to use a given function or class, we can invoke the  `help` function. As an example, let's explore the usage instructions for `ndarray`'s `ones_like` function.

```{.python .input}
help(np.ones_like)
```

From the documentation, we can see that the `ones_like` function creates a new array with the same shape as the supplied `ndarray` and sets all the elements to `1`. Whenever possible, you should run a quick test to confirm your interpretation:

```{.python .input}
x = np.array([[0, 0, 0], [2, 2, 2]])
np.ones_like(x)
```

In the Jupyter notebook, we can use `?` to display the document in another window. For example, `np.random.uniform?` will create content that is almost identical to `help(np.random.uniform)`, displaying it in a new browser window. In addition, if we use two question marks, such as `np.random.uniform??`, the code implementing the function will also be displayed.


## API Documentation

For further details on the API details check the MXNet website at  [http://mxnet.apache.org/](http://mxnet.apache.org/). You can find the details under the appropriate headings (also for programming languages other than Python).

## Summary

* The official documentation provides plenty of descriptions and examples that are beyond this book.
* We can look up documentation for the usage of MXNet API by calling the `dir` and `help` functions, or checking the MXNet website.


## Exercises

1. Look up `ones_like` and `autograd` on the MXNet website.
2. What are all the possible outputs after running `np.random.choice(4, 2)`?
3. Can you rewrite `np.random.choice(4, 2)` by using the `np.random.randint` function?


## [Discussions](https://discuss.mxnet.io/t/2322)

![](../img/qr_lookup-api.svg)
