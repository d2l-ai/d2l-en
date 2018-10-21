# Checking MXNet documentation

Constrained by the length of this book, it is impossible for us to list all of the concerns regarding MXNet functions and classes.  However, readers are encouraged to seek other documentation to improve their knowledge. 

## Finding all the functions and classes in the module

If you want to know which functions and classes can be called in a module, we use the `dir` function. Next, we will print all the members or properties in the `nd.random` module.

```{.python .input  n=1}
from mxnet import nd

print(dir(nd.random))
```

Generally speaking, we can ignore functions that start and end with `__` (special objects in Python) or functions that start with a single `_`(usually internal functions). According to the remaining member names, we can then hazard a  guess that this module offers a generation method for various random numbers, including uniform distribution sampling (`uniform`), normal distribution sampling (`normal`), and Poisson sampling  (`poisson`).

## Finding the usage of specific functions and classes

For specific function or class usage, we can use the  `help` function. Let's take a look at the usage of the `ones_like,  such as the NDArray function as an example. ` 

```{.python .input}
help(nd.ones_like)
```

From the documentation, we learned that the `ones_like` function creates a new one with the same shape as the NDArray and an element of 1. Let's verify it:

```{.python .input}
x = nd.array([[0, 0, 0], [2, 2, 2]])
y = x.ones_like()
y
```

In the Jupyter notebook, we can use `?` to display the document in another window. For example, `nd.ones_like?` will create content that is almost identical to `help(nd.ones_like)`, but will be displayed in an extra window. In addition, if we use two `nd.ones_like??`, the function implementation code will also be displayed.  


## Checking documentation on the MXNet website

我们也可以在MXNet的网站上查阅相关文档。访问MXNet网站 [http://mxnet.apache.org/](http://mxnet.apache.org/) （如图2.1所示），点击网页顶部的下拉菜单“API”可查阅各个前端语言的接口。此外，我们也可以在网页右上方含“Search”字样的搜索框中直接搜索函数或类名称。

![MXNet's official website (mxnet.apache.org). Click on the drop-down menu "API" at the top to see the APIs for each front-end language. We can also search for API names directly in the search box located at the top right of the page, with the word “search” written inside. ](../img/mxnet-website.png)

Figure 2.2 shows the documentation for the `ones_like` function on the MXNet website.

Documentation for `ones_like` function on the ![MXNet website. ](../img/ones_like.png)


## Summary

* If there is something new about about MXNet API for specific users, they can individually consult related documentation.  
* You can view MXNet documentation through the `dir` and `help` functions, or on the MXNet official website.  


## exercise

* Check out the other operations supported by NDArray.


## Scan the QR code to get to the [ ](https://discuss.gluon.ai/t/topic/7116) forum

![](../img/qr_lookup-api.svg)
