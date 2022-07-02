```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Documentation
:begin_tab:`mxnet`
While we cannot possibly introduce every single MXNet function and class 
(and the information might become outdated quickly), 
the [API documentation](https://mxnet.apache.org/versions/1.8.0/api) 
and additional [tutorials](https://mxnet.apache.org/versions/1.8.0/api/python/docs/tutorials/) and examples 
provide such documentation. 
This section provides some guidance for how to explore the MXNet API.
:end_tab:

:begin_tab:`pytorch`
While we cannot possibly introduce every single PyTorch function and class 
(and the information might become outdated quickly), 
the [API documentation](https://pytorch.org/docs/stable/index.html) and additional [tutorials](https://pytorch.org/tutorials/beginner/basics/intro.html) and examples 
provide such documentation.
This section provides some guidance for how to explore the PyTorch API.
:end_tab:

:begin_tab:`tensorflow`
While we cannot possibly introduce every single TensorFlow function and class 
(and the information might become outdated quickly), 
the [API documentation](https://www.tensorflow.org/api_docs) and additional [tutorials](https://www.tensorflow.org/tutorials) and examples 
provide such documentation. 
This section provides some guidance for how to explore the TensorFlow API.
:end_tab:


## Functions and Classes in a Module

In order to know which functions and classes can be called in a module,
we invoke the `dir` function. For instance, we can
(**query all properties in the module for generating random numbers**):

```{.python .input  n=1}
%%tab mxnet
from mxnet import np
print(dir(np.random))
```

```{.python .input  n=1}
%%tab pytorch
import torch
print(dir(torch.distributions))
```

```{.python .input  n=1}
%%tab tensorflow
import tensorflow as tf
print(dir(tf.random))
```

Generally, we can ignore functions that start and end with `__` (special objects in Python) 
or functions that start with a single `_`(usually internal functions). 
Based on the remaining function or attribute names, 
we might hazard a guess that this module offers 
various methods for generating random numbers, 
including sampling from the uniform distribution (`uniform`), 
normal distribution (`normal`), and multinomial distribution (`multinomial`).

## Specific Functions and Classes

For more specific instructions on how to use a given function or class,
we can invoke the  `help` function. As an example, let's
[**explore the usage instructions for tensors' `ones` function**].

```{.python .input}
%%tab mxnet
help(np.ones)
```

```{.python .input}
%%tab pytorch
help(torch.ones)
```

```{.python .input}
%%tab tensorflow
help(tf.ones)
```

From the documentation, we can see that the `ones` function 
creates a new tensor with the specified shape 
and sets all the elements to the value of 1. 
Whenever possible, you should (**run a quick test**) 
to confirm your interpretation:

```{.python .input}
%%tab mxnet
np.ones(4)
```

```{.python .input}
%%tab pytorch
torch.ones(4)
```

```{.python .input}
%%tab tensorflow
tf.ones(4)
```

In the Jupyter notebook, we can use `?` to display the document in another
window. For example, `list?` will create content
that is almost identical to `help(list)`,
displaying it in a new browser window.
In addition, if we use two question marks, such as `list??`,
the Python code implementing the function will also be displayed.

The official documentation provides plenty of descriptions and examples that are beyond this book. 
Our emphasis lies on covering important use cases 
that will allow you to get started quickly with practical problems, 
rather than completeness of coverage. 
We also encourage you to study the source code of the libraries 
to see examples of high quality implementations for production code. 
By doing this you will become a better engineer 
in addition to becoming a better scientist.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/38)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/39)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/199)
:end_tab:
