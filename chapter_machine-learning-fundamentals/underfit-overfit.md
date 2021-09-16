```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Underfitting and Overfitting 
:label:`sec_polynomial`

In this section we test out some of the concepts that we saw previously. To keep matters simple, we use polynomial regression as our toy example.

```{.python .input  n=3}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import math
```

```{.python .input  n=5}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

### Generating the Dataset

First we need data. Given $x$, we will [**use the following cubic polynomial to generate the labels**] on training and test data:

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

The noise term $\epsilon$ obeys a normal distribution
with a mean of 0 and a standard deviation of 0.1.
For optimization, we typically want to avoid
very large values of gradients or losses.
This is why the *features*
are rescaled from $x^i$ to $\frac{x^i}{i!}$.
It allows us to avoid very large values for large exponents $i$.
We will synthesize 100 samples each for the training set and test set.

```{.python .input  n=6}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()        
        p, n = max(3, self.num_inputs), num_train + num_val
        w = d2l.tensor([1.2, -3.4, 5.6] + [0]*(p-3))
        if tab.selected('mxnet') or tab.selected('pytorch'):
            x = d2l.randn(n, 1)
            noise = d2l.randn(n, 1) * 0.1
        if tab.selected('tensorflow'):
            x = d2l.normal((n, 1))
            noise = d2l.normal((n, 1)) * 0.1
        X = d2l.concat([x ** (i+1) / math.gamma(i+2) for i in range(p)], 1)
        self.y = d2l.matmul(X, d2l.reshape(w, (-1, 1))) + noise
        self.X = X[:,:num_inputs]
        
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

Again, monomials stored in `poly_features`
are rescaled by the gamma function,
where $\Gamma(n)=(n-1)!$.
[**Take a look at the first 2 samples**] from the generated dataset.
The value 1 is technically a feature,
namely the constant feature corresponding to the bias.

### [**Third-Order Polynomial Function Fitting (Normal)**]

We will begin by first using a third-order polynomial function, which is the same order as that of the data generation function.
The results show that this model's training and test losses can be both effectively reduced.
The learned model parameters are also close
to the true values $w = [1.2, -3.4, 5.6], b=5$.

```{.python .input  n=7}
%%tab all
def train(p):
    if tab.selected('mxnet') or tab.selected('tensorflow'):
        model = d2l.LinearRegression(lr=0.01)
    if tab.selected('pytorch'):
        model = d2l.LinearRegression(p, lr=0.01)
    model.board.ylim = [1, 1e2]
    data = Data(200, 200, p, 20)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    print(model.get_w_b())
    
train(p=3)
```

### [**Linear Function Fitting (Underfitting)**]

Let's take another look at linear function fitting.
After the decline in early epochs,
it becomes difficult to further decrease
this model's training loss.
After the last epoch iteration has been completed,
the training loss is still high.
When used to fit nonlinear patterns
(like the third-order polynomial function here)
linear models are liable to underfit.

```{.python .input  n=8}
%%tab all
train(p=1)
```

### [**Higher-Order Polynomial Function Fitting  (Overfitting)**]

Now let's try to train the model
using a polynomial of too high degree.
Here, there is insufficient data to learn that
the higher-degree coefficients should have values close to zero.
As a result, our overly-complex model
is so susceptible that it is being influenced
by noise in the training data.
Though the training loss can be effectively reduced,
the test loss is still much higher.
It shows that
the complex model overfits the data.

```{.python .input  n=9}
%%tab all
train(p=10)
```

In the subsequent sections, we will continue
to discuss overfitting problems
and methods for dealing with them,
such as weight decay and dropout.


## Summary

* Since the generalization error cannot be estimated based on the training error, simply minimizing the training error will not necessarily mean a reduction in the generalization error. Machine learning models need to be careful to safeguard against overfitting so as to minimize the generalization error.
* A validation set can be used for model selection, provided that it is not used too liberally.
* Underfitting means that a model is not able to reduce the training error. When training error is much lower than validation error, there is overfitting.
* We should choose an appropriately complex model and avoid using insufficient training samples.


## Exercises

1. Can you solve the polynomial regression problem exactly? Hint: use linear algebra.
1. Consider model selection for polynomials:
    1. Plot the training loss vs. model complexity (degree of the polynomial). What do you observe? What degree of polynomial do you need to reduce the training loss to 0?
    1. Plot the test loss in this case.
    1. Generate the same plot as a function of the amount of data.
1. What happens if you drop the normalization ($1/i!$) of the polynomial features $x^i$? Can you fix this in some other way?
1. Can you ever expect to see zero generalization error?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
