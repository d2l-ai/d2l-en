```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# The Base Classification Model
:label:`sec_classification`

You may have noticed that the implementations from scratch and the concise implementation using framework functionality were quite similar in the case of regression. The same is true for classification. Since a great many models in this book deal with classification, it is worth adding some functionality to support this setting specifically. This section provides a base class for classification models to simplify future code.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

## The `Classification` Class

We define the `Classification` class below. In the `validation_step` we report both the loss value and the classification accuracy on a validation batch. We draw an update for every `num_val_batches` batches. This has the benefit of generating the averaged loss and accuracy on the whole validation data. These average numbers are not exact correct if the last batch contains fewer examples, but we ignore this minor difference to keep the code simple.

```{.python .input}
%%tab all
class Classification(d2l.Module):  #@save
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

By default we use a Stochastic Gradient Descent optimizer, operating on minibatches, just as we did in the context of linear regression.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params,  'sgd', {'learning_rate': self.lr})
```

```{.python .input}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

## Accuracy

Given the predicted probability distribution `y_hat`,
we typically choose the class with the highest predicted probability
whenever we must output a hard prediction.
Indeed, many applications require that we make a choice.
For instance, Gmail must categorize an email into "Primary", "Social", "Updates", "Forums", or "Spam".
It might estimate probabilities internally,
but at the end of the day it has to choose one among the classes.

When predictions are consistent with the label class `y`, they are correct.
The classification accuracy is the fraction of all predictions that are correct.
Although it can be difficult to optimize accuracy directly (it is not differentiable),
it is often the performance measure that we care about the most. It is often *the*
relevant quantity in benchmarks. As such, we will nearly always report it when training classifiers.

Accuracy is computed as follows:
First, if `y_hat` is a matrix,
we assume that the second dimension stores prediction scores for each class.
We use `argmax` to obtain the predicted class by the index for the largest entry in each row.
Then we [**compare the predicted class with the ground-truth `y` elementwise.**]
Since the equality operator `==` is sensitive to data types,
we convert `y_hat`'s data type to match that of `y`.
The result is a tensor containing entries of 0 (false) and 1 (true).
Taking the sum yields the number of correct predictions.

```{.python .input}
%%tab all
@d2l.add_to_class(Classification)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if len(params.keys()) else self.get_scratch_params()
```

## Summary

Classification is a sufficiently frequently used problem type that it warrants its own convenience functions. Note that there is a difference between (classification) accuracy that we want to minimize and the logistic loss function that we are actually minimizing. Fortunately, our specific choice of loss function ensures that minimizing it will also lead to maximum accuracy. This is the case since the maximum likelihood estimator is consistent. It follows as a special case of the Cramer-Rao bound :cite:`cramer1946mathematical,radhakrishna1945information`. For more work on consistency see also :cite:`zhang2004statistical`.

More generally, though, the decision of which category to pick is far from trivial. For instance, when deciding where to assign an e-mail to, mistaking a "Primary" e-mail for a "Social" e-mail might be undesirable but far less disastrous than moving it to the spam folder (and later automatically deleting it). As such, we will tend to err on the side of caution with regard to assigning any e-mail to the "Spam" folder, rather than picking the most likely category.

## Exercises

1. Denote by $L_v$ the validation loss, and let $L_v^q$ be its quick and dirty estimate computed by the loss function averaging in this section. Lastly, denote by $l_v^b$ the loss on the last minibatch. Express $L_v$ in terms of $L_v^q$, $l_v^b$, and the sample and minibatch sizes.
1. Show that the quick and dirty estimate $L_v^q$ is unbiased. That is, show that $E[L_v] = E[L_v^q]$. Why would you still want to use $L_v$ instead?
1. Given a multiclass classification loss, denoting by $l(y,y')$ the penalty of estimating $y'$ when we see $y$ and given a probabilty $p(y|x)$, formulate the rule for an optimal selection of $y'$. Hint: express the expected loss, using $l$ and $p(y|x)$.

