```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# The Base Classification Model
:label:`sec_classification`

You may have noticed that our from scratch and concise implementation of linear regression are very similar to each other. The implementation of softmax regression will be similar as well. A large part of of models in this book is about classification. In this section, we will provide a base class for classification models to simplify our future code.

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

We define the `Classification` class in the following code block. In the `validation_step` we report both the loss value and the classification accuracy on a validation batch. Note that we draw points for every `num_val_batches` batches, so it means we report the averaged loss and accuracy on the whole validation datasets. These average numbers are not exact correct if the last batch contains less examples, but we ignore this tiny difference for code simplicity.

```{.python .input}
%%tab all
class Classification(d2l.Module):  #@save
    def validation_step(self, batch):
        X, y = batch
        y_hat = self(X)
        for k, v in (('val_loss', self.loss(y_hat, y)),
                     ('val_acc', self.accuracy(y_hat, y))):
            self.board.draw(self.trainer.epoch+1, v, k,
                            every_n=self.trainer.num_val_batches)    
```

Again, we use minibatch SGD as the optimizer.

```{.python .input}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.collect_params()
    if isinstance(params, (tuple, list)):
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
Gmail must categorize an email into "Primary", "Social", "Updates", or "Forums".
It might estimate probabilities internally,
but at the end of the day it has to choose one among the classes.

When predictions are consistent with the label class `y`, they are correct.
The classification accuracy is the fraction of all predictions that are correct.
Although it can be difficult to optimize accuracy directly (it is not differentiable),
it is often the performance measure that we care most about,
and we will nearly always report it when training classifiers.

To compute accuracy we do the following.
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
def accuracy(self, y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return d2l.reduce_mean(d2l.astype(cmp, d2l.float32))
```

## Summary

- The `Classification` class contains shared methods for classification models. We only need to define the model, compute the predictions, and the loss function for future models
