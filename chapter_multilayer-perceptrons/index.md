# Multilayer Perceptrons
:label:`chap_perceptrons`

In this chapter, we will introduce your first truly *deep* network.
The simplest deep networks are called multilayer perceptrons,
and they consist of multiple layers of neurons
each fully connected to those in the layer below
(from which they receive input)
and those above (which they, in turn, influence).
Although automatic differentiation
significantly simplifies the implementation of deep learning algorithms,
we will dive deep into how these gradients
are calculated in deep networks.
Then we will
be ready to 
discuss issues relating to numerical stability and parameter initialization
that are key to successfully training deep networks.

```toc
:maxdepth: 2

mlp
mlp-implementation
backprop
numerical-stability-and-init
generalization-deep
dropout
kaggle-house-price
```

