#  Preliminaries
:label:`chapter_preliminaries`

To get started with deep learning,
we will need to develop a few basic skills.
All machine learning is concerned
with extracting information from data.
So we will begin by learning the practical skills
for storing, manipulating, and preprocessing data with the NumPy interface of MXNet.

Moreover, machine learning typically requires
working with large datasets, which we can think of as tables,
where the rows correspond to examples
and the columns correspond to attributes.
Linear algebra gives us a powerful set of techniques
for working with tabular data.
We will not go too far into the weeds but rather focus on the basic
of matrix operations and their implementation.

Additionally, deep learning is all about optimization.
We have a model with some parameters and
we want to find those that fit our data *the best*.
Determining which way to move each parameter at each step of an algorithm
requires a little bit of calculus, which will be briefly introduced.
Fortunately, MXNet's `autograd` package automatically computes differentation for us,
and we will cover it next.

Next, machine learning is concerned with making predictions:
what is the likely value of some unknown attribute,
given the information that we observe?
To reason rigorously under uncertainty
we will need to invoke the language of probability and statistics.

In the end, MXNet's documentation provides
plenty of descriptions and examples that are beyond this book. 
To conclude the chapter, we will show you how to look up documentation for
the needed information.

```toc
:maxdepth: 2

ndarray
linear-algebra
autograd
probability
lookup-api
```

