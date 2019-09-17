#  The Preliminaries
:label:`chapter_crashcourse`

To get started with deep learning,
we will need to develop a few basic skills.
All machine learning is concerned
with extracting information from data.
So we will begin by learning the practical skills
for storing and manipulating data with Apache MXNet.
Moreover machine learning typically requires
working with large datasets, which we can think of as tables,
where the rows correspond to examples
and the columns correspond to attributes.
Linear algebra gives us a powerful set of techniques
for working with tabular data.
We won't go too far into the weeds but rather focus on the basic
of matrix operations and their implementation in Apache MXNet.
Additionally, deep learning is all about optimization.
We have a model with some parameters and
we want to find those that fit our data the *best*.
Determining which way to move each parameter at each step of an algorithm
requires a little bit of calculus.
Fortunately, Apache MXNet's autograd package covers this for us,
and we will cover it next.
Next, machine learning is concerned with making predictions:
*what is the likely value of some unknown attribute,
given the information that we observe?*
To reason rigorously under uncertainty
we will need to invoke the language of probability and statistics.
To conclude the chapter, we will present
your first basic classifier, *Naive Bayes*.

```toc
:maxdepth: 2

ndarray
linear-algebra
autograd
probability
naive-bayes
lookup-api
```

