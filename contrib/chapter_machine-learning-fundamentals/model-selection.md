```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Model Selection
:label:`sec_model_selection`

As machine learning scientists,
our goal is to discover *patterns*.
But how can we be sure that we have
truly discovered a *general* pattern
and not simply memorized our data?
For example, imagine that we wanted to hunt
for patterns among genetic markers
linking patients to their dementia status,
where the labels are drawn from the set
$\{\text{dementia}, \text{mild cognitive impairment}, \text{healthy}\}$.
Because each person's genes identify them uniquely
(ignoring identical siblings),
it is possible to memorize the entire dataset.

We do not want our model to say
*"That's Bob! I remember him! He has dementia!"*
The reason why is simple.
When we deploy the model in the future,
we will encounter patients
that the model has never seen before.
Our predictions will only be useful
if our model has truly discovered a *general* pattern.

To recapitulate more formally,
our goal is to discover patterns
that capture regularities in the underlying population
from which our training set was drawn.
If we are successful in this endeavor,
then we could successfully assess risk
even for individuals that we have never encountered before.
This problem---how to discover patterns that *generalize*---is
the fundamental problem of machine learning.

The danger is that when we train models,
we access just a small sample of data.
The largest *labeled* public image datasets, such as Imagenet :cite:`Deng.Dong.Socher.ea.2009` contain
roughly one million images. Unlabeled image collections such as the Flickr YFC100M dataset can be significantly larger, containing 100 million images :cite:`Thomee.Shamma.Friedland.ea.2016`. While both numbers seem large, they are tiny compared to the space of all possible images that one could take at, say, 1 Megapixel resolution.
Worse still, we frequently must learn from only hundreds of examples. For instance,
a hospital might only have data of 100 occurrences of an infrequent disease.
When working with finite samples, we run the risk
that we might discover apparent associations
that turn out not to hold up when we collect more data.

The phenomenon of fitting our training data
more closely than we fit the underlying distribution is called *overfitting*, and the techniques used to combat overfitting are called *regularization*. While the following is no substitute for a proper introduction to statistical learning theory :cite:`Vapnik.1998,Boucheron.Bousquet.Lugosi.2005`, it should at least make you aware of some of the phenomena that arise in learning. Overfitting is really quite prevalent. In the previous sections, you might have observed
overfitting while experimenting with the Fashion-MNIST dataset.
If you altered the model structure or the hyperparameters during the experiment, you might have noticed that with enough neurons, layers, and training epochs, the model can eventually reach perfect accuracy on the training set, while the accuracy on test data deteriorates.


## Training Error and Generalization Error

In order to discuss this phenomenon more formally,
we need to differentiate between training error and generalization error.
The *training error* $R_\mathrm{emp}$ is the error of our model
as calculated on the training dataset,
while *generalization error* $R$ is the expectation of our model's error
were we to apply it to an infinite stream of additional data examples
drawn from the same underlying data distribution as our original sample.
They are defined as follows:

$$R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] = \frac{1}{m} \sum_{i=1}^m l(\mathbf{x}_i, \mathbf{y}_i, f(\mathbf{x}_i))
\text{ and }
R[p, f] = E_{(\mathbf{x}, \mathbf{y}) \sim p} [l(\mathbf{x}, \mathbf{y}, f(\mathbf{x}))]$$

Problematically, we can never calculate the generalization error $R$ exactly.
That is because the stream of infinite data is an imaginary object.
In practice, we must *estimate* the generalization error
by applying our model to an independent test set
constituted of a random selection of examples $\mathbf{X}'$ and labels $\mathbf{Y}'$
that were withheld from our training set. This yields $R_\mathrm{emp}[\mathbf{X}', \mathbf{Y}', f]$.

The following three thought experiments
will help illustrate this situation better.
Consider a college student trying to prepare for his final exam.
A diligent student will strive to practice well
and test his abilities using exams from previous years.
Nonetheless, doing well on past exams is no guarantee
that he will excel when it matters.
For instance, the student might try to prepare
by rote learning the answers to the exam questions.
This requires the student to memorize many things.
She might even remember the answers for past exams perfectly.
Another student might prepare by trying to understand
the reasons for giving certain answers.
While this tends to work well for drivers license exams,
it has poor outcomes when the set of exam questions is more
varied and drawn from a larger, possibly infinite pool.

Likewise, consider a model that simply uses a lookup table to answer questions. If the set of allowable inputs is discrete and reasonably small, then perhaps after viewing *many* training examples, this approach would perform well. Still this model has no ability to do better than random guessing when faced with examples that it has never seen before.
In reality the input spaces are far too large to memorize the answers corresponding to every conceivable input. For example, consider the black and white $28\times28$ images. If each pixel can take one among $256$ grayscale values, then there are $256^{784} \approx 10^{1888}$ possible images. That means that there are far more low-resolution grayscale thumbnail-sized images than the approximately $10^{82}$ atoms in the universe. Even if we could encounter such data, we could never afford to store the lookup table. This explosion in the number of required samples is closely related to the curse of dimensionality where simple problems become rather difficult once the data is high dimensional :cite:`Friedman.1997`.

Last, consider the problem of trying
to classify the outcomes of coin tosses (class 0: heads, class 1: tails)
based on some contextual features that might be available.
Suppose that the coin is fair.
No matter what algorithm we come up with,
the generalization error will always be $\frac{1}{2}$.
However, for most algorithms,
we should expect our training error to be considerably lower,
depending on the luck of the draw,
even if we did not have any features!
Consider the dataset {0, 1, 1, 1, 0, 1}.
Our feature-less algorithm would have to fall back on always predicting
the *majority class*, which appears from our limited sample to be *1*.
In this case, the model that always predicts class 1
will incur an error of $\frac{1}{3}$,
considerably better than our generalization error.
As we increase the amount of data,
the probability that the fraction of heads
will deviate significantly from $\frac{1}{2}$ diminishes,
and our training error would come to match the generalization error.

### Statistical Learning Theory

Since generalization is the fundamental problem in machine learning,
you might not be surprised to learn
that many mathematicians and theorists have dedicated their lives
to developing formal theories to describe this phenomenon.
In their [epoynmous theorem](https://en.wikipedia.org/wiki/Glivenko%E2%80%93Cantelli_theorem) Glivenko and Cantelli derived the rate at which the training error
converges to the generalization error :cite:`Glivenko.1933,Cantelli.1933`.
In a series of seminal papers, Vapnik and Chervonenkis
extended this theory to more general classes of functions
:cite:`Vapnik.Chervonenkis.1964,Vapnik.Chervonenkis.1968,Vapnik.Chervonenkis.1971,Vapnik.Chervonenkis.1981,Vapnik.Chervonenkis.1991,Vapnik.Chervonenkis.1974*1`.
This work laid the foundations of statistical learning theory.

In the standard supervised learning setting, which we have addressed
up until now and will stick with throughout most of this book,
we assume that both the training data and the test data
are drawn *independently* from *identical* distributions.
This is commonly called the *IID assumption*. It means that
all samples are drawn from the same distribution. It also means that,
knowing all $n-1$ samples makes it no easier for us to predict the $n$-th sample
than it is to predict the first one.

Being a good machine learning scientist requires thinking critically,
and already you should be poking holes in this assumption,
coming up with common cases where the assumption fails.
What if we train a mortality risk predictor
on data collected from patients at UCSF Medical Center,
and apply it on patients at Massachusetts General Hospital?
These distributions are simply not identical. This is a well-studied
problem in statistics :cite:`Rosenbaum.Rubin.1983`.
Moreover, draws might be correlated in time.
What if we are classifying the topics of Tweets?
The news cycle would create temporal dependencies
in the topics being discussed, violating any assumptions of independence.

Sometimes we can get away with minor violations of the IID assumption
and our models will continue to work remarkably well.
After all, nearly every real-world application
involves at least some minor violation of the IID assumption,
and yet we have many useful tools for
various applications such as
face recognition,
speech recognition, and language translation. :cite:`Yu.1994` provides
a quantitative handle on this behavior.

Other violations are sure to cause trouble.
Imagine, for example, if we try to train
a face recognition system by training it
exclusively on university students
and then want to deploy it as a tool
for monitoring geriatrics in a nursing home population.
This is unlikely to work well since college students
tend to look considerably different from the elderly.

In subsequent chapters, we will discuss problems
arising from violations of the IID assumption.
For now, even taking the IID assumption for granted,
understanding generalization is a formidable problem.
Moreover, elucidating the precise theoretical foundations
that might explain why deep neural networks generalize as well as they do
continues to vex the greatest minds in learning theory :cite:`Frankle.Carbin.2018,Bartlett.Montanari.Rakhlin.2021,Nagarajan.Kolter.2019,Kawaguchi.Kaelbling.Bengio.2017`.

When we train our models, we attempt to search for a function
that fits the training data as well as possible.
If the function is so flexible that it can catch on to spurious patterns
just as easily as to true associations,
then it might perform *too well* without producing a model
that generalizes well to unseen data.
This is precisely what we want to avoid or at least control.
Many of the techniques in deep learning are heuristics and tricks
aimed at guarding against overfitting (:numref:`sec_weight_decay`, :numref:`sec_dropout`, :numref:`sec_batch_norm`).

### Model Complexity

When we have simple models and abundant data,
we expect the generalization error to resemble the training error.
When we work with more complex models and fewer examples,
we expect the training error to go down but the generalization gap to grow.
What precisely constitutes model complexity is a complex matter.
Many factors govern whether a model will generalize well.
For example a model with more parameters might be considered more complex in general.
Note, though, that this is not necessarily true. For instance, kernel methods operate in spaces with infinite numbers of parameters, yet they exhibit very well-controlled model complexity :cite:`Scholkopf.Smola.2002`.
Instead, a better way to think about this is that a
model whose parameters can take a wider range of values
might be more complex.
Often with neural networks, we think of a model
that takes more training iterations as more complex,
and one subject to *early stopping* (fewer training iterations) as less complex :cite:`Prechelt.1998`.

It can be difficult to compare the complexity among members
of substantially different model classes
(say, decision trees vs. neural networks).
For now, a simple rule of thumb is quite useful:
a model that can readily explain arbitrary facts
is what statisticians view as complex,
whereas one that has only a limited expressive power
but still manages to explain the data well
is probably closer to the truth :cite:`Vapnik.Levin.Le-Cun.1994`.
In philosophy, this is closely related to Popper's
criterion of falsifiability
of a scientific theory: a theory is good if it fits data
and if there are specific tests that can be used to disprove it.
This is important since all statistical estimation is
*post hoc*,
i.e., we estimate after we observe the facts,
hence vulnerable to the associated fallacy :cite:`Corfield.Scholkopf.Vapnik.2009`.
For now, we will put the philosophy aside and stick to more tangible issues.

In this section, to give you some intuition,
we will focus on a few factors that tend
to influence the generalizability of a model class:

1. The number of tunable parameters. When the number of tunable parameters, sometimes called the *degrees of freedom*, is large, models tend to be more susceptible to overfitting :cite:`Murata.Yoshizawa.Amari.1994`.
1. The values taken by the parameters. When weights can take a wider range of values, models can be more susceptible to overfitting :cite:`Krogh.Hertz.1992`.
1. The number of training examples. It is trivially easy to overfit a dataset containing only one or two examples even if your model is simple. But overfitting a dataset with millions of examples requires an extremely flexible model :cite:`Henighan.Kaplan.Katz.ea.2020`.

## Model Selection

In machine learning, we usually select our final model
after evaluating several candidate models.
This process is called *model selection*.
Sometimes the models subject to comparison
are fundamentally different in nature
(say, decision trees vs. linear models).
At other times, we are comparing
members of the same class of models
that have been trained with different hyperparameter settings.

With MLPs, for example,
we may wish to compare models with
different numbers of hidden layers,
different numbers of hidden units,
and various choices of the activation functions
applied to each hidden layer. For a particularly elegant
strategy to accomplish this for computer vision see :cite:`Radosavovic.Kosaraju.Girshick.ea.2020`.
In order to determine the best among our candidate models,
we will typically employ a validation dataset.


### Validation Dataset

In principle we should not touch our test set
until after we have chosen all our hyperparameters.
Were we to use the test data in the model selection process,
there is a risk that we might overfit the test data.
Then we would be in serious trouble.
If we overfit our training data,
there is always the evaluation on test data to keep us honest.
But if we overfit the test data, how would we ever know?
See e.g. :cite:`Ong.Smola.Williamson.ea.2005` for an example how
this can lead to absurd results even for models where the complexity
can be tightly controlled.

Thus, we should never rely on the test data for model selection.
And yet we cannot rely solely on the training data
for model selection either because
we cannot estimate the generalization error
on the very data that we use to train the model.


In practical applications, the picture gets muddier.
While ideally we would only touch the test data once,
to assess the very best model or to compare
a small number of models to each other,
real-world test data is seldom discarded after just one use.
We can seldom afford a new test set for each round of experiments.
In fact, recycling benchmark data for decades can have a significant impact on the
development of algorithms, e.g. for [image classification](https://paperswithcode.com/sota/image-classification-on-imagenet) and [optical character recognition](https://paperswithcode.com/sota/image-classification-on-mnist).

The common practice to address the problem of `training on the test set`
is to split our data three ways,
incorporating a *validation set*
in addition to the training and test datasets.
The result is a murky practice where the boundaries
between validation and test data are worryingly ambiguous.
Unless explicitly stated otherwise, in the experiments in this book
we are really working with what should rightly be called
training data and validation data, with no true test sets.
Therefore, the accuracy reported in each experiment of the book is really
the validation accuracy and not a true test set accuracy.

### $K$-Fold Cross-Validation

When training data is scarce,
we might not even be able to afford to hold out
enough data to constitute a proper validation set.
One popular solution to this problem is to employ
$K$*-fold cross-validation*.
Here, the original training data is split into $K$ non-overlapping subsets.
Then model training and validation are executed $K$ times,
each time training on $K-1$ subsets and validating
on a different subset (the one not used for training in that round).
Finally, the training and validation errors are estimated
by averaging over the results from the $K$ experiments.

## Underfitting or Overfitting?

When we compare the training and validation errors,
we want to be mindful of two common situations.
First, we want to watch out for cases
when our training error and validation error are both substantial
but there is a little gap between them.
If the model is unable to reduce the training error,
that could mean that our model is too simple
(i.e., insufficiently expressive)
to capture the pattern that we are trying to model.
Moreover, since the *generalization gap*
between our training and validation errors is small,
we have reason to believe that we could get away with a more complex model.
This phenomenon is known as *underfitting* (note, though, that it could also
mean that the problem is simply very difficult).

On the other hand, as we discussed above,
we want to watch out for the cases
when our training error is significantly lower
than our validation error, indicating severe *overfitting*.
Note that overfitting is not always a bad thing.
With deep learning especially, it is well known
that the best predictive models often perform
far better on training data than on holdout data.
Ultimately, we usually care more about the validation error
than about the gap between the training and validation errors.

Whether we overfit or underfit can depend
both on the complexity of our model
and the size of the available training datasets,
two topics that we discuss below.

### Model Complexity

To illustrate some classical intuition
about overfitting and model complexity,
we give an example using polynomials.
Given training data consisting of a single feature $x$
and a corresponding real-valued label $y$,
we try to find the polynomial of degree $d$

$$\hat{y}= \sum_{i=0}^d x^i w_i$$

to estimate the labels $y$.
This is just a linear regression problem
where our features are given by the powers of $x$,
the model's weights are given by $w_i$,
and the bias is given by $w_0$ since $x^0 = 1$ for all $x$.
Since this is just a linear regression problem,
we can use the squared error as our loss function.


A higher-order polynomial function is more complex
than a lower-order polynomial function,
since the higher-order polynomial has more parameters
and the model function's selection range is wider.
Fixing the training dataset,
higher-order polynomial functions should always
achieve lower (at worst, equal) training error
relative to lower degree polynomials.
In fact, whenever the data examples each have a distinct value of $x$,
a polynomial function with degree equal to the number of data examples
can fit the training set perfectly.
We visualize the relationship between polynomial degree
and underfitting vs. overfitting in :numref:`fig_capacity_error`.

![Influence of model complexity on underfitting and overfitting](../img/capacity-vs-error.svg)
:label:`fig_capacity_error`

Much of the intuition of this arises from Statistical Learning Theory. One of the guarantees it
provides :cite:`Vapnik.1998` is that the gap between empirical risk and expected risk is bounded by

$$\Pr\left(R[p, f] - R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \epsilon\right) \geq 1-\delta
\ \text{for}\ \epsilon \geq c \sqrt{(\mathrm{VC} - \log \delta)/n}.$$

Here $\delta > 0$ is the probability that the bound is violated and $\mathrm{VC}$ is the Vapnik-Chervonenkis (VC)
dimension of the set of functions that we want to fit. For instance, for polynomials of degree $d$ the VC dimension is $d+1$. Lastly, $c > 0$ is a constant that depends only on the scale of the loss that can be incurred. In short, this shows that our bound becomes increasingly loose as we pick more complex models and that the number of free parameters should not increase more rapidly than the dataset size $n$ increases. See :cite:`Boucheron.Bousquet.Lugosi.2005` for a detailed discussion and for much more advanced ways of measuring function complexity.

### Dataset Size

As the above bound already indicates, the other big consideration to bear in mind is the dataset size.
Fixing our model, the fewer samples we have in the training dataset,
the more likely (and more severely) we are to encounter overfitting.
As we increase the amount of training data,
the generalization error typically decreases.
Moreover, in general, more data never hurts.
For a fixed task and data distribution, model complexity should not
increase more rapidly than the amount of data does.
Given more data, we might profitably attempt to fit a more complex model.
Absent sufficient data, simpler models may be more difficult to beat.
For many tasks, deep learning only outperforms linear models
when many thousands of training examples are available.
In part, the current success of deep learning
owes to the current abundance of massive datasets
due to Internet companies, cheap storage, connected devices,
and the broad digitization of the economy.


## Summary

This section explored some of the theoretical underpinnings of machine learning. Making these work for modern deep learning is still very much a work in progress. Simply minimizing the training error will not necessarily mean a reduction in the generalization error. Machine learning models need to be careful to safeguard against overfitting so as to minimize the generalization error. Nonetheless, we provided some basic intuition how to control the generalization error. For instance, we can to resort to validation sets or statistical bounds.

A few rules of thumb: 1) A validation set can be used for model selection, provided that it is not used too liberally. 2) A more complex model requires more data, where the amount of data should scale up at least as rapidly as the model complexity. 3) More parameters can mean more complex models, but there are ways where this need not be the case, e.g. by controlling the magnitude. 4) More data makes everything better. As long as the data is drawn from the same distribution. 5) Check your assumptions.

## Exercises

1. Can you solve the problem of polynomial regression exactly?
1. Give at least five examples where dependent random variables make treating the problem as IID data inadvisable.
1. Can you ever expect to see zero training error? Under which circumstances would you see zero generalization error?
1. Why is $k$-fold crossvalidation very expensive to compute?
1. Why is the $k$-fold crossvalidation error estimate biased?
1. The VC dimension is defined as the maximum number of points that can be classified with arbitrary labels $\{\pm 1\}$ by a function of a class of functions. Why might this not be a good idea to measure how complex the class of functions is? Hint: what about the magnitude of the functions?
1. Your manager gives you a difficult dataset on which your current algorithm doesn't perform so well. How would you justify to him that you need more data? Hint: you cannot increase the data but you can decrease it.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
