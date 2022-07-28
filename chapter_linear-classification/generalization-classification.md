# Generalization in Classification

:label:`chap_classification_generalization`



So far, we have focused on how to tackle multiclass classification problems
by training (linear) neural networks with multiple outputs and softmax functions.
Interpreting our model's outputs as probabilistic predictions,
we motivated and derived the cross-entropy loss function,
which calculates the negative log likelihood
that our model (for a fixed set of parameters)
assigns to the actual labels.
And finally, we put these tools into practice
by fitting our model to the training set.
However, as always, our goal is to learn *general patterns*,
as assessed empirically on previously unseen data (the test set).
High accuracy on the training set means nothing.
Whenever each of our inputs is unique
(and indeed this is true for most high-dimensional datasets),
we can attain perfect accuracy on the training set
by just memorizing the dataset on the first training epoch,
and subsequently looking up the label whenever we see a new image.
And yet, memorizing the exact labels
associated with the exact training examples
does not tell us how to classify new examples.
Absent further guidance, we might have to fall back
on random guessing whenever we encounter new examples.

A number of burning questions demand immediate attention:
1. How many test examples do we need to precisely estimate
   the accuracy of our classifiers on the underlying population?
1. What happens if we keep evaluating models on the same test repeatedly?
1. Why should we expect that fitting our linear models to the training set
   should fare any better than our naive memorization scheme?


While :numref:`sec_generalization_basics` introduced
the basics of overfitting and generalization
in the context of linear regression,
this chapter will go a little deeper,
introducing some of the foundational ideas
of statistical learning theory.
It turns out that we often can guarantee generalization *a priori*:
for many models,
and for any desired upper bound
on the generalization gap $\epsilon$,
we can often determine some required number of samples $n$
such that if our training set contains at least $n$
samples, then our empirical error will lie
within $\epsilon$ of the true error,
*for any data generating distribution*.
Unfortunately, it also turns out
that while these sorts of guarantees provide
a profound set of intellectual building blocks,
they are of limited practical utility
to the deep learning practitioner.
In short, these guarantees suggest
that ensuring generalization
of deep neural networks *a priori*
requires an absurd number of examples
(perhaps trillions or more),
even when we find that on the tasks we care about
that deep neural networks typically to generalize
remarkably well with far fewer examples (thousands).
Thus deep learning practitioners often forgo
a priori guarantees altogether,
instead employing methods on the basis
that they have generalized well
on similar problems in the past,
and certifying generalization *post hoc*
through empirical evaluations.
When we get to :numref:`chap_perceptrons`,
we will revisit generalization
and provide a light introduction
to the vast scientific literature
that has sprung in attempts
to explain why deep neural networks generalize in practice.

## The Test Set

Since we have already begun to rely on test sets
as the gold standard method
for assessing generalization error,
let's get started by discussing
the properties of such error estimates.
Let's focus on a fixed classifier $f$,
without worrying about how it was obtained.
Moreover suppose that we possess
a *fresh* dataset of examples $\mathcal{D} = {(\mathbf{x}^{(i)},y^{(i)})}_{i=1}^n$
that were not used to train the classifier $f$.
The *empirical error* of our classifier $f$ on $\mathcal{D}$
is simply the fraction of instances
for which the prediction $f(\mathbf{x}^{(i)})$
disagrees with the true label $y^{(i)}$,
and is given by the following expression:

$$\epsilon_\mathcal{D}(f) = \frac{1}{n}\sum_{i=1}^n \mathbf{1}(f(\mathbf{x}^{(i)}) \neq y^{(i)}).$$

By contrast, the *population error*
is the *expected* fraction
of examples in the underlying population
(some distribution $P(X,Y)$  characterized
by probability density function $p(\mathbf{x},y)$
for which our classifier disagrees
with the true label:

$$\epsilon(f) =  E_{(\mathbf{x}, y) \sim P} \mathbf{1}(f(\mathbf{x}) \neq y) =
\int\int \mathbf{1}(f(\mathbf{x}) \neq y) p(\mathbf{x}, y) \;d\mathbf{x} dy.$$

While $\epsilon(f)$ is the quantity that we actually care about,
we cannot observe it directly,
just as we cannot directly
observe the average height in a large population
without measuring every single person.
We can only estimate this quantity based on samples.
Because our test set $\mathcal{D}$
is statistically representative
of the underlying population,
we can view $\epsilon_\mathcal{D}(f)$ as a statistical
estimator of the population error $\epsilon(f)$.
Moreover, because our quantity of interest $\epsilon(f)$
is an expectation (of the random variable $\mathbf{1}(f(X) \neq Y)$)
and the corresponding estimator $\epsilon_\mathcal{D}(f)$
is the sample average,
estimating the popullation error
is simply the classic problem of mean estimation,
which you may recall from :numref:`sec_prob`.

An important classical result from probability theory
called the *central limit theorem* guarantees
that whenever we possess $n$ random samples $a_1, ..., a_n$
drawn from any distribution with mean $\mu$ and standard deviation $\sigma$,
as the number of samples $n$ approaches infinity,
the sample average $\hat{\mu}$ approximately
tends towards a normal distribution centered
at the true mean and with standard deviation $\sigma/\sqrt{n}$.
Already, this tells us something important:
as the number of examples grows large,
our test error $\epsilon_\mathcal{D}(f)$
should approach the true error $\epsilon(f)$
at a rate of $\mathcal{O}(1/\sqrt{n})$.
Thus, to estimate our test error twice as precisely,
we must collect four times as large a test set.
To reduce our test error by a factor of one hundred,
we must collect ten thousand times as large a test set.
In general, such a rate of $\mathcal{O}(1/\sqrt{n})$
is often the best we can hope for in statistics.

Now that we know something about the asymptotic rate
at which our test error $\epsilon_\mathcal{D}(f)$ converges to the true error $\epsilon(f)$,
we can zoom in on some important details.
Recall that the random variable of interest
$\mathbf{1}(f(X) \neq Y)$
can only take values $0$ and $1$
and thus is a Bernoulli random variable,
characterized by a parameter
indicating the probability that it takes value $1$.
Here, $1$ means that our classifier made an error,
so the parameter of our random variable
is actually the true error rate $\epsilon(f)$.
The variance $\sigma^2$ of a Bernoulli
depends on its parameter (here, $\epsilon(f)$)
according to the expression $\epsilon(f)(1-\epsilon(f))$.
While $\epsilon(f)$ is initially unknown,
we know that it cannot be greater than $1$.
A little investigation of this function
reveals that our variance is highest
when the true error rate is close to $0.5$
and can be far lower when it is
close to $0$ or close to $1$.
This tells us that the asymptotic standard deviation
of our estimate $\epsilon_\mathcal{D}(f)$ of the error $\epsilon(f)$
(over the choice of the $n$ test samples)
cannot be any greater than $\sqrt{0.25/n}$.

If we ignore the fact that this rate characterizes
behavior as the test set size approaches infinity
rather than when we possess finite samples,
this tells us that if we want our test error $\epsilon_\mathcal{D}(f)$
to approximate the population error $\epsilon(f)$
such that one standard deviation corresponds
to an interval of $\pm 0.01$,
then we should collect roughly 2500 samples.
If we want to fit two standard deviations
in that range and thus be 95%
that $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$,
then we will need 10000 samples!

This turns out to be the size of the test sets
for many popular benchmarks in machine learning.
You might be surprised to find out that thousands
of applied deep learning papers get published every year
making a big deal out of error rate improvements of $0.01$ or less.
Of course, when the error rates are much closer to $0$,
then an improvement of $0.01$ can indeed be a big deal.


One pesky feature of our analysis thus far
is that it really only tells us about asymptotics,
i.e., how the relationship between $\epsilon_\mathcal{D}$ and $\epsilon$
evolves as our sample size goes to infinity.
Fortunately, because our random variable is bounded,
we can obtain valid finite sample bounds
by applying an inequality due to Hoeffding (1963):

$$P(\epsilon_\mathcal{D}(f) - \epsilon(f) \geq t) < \exp\left( - 2n t^2 \right).$$

Solving for the smallest dataset size
that would allow us to conclude
with 95% confidence that the distance $t$
between our estimate $\epsilon_\mathcal{D}(f)$
and the true error rate $\epsilon(f)$
does not exceed $0.01$,
you will find that roughly $15000$ examples are required
as compared to the $10000$ examples suggested
by the asymptotic analysis above.
If you go deeper into statistics
you will find that this trend holds generally.
Guarantees that hold even in finite samples
are typically slightly more conservative.
Note that in the scheme of things,
these numbers are not so far apart,
reflecting the general usefulness
of asymptotic analysis for giving
us ballpark figures even if not
guarantees we can take to court.

## Test Set Reuse

In some sense, you are now set up to succeed
at conducting empirical machine learning research.
Nearly all practical models are developed
and validated based on test set performance
and you are now a master of the test set.
For any fixed classifier $f$,
you know to evaluate its test error $\epsilon_\mathcal{D}(f)$,
and know precisely what can (and can't)
be said about its population error $\epsilon(f)$.

So let's say that you take this knowledge
and prepare to train your first model $f_1$.
Knowing just how confident you need to be
in the performance of your classifier's error rate
you apply our analysis above to determine
an appropriate number of examples
to set aside for the test set.
Moreover, let's assume that you took the lessons from
:numref:`sec_generalization_basics` to heart
and made sure to preserve the sanctity of the test set
by conducting all of your preliminary analysis,
hyperparameter tuning, and even selection
among multiple competing model architectures
on a validation set.
Finally you evaluate your model $f_1$
on the test set and report an unbiased
estimate of the population error
with an associated confidence interval.

So far everything seems to be going well.
However, that night you wake up at 3am
with a brilliant idea for a new modeling approach.
The next day, you code up your new model,
tune its hyperparameters on the validation set
and not only are you getting your new model $f_2$ to work
but it's error rate appears to be much lower than $f_1$'s.
However, the thrill of discovery suddenly fades
as you prepare for the final evaluation.
You don't have a test set!

Even though the original test set $\mathcal{D}$
is still sitting on your server,
you now face two formidable problems.
First, when you collected your test set,
you determined the required level of precision
under the assumption that you were evaluating
a single classifier $f$.
However, if you get into the business
of evaluating multiple classifiers $f_1, ..., f_k$
on the same test set,
you must consider the problem of false discovery.
Before, you might have been 95% sure
that $\epsilon_\mathcal{D}(f) \in \epsilon(f) \pm 0.01$
for a single classifier $f$
and thus the probability of a misleading result
was a mere 5%.
With $k$ classifiers in the mix,
it can be hard to guarantee
that there is not even one among them
whose test set performance is misleading.
With 20 classifiers under consideration,
you might have no power at all
to rule out the possibility
that at least one among them
received a misleading score.
This problem relates to multiple hypothesis testing,
which despite a vast literature in statistics,
remains a persistent problem plaguing scientific research.


If that's not enough to worry you,
there's a special reason to distrust
the results that you get on subsequent evaluations.
Recall that our analysis of test set performance
rested on the assumption that the classifier
was chosen absent any contact with the test set
and thus we could view the test set
as drawn randomly from the underlying population.
Here, not only are you testing multiple functions,
the subsequent function $f_2$ was chosen
after you observed the test set performance of $f_1$.
Once information from the test set has leaked to the modeler,
it can never be a true test set again in the strictest sense.
This problem is called *adaptive overfitting* and has recently emerged
as a topic of intense interest to learning theorists and statisticians
:cite:`dwork2015preserving`.
Fortunately, while it is possible
to leak all information out of a holdout set,
and the theoretical worst case scenarios are bleak,
these analyses may be too conservative.
In practice, take care to create real test sets,
to consult them as infrequently as possible,
to account for multiple hypothesis testing
when reporting confidence intervals,
and to dial up your vigilance more aggressively
when the stakes are high and your dataset size is small.
When running a series of benchmark challenges,
it's often good practice to maintain
several test sets so that after each round,
the old test set can be demoted to a validation set.





## Statistical Learning Theory

At once, *test sets are all that we really have*,
and yet this fact seems strangely unsatisfying.
First, we seldom possess a *true test set*---unless
we are the ones creating the dataset,
someone else has probably already evaluated
their own classifier on our ostensible "test set".
And even when we get first dibs,
we soon find ourselves frustrated, wishing we could
evaluate our subsequent modeling attempts
without the gnawing feeling
that we cannot trust our numbers.
Moreover, even a true test set can only tell us *post hoc*
whether a classifier has in fact generalized to the population,
not whether we have any reason to expect *a priori*
that it should generalize.

With these misgivings in mind,
you might now be sufficiently primed
to see the appeal of *statistical learning theory*,
the mathematical subfield of machine learning
whose practitioners aim to elucidate the
fundamental principles that explain
why/when models trained on empirical data
can/will generalize to unseen data.
One of the primary aims for several decades
of statistical learning researchers
has been to bound the generalization gap,
relating the properties of the model class,
the number of samples in the dataset.

Learning theorists aim to bound the difference
between the *empirical error* $\epsilon_\mathcal{S}(f_\mathcal{S})$
of a learned classifier $f_\mathcal{S}$,
both trained and evaluated
on the training set $\mathcal{S}$,
and the true error $\epsilon(f_\mathcal{S})$
of that same classifier on the underlying population.
This might look similar to the evaluation problem
that we just addressed but there's a major difference.
Before, the classifier $f$ was fixed
and we only needed a dataset
for evaluative purposes.
And indeed, any fixed classifier does generalize:
its error on a (previously unseen) dataset
is an unbiased estimate of the population error.
But what can we say when a classifier
is trained and evaluated on the same dataset?
Can we ever be confident that the training error
will be close to the testing error?


Suppose that our learned classifier $f_\mathcal{S}$ must be chosen
among some pre-specified set of functions $\mathcal{F}$.
Recall from our discussion of test sets
that while it's easy to estimate
the error of a single classifier,
things get hairy when we begin
to consider collections of classifiers.
Even if the empirical error
of any one (fixed) classifier
will be close to its true error
with high probability,
once we consider a collection of classifiers,
we need to worry about the possibility
that *just one* classifier in the set
will receive a badly misestimated error.
The worry is that if just one classifier
in our collection receives
a misleadingly low error
then we might pick it
and thereby grossly underestimate
the population error.
Moreover, even for linear models,
because their parameters are continuously valued,
we are typically choosing among
an infinite class of functions ($|\mathcal{F}| = \infty$).

One ambitious solution to the problem
is to develop analytic tools
for proving uniform convergence, i.e.,
that with high probability,
the empirical error rate for every classifier in the class $f\in\mathcal{F}$
will *simultaneously* converge to its true error rate.
In other words, we seek a theoretical principle
that would allow us to state that
with probability at least $1-\delta$
(for some small $\delta$)
no classifier's error rate $\epsilon(f)$
(among all classifiers in the class $\mathcal{F}$)
will be misestimated by more
than some  small amount $\alpha$.
Clearly, we cannot make such statements
for all model classes $\mathcal{F}$.
Recall the class of memorization machines
that always achieve empirical error $0$
but never outperform random guessing
on the underlying population.

In a sense the class of memorizers is too flexible.
No such a uniform convergence result could possibly hold.
On the other hand, a fixed classifier is useless---it
generalizes perfectly, but fits neither
the training data nor the test data.
The central question of learning
has thus historically been framed as a tradeoff
between more flexible (higher variance) model classes
that better fit the training data but risk overfitting,
versus more rigid (higher bias) model classes
that generalize well but risk underfitting.
A central question in learning theory
has been to develop the appropriate
mathematical analysis to quantify
where a model sits along this spectrum,
and to provide the associated guarantees.

In a series of seminal papers,
Vapnik and Chervonenkis extended
the theory on the convergence
of relative frequencies
to more general classes of functions
:cite:`VapChe64,VapChe68,VapChe71,VapChe74b,VapChe81,VapChe91`.
One of the key contributions of this line of work
is the Vapnik-Chervonenkis (VC) dimension,
which measures (one notion of)
the complexity (flexibility) of a model class.
Moreover, one of their key results bounds
the difference between the empirical error
and the population error as a function
of the VC dimension and the number of samples:

$$P\left(R[p, f] - R_\mathrm{emp}[\mathbf{X}, \mathbf{Y}, f] < \alpha\right) \geq 1-\delta
\ \text{ for }\ \alpha \geq c \sqrt{(\mathrm{VC} - \log \delta)/n}.$$

Here $\delta > 0$ is the probability that the bound is violated,
$\alpha$ is the upper bound on the generalization gap,
and $n$ is the dataset size.
Lastly, $c > 0$ is a constant that depends
only on the scale of the loss that can be incurred.
One use of the bound might be to plug in desired
values of $\delta$ and $\alpha$
to determine how many samples to collect.
The VC dimension quantifies the largest
number of data points for which we can assign
any arbitrary (binary) labeling
and for each find some model $f$ in the class
that agrees with that labeling.
For example, linear models on $d$-dimensional inputs
have VC dimension $d+1$.
It's easy to see that a line can assign
any possible labeling to three points in two dimensions,
but not to four.
Unfortunately, the theory tends to be
overly pessimistic for more complex models
and obtaining this guarantee typically requires
far more examples than are actually required
to achieve the desired error rate.
Note also that fixing the model class and $\delta$,
our error rate again decays
with the usual $\mathcal{O}(1/\sqrt{n})$ rate.
It seems unlikely that we could do better in terms of $n$.
However, as we vary the model class,
VC dimension can present
a pessimistic picture
of the generalization gap.





## Summary

The most straightforward way to evaluate a model
is to consult a test set comprised of previously unseen data.
Test set evaluations provide an unbiased estimate of the true error
and converge at the desired $\mathcal{O}(1/\sqrt{n})$ rate as the test set grows.
We can provide approximate confidence intervals
based on exact asymptotic distributions
or valid finite sample confidence intervals
based on (more conservative) finite sample guarantees.
Indeed test set evaluation is the bedrock
of modern machine learning research.
However, test sets are seldom true test sets
(used by multiple researchers again and again).
Once the same test set is used
to evaluate multiple models,
controlling for false discovery can be difficult.
This can cause huge problems in theory.
In practice, the significance of the problem
depends on the size of the holdout sets in question
and whether they are merely being used to choose hyperparameters
or if they are leaking information more directly.
Nevertheless, it's good practice to curate real test sets (or multiple)
and to be as conservative as possible about how often they are used.


Hoping to provide a more satisfying solution,
statistical learning theorists have developed methods
for guaranteeing uniform convergence over a model class.
If indeed every model's empirical error
converges to its true error simultaneously,
then we are free to choose the model that performs
best, minimizing the training error,
knowing that it too will perform similarly well
on the holdout data.
Crucially, any of such results must depend
on some property of the model class.
Vladimir Vapnik and Alexey Chernovenkis
introduced the VC dimension,
presenting uniform convergence results
that hold for all models in a VC class.
The training errors for all models in the class
are (simultaneously) guaranteed
to be close to their true errors,
and guaranteed to grow closer
at $\mathcal{O}(1/\sqrt{n})$ rates.
Following the revolutionary discovery of VC dimension,
numerous alternative complexity measures have been proposed,
each facilitating an analogous generalization guarantee.
See :citet:`boucheron2005theory` for a detailed discussion
of several advanced ways of measuring function complexity.
Unfortunately, while these complexity measures
have become broadly useful tools in statistical theory,
they turn out to be powerless
(as straightforwardly applied)
for explaining why deep neural networks generalize.
Deep neural networks often have millions of parameters (or more),
and can easily assign random labels to large collections of points.
Nevertheless, they generalize well on practical problems
and, surprisingly, they often generalize better,
when they are larger and deeper,
despite incurring larger VC dimensions.
In the next chapter, we will revisit generalization
in the context of deep learning.

## Exercises

1. If we wish to estimate the error of a fixed model $f$
   to within $0.0001$ with probability greater than 99.9%,
   how many samples do we need?
1. Suppose that somebody else possesses a labeled test set
   $\mathcal{D}$ and only makes available the unlabeled inputs (features).
   Now suppose that you can only access the test set labels
   by running a model $f$ (no restrictions placed on the model class)
   on each of the unlabeled inputs
   and receiving the corresponding error $\epsilon_\mathcal{D}(f)$.
   How many models would you need to evaluate
   before you leak the entire test set
   and thus could appear to have error $0$,
   regardless of your true error?
1. What is the VC dimension of the class of $5^\mathrm{th}$-order polynomials?
1. What is the VC dimension of axis-aligned rectangles on two-dimensional data?

[Discussions](https://discuss.d2l.ai/t/6829)
