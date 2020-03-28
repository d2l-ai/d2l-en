# Considering the Environment

In the previous chapters, we worked through 
a number of hands-on applications of machine learning,
fitting models to a variety of datasets.
And yet, we never stopped to contemplate
either where data comes from in the first place,
or what we plan to ultimately *do* 
with the outputs from our models.
Too often, machine learning developers
in possession of data rush to develop models
without pausing to consider these fundamental issues.

Many failed machine learning deployments 
can be traced back to this pattern.
Sometimes models appear to perform marvelously
as measured by test set accuracy
but fail catastrophically in deployment
when the distribution of data suddenly shifts.
More insidiously, sometimes the very deployment of a model
can be the catalyst that perturbs the data distribution.
Say, for example, that we trained a model 
to predict who will repay vs default on a loan,
finding that an applicant's choice of footware 
was associated with the risk of default
(Oxfords indicate repayment, sneakers indicate default).
We might be inclined to thereafter grant loans
to all applicants wearing Oxfords
and to deny all applicants wearing sneakers.

In this case, our ill-considered leap from 
pattern recognition to decision-making
and our failure to critically consider the environment
might have disastrous consequences.
For starters, as soon as we began 
making decisions based on footware,
customers would catch on and change their behavior.
Before long, all applicants would be wearing Oxfords,
without any coinciding improvement in credit-worthiness.
Take a minute to digest this because similar issues abound
in many applications of machine learning: 
by introducing our model-based decisions to the environment,
we might break the model.

While we cannot possible give these topics 
a complete treatment in one section,
we aim here to expose some common concerns,
and to stimulate the critical thinking
required to detect these situations early,
mitigate damage, and use machine learning responsibly.
Some of the solutions are simple
(ask for the "right" data)
some are technically difficult
(implement a reinforcement learning system),
and others require that step outside the realm of
statistical prediction altogether and 
grapple with difficult philosophical questions
concerning the ethical application of algorithms.


## Distribution Shift

To begin, we stick with the passive predictions setting
considering the various ways that data distributions might shift
and what might be done to salvage model performance.
In one classic setup, we assume that our training data
was sampled from some distribution $p_S(\mathbf{x},y)$
but that our test data will consist 
of unlabeled examples drawn from 
some different distribution $p_T(\mathbf{x},y)$.
Already, we must confront a sobering reality.
Absent any assumptions on how $p_S$ 
and $p_T$ relate to each other,
learning a robust classifier is impossible.

Consider a binary classification problem,
where we wish to distinguish between dogs and cats.
If the distribution can can shift in arbitrary ways,
then our setup permits the pathological case
in which the distribution over inputs remained
constant: $p_S(\mathbf{x}) = p_T(\mathbf{x})$
but the labels are all flipped 
$p_S(y | \mathbf{x}) = 1 - $p_T(y | \mathbf{x})$. 
In other words, if God can suddenly decide
that in the future all "cats" are now dogs
and what we previously called "dogs" are now cats---without
any change in the distribution of inputs $p(\mathbf{x})$,
then we cannot possibly distinguish this setting 
from one in which the distribution did not change at all. 

Fortunately, under some restricted assumptions
on the ways our data might change in the future,
principled algorithms can detect shift 
and sometimes even adapt on the fly, 
improving on the accuracy of the original classifier.


### Covariate Shift

Among categories of distribution shift,
*covariate shift* may be the most widely studied.
Here, we assume that while the distribution of inputs 
may change over time, the labeling function, 
i.e., the conditional distribution 
$P(y \mid \mathbf{x})$ does not change.
Statisticians call this *covariate shift*
because the problem arises due to a 
a shift in the distribution of the *covariates* (the features).
While we can sometimes reason about distribution shift
without invoking causality, we note that covariate shift
is the natural assumption to invoke in settings
where we believe that $\mathbf{x}$ causes $y$.

Consider the challenge of distinguishing cats and dogs.
Our training data might consist of images of the following kind:

|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat3.jpg)|![](../img/cat2.jpg)|![](../img/dog1.jpg)|![](../img/dog2.jpg)|

At test time we are asked to classify the following images:

|cat|cat|dog|dog|
|:---------------:|:---------------:|:---------------:|:---------------:|
|![](../img/cat-cartoon1.png)|![](../img/cat-cartoon2.png)|![](../img/dog-cartoon1.png)|![](../img/dog-cartoon2.png)|

The training set consists of photos,
while the test set contains only cartoons.
Training on a dataset with substantially different
characteristics from the test set
can spell trouble absent a coherent plan 
for how to adapt to the new domain.


### Label Shift
 
*Label shift* describes the converse problem.
Here, we assume that the label marginal $P(y)$
can change (inducing a change in $P(\mathbf{x})$)
but the class-conditional distribution 
$P(\mathbf{x} \mid y)$ remains fixed across domains.
Label shift is a reasonable assumption to make
when we believe that $y$ causes $\mathbf{x}$.
For example, we may want to predict diagnoses 
given their symptoms (or other manifestations),
even as the relative prevalence of diagnoses 
are changing over time.
Label shift is the appropriate assumption here
because diseases cause symptoms.
In some degenerate cases the label shift 
and covariate shift assumptions can hold simultaneously.
For example, when the label is deterministic,
then covariate shift assumption will be satisfied,
even when $y$ causes $\mathbf{x}$.
Interestingly, in these cases,
it is often advantageous to work with the methods
that flow from the label shift assumption.
That is because these methods tend 
to involve manipulating objects that look like the label
(which is often low-dimensional),
as opposed to objects that look like the input,
which tends (in deep learning) tends to be high-dimensional.



### Concept Shift

We may also encounter the related problem of *concept shift*,
which arises when the very definitions of labels can change.
This sounds weird---a *cat* is a *cat*, no?
However, other categories are subject to changes in usage over time.
Diagnostic criteria for mental illness,
what passes for fashionable, and job titles,
are all subject to considerable 
amounts of *concept shift*.
It turns out that if we navigate around the United States,
shifting the source of our data by geography,
we will find considerable concept shift regarding
the distribution of names for *soft drinks*
as shown in :numref:`fig_popvssoda`.

![Concept shift on soft drink names in the United States.](../img/popvssoda.png)
:width:`400px`
:label:`fig_popvssoda`

If we were to build a machine translation system,
the distribution $P(y \mid x)$ might be different
depending on our location.
This problem can be tricky to spot.
We might hope to exploit knowledge
that shift only takes place gradually
(either in a temporal or geographic sense).



### Examples

Before delving into formalism and algorithms,
we can discuss a some concrete situations
where covariate or concept shift might not be obvious.


#### Medical Diagnostics

Imagine that you want to design an algorithm to detect cancer.
You collect data from healthy and sick people
and you train your algorithm.
It works fine, giving you high accuracy
and you conclude that you’re ready
for a successful career in medical diagnostics.
*Not so fast.*

The distributions that gave rise to the training data
and those you will encounter in the wild might differ considerably.
This happened to an unfortunate startup
that one of us worked with years ago.
They were developing a blood test for a disease
that predominantly affects older men 
and hoped to study it using blood samples
that they had collected from patients.
However, it's considerably more difficult
to obtain blood samples from healthy men 
than sick patients already in the system.
To compensate, the startup solicited
blood donations from students on a university campus
to serve as healthy controls in developing their test.
Then they asked whether we could help them
to build a classifier for detecting the disease.

As we explained to them,
it would indeed be easy to distinguish
between the healthy and sick cohorts
with near-perfect accuracy.
However, that's because the test subjects 
differed in age, hormone levels,
physical activity, diet, alcohol consumption,
and many more factors unrelated to the disease.
This was unlikely to be the case with real patients.
Due to their sampling procedure,
we could expect to encounter extreme covariate shift. 
Moreover, this case was unlikely to be
correctable via conventional methods.
In short, they wasted a significant sum of money.

#### Self Driving Cars

Say a company wanted to leverage machine learning
for developing self-driving cars.
One key component here is a roadside detector.
Since real annotated data is expensive to get,
they had the (smart and questionable) idea
to use synthetic data from a game rendering engine
as additional training data.
This worked really well on "test data" 
drawn from the rendering engine.
Alas, inside a real car it was a disaster.
As it turned out, the roadside had been rendered
with a very simplistic texture.
More importantly, *all* the roadside had been rendered
with the *same* texture and the roadside detector
learned about this "feature" very quickly.

A similar thing happened to the US Army
when they first tried to detect tanks in the forest.
They took aerial photographs of the forest without tanks,
then drove the tanks into the forest
and took another set of pictures.
The classifier appeared to work *perfectly*.
Unfortunately, it had merely learned
how to distinguish trees with shadows
from trees without shadows---the first set
of pictures was taken in the early morning,
the second one at noon.

#### Nonstationary distributions

A much more subtle situation arises
when the distribution changes slowly
and the model is not updated adequately.
Here are some typical cases:

* We train a computational advertising model and then fail to update it frequently (e.g., we forget to incorporate that an obscure new device called an iPad was just launched).
* We build a spam filter. It works well at detecting all spam that we have seen so far. But then the spammers wisen up and craft new messages that look unlike anything we have seen before.
* We build a product recommendation system. It works throughout the winter but then continues to recommend Santa hats long after Christmas.

#### More Anecdotes

* We build a face detector. It works well on all benchmarks. Unfortunately it fails on test data---the offending examples are close-ups where the face fills the entire image (no such data was in the training set).
* We build a web search engine for the USA market and want to deploy it in the UK.
* We train an image classifier by compiling a large dataset where each among a large set of classes is equally represented in the dataset, say 1000 categories, represented by 1000 images each. Then we deploy the system in the real world, where the actual label distribution of photographs is decidedly non-uniform.

In short, there are many cases 
where training and test distributions
$p(\mathbf{x}, y)$ are different.
In some cases, we get lucky and the models work
despite covariate, label, or concept shift.
In other cases, we can do better by employing
principled strategies to cope with the shift.
The remainder of this section grows considerably more technical.
The impatient reader could continue on to the next section
as this material is not prerequisite to subsequent concepts.

### Covariate Shift Correction

Assume that we want to estimate 
some dependency $P(y \mid \mathbf{x})$
for which we have labeled data $(\mathbf{x}_i, y_i)$.
Unfortunately, the observations $x_i$ are drawn
from some *target* distribution $q(\mathbf{x})$
rather than the *source* distribution $p(\mathbf{x})$.
To make progress, we need to reflect about what exactly
is happening during training:
we iterate over training data and associated labels
$\{(\mathbf{x}_1, y_1), \ldots, (\mathbf{x}_n, y_n)\}$
and update the weight vectors of the model after every minibatch.
We sometimes additionally apply some penalty to the parameters,
using weight decay, dropout, or some other related technique.
This means that we largely minimize the loss on the training.

$$
\mathop{\mathrm{minimize}}_w \frac{1}{n} \sum_{i=1}^n l(x_i, y_i, f(x_i)) + \mathrm{some~penalty}(w).
$$

Statisticians call the first term an *empirical average*,
i.e., an average computed over the data drawn from $P(x) P(y \mid x)$.
If the data is drawn from the "wrong" distribution $q$,
we can correct for that by using the following simple identity:

$$
\begin{aligned}
\int p(\mathbf{x}) f(\mathbf{x}) dx 
& = \int q(\mathbf{x}) f(\mathbf{x}) \frac{p(\mathbf{x})}{q(\mathbf{x})} dx.
\end{aligned}
$$

In other words, we need to re-weight each instance
by the ratio of probabilities
that it would have been drawn from the correct distribution
$\beta(\mathbf{x}) := p(\mathbf{x})/q(\mathbf{x})$.
Alas, we do not know that ratio,
so before we can do anything useful we need to estimate it.
Many methods are available,
including some fancy operator-theoretic approaches
that attempt to recalibrate the expectation operator directly
using a minimum-norm or a maximum entropy principle.
Note that for any such approach, we need samples
drawn from both distributions---the "true" $p$, e.g.,
by access to training data, and the one used
for generating the training set $q$ (the latter is trivially available).
Note however, that we only need samples $\mathbf{x} \sim q(\mathbf{x})$;
we do not to access labels $y \sim q(y)$.

In this case, there exists a very effective approach
that will give almost as good results: logistic regression.
This is all that is needed to compute estimate probability ratios.
We learn a classifier to distinguish 
between data drawn from $p(\mathbf{x})$
and data drawn from $q(\mathbf{x})$.
If it is impossible to distinguish 
between the two distributions
then it means that the associated instances 
are equally likely to come from 
either one of the two distributions.
On the other hand, any instances 
that can be well discriminated
should be significantly overweighted 
or underweighted accordingly.
For simplicity’s sake assume that we have 
an equal number of instances from both distributions, 
denoted by $\mathbf{x}_i \sim p(\mathbf{x})$ 
and $\mathbf{x}_i' \sim q(\mathbf{x})$, respectively.
Now denote by $z_i$ labels which are 1
for data drawn from $p$ and -1 for data drawn from $q$.
Then the probability in a mixed dataset is given by

$$P(z=1 \mid \mathbf{x}) = \frac{p(\mathbf{x})}{p(\mathbf{x})+q(\mathbf{x})} \text{ and hence } \frac{P(z=1 \mid \mathbf{x})}{P(z=-1 \mid \mathbf{x})} = \frac{p(\mathbf{x})}{q(\mathbf{x})}.$$

Hence, if we use a logistic regression approach,
where $P(z=1 \mid \mathbf{x})=\frac{1}{1+\exp(-f(\mathbf{x}))}$.
it follows that

$$
\beta(\mathbf{x}) = \frac{1/(1 + \exp(-f(\mathbf{x})))}{\exp(-f(\mathbf{x}))/(1 + \exp(-f(\mathbf{x})))} = \exp(f(\mathbf{x})).
$$

As a result, we need to solve two problems:
first one to distinguish between
data drawn from both distributions,
and then a reweighted minimization problem
where we weigh terms by $\beta$, e.g., via the head gradients.
Here's a prototypical algorithm for that purpose
which uses an unlabeled training set $X$ and test set $Z$:

1. Generate training set with $\{(\mathbf{x}_i, -1) ... (\mathbf{z}_j, 1)\}$.
1. Train binary classifier using logistic regression to get function $f$.
1. Weigh training data using $\beta_i = \exp(f(\mathbf{x}_i))$ or better $\beta_i = \min(\exp(f(\mathbf{x}_i)), c)$.
1. Use weights $\beta_i$ for training on $X$ with labels $Y$.

Note that this method relies on a crucial assumption.
For this scheme to work, we need that each data point
in the target (test time) distribution
had nonzero probability of occurring at training time.
If we find a point where $q(\mathbf{x}) > 0$ but $p(\mathbf{x}) = 0$,
then the corresponding importance weight should be infinity.

*Generative Adversarial Networks*
use a very similar idea to that described above
to engineer a *data generator* that outputs data
that cannot be distinguished
from examples sampled from a reference dataset.
In these approaches, we use one network, $f$
to distinguish real versus fake data
and a second network $g$ that tries 
to fool the discriminator $f$
into accepting fake data as real.
We will discuss this in much more detail later.


### Label Shift Correction

Assume that we are dealing with a 
$k$-way multiclass classification task.
When the distribution of labels shifts over time,
$p(y) \neq q(y)$ but the class-conditional distributions 
stay the same $p(\mathbf{x})=q(\mathbf{x})$.
Here, our importance weights will correspond to the
label likelihood ratios $q(y)/p(y)$.
One nice thing about label shift is that
if we have a reasonably good model 
(on the source distribution)
then we can get consistent estimates of these weights
without ever having to deal with the ambient dimension
In deep learning, the inputs tend 
to be high-dimensional objects like images,
while the labels are often simpler objects like categories.

To estimate calculate the target label distribution,
we first take our reasonably good off the shelf classifier
(typically trained on the training data)
and compute its confusion matrix using the validation set
(also from the training distribution).
The confusion matrix C, is simply a $k \times k$ matrix,
where each column corresponds to the *actual* label
and each row corresponds to our model's predicted label.
Each cell's value $c_{ij}$ is the fraction of predictions
where the true label was $j$ *and* our model predicted $y$.

Now, we cannot calculate the confusion matrix
on the target data directly,
because we do not get to see the labels for the examples
that we see in the wild,
unless we invest in a complex real-time annotation pipeline.
What we can do, however, is average all of our models predictions
at test time together, yielding the mean model output $\mu_y$.

It turns out that under some mild conditions---if 
our classifier was reasonably accurate in the first place,
and if the target data contains only classes of images 
that we have seen before,
and if the label shift assumption holds in the first place
(the strongest assumption here),
then we can recover the test set label distribution
by solving a simple linear system $C \cdot q(y) = \mu_y$.
If our classifier is sufficiently accurate to begin with,
then the confusion $C$ will be invertible,
and we get a solution $q(y) = C^{-1} \mu_y$.
Here we abuse notation a bit, using $q(y)$
to denote the vector of label frequencies.
Because we observe the labels on the source data,
it is easy to estimate the distribution $p(y)$.
Then for any training example $i$ with label $y$,
we can take the ratio of our estimates $\hat{q}(y)/\hat{p}(y)$
to calculate the weight $w_i$,
and plug this into the weighted risk minimization algorithm above.


### Concept Shift Correction

Concept shift is much harder to fix in a principled manner.
For instance, in a situation where suddenly the problem changes
from distinguishing cats from dogs to one of
distinguishing white from black animals,
it will be unreasonable to assume
that we can do much better than just collecting new labels
and training from scratch.
Fortunately, in practice, such extreme shifts are rare.
Instead, what usually happens is that the task keeps on changing slowly.
To make things more concrete, here are some examples:

* In computational advertising, new products are launched, 
old products become less popular. This means that the distribution over ads and their popularity changes gradually and any click-through rate predictor needs to change gradually with it.
* Traffic cameras lenses degrade gradually due to environmental wear, affecting image quality progressively.
* News content changes gradually (i.e., most of the news remains unchanged but new stories appear).

In such cases, we can use the same approach that we used for training networks to make them adapt to the change in the data. In other words, we use the existing network weights and simply perform a few update steps with the new data rather than training from scratch.

## A Taxonomy of Learning Problems

Armed with knowledge about how to deal with changes in $p(x)$ and in $P(y \mid x)$, we can now consider some other aspects of machine learning problems formulation.


* **Batch Learning.** Here we have access to training data and labels $\{(x_1, y_1), \ldots, (x_n, y_n)\}$, which we use to train a network $f(x, w)$. Later on, we deploy this network to score new data $(x, y)$ drawn from the same distribution. This is the default assumption for any of the problems that we discuss here. For instance, we might train a cat detector based on lots of pictures of cats and dogs. Once we trained it, we ship it as part of a smart catdoor computer vision system that lets only cats in. This is then installed in a customer's home and is never updated again (barring extreme circumstances).
* **Online Learning.** Now imagine that the data $(x_i, y_i)$ arrives one sample at a time. More specifically, assume that we first observe $x_i$, then we need to come up with an estimate $f(x_i, w)$ and only once we have done this, we observe $y_i$ and with it, we receive a reward (or incur a loss), given our decision. Many real problems fall into this category. E.g. we need to predict tomorrow's stock price, this allows us to trade based on that estimate and at the end of the day we find out whether our estimate allowed us to make a profit. In other words, we have the following cycle where we are continuously improving our model given new observations.

$$
\mathrm{model} ~ f_t \longrightarrow
\mathrm{data} ~ x_t \longrightarrow
\mathrm{estimate} ~ f_t(x_t) \longrightarrow
\mathrm{observation} ~ y_t \longrightarrow
\mathrm{loss} ~ l(y_t, f_t(x_t)) \longrightarrow
\mathrm{model} ~ f_{t+1}
$$

* **Bandits.** They are a *special case* of the problem above. While in most learning problems we have a continuously parametrized function $f$ where we want to learn its parameters (e.g., a deep network), in a bandit problem we only have a finite number of arms that we can pull (i.e., a finite number of actions that we can take). It is not very surprising that for this simpler problem stronger theoretical guarantees in terms of optimality can be obtained. We list it mainly since this problem is often (confusingly) treated as if it were a distinct learning setting.
* **Control (and nonadversarial Reinforcement Learning).** In many cases the environment remembers what we did. Not necessarily in an adversarial manner but it'll just remember and the response will depend on what happened before. E.g. a coffee boiler controller will observe different temperatures depending on whether it was heating the boiler previously. PID (proportional integral derivative) controller algorithms are a popular choice there. Likewise, a user's behavior on a news site will depend on what we showed him previously (e.g., he will read most news only once). Many such algorithms form a model of the environment in which they act such as to make their decisions appear less random (i.e., to reduce variance).
* **Reinforcement Learning.** In the more general case of an environment with memory, we may encounter situations where the environment is trying to *cooperate* with us (cooperative games, in particular for non-zero-sum games), or others where the environment will try to *win*. Chess, Go, Backgammon or StarCraft are some of the cases. Likewise, we might want to build a good controller for autonomous cars. The other cars are likely to respond to the autonomous car's driving style in nontrivial ways, e.g., trying to avoid it, trying to cause an accident, trying to cooperate with it, etc.

One key distinction between the different situations above is that the same strategy that might have worked throughout in the case of a stationary environment, might not work throughout when the environment can adapt. For instance, an arbitrage opportunity discovered by a trader is likely to disappear once he starts exploiting it. The speed and manner at which the environment changes determines to a large extent the type of algorithms that we can bring to bear. For instance, if we *know* that things may only change slowly, we can force any estimate to change only slowly, too. If we know that the environment might change instantaneously, but only very infrequently, we can make allowances for that. These types of knowledge are crucial for the aspiring data scientist to deal with concept shift, i.e., when the problem that he is trying to solve changes over time.


## Fairness, Accountability, and Transparency in Machine Learning

Finally, it is important to remember
that when you deploy machine learning systems
you are not merely optimizing a predictive model---you 
are typically providing a tool that will
be used to (partially or fully) automate decisions.
This technical systems can impact the lives
of individuals subject to the resulting decisions.
The leap from considering predictions to decisions
raises not only new technical questions,
but also a slew of ethical questions 
that must be carefully considered.
If we are deploying a medical diagnostic system,
we need to know for which populations 
it may work and which it may not.
Overlooking foreseeable risks to the welfare of 
a subpopulation could cause us to administer inferior care.
Moreover, once we contemplate decision-making systems,
we must step back and reconsider how we evaluate our technology.
Among other consequences of this change of scope,
we will find that *accuracy* is seldom the right metric.
For instance, when translating predictions into actions,
we will often want to take into account 
the potential cost sensitivity of erring in various ways.
If one way of misclassifying an image 
could be perceived as a racial sleight,
while misclassification to a different category
would be harmless, then we might want to adjust 
our thresholds accordingly, accounting for societal values
in designing the decision-making protocol.
We also want to be careful about 
how prediction systems can lead to feedback loops.
For example, consider predictive policing systems,
which allocate patrol officers 
to areas with high forecasted crime.
It's easy to see how a worrying pattern can emerge:

 1. Neighborhoods with more crime get more patrols. 
 1. Consequently, more crimes are discovered in these neighborhoods, entering the training data available for future iterations.
 1. Exposed to more positives, the model predicts yet more crime in these neighborhoods.
 1. In the next iteration, the updated model targets the same neighborhood even more heavily leading to yet more crimes discovered, etc.

Often, the various mechanisms by which 
a model's predictions become coupled to its training data 
are unaccounted for in the modeling process. 
This can lead to what researchers call "runaway feedback loops." 
Additionally, we want to be careful about 
whether we are addressing the right problem in the first place. 
Predictive algorithms now play an outsize role 
in mediating the dissemination of information.
Should the news that an individual encounters
be determined by the set of Facebook pages they have *Liked*? 
These are just a few among the many pressing ethical dilemmas
that you might encounter in a career in machine learning.



## Summary

* In many cases training and test set do not come from the same distribution. This is called covariate shift.
* Under the corresponding assumptions, *covariate* and *label* shift can be detected and corrected for at test time. Failure to account for this bias can become problematic at test time.
* In some cases, the environment may *remember* automated actions and respond in surprising ways. We must account for this possibility when building models and continue to monitor live systems, open to the possibility that our models and the environment will become entangled in unanticipated ways.

## Exercises

1. What could happen when we change the behavior of a search engine? What might the users do? What about the advertisers?
1. Implement a covariate shift detector. Hint: build a classifier.
1. Implement a covariate shift corrector.
1. What could go wrong if training and test set are very different? What would happen to the sample weights?

## [Discussions](https://discuss.mxnet.io/t/2347)

![](../img/qr_environment.svg)
