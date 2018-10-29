# Probability and statistics


In some form or another, machine learning is all about making predictions.
We might want to predict the *probability* of a patient suffering a heart attack in the next year,
given their clinical history.
In anomaly detection, we might want to assess how *likely* a set of readings from an airplane's jet engine would be,
were it operating normally.
In reinforcement learning, we want an agent to act intelligently in an environment.
This means we need to think about the probability of getting a high reward under each of the available action.
And when we build recommender systems we also need to think about probability.
For example, if we *hypothetically* worked for a large online bookseller,
we might want to estimate the probability that a particular user would buy a particular book, if prompted.
For this we need to use the language of probability and statistics.
Entire courses, majors, theses, careers, and even departments, are devoted to probability.
So our goal here isn't to teach the whole subject.
Instead we hope to get you off the ground,
to teach you just enough that you know everything necessary to start building your first machine learning models
and to have enough of a flavor for the subject that you can begin to explore it on your own if you wish.


We've talked a lot about probabilities so far without articulating what precisely they are or giving a concrete example. Let's get more serious by considering the problem of distinguishing cats and dogs based on photographs. This might sound simpler but it's actually a formidable challenge. To start with, the difficulty of the problem may depend on the resolution of the image.

| 10px | 20px | 40px | 80px | 160px |
|:----:|:----:|:----:|:----:|:-----:|
|![](../img/whitecat10.jpg)|![](../img/whitecat20.jpg)|![](../img/whitecat40.jpg)|![](../img/whitecat80.jpg)|![](../img/whitecat160.jpg)|
|![](../img/whitedog10.jpg)|![](../img/whitedog20.jpg)|![](../img/whitedog40.jpg)|![](../img/whitedog80.jpg)|![](../img/whitedog160.jpg)|

While it's easy for humans to recognize cats and dogs at 320 pixel resolution,
it becomes challenging at 40 pixels
and next to impossible at 10 pixels.
In other words, our ability to tell cats and dogs apart at a large distance (and thus low resolution)
might approach uninformed guessing.
Probability gives us a formal way of reasoning about our level of certainty.
If we are completely sure that the image depicts a cat,
we say that the *probability* that the corresponding label $l$ is $\mathrm{cat}$,
denoted $P(l=\mathrm{cat})$ equals 1.0.
If we had no evidence to suggest that $l =\mathrm{cat}$ or that $l = \mathrm{dog}$,
then we might say that the two possibilities were equally $likely$
expressing this as $P(l=\mathrm{cat}) = 0.5$.
If we were reasonably confident, but not sure that the image depicted a cat,
we might assign a probability $.5  < P(l=\mathrm{cat}) < 1.0$.

Now consider a second case:
given some weather monitoring data,
we want to predict the probability that it will rain in Taipei tomorrow.
If it's summertime, the rain might come with probability $.5$
In both cases, we have some value of interest.
And in both cases we are uncertain about the outcome.
But there's a key difference between the two cases.
In this first case, the image is in fact either a dog or a cat,
we just don't know which.
In the second case, the outcome may actually be a random event,
if you believe in such things (and most physicists do).
So probability is a flexible language for reasoning about our level of certainty,
and it can be applied effectively in a broad set of contexts.

## Basic probability theory

Say that we cast a die and want to know
what the chance is of seeing a $1$
rather than another digit.
If the die is fair, all six outcomes $\mathcal{X} = \{1, \ldots, 6\}$
are equally likely to occur,
hence we would see a $1$ in $1$ out of $6$ cases.
Formally we state that $1$ occurs with probability $\frac{1}{6}$.

For a real die that we receive from a factory,
we might not know those proportions
and we would need to check whether it is tainted.
The only way to investigate the die is by casting it many times
and recording the outcomes.
For each cast of the die,
we'll observe a value $\{1, 2, \ldots, 6\}$.
Given these outcomes, we want to investigate the probability of observing each outcome.

One natural approach for each value is to take the individual count for that value
and to divide it by the total number of tosses.
This gives us an *estimate* of the probability of a given event.
The law of large numbers tell us that as the number of tosses grows this estimate will draw closer and closer to the true underlying probability.
Before going into the details of what's going here, let's try it out.

To start, let's import the necessary packages:

```{.python .input}
import mxnet as mx
from mxnet import nd
```

Next, we'll want to be able to cast the die.
In statistics we call this process of drawing examples from probability distributions *sampling*.
The distribution which assigns probabilities to a number of discrete choices is called
the *multinomial* distribution.
We'll give a more formal definition of *distribution* later,
but at a high level, think of it as just an assignment of probabilities to events.
In MXNet, we can sample from the multinomial distribution via the aptly named `nd.random.multinomial` function.
The function can be called in many ways, but we'll focus on the simplest.
To draw a single sample, we simply give pass in a vector of probabilities.

```{.python .input}
probabilities = nd.ones(6) / 6
nd.random.multinomial(probabilities)
```

If you run the sampler a bunch of times,
you'll find that you get out random values each time.
As with estimating the fairness of a die,
we often want to generate many samples from the same distribution.
It would be really slow to do this with a Python `for` loop,
so `random.multinomial` supports drawing multiple samples at once,
returning an array of independent samples in any shape we might desire.

```{.python .input}
print(nd.random.multinomial(probabilities, shape=(10)))
print(nd.random.multinomial(probabilities, shape=(5,10)))
```

Now that we know how to sample rolls of a die,
we can simulate 1000 rolls. We can then go through and count, after each of the 1000 rolls,
how many times each number was rolled.

```{.python .input}
rolls = nd.random.multinomial(probabilities, shape=(1000))
counts = nd.zeros((6,1000))
totals = nd.zeros(6)
for i, roll in enumerate(rolls):
    totals[int(roll.asscalar())] += 1
    counts[:, i] = totals
```

To start, we can inspect the final tally at the end of $1000$ rolls.

```{.python .input}
totals / 1000
```

As you can see, the lowest estimated probability for any of the numbers is about $.15$
and the highest estimated probability is $0.188$.
Because we generated the data from a fair die,
we know that each number actually has probability of $1/6$, roughly $.167$,
so these estimates are pretty good.
We can also visualize how these probabilities converge over time
towards reasonable estimates.

To start let's take a look at the `counts`
array which has shape `(6, 1000)`.
For each time step (out of 1000),
counts, says how many times each of the numbers has shown up.
So we can normalize each $j$-th column of the counts vector by the number of tosses
to give the `current` estimated probabilities at that time.
The counts object looks like this:

```{.python .input}
counts
```

Normalizing by the number of tosses, we get:

```{.python .input}
x = nd.arange(1000).reshape((1,1000)) + 1
estimates = counts / x
print(estimates[:,0])
print(estimates[:,1])
print(estimates[:,100])
```

As you can see, after the first toss of the die, we get the extreme estimate that one of the numbers will be rolled with probability $1.0$ and that the others have probability $0$. After $100$ rolls, things already look a bit more reasonable.
We can visualize this convergence by using the plotting package `matplotlib`. If you don't have it installed, now would be a good time to [install it](https://matplotlib.org/).

```{.python .input}
%matplotlib inline
from matplotlib import pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

plt.figure(figsize=(8, 6))
for i in range(6):
    plt.plot(estimates[i, :].asnumpy(), label=("P(die=" + str(i) +")"))

plt.axhline(y=0.16666, color='black', linestyle='dashed')
plt.legend()
plt.show()   
```

Each solid curve corresponds to one of the six values of the die
and gives our estimated probability that the die turns up that value
as assessed after each of the 1000 turns.
The dashed black line gives the true underlying probability.
As we get more data, the solid curves converge towards the true answer.

In our example of casting a die, we introduced the notion of a **random variable**.
A random variable, which we denote here as $X$ can be pretty much any quantity and is not determistic.
Random variables could take one value among a set of possibilites.
We denote sets with brackets, e.g., $\{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}$.
The items contained in the set are called *elements*,
and we can say that an element $x$ is *in* the set S, by writing $x \in S$.
The symbol $\in$ is read as "in" and denotes membership.
For instance, we could truthfully say $\mathrm{dog} \in \{\mathrm{cat}, \mathrm{dog}, \mathrm{rabbit}\}$.
When dealing with the rolls of die, we are concerned with a variable $X \in \{1, 2, 3, 4, 5, 6\}$.

Note that there is a subtle difference between discrete random variables, like the sides of a dice,
and continuous ones, like the weight and the height of a person.
There's little point in asking whether two people have exactly the same height.
If we take precise enough measurements you'll find that no two people on the planet have the exact same height.
In fact, if we take a fine enough measurement,
you will not have the same height when you wake up and when you go to sleep.
So there's no purpose in asking about the probability
that someone is $2.00139278291028719210196740527486202$ meters tall.
Given the world population of humans the probability is virtually 0.
It makes more sense in this case to ask whether someone's height falls into a given interval,
say between 1.99 and 2.01 meters.
In these cases we quantify the likelihood that we see a value as a *density*.
The height of exactly 2.0 meters has no probability, but nonzero density.
In the interval between any two different heights we have nonzero probability.


There are a few important axioms of probability that you'll want to remember:

* For any event $z$, the probability is never negative, i.e. $\Pr(Z=z) \geq 0$.
* For any two events $Z=z$ and $X=x$ the union is no more likely than the sum of the individual events, i.e. $\Pr(Z=z \cup X=x) \leq \Pr(Z=z) + \Pr(X=x)$.
* For any random variable, the probabilities of all the values it can take must sum to 1, i.e. $\sum_{i=1}^n P(Z=z_i) = 1$.
* For any two mutually exclusive events $Z=z$ and $X=x$, the probability that either happens is equal to the sum of their individual probabilities that $\Pr(Z=z \cup X=x) = \Pr(Z=z) + \Pr(X=z)$.

## Dealing with multiple random variables

Very often, we'll want consider more than one random variable at a time.
For instance, we may want to model the relationship between diseases and symptoms.
Given a disease and symptom, say 'flu' and 'cough',
either may or may not occur in a patient with some probability.
While we hope that the probability of both would be close to zero,
we may want to estimate these probabilities and their relationships to each other
so that we may apply our inferences to effect better medical care.

As a more complicated example, images contain millions of pixels, thus millions of random variables.
And in many cases images will come with a label, identifying objects in the image.
We can also think of the label as a random variable.
We can even get crazy and think of all the metadata as random variables
such as location, time, aperture, focal length, ISO, focus distance, camera type, etc.
All of these are random variables that occur jointly.
When we deal with multiple random variables,
there are several quantities of interest.
The first is called the joint distribution $\Pr(A, B)$.
Given any elements $a$ and $b$,
the joint distribution lets us answer,
what is the probability that $A=a$ and $B=b$ simulataneously?
It might be clear that for any values $a$ and $b$, $\Pr(A,B) \leq \Pr(A=a)$.

This has to be the case, since for $A$ and $B$ to happen,
$A$ has to happen *and* $B$ also has to happen (and vice versa).
Thus $A,B$ cannot be more likely than $A$ or $B$ individually.
This brings us to an interesting ratio: $0 \leq \frac{\Pr(A,B)}{\Pr(A)} \leq 1$.
We call this a **conditional probability** and denote it by $\Pr(B|A)$,
the probability that $B$ happens, provided that $A$ has happened.

Using the definition of conditional probabilities,
we can derive one of the most useful and celebrated equations in statistics - Bayes' theorem.
It goes as follows: By construction, we have that $\Pr(A, B) = \Pr(B|A) \Pr(A)$.
By symmetry, this also holds for $\Pr(A,B) = \Pr(A|B) \Pr(B)$.
Solving for one of the conditional variables we get:
$$\Pr(A|B) = \frac{\Pr(B|A) \Pr(A)}{\Pr(B)}$$

This is very useful if we want to infer one thing from another,
say cause and effect but we only know the properties in the reverse direction.
One important operation that we need to make this work is **marginalization**, i.e.,
the operation of determining $\Pr(A)$ and $\Pr(B)$ from $\Pr(A,B)$.
We can see that the probability of seeing $A$ amounts to accounting
for all possible choices of $B$ and aggregating the joint probabilities over all of them, i.e.

$$\Pr(A) = \sum_{B'} \Pr(A,B') \text{ and } \Pr(B) = \sum_{A'} \Pr(A',B)$$

A really useful property to check is for **dependence** and **independence**.
Independence is when the occurrence of one event does not influence the occurrence of the other.
In this case $\Pr(B|A) = \Pr(B)$. Statisticians typically use $A \perp\!\!\!\perp B$ to express this.
From Bayes Theorem it follows immediately that also $\Pr(A|B) = \Pr(A)$.
In all other cases we call $A$ and $B$ dependent.
For instance, two successive rolls of a dice are independent.
On the other hand, the position of a light switch and the brightness in the room are not
(they are not perfectly deterministic, though,
since we could always have a broken lightbulb, power failure, or a broken switch).

Let's put our skills to the test.
Assume that a doctor administers an AIDS test to a patient.
This test is fairly accurate and it fails only with 1% probability
if the patient is healthy by reporting him as diseased. Moreover,
it never fails to detect HIV if the patient actually has it.
We use $D$ to indicate the diagnosis and $H$ to denote the HIV status.
Written as a table the outcome $\Pr(D|H)$ looks as follows:

|      outcome| HIV positive | HIV negative |
|:------------|-------------:|-------------:|
|Test positive|            1 |         0.01 |
|Test negative|            0 |         0.99 |

Note that the column sums are all one (but the row sums aren't),
since the conditional probability needs to sum up to $1$, just like the probability.
Let us work out the probability of the patient having AIDS if the test comes back positive.
Obviously this is going to depend on how common the disease is, since it affects the number of false alarms.
Assume that the population is quite healthy, e.g. $\Pr(\text{HIV positive}) = 0.0015$.
To apply Bayes Theorem we need to determine

$$\begin{aligned}
\Pr(\text{Test positive}) =& \Pr(D=1|H=0) \Pr(H=0) + \Pr(D=1|H=1) \Pr(H=1) \\
=& 0.01 \cdot 0.9985 + 1 \cdot 0.0015 \\
=& 0.011485
\end{aligned}
$$

Hence we get

$$\begin{aligned}
\Pr(H = 1|D = 1) =& \frac{\Pr(D=1|H=1) \Pr(H=1)}{\Pr(D=1)} \\
=& \frac{1 \cdot 0.0015}{0.011485} \\
=& 0.131
\end{aligned}
$$

In other words, there's only a 13.1% chance that the patient actually has AIDS, despite using a test that is 99% accurate! As we can see, statistics can be quite counterintuitive.

## Conditional independence

What should a patient do upon receiving such terrifying news?
Likely, he/she would ask the physician to administer another test to get clarity.
The second test has different characteristics (it isn't as good as the first one).

|     outcome |  HIV positive |  HIV negative |
|:------------|--------------:|--------------:|
|Test positive|          0.98 |          0.03 |
|Test negative|          0.02 |          0.97 |

Unfortunately, the second test comes back positive, too.
Let us work out the requisite probabilities to invoke Bayes' Theorem.

* $\Pr(D_1 = 1 \text{ and } D_2 = 1|H = 0) = 0.01 \cdot 0.03 = 0.0003$
* $\Pr(D_1 = 1 \text{ and } D_2 = 1|H = 1) = 1 \cdot 0.98 = 0.98$
* $\Pr(D_1 = 1 \text{ and } D_2 = 1) = 0.0001 \cdot 0.9985 + 0.98 \cdot 0.0015 = 0.00176955$
* $\Pr(H = 1|D_1 = 1 \text{ and } D_2 = 1) = \frac{0.98 \cdot 0.0015}{0.00176955} = 0.831$

That is, the second test allowed us to gain much higher confidence that not all is well.
Despite the second test being considerably less accurate than the first one,
it still improved our estimate quite a bit.
*Why couldn't we just run the first test a second time?*
After all, the first test was more accurate.
The reason is that we needed a second test that confirmed *independently* of the first test that things were dire, indeed. In other words, we made the tacit assumption that $\Pr(D_1, D_2|H) = \Pr(D_1|H) \Pr(D_2|H)$. Statisticians call such random variables **conditionally independent**. This is expressed as $D_1 \perp\!\!\!\perp D_2 | H$.

## Summary

So far we covered probabilities, independence, conditional independence, and how to use this to draw some basic conclusions. This is already quite powerful. In the next section we will see how this can be used to perform some basic estimation using a Naive Bayes classifier.

```{.python .input}

```
