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

plt.figure(figsize=(15, 8))
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

<!-- What we can see is that the red curves pretty well capture the behavior of the 10 random traces of averages. This is the case since we are averaging numbers and their aggregate behavior is like that of a number with a lot less uncertainty. Looking at the red curves, they are given by $f(x) = \pm 1/\sqrt{x}$. (The reader might cry foul by noting that we just added Gaussian random variables which, quite obviously, lead to yet another Gaussian random variable. That said, the curves for sums of other random variables, such as $\{0, 1\}$ valued objects look identical in the limit.) -->


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
* For any random variable, the probabilities of all the values it can take must sum to 1 $\sum_{i=1}^n P(Z=z_i) = 1$.
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

|             | HIV positive | HIV negative |
|:------------|-------------:|-------------:|
|Test positive|            1 |         0.01 |
|Test negative|            0 |         0.99 |

Note that the column sums are all one (but the row sums aren't),
since the conditional probability needs to sum up to $1$, just like the probability.
Let us work out the probability of the patient having AIDS if the test comes back positive.
Obviously this is going to depend on how common the disease is, since it affects the number of false alarms.
Assume that the population is quite healthy, e.g. $\Pr(\text{HIV positive}) = 0.0015$.
To apply Bayes Theorem we need to determine

$$\begin{eqnarray}
\Pr(\text{Test positive}) =& \Pr(D=1|H=0) \Pr(H=0) + \Pr(D=1|H=1) \Pr(H=1) \\
=& 0.01 \cdot 0.9985 + 1 \cdot 0.0015 \\
=& 0.011485
\end{eqnarray}
$$

Hence we get

$$\begin{eqnarray}
\Pr(H = 1|D = 1) =& \frac{\Pr(D=1|H=1) \Pr(H=1)}{\Pr(D=1)} \\
=& \frac{1 \cdot 0.0015}{0.011485} \\
=& 0.131
\end{eqnarray}$$

In other words, there's only a 13.1% chance that the patient actually has AIDS, despite using a test that is 99% accurate! As we can see, statistics can be quite counterintuitive.

## Conditional independence

What should a patient do upon receiving such terrifying news?
Likely, he/she would ask the physician to administer another test to get clarity.
The second test has different characteristics (it isn't as good as the first one).

|             |  HIV positive |  HIV negative |
|:------------|------------------------:|------------------------:|
|Test positive| 0.98 | 0.03 |
|Test negative| 0.02 | 0.97 |

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

## Naive Bayes classification

Conditional independence is useful when dealing with data, since it simplifies a lot of equations.
A popular algorithm is the Naive Bayes Classifier.
The key assumption in it is that the attributes are all independent of each other, given the labels.
In other words, we have:

$$p(x|y) = \prod_i p(x_i|y)$$

Using Bayes Theorem this leads to the classifier $p(y|x) = \frac{\prod_i p(x_i|y) p(y)}{p(x)}$. Unfortunately, this is still intractable, since we don't know $p(x)$. Fortunately, we don't need it, since we know that $\sum_y p(y|x) = 1$, hence we can always recover the normalization from
$$p(y|x) \propto \prod_i p(x_i|y) p(y).$$
After all that math, it's time for some code to show how to use a Naive Bayes classifier for distinguishing digits on the MNIST classification dataset.

The problem is that we don't actually know $p(y)$ and $p(x_i|y)$. So we need to *estimate* it given some training data first. This is what is called *training* the model. In the case of 10 possible classes we simply compute $n_y$, i.e. the number of occurrences of class $y$ and then divide it by the total number of occurrences. E.g. if we have a total of 60,000 pictures of digits and digit 4 occurs 5800 times, we estimate its probability as $\frac{5800}{60000}$. Likewise, to get an idea of $p(x_i|y)$ we count how many times pixel $i$ is set for digit $y$ and then divide it by the number of occurrences of digit $y$. This is the probability that that very pixel will be switched on.

```{.python .input}
import numpy as np

# we go over one observation at a time (speed doesn't matter here)
def transform(data, label):
    return (nd.floor(data/128)).astype(np.float32), label.astype(np.float32)
mnist_train = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = mx.gluon.data.vision.MNIST(train=False, transform=transform)

# Initialize the count statistics for p(y) and p(x_i|y)
# We initialize all numbers with a count of 1 to ensure that we don't get a
# division by zero.  Statisticians call this Laplace smoothing.
ycount = nd.ones(shape=(10))
xcount = nd.ones(shape=(784, 10))

# Aggregate count statistics of how frequently a pixel is on (or off) for
# zeros and ones.
for data, label in mnist_train:
    x = data.reshape((784,))
    y = int(label)
    ycount[y] += 1
    xcount[:, y] += x

# normalize the probabilities p(x_i|y) (divide per pixel counts by total
# count)
for i in range(10):
    xcount[:, i] = xcount[:, i]/ycount[i]

# likewise, compute the probability p(y)
py = ycount / nd.sum(ycount)
```

Now that we computed per-pixel counts of occurrence for all pixels, it's time to see how our model behaves. Time to plot it. We show the estimated probabilities of observing a switched-on pixel. These are some mean looking digits.

```{.python .input}
import matplotlib.pyplot as plt
fig, figarr = plt.subplots(1, 10, figsize=(15, 15))
for i in range(10):
    figarr[i].imshow(xcount[:, i].reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[i].axes.get_xaxis().set_visible(False)
    figarr[i].axes.get_yaxis().set_visible(False)

plt.show()
print(py)
```

Now we can compute the likelihoods of an image, given the model. This is statistican speak for $p(x|y)$, i.e. how likely it is to see a particular image under certain conditions (such as the label). Since this is computationally awkward (we might have to multiply many small numbers if many pixels have a small probability of occurring), we are better off computing its logarithm instead. That is, instead of $p(x|y) = \prod_{i} p(x_i|y)$ we compute $\log p(x|y) = \sum_i \log p(x_i|y)$.

$$\begin{eqnarray}
l_y :=& \sum_i \log p(x_i|y) \\
 = & \sum_i x_i \log p(x_i = 1|y) + (1-x_i) \log \left(1-p(x_i=1|y)\right)
 \end{eqnarray}$$

To avoid recomputing logarithms all the time, we precompute them for all pixels.

```{.python .input}
logxcount = nd.log(xcount)
logxcountneg = nd.log(1-xcount)
logpy = nd.log(py)

fig, figarr = plt.subplots(2, 10, figsize=(15, 3))

# show 10 images
ctr = 0
for data, label in mnist_test:
    x = data.reshape((784,))
    y = int(label)

    # we need to incorporate the prior probability p(y) since p(y|x) is
    # proportional to p(x|y) p(y)
    logpx = logpy.copy()
    for i in range(10):
        # compute the log probability for a digit
        logpx[i] += nd.dot(logxcount[:, i], x) + nd.dot(logxcountneg[:, i], 1-x)
    # normalize to prevent overflow or underflow by subtracting the largest
    # value
    logpx -= nd.max(logpx)
    # and compute the softmax using logpx
    px = nd.exp(logpx).asnumpy()
    px /= np.sum(px)

    # bar chart and image of digit
    figarr[1, ctr].bar(range(10), px)
    figarr[1, ctr].axes.get_yaxis().set_visible(False)
    figarr[0, ctr].imshow(x.reshape((28, 28)).asnumpy(), cmap='hot')
    figarr[0, ctr].axes.get_xaxis().set_visible(False)
    figarr[0, ctr].axes.get_yaxis().set_visible(False)
    ctr += 1
    if ctr == 10:
        break

plt.show()
```

As we can see, this classifier is both incompetent and overly confident of its incorrect estimates. That is, even if it is horribly wrong, it generates probabilities close to 1 or 0. Not a classifier we should use very much nowadays any longer. While Naive Bayes classifiers used to be popular in the 80s and 90s, e.g. for spam filtering, their heydays are over. The poor performance is due to the incorrect statistical assumptions that we made in our model: we assumed that each and every pixel are *independently* generated, depending only on the label. This is clearly not how humans write digits, and this wrong assumption led to the downfall of our overly naive (Bayes) classifier.

## Sampling

Random numbers are just one form of random variables, and since computers are particularly good with numbers, pretty much everything else in code ultimately gets converted to numbers anyway. One of the basic tools needed to generate random numbers is to sample from a distribution. Let's start with what happens when we use a random number generator.

```{.python .input}
import random
for i in range(10):
    print(random.random())
```

### Uniform Distribution

These are some pretty random numbers. As we can see, their range is between 0 and 1, and they are evenly distributed. That is, there is (actually, should be, since this is not a *real* random number generator) no interval in which numbers are more likely than in any other. In other words, the chances of any of these numbers to fall into the interval, say $[0.2,0.3)$ are as high as in the interval $[.593264, .693264)$. The way they are generated internally is to produce a random integer first, and then divide it by its maximum range. If we want to have integers directly, try the following instead. It generates random numbers between 0 and 100.

```{.python .input}
for i in range(10):
    print(random.randint(1, 100))
```

What if we wanted to check that ``randint`` is actually really uniform. Intuitively the best strategy would be to run it, say 1 million times, count how many times it generates each one of the values and to ensure that the result is uniform.

```{.python .input}
import math

counts = np.zeros(100)
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
axes = axes.reshape(6)
# mangle subplots such that we can index them in a linear fashion rather than
# a 2d grid

for i in range(1, 1000001):
    counts[random.randint(0, 99)] += 1
    if i in [10, 100, 1000, 10000, 100000, 1000000]:
        axes[int(math.log10(i))-1].bar(np.arange(1, 101), counts)
plt.show()
```

What we can see from the above figures is that the initial number of counts looks *very* uneven. If we sample fewer than 100 draws from a distribution over 100 outcomes this is pretty much expected. But even for 1000 samples there is a significant variability between the draws. What we are really aiming for is a situation where the probability of drawing a number $x$ is given by $p(x)$.

### The categorical distribution

Quite obviously, drawing from a uniform distribution over a set of 100 outcomes is quite simple. But what if we have nonuniform probabilities? Let's start with a simple case, a biased coin which comes up heads with probability 0.35 and tails with probability 0.65. A simple way to sample from that is to generate a uniform random variable over $[0,1]$ and if the number is less than $0.35$, we output heads and otherwise we generate tails. Let's try this out.

```{.python .input}
# number of samples
n = 1000000
y = np.random.uniform(0, 1, n)
x = np.arange(1, n+1)
# count number of occurrences and divide by the number of total draws
p0 = np.cumsum(y < 0.35) / x
p1 = np.cumsum(y >= 0.35) / x

plt.figure(figsize=(15, 8))
plt.semilogx(x, p0)
plt.semilogx(x, p1)
plt.show()
```

As we can see, on average this sampler will generate 35% zeros and 65% ones. Now what if we have more than two possible outcomes? We can simply generalize this idea as follows. Given any probability distribution, e.g.
$p = [0.1, 0.2, 0.05, 0.3, 0.25, 0.1]$ we can compute its cumulative distribution (python's ``cumsum`` will do this for you) $F = [0.1, 0.3, 0.35, 0.65, 0.9, 1]$. Once we have this we draw a random variable $x$ from the uniform distribution $U[0,1]$ and then find the interval where $F[i-1] \leq x < F[i]$. We then return $i$ as the sample. By construction, the chances of hitting interval $[F[i-1], F[i])$ has probability $p(i)$.

Note that there are many more efficient algorithms for sampling than the one above. For instance, binary search over $F$ will run in $O(\log n)$ time for $n$ random variables. There are even more clever algorithms, such as the [Alias Method](https://en.wikipedia.org/wiki/Alias_method) to sample in constant time, after $O(n)$ preprocessing.

### The Normal distribution

The Normal distribution (aka the Gaussian distribution) is given by $p(x) = \frac{1}{\sqrt{2 \pi}} \exp\left(-\frac{1}{2} x^2\right)$. Let's plot it to get a feel for it.

```{.python .input}
x = np.arange(-10, 10, 0.01)
p = (1/math.sqrt(2 * math.pi)) * np.exp(-0.5 * x**2)
plt.figure(figsize=(10, 5))
plt.plot(x, p)
plt.show()
```

Sampling from this distribution is a lot less trivial. First off, the support is infinite, that is, for any $x$ the density $p(x)$ is positive. Secondly, the density is nonuniform. There are many tricks for sampling from it - the key idea in all algorithms is to stratify $p(x)$ in such a way as to map it to the uniform distribution $U[0,1]$. One way to do this is with the probability integral transform.

Denote by $F(x) = \int_{-\infty}^x p(z) dz$ the cumulative distribution function (CDF) of $p$. This is in a way the continuous version of the cumulative sum that we used previously. In the same way we can now define the inverse map $F^{-1}(\xi)$, where $\xi$ is drawn uniformly. Unlike previously where we needed to find the correct interval for the vector $F$ (i.e. for the piecewise constant function), we now invert the function $F(x)$.

In practice, this is slightly more tricky since inverting the CDF is hard in the case of a Gaussian. It turns out that the *twodimensional* integral is much easier to deal with, thus yielding two normal random variables than one, albeit at the price of two uniformly distributed ones. For now, suffice it to say that there are built-in algorithms to address this.

The normal distribution has yet another desirable property. In a way all distributions converge to it, if we only average over a sufficiently large number of draws from any other distribution. To understand this in a bit more detail, we need to introduce three important things: expected values, means and variances.

* The expected value $\mathbb{E}_{x \sim p(x)}[f(x)]$ of a function $f$ under a distribution $p$ is given by the integral $\int_x p(x) f(x) dx$. That is, we average over all possible outcomes, as given by $p$.
* A particularly important expected value is that for the function $f(x) = x$, i.e. $\mu := \mathbb{E}_{x \sim p(x)}[x]$. It provides us with some idea about the typical values of $x$.
* Another important quantity is the variance, i.e. the typical deviation from the mean
$\sigma^2 := \mathbb{E}_{x \sim p(x)}[(x-\mu)^2]$. Simple math shows (check it as an exercise) that
$\sigma^2 = \mathbb{E}_{x \sim p(x)}[x^2] - \mathbb{E}^2_{x \sim p(x)}[x]$.

The above allows us to change both mean and variance of random variables. Quite obviously for some random variable $x$ with mean $\mu$, the random variable $x + c$ has mean $\mu + c$. Moreover, $\gamma x$ has the variance $\gamma^2 \sigma^2$. Applying this to the normal distribution we see that one with mean $\mu$ and variance $\sigma^2$ has the form $p(x) = \frac{1}{\sqrt{2 \sigma^2 \pi}} \exp\left(-\frac{1}{2 \sigma^2} (x-\mu)^2\right)$. Note the scaling factor $\frac{1}{\sigma}$ - it arises from the fact that if we stretch the distribution by $\sigma$, we need to lower it by $\frac{1}{\sigma}$ to retain the same probability mass (i.e. the weight under the distribution always needs to integrate out to 1).

Now we are ready to state one of the most fundamental theorems in statistics, the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem). It states that for sufficiently well-behaved random variables, in particular random variables with well-defined mean and variance, the sum tends toward a normal distribution. To get some idea, let's repeat the experiment described in the beginning, but now using random variables with integer values of $\{0, 1, 2\}$.

```{.python .input}
# generate 10 random sequences of 10,000 random normal variables N(0,1)
tmp = np.random.uniform(size=(10000,10))
x = 1.0 * (tmp > 0.3) + 1.0 * (tmp > 0.8)
mean = 1 * 0.5 + 2 * 0.2
variance = 1 * 0.5 + 4 * 0.2 - mean**2
print('mean {}, variance {}'.format(mean, variance))
# cumulative sum and normalization
y = np.arange(1,10001).reshape(10000,1)
z = np.cumsum(x,axis=0) / y

plt.figure(figsize=(10,5))
for i in range(10):
    plt.semilogx(y,z[:,i])

plt.semilogx(y,(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.semilogx(y,-(variance**0.5) * np.power(y,-0.5) + mean,'r')
plt.show()   
```

This looks very similar to the initial example, at least in the limit of averages of large numbers of variables. This is confirmed by theory. Denote by mean and variance of a random variable the quantities

$$\mu[p] := \mathbf{E}_{x \sim p(x)}[x] \text{ and } \sigma^2[p] := \mathbf{E}_{x \sim p(x)}[(x - \mu[p])^2]$$

Then we have that $\lim_{n\to \infty} \frac{1}{\sqrt{n}} \sum_{i=1}^n \frac{x_i - \mu}{\sigma} \to \mathcal{N}(0, 1)$. In other words, regardless of what we started out with, we will always converge to a Gaussian. This is one of the reasons why Gaussians are so popular in statistics.


### More distributions

Many more useful distributions exist. We recommend consulting a statistics book or looking some of them up on Wikipedia for further detail.

* **Binomial Distribution** It is used to describe the distribution over multiple draws from the same distribution, e.g. the number of heads when tossing a biased coin (i.e. a coin with probability $\pi$ of returning heads) 10 times. The probability is given by $p(x) = {n \choose x} \pi^x (1-\pi)^{n-x}$.
* **Multinomial Distribution** Obviously we can have more than two outcomes, e.g. when rolling a dice multiple times. In this case the distribution is given by $p(x) = \frac{n!}{\prod_{i=1}^k x_i!} \prod_{i=1}^k \pi_i^{x_i}$.
* **Poisson Distribution** It is used to model the occurrence of point events that happen with a given rate, e.g. the number of raindrops arriving within a given amount of time in an area (weird fact - the number of Prussian soldiers being killed by horses kicking them followed that distribution). Given a rate $\lambda$, the number of occurrences is given by $p(x) = \frac{1}{x!} \lambda^x e^{-\lambda}$.
* **Beta, Dirichlet, Gamma, and Wishart Distributions** They are what statisticians call *conjugate* to the Binomial, Multinomial, Poisson and Gaussian respectively. Without going into detail, these distributions are often used as priors for coefficients of the latter set of distributions, e.g. a Beta distribution as a prior for modeling the probability for binomial outcomes.  

For whinges or inquiries, [open an issue on  GitHub.](https://github.com/zackchase/mxnet-the-straight-dope)
