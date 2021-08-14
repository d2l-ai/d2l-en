# Probability
:label:`sec_prob`

In some form or another, machine learning is all about making predictions.
We might want to predict the *probability* of a patient suffering a heart attack in the next year, given their clinical history. In anomaly detection, we might want to assess how *likely* a set of readings from an airplane's jet engine would be, were it operating normally. In reinforcement learning, we want an agent to act intelligently in an environment. This means we need to think about the probability of getting a high reward under each of the available actions. And when we build recommender systems we also need to think about probability. For example, say *hypothetically* that we worked for a large online bookseller. We might want to estimate the probability that a particular user would buy a particular book. For this we need to use the language of probability.
Entire courses, majors, theses, careers, and even departments, are devoted to probability. So naturally, our goal in this section is not to teach the whole subject. Instead we hope to get you off the ground, to teach you just enough that you can start building your first deep learning models, and to give you enough of a flavor for the subject that you can begin to explore it on your own if you wish.

We have already invoked probabilities in previous sections without articulating what precisely they are or giving a concrete example. Let's get more serious now by considering the first case: distinguishing cats and dogs based on photographs. This might sound simple but it is actually a formidable challenge. To start with, the difficulty of the problem may depend on the resolution of the image.

![Images of varying resolutions ($10 \times 10$, $20 \times 20$, $40 \times 40$, $80 \times 80$, and $160 \times 160$ pixels).](../img/cat-dog-pixels.png)
:width:`300px`
:label:`fig_cat_dog`

As shown in :numref:`fig_cat_dog`,
while it is easy for humans to recognize cats and dogs at the resolution of $160 \times 160$ pixels,
it becomes challenging at $40 \times 40$ pixels and next to impossible at $10 \times 10$ pixels. In
other words, our ability to tell cats and dogs apart at a large distance (and thus low resolution) might approach uninformed guessing. Probability gives us a
formal way of reasoning about our level of certainty.
If we are completely sure
that the image depicts a cat, we say that the *probability* that the corresponding label $y$ is "cat", denoted $P(y=$ "cat"$)$ equals $1$.
If we had no evidence to suggest that $y =$ "cat" or that $y =$ "dog", then we might say that the two possibilities were equally
*likely* expressing this as $P(y=$ "cat"$) = P(y=$ "dog"$) = 0.5$. If we were reasonably
confident, but not sure that the image depicted a cat, we might assign a
probability $0.5  < P(y=$ "cat"$) < 1$.

Now consider the second case: given some weather monitoring data, we want to predict the probability that it will rain in Taipei tomorrow. If it is summertime, the rain might come with probability 0.5.

In both cases, we have some value of interest. And in both cases we are uncertain about the outcome.
But there is a key difference between the two cases. In this first case, the image is in fact either a dog or a cat, and we just do not know which. In the second case, the outcome may actually be a random event, if you believe in such things (and most physicists do). So probability is a flexible language for reasoning about our level of certainty, and it can be applied effectively in a broad set of contexts.

## Basics

Say that we cast a die and want to know what the chance is of seeing a 1 rather than another digit. If the die is fair, all the six outcomes $\{1, \ldots, 6\}$ are equally likely to occur, and thus we would see a $1$ in one out of six cases. Formally we state that $1$ occurs with probability $\frac{1}{6}$.

For a real die that we receive from a factory, we might not know those proportions and we would need to check whether it is tainted. The only way to investigate the die is by casting it many times and recording the outcomes. For each cast of the die, we will observe a value in $\{1, \ldots, 6\}$. Given these outcomes, we want to investigate the probability of observing each outcome.

One natural approach for each value is to take the
individual count for that value and to divide it by the total number of tosses.
This gives us an *estimate* of the probability of a given *event*. The *law of
large numbers* tell us that as the number of tosses grows this estimate will draw closer and closer to the true underlying probability. Before going into the details of what is going here, let's try it out. As always, we start by importing the necessary packages.

```{.python .input}
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
npx.set_np()
```

```{.python .input  n=10}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
from torch.distributions.multinomial import Multinomial
import numpy as np
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import numpy as np
```

Next, we want to be able to cast the die. In statistics we call this process
of drawing examples from probability distributions *sampling*.
The distribution
that assigns probabilities to a number of discrete choices is called the
*multinomial distribution*. We will give a more formal definition of
*distribution* later, but at a high level, think of it as just an assignment of
probabilities to events.

To draw a single sample, we simply supply it with a vector of probabilities.
The output is another vector of the same length:
its value at index $i$ is the number of times the sampling outcome corresponds to $i$.

```{.python .input}
fair_probs = [1.0 / 6] * 6
multinomial(1, fair_probs)
```

```{.python .input  n=2}
#@tab pytorch
fair_probs = torch.ones([6]) / 6
Multinomial(1, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
fair_probs = tf.ones(6) / 6
tfd.Multinomial(1, fair_probs).sample()
```

Each time you run the sampler you will receive a new random value that will likely
differ from the previous one. To assess the fairness of the die, we often need to
draw many samples from the same distribution. The code below draws 10 samples and 
tallies up the counts.

```{.python .input}
multinomial(10, fair_probs)
```

```{.python .input  n=3}
#@tab pytorch
Multinomial(10, fair_probs).sample()
```

```{.python .input}
#@tab tensorflow
tfd.Multinomial(10, fair_probs).sample()
```

Even though we drew the samples from a fair dice, the numbers don't look particularly even
at all. This is due to the fact that we only drew a small number of samples. Let's see what
happens when we draw 1000 rolls. For better intuition we calculate the relative frequency (count/total count) as the estimate of the true probability.

```{.python .input}
counts = multinomial(1000, fair_probs).astype(np.float32)
counts / 1000
```

```{.python .input  n=4}
#@tab pytorch
counts = Multinomial(1000, fair_probs).sample()
counts / 1000  # Relative frequency as the estimate
```

```{.python .input}
#@tab tensorflow
counts = tfd.Multinomial(1000, fair_probs).sample()
counts / 1000
```

Because we generated the data from a fair die, we know that each outcome has true probability $\frac{1}{6} \approx 0.167$. In this light the above estimates look quite reasonable. Let's get some more intuition as to what the typical behavior is by studying the curve as a function of sample size.

```{.python .input}
counts = multinomial(1, fair_probs, size=5000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i], label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input  n=13}
#@tab pytorch
counts = Multinomial(1, fair_probs).sample((5000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i], label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input  n=7}
#@tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(5000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(estimates[:, i], label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Each solid curve corresponds to one of the six values of the die and gives our estimated probability that the die turns up that value as assessed after each group of experiments.
The dashed black line gives the true underlying probability.
As we get more data by conducting more experiments, the $6$ solid curves converge towards the true probability, but how quickly does this happen? The Central Limit Theorem suggests that this occurs at the rate of $1/\sqrt{n}$, i.e., the deviation between truth and estimate decreases by a fixed multiple of $1/\sqrt{n}$, where $n$ is the number of samples that we've drawn. For a more detailed discussion, see e.g. :cite:`wasserman2013all`. Let's try this out in practice, using the random samples we drew before. We plot the deviation from $\frac{1}{6}$ and scale it by $\sqrt{n}$. If our theory is correct, we should see straight lines.

```{.python .input  n=14}
deviation = (estimates - (1/6.0)) * (np.arange(1,5001)**0.5).reshape((5000,1))
d2l.set_figsize((6, 4.5))
for i in range(6):
    d2l.plt.plot(deviation[:, i], label=(str(i + 1)))
d2l.plt.axhline(y=0, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Scaled deviation')
d2l.plt.legend();
```

While the lines aren't quite straight, they are very much of bounded range, even acros a large number of samples. Going into more details is significantly beyond the scope of this book but it should give you an intuition of both the power of statistical tools (the range kind-of works) and also their limitation (each individual curve doesn't look too straight at all). 

##  Axioms and Tools

Statisticans have their own formal tools and language to refer to random events. Since much of machine learning is about such events we need to understand at least the basics. The definitions below will allow us to be more precise about what we mean by a sample, an event, a distribution or a density. 

###  Probability

When dealing with the rolls of a die,
we call the set $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ the *sample space* or *outcome space*, where each element is an *outcome*.
An *event* is a set of outcomes from a given sample space.
For instance, "seeing a $5$" ($\mathcal{A} = \{5\}$) and "seeing an odd number" ($\mathcal{B} = \{1, 3, 5\}$) are both valid events of rolling a die. 
Note that if the outcome $z$ of a random experiment satisfies $z \in \mathcal{A}$,
then event $\mathcal{A}$ has occurred.
That is to say, if $z = 3$, then $\mathcal{A}$ did not occur but $\mathcal{B}$ occurred, since $z$ is odd but not $5$. 

Formally, *probability* can be thought of as a function that maps a set to a real value.
The probability of an event $\mathcal{A}$ in the given sample space $\mathcal{S}$,
denoted by $P(\mathcal{A})$, satisfies the following properties:

* For any event $\mathcal{A}$, its probability is never negative, i.e., $P(\mathcal{A}) \geq 0$;
* The probability of the entire sample space is $1$, i.e., $P(\mathcal{S}) = 1$;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* ($\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any one of them happens equals the sum of their individual probabilities, i.e., $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

:cite:`kolmogorov1933sulla` proposed the above as the three axioms of probability theory. They are quite useful. For instance, let $\mathcal{A}$ be the entire sample space and let $\mathcal{A}' = \emptyset$. Then it follows immediately from $1 \geq P(\mathcal{A}) + P(\mathcal{A}')$ that $P(\emptyset) = 0$. In plain English---impossible events have zero probability of occurring. 

### Random Variables

In our experiment of casting a die, we introduced the notion of a *random variable*. A random variable can be pretty much any quantity. It can take on a value among a set of (possibly many) possibilities.
Consider a random variable $X$ whose value is in the sample space $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$ of rolling a die. We can denote the event "seeing a $5$" as $\mathcal{A} := \{X = 5\}$ or $X = 5$, and its probability as $P(\{X = 5\})$ or $P(X = 5)$.

By $P(X = a)$, we make a distinction between the random variable $X$ and the values (e.g., $a$) that $X$ can take.
However, such pedantry results in a cumbersome notation and we omit it unless necessary.
On the one hand, we can just denote $P(X)$ as the *distribution* over the random variable $X$:
the distribution tells us the probability that $X$ takes any value.
On the other hand,
we can simply write $P(a)$ to denote the probability that a random variable takes the value $a$.
Since an event in probability theory is a set of outcomes from the sample space,
we can specify a range of values for a random variable to take.
For example, $P(1 \leq X \leq 3)$ denotes the probability of the event $\{1 \leq X \leq 3\}$.
By definition it is also the probability that the random variable $X$ can take a value from $\{1, 2, 3\}$.

Note that there is a subtle difference between *discrete* random variables, like the sides of a die, and *continuous* ones, like the weight and the height of a person. There is little point in asking whether two people have *exactly* the same height. If we take precise enough measurements you will find that no two people on the planet have the exact same height. In fact, if we take a fine enough measurement, you will not have the same height when you wake up and when you go to sleep. So there is no point in asking about the probability
that someone is 1.801392782910287192 meters tall. Given the world population of 8 billion humans the probability is virtually 0. It makes more sense in this case to ask whether someone's height falls into a given interval, say between 1.79 and 1.81 meters. In these cases we quantify the likelihood that we see a value as a *density*. The height of exactly 1.80 meters has no probability, but nonzero density. In fact, the probability is then given by the *integral* over the density in the interval of $[1.79, 1.81]$ meters. 
For more details on continuous random variables see :numref:`sec_random_variables`. 

## Multiple Random Variables

Frequently we will want to consider more than one random variable at a time.
For instance, we may want to model the relationship between diseases and symptoms. Given diseases, say (flu, COVID-19) and symptoms, say (cough, sneezing, fever), combinations of the two such as (flu, fever) may or may not occur in a patient. Our goal might be to estimate their relationships for better medical care. 

As a more complicated example, images contain millions of pixels, thus millions of random variables. And in many cases images will come with a
label, identifying objects in the image. We can also think of the label as another
random variable. We can even think of all the metadata such as location, time, aperture, focal length, ISO, focus distance, and camera type as random variables.
All of these are random variables that occur jointly. When we deal with multiple random variables, there are several quantities of interest.

Given values $a$ and $b$, we may ask what the *joint probability* is that both values occur simultaneously, i.e. $P(A = a, B = b)$. This lets us determine, for instance whether $a$ implies $b$ or vice versa or whether they are related at all. Note that for any values $a$ and $b$ we have that $P(A=a, B=b) \leq P(A=a)$ and $P(A=a, B=b) \leq P(B = b)$. 
This has to be the case, since for $A=a$ and $B=b$ to happen, $A=a$ has to happen *and* $B=b$ also has to happen. Thus, the intersection of $\{A=a\}$ and $\{B=b\}$ cannot be more likely than $\{A=a\}$ or $\{B=b\}$ individually.

Due to the above inequality we know that $0 \leq \frac{P(A=a, B=b)}{P(A=a)} \leq 1$. We call this ratio the *conditional probability* and denote it by $P(B=b|A=a)$: it is the probability of $B=b$, provided that
$A=a$ holds. Conditional probabilities behave just like unconditional ones, as long as we leave the conditioning unchanged. For instance, for disjoint events $\mathcal{B}$ and $\mathcal{B}'$ we have that 
$P(\mathcal{B} \cup \mathcal{B}'|A = a) = P(\mathcal{B}|A = a) + P(\mathcal{B}'|A = a)$. Note that for convenience one typically writes $P(A, B)$ and $P(A|B)$ whenever the specific value is less important. 

Using the definition of conditional probabilities, we can derive one of the most useful and celebrated equations in statistics: *Bayes' Rule*.
By construction, we have that $P(A, B) = P(B|A) P(A)$ and $P(A, B) = P(A|B) P(B)$. Combining both equations yields 
$P(B|A) P(A) = P(A|B) P(B)$ and hence 

$$P(A|B) = \frac{P(B|A) P(A)}{P(B)}.$$

This simple equation has rather profound implications as it allows us to reverse the order of conditioning. For instance, given the prevalence of symptoms for a given disease, the overall prevalence of diseases and of symptoms respectively, it allows us to infer the probability of a disease, given the manifestation of certain symptoms. This is  what a physician would do when diagnosing a patient. 

In some cases we might not have direct access to $P(B)$, such as the prevalence of symptoms. In this case a simplified version of Bayes' rule comes in handy:

$$P(A|B) \propto P(B|A) P(A)$$

Since we know that $P(A|B)$ must be normalized to $1$, i.e., $\sum_a P(A=a|B) = 1$, we can use it to compute

$$P(A|B) = \frac{P(B|A) P(A)}{\sum_b P(B=b|A) P(A)}.$$

In statistics $P(A)$ is typically referred as a *prior*, $P(B|A)$ as the likelihood of $B$ occurring, given $A$, and $P(A|B)$ as the *posterior* probability of $A$, given $B$, in short, *the posterior*. The above equation states that an unnormalized posterior can be normalized easily. This has many useful applications. For instance, in a speech recognition model $A$ might be the text and $B$ might be the audio file. There we can compute the (normalized) posterior probability for a specific string by summing over all possible strings and looking at their relative weight. 

Note that $\sum_b P(b=B|A) = 1$ also allows us to *marginalize* over random variables. That is, we can drop variables from a joint distribution such as $P(A, B)$. After all, we have that 

$$\sum_b P(A, B=b) = P(A) \sum_b P(b = B|A) = P(A).$$

Another important property to check for is whether random variables are independent. The latter means that the value of $A$ doesn't affect $B$ and vice versa. As such we have that $P(A|B) = P(A)$ and thus $P(A,B) = P(A|B) P(B) = P(A) P(B)$. Statisticians typically write this as $A \perp B$. Independence is a useful property. For instance, two successive rolls of a dice are independent (unless the dealer is cheating). Conversely, we call random variables dependent whenever $P(A,B) \neq P(A) P(B)$. Dependence allows us to draw meaningful statistical conclusions. For instance, medical diagnosis only works because symptoms are dependent on the diseases. 

Lastly, there's *conditional* independence and dependence. Two random variables $A$ and $B$ are *conditionally independent* given another random variable $C$
if and only if $P(A, B|C) = P(A|C)P(B|C)$. For instance, while the choices of two people to bring an umbrella might be highly dependent on each other, they might be independent, given the weather report of the day. After all, it is reasonable to assume that both will make their own independent decision as to whether to bring an umbrella, given the forecast. A concise notation for independence is $A \perp B | C$.

## An Example
:label:`subsec_probability_hiv_app`

Let's put our skills to the test. Assume that a doctor administers an HIV test to a patient. This test is fairly accurate and it fails only with 1% probability if the patient is healthy but reporting him as diseased. Moreover,
it never fails to detect HIV if the patient actually has it. We use $D_1 \in \{0, 1\}$ to indicate the diagnosis ($0$ if negative and $1$ if positive) and $H \in \{0, 1\}$ to denote the HIV status.

| Conditional probability | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 | H)$        |     1 |  0.01 |
| $P(D_1 = 0 | H)$        |     0 |  0.99 |

Note that the column sums are all 1 (but the row sums don't), since they are conditional probabilities. Let's compute the probability of the patient having HIV if the test comes back positive, i.e., $P(H = 1|D_1 = 1)$. Intuitively this is going to depend on how common the disease is, since it affects the number of false alarms. Assume that the population is quite healthy, e.g., $P(H=1) = 0.0015$. To apply Bayes' theorem, we need to apply marginalization and the multiplication rule to determine

$$\begin{aligned}
P(D_1 = 1) 
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1|H=0) P(H=0) + P(D_1=1|H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

This leads us to 

$$P(H = 1|D_1 = 1) = \frac{P(D_1=1|H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

In other words, there is only a 13.06% chance that the patient
actually has HIV, despite using a very accurate test.
As we can see, probability can be counterintuitive.
What should a patient do upon receiving such terrifying news? Likely, the patient
would ask the physician to administer another test to get clarity. The second
test has different characteristics and it is not as good as the first one.

| Conditional probability | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1|H)$          |  0.98 |  0.03 |
| $P(D_2 = 0|H)$          |  0.02 |  0.97 |

Unfortunately, the second test comes back positive, too.
Let's calculate the requisite probabilities to invoke Bayes' theorem
by assuming conditional independence:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1|H = 0) 
& = P(D_1 = 1|H = 0) P(D_2 = 1|H = 0)  
=& 0.0003 \\
P(D_1 = 1, D_2 = 1|H = 1) 
& = P(D_1 = 1|H = 1) P(D_2 = 1|H = 1)  
=& 0.98
\end{aligned}
$$

Now we can apply marginalization to obtain the probability that both tests come back positive:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1) 
=& P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
=& P(D_1 = 1, D_2 = 1| H = 0)P(H=0) + P(D_1 = 1, D_2 = 1|H = 1)P(H=1)\\
=& 0.00176955
\end{aligned}
$$

Finally, the probability of the patient having HIV given both tests being positive is

$$P(H = 1| D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1|H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

That is, the second test allowed us to gain much higher confidence that not all is well. Despite the second test being considerably less accurate than the first one, it still significantly improved our estimate. The assumption of boths tests being conditional independent of each other was crucial for our ability to generate a more accurate estimate. Take the extreme case where we run the same test twice. In this situation we would expect the same outcome in both times, hence no additional insight is gained from running the same test again. 
The astute reader might have noticed that the diagnosis behaved like a classifier hiding in plain sight where our ability to decide whether a patient is healthy increases as we obtain more features (test outcomes). We will pick up this idea in :ref:`sec_naive_bayes` where we will inroduce Naive Bayes Classifiers, using the approximation that all features occur independently. 

## Expectations

Quite often probabilities *per se* are insufficient to provide us with the relevant insight to make decisions. For instance, when we want to decide whether to make an investment we should assess the expected return and the risk profile. For instance, with 50% probability the investment might fail, with 40% probability it might return twice the investment and with 10% it might return 10 times the money invested. To calculate the expected return we sum over all returns multiplied by the probability that they will occur. This yields $0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$. Hence the expected return is 1.8 times the initial investment. 

In general, the *expectation* (or average) of the random variable $X$ is defined as

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

Likewise, for densities we obtain $E[X] = \int x dp(x)$. Rather than $x$ we are 
often interested in the expected value of a function $f$ under the distribution. In this case we arrive at 

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \text{ and } E_{x \sim P}[f(x)] = \int f(x) dp(x)$$

for discrete probabilities and densities respectively. Returning to the investment example from above, $f$ might be the happiness associated with the return. Assuming that the happiness for a total loss is $-1$, and for returns of 1, 2, and 10 is 1, 2 and 4 respectively, we can see that the expected happiness of investing is $0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$. As such, we would be well advised to keep the money in the bank. 

For financial decisions we might also want to measure how *risky* an investment is. But how should we quantify it? One option is to look at how far the actual returns deviate from their expected value. Alas, in expectation the deviations all vanish. A better strategy is to square the deviations. This penalizes larger deviations more. This quantity is called the variance of a random variable:

$$\mathrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

Its square root is called the *standard deviation*. Here the equality follows by expanding $(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$ and taking expectations for each term. Lastly, the variance of a function of a random variable is defined analogously as 

$$\mathrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$ 

Returning to our investment example we can now compute the variance of the investment. It is given by 
$0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$. For all intents and purposes this is a risky investment. Note that by mathematical convention mean and variance are often referenced as $\mu$ and $\sigma^2$. This is particularly common whenever we use it to parametrize a Gaussian distribution. For more information about the latter see e.g., :ref:`sec_distributions`.

In the same way as we introduced expectations and variance for *scalar* random variables, we can do so for vector-valued ones. Expectations are easy, since we can apply them element-wise. For instance, $\mathbf{\mu} := E_{\mathbf{x} \sim P}[\mathbf{x}]$ has coordinates $\mu_i = E_{\mathbf{x} \sim P}[x_i]$. Covariances are more complicated. We resolve the problem by taking expectations of the *outer product* of the difference between random variables and their mean. 

$$\Sigma := \mathrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \mathbf{\mu}) (\mathbf{x} - \mathbf{\mu})^\top\right]$$

This matrix $\Sigma$ is referred to as the covariance matrix. An easy way to see its effect is to consider some vector $\mathbf{v}$ of the same size as $\mathbf{x}$. It follows that 

$$\mathbf{v}^\top \Sigma \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \mathbf{\mu}) (\mathbf{x} - \mathbf{\mu})^\top \mathbf{v}\right] = \mathrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}]$$

As such, $\Sigma$ allows us to compute the variance for any linear function of $\mathbf{x}$ by a simple matrix multiplication. The off-diagonal elements tell us how correlated coordinates are: a value of 0 means no correlation, whereas a large positive value means that they are strongly correlated. 

## Summary and Discussion

In this section we saw that probability can be used to encode uncertainty associated with the problem itself and also with the model. These aspects are known as aleatoric and epistemic uncertainty respectively. See e.g., :cite:`der2009aleatory` for a review on this aspect of [Uncertainty Quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification). While epistemic uncertainty can be addressed by observing more data, such progress is impossible in the aleatoric case. After all, no matter how long we watch someone tossing a coin, we will never be more or less than 50% certain that we'll see heads. 

On the topic of estimation we saw that sampling from a probability distribution provides us with information to determine the distribution to some extent. That said, the rate at which this is possible can be quite slow. In particular we saw that the rate of $1/\sqrt{n}$ is a good quantifier of the amount of uncertainty left after we observe $n$ observations. This means that by going from 10 to 1000 observations (usually a very achievable task) we see a tenfold reduction of uncertainty, whereas the next 1000 observations help comparatively little, offering only a 1.41 times reduction. This is a persistent feature of machine learning: while there are often easy gains, it takes a very large amount of data, and often with it an enormous amount of computation to make even further gains. For an empirical review of this fact for large scale language models see :cite:`revels2016forward`. 

We also sharpened our language and tools for statistical modeling. In the process of that we learned about conditional probabilities and about one of the most important equations in statistics---Bayes' rule. It is an effective tool for decoupling information conveyed by data through a likelihood term $P(B|A)$ that addresses how well observations $B$ match a choice of parameters $A$, and a prior probability $P(A)$ which governs how plausible a particular choice of $A$ was in the first place. In particular, we saw how this affects our ability to diagnose diseases, based on the efficacy of the test *and* the prevalence of the disease itself (i.e., our prior). We strongly recommend that you play with the equations to see how the analysis changes if you change the prior. 

Lastly, we introduced a first set of nontrivial questions about the effect of a specific probability distribution, namely expectations and variances. While there are many more than just linear and quadratic expectations for a probability distribution, these two already provide a good deal of knowledge about the possible behavior of the distribution. For instance, [Chebyshev's Inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality) states that $P(|X - \mu| \geq k \sigma) \leq 1/k^2$, where $\mu$ is the expectation, $\sigma^2$ is the variance of the distribution and $k > 1$ is a confidence parameter of our choosing. It tells us that draws from a distribution lie with at least 50% probability within a $2 \sigma$ interval of the expectation. 


## Exercises

1. Give an example where observing more data can reduce the amount of uncertainty about the outcome to an arbitrarily low level. 
1. Give an example where observing more data will only reduce the amount of uncertainty up to a point and then no further. Explain why this is the case and where you expect this point to occur.
1. We empirically demonstrated convergence to the mean for the roll of a dice. Calculate the variance of the estimate of the probability that we see $1$ after drawing $n$ samples. 
    1. How does the variance scale with the number of observations? 
    1. Use Chebyshev's Inequality to bound the deviation from the expectation. 
    1. How does it relate to the Central Limit Theorem?
1. Assume that we draw $n$ samples $x_i$ from a probability distribution with zero mean and unit variance. Compute the averages $z_m := m^{-1} \sum_{i=1}^m x_i$. Can we apply Chebyshev's Inequality for every $z_m$ independently? Why not?
1. Given two events with probability $P(\mathcal{A})$ and $P(\mathcal{B})$, compute upper and lower bounds on $P(\mathcal{A} \cup \mathcal{B})$ and $P(\mathcal{A} \cap \mathcal{B})$. Hint: graph the situation using a [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram).
1. Assume that we have a sequence of random variables, say $A$, $B$, and $C$, where $B$ only depends on $A$, and $C$ only depends on $B$, can you simplify the joint probability $P(A, B, C)$? Hint: this is a [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain).
1. In :numref:`subsec_probability_hiv_app`, assume that the outcomes of the two tests are not independent. In particular assume that each test on its own has a false positive rate of 10% and a false negative rate of 1%. That is, assume that $P(D =1|H=0) = 0.1$ and that $P(D = 0|H=1) = 0.01$. Moreover, assume that for $H = 1$ (infected) the test outcomes are conditionally independent, i.e.; that $P(D_1, D_2|H=1) = P(D_1|H=1) P(D_2|H=1)$ but that for healthy patients the outcomes are coupled via $P(D_1 = D_2 = 1|H=0) = 0.02$. 
    1. Work out the joint probability table for $D_1$ and $D_2$, given $H=0$ based on the information you have so far.
    1. Derive the probability of the patient being positive ($H=1$) after one test returns positive. You can assume the same baseline probability $P(H=1) = 0.0015$ as before. 
    1. Derive the probability of the patient being positive ($H=1$) after both tests return positive.
1. Assume that you are an asset manager for an investment bank and you have a choice of stocks $s_i$ to invest in. Your portfolio needs to add up to $1$ with weights $\alpha_i$ for each stock. The stocks have an average return $\mathbf{\mu} = E_{s \sim P}[\mathbf{s}]$ and covariance $\Sigma = \mathrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Compute the expected return for a given portfolio $\alpha$.
    1. If you wanted to maximize the return of the portfolio, how should you choose your investment?
    1. Compute the *variance* of the portfolio. 
    1. Formulate an optimization problem of maximizing the return while keeping the variance constrained to an upper bound. This is the Nobel-Prize winning [Markovitz portfolio](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`mangram2013simplified`. To solve it you will need a quadratic programming solver, something way beyond the scope of this book.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/36)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/37)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/198)
:end_tab:
