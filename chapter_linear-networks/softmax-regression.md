# Softmax Regression
:label:`sec_softmax`

In :numref:`sec_linear_regression`, we introduced linear regression,
working through implementations from scratch in :numref:`sec_linear_scratch`
and again using Gluon in :numref:`sec_linear_gluon` to do the heavy lifting.

Regression is the hammer we reach for when 
we want to answer *how much?* or *how many?* questions. 
If you want to predict the number of dollars (the *price*) 
at which a house will be sold, 
or the number of wins a baseball team might have, 
or the number of days that a patient will remain hospitalized before being discharged, 
then you are probably looking for a regression model.

In practice, we are more often interested in classification: 
asking not *how much?* but *which one?*

* Does this email belong in the spam folder or the inbox*?
* Is this customer more likley *to sign up* or *not to sign up* for a subscription service?*
* Does this image depict a donkey, a dog, a cat, or a rooster?
* Which movie is Aston most likely to watch next?

Colloquially, machine learning practitioners 
overload the word *classification* 
to describe two subtly different problems: 
(i) those where we are interested only in 
*hard* assignments of examples to categories; 
and (ii) those where we wish to make *soft assignments*, 
i.e., to assess the *probability* that each category applies. 
The distinction tends to get blurred, in part, 
because often, even when we only care about hard assignments, 
we still use models that make soft assignments.


## Classification Problems

To get our feet wet, let us start off with 
a simple image classification problem. 
Here, each input consists of a $2\times2$ grayscale image. 
We can represent each pixel value with a single scalar, 
giving us four features $x_1, x_2, x_3, x_4$. 
Further, let us assume that each image belongs to one 
among the categories "cat", "chicken" and "dog".

Next, we have to choose how to represent the labels. 
We have two obvious choices. 
Perhaps the most natural impulse would be to choose $y \in \{1, 2, 3\}$, 
where the integers represent {dog, cat, chicken} respectively. 
This is a great way of *storing* such information on a computer.
If the categories had some natural ordering among them, 
say if we were trying to predict {baby, toddler, adolescent, young adult, adult, geriatric}, 
then it might even make sense to cast this problem as regression 
and keep the labels in this format.

But general classification problems do not come with natural orderings among the classes. 
Fortunately, statisticians long ago invented a simple way 
to represent categorical data: the *one hot encoding*. 
A one-hot encoding is a vector with as many components as we have categories.
The component corresponding to particular instance's category is set to 1
and all other components are set to 0.

$$y \in \{(1, 0, 0), (0, 1, 0), (0, 0, 1)\}$$

In our case, $y$ would be a three-dimensional vector, 
with $(1,0,0)$ corresponding to "cat", $(0,1,0)$ to "chicken" and $(0,0,1)$ to "dog". 


### Network Architecture

In order to estimate the conditional probabilities associated with each classes, 
we need a model with multiple outputs, one per class. 
To address classification with linear models, 
we will need as many linear functions as we have outputs.
Each output will correpsond to its own linear function. 
In our case, since we have 4 features and 3 possible output categories, 
we will need 12 scalars to represent the weights, 
($w$ with subscripts) and 3 scalars to represent the biases ($b$ with subscripts). 
We compute these three *logits*, $o_1, o_2$, and $o_3$, for each input:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{12} + x_3 w_{13} + x_4 w_{14} + b_1,\\
o_2 &= x_1 w_{21} + x_2 w_{22} + x_3 w_{23} + x_4 w_{24} + b_2,\\
o_3 &= x_1 w_{31} + x_2 w_{32} + x_3 w_{33} + x_4 w_{34} + b_3.
\end{aligned}
$$

We can depict this calculation with the neural network diagram below.
Just as in linear regression, softmax regression is also a single-layer neural network. 
And since the calculation of each output, $o_1, o_2$, and $o_3$, 
depends on all inputs, $x_1$, $x_2$, $x_3$, and $x_4$, 
the output layer of softmax regression can also be described as fully-connected layer.

![Softmax regression is a single-layer neural network.  ](../img/softmaxreg.svg)


To express the model more compactly, we can use linear algebra notation. 
In vector form, we arrive at $\mathbf{o} = \mathbf{W} \mathbf{x} + \mathbf{b}$,
a form better suited both for mathematics, and for writing code. 
Note that we have gathered all of our weights into a $3\times4$ matrix 
and that for a given example $\mathbf{x}$,
our outputs are given by a matrix-vector product of our weights by our inputs 
plus our biases $\mathbf{b}$.


### Softmax Operation

The main approach that we are going to take here
is to interpret the outputs of our model as probabilities. 
We will optimize our parameters to produce probabilities 
that maximize the likelihood of the observed data.
Then, to generate predictions, we will set a threshold,
for example, choosing the *argmax* of the predicted probabilities. 

Put formally, we would like outputs $\hat{y}_k$ 
that we can interpret as the probability 
that a given item belongs to class $k$. 
Then we can choose the class with the largest output value 
as our prediction $\operatorname*{argmax}_k y_k$. 
For example, if $\hat{y}_1$, $\hat{y}_2$, and $\hat{y}_3$ 
are $0.1$, $.8$, and $0.1$, respectively,
then we predict category $2$, which (in our example) represents "chicken".

You might be tempted to suggest that we interpret 
the logits $o$ directly as our outputs of interest. 
However, there are some problems with directly
interpreting the output of the linear layer as a probability.
Nothing constrains these numbers to sum to 1. 
Moreover, depending on the inputs, they can take negative values. 
These violate basic axioms of probability presented in :num_ref:`sec_prob`

To interpret our outputs as probabilities, 
we must guarantee that (even on new data),
they will be nonnegative and sum up to 1. 
Moreover, we need a training objective that encourages 
the model to estimate faithfully *probabilities*.
Of all instances when a classifier outputs $.5$,
we hope that half of those examples 
will *actually* belong to the predicted class.
This is a property called *calibration*.

The *softmax function*, invented in 1959 by the social scientist
R Duncan Luce in the context of *choice models* does precisely this. 
To transform our logits such that they become nonnegative and sum to $1$,
while requiring that the model remains differentiable,
we first exponentiate each logit (ensuring non-negativity)
and then divide by their sum (ensuring that they sum to $1$). 

$$
\hat{\mathbf{y}} = \mathrm{softmax}(\mathbf{o})\quad \text{where}\quad
\hat{y}_i = \frac{\exp(o_i)}{\sum_j \exp(o_j)}
$$

It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ 
with $0 \leq \hat{y}_i \leq 1$ for all $i$.
Thus, $\hat{y}$ is a proper probability distribution 
and the values of $\hat{\mathbf{y}}$ can be interpreted accordingly.
Note that the softmax operation does not change the ordering among the logits,
and thus we can still pick out the most likely class by:

$$
\hat{\imath}(\mathbf{o}) = \operatorname*{argmax}_i o_i = \operatorname*{argmax}_i \hat y_i
$$

The logits $\mathbf{o}$ then are simply the pre-softmax values 
that determining the probabilities assigned to each category. 
Summarizing it all in vector notation we get 
${\mathbf{o}}^{(i)} = \mathbf{W} {\mathbf{x}}^{(i)} + {\mathbf{b}}$,
where ${\hat{\mathbf{y}}}^{(i)} = \mathrm{softmax}({\mathbf{o}}^{(i)})$.


### Vectorization for Minibatches

To improve computational efficiency and take advantage of GPUs, 
we typically carry out vector calculations for mini-batches of data. 
Assume that we are given a mini-batch $\mathbf{X}$ of examples 
with dimensionality $d$ and batch size $n$. 
Moreover, assume that we have $q$ categories (outputs). 
Then the minibatch features $\mathbf{X}$ are in $\mathbb{R}^{n \times d}$,
weights $\mathbf{W} \in \mathbb{R}^{d \times q}$,
and the bias satisfies $\mathbf{b} \in \mathbb{R}^q$.

$$
\begin{aligned}
\mathbf{O} &= \mathbf{X} \mathbf{W} + \mathbf{b} \\
\hat{\mathbf{Y}} & = \mathrm{softmax}(\mathbf{O})
\end{aligned}
$$

This accelerates the dominant operation into 
a matrix-matrix product $\mathbf{W} \mathbf{X}$ 
vs the matrix-vector products we would be executing 
if we processed one example at a time. 
The softmax itself can be computed 
by exponentiating all entries in $\mathbf{O}$ 
and then normalizing them by the sum.


## Loss Function
:label:`section_cross_entropy`

Next, we need a *loss function* to measure 
the quality of our predicted probabilities.
We will rely on *likelihood maximization*,
the very same concept that we encountered
when providing a probabilistic justification 
for the least squares objective in linear regression 
(:numref:`sec_linear_regression`).

### Log-Likelihood

The softmax function gives us a vector $\hat{\mathbf{y}}$, 
which we can interpret as estimated conditional probabilities
of each class given the input $x$, e.g.,
$\hat{y}_1$ = $\hat{p}(y=\mathrm{cat}|\mathbf{x})$. 
We can compare the estimates with reality
by checking how probable the *actual* classes are
according to our model, given the features.

$$
p(Y|X) = \prod_{i=1}^n p(y^{(i)}|x^{(i)})
\text{ and thus }
-\log p(Y|X) = \sum_{i=1}^n -\log p(y^{(i)}|x^{(i)})
$$


Maximizing $p(Y|X)$ (and thus equivalently minimizing $-\log p(Y|X)$)
corresponds to predicting the label well.
This yields the loss function 
(we dropped the superscript $(i)$ to avoid notation clutter):

$$
l = -\log p(y|x) = - \sum_j y_j \log \hat{y}_j
$$

For reasons explained later on, this loss function 
is commonly called the *cross-entropy* loss.
Here, we used that by construction $\hat{y}$ 
is a discrete probability distribution
and that the vector $\mathbf{y}$ is a one-hot vector.
Hence the the sum over all coordinates $j$ vanishes for all but one term.
Since all $\hat{y}_j$ are probabilities, 
their logarithm is never larger than $0$.
Consequently, the loss function cannot be minimized any further
if we correctly predict $y$ with *certainty*, 
i.e., if $p(y|x) = 1$ for the correct label.
Note that this is often not possible. 
For example, there might be label noise in the dataset
(some examples may be mislabeled).
It may also not be possible when the input features 
are not sufficiently informative
to classify every example perfectly.

### Softmax and Derivatives

Since the softmax and the corresponding loss are so common, 
it is worth while understanding a bit better how it is computed. 
Plugging $o$ into the definition of the loss $l$ 
and using the definition of the softmax we obtain:

$$
l = -\sum_j y_j \log \hat{y}_j = \sum_j y_j \log \sum_k \exp(o_k) - \sum_j y_j o_j
= \log \sum_k \exp(o_k) - \sum_j y_j o_j
$$

To understand a bit better what is going on, 
consider the derivative with respect to $o$. We get

$$
\partial_{o_j} l = \frac{\exp(o_j)}{\sum_k \exp(o_k)} - y_j = \mathrm{softmax}(\mathbf{o})_j - y_j = P(y = j|x) - y_j
$$

In other words, the gradient is the difference 
between the probability assigned to the true class by our model, 
as expressed by the probability $p(y|x)$, 
and what actually happened, as expressed by $y$. 
In this sense, it is very similar to what we saw in regression, 
where the gradient was the difference 
between the observation $y$ and estimate $\hat{y}$. This is not coincidence. 
In any [exponential family](https://en.wikipedia.org/wiki/Exponential_family) model, 
the gradients of the log-likelihood are given by precisely this term. 
This fact makes computing gradients easy in practice.

### Cross-Entropy Loss

Now consider the case where we observe not just a single outcome 
but an entire distribution over outcomes. 
We can use the same representation as before for $y$. 
The only difference is that rather than a vector containing only binary entries, 
say $(0, 0, 1)$, we now have a generic probability vector, say $(0.1, 0.2, 0.7)$.
The math that we used previously to define the loss $l$ still works out fine, 
just that the interpretation is slightly more general. 
It is the expected value of the loss for a distribution over labels.

$$
l(\mathbf{y}, \hat{\mathbf{y}}) = - \sum_j y_j \log \hat{y}_j
$$

This loss is called the cross-entropy loss and it is 
one of the most commonly used losses for multiclass classification. 
We can demystify the name by introducing the basics of information theory. 

## Information Theory Basics

Information theory deals with the problem of encoding, decoding, transmitting 
and manipulating information (aka data) in as concise form as possible.

### Entropy

The central idea in information theory is to quantify the information content in data.
This quantity places a hard limit on our ability to compress the data.
In information theory, this quantity is called the [entropy](https://en.wikipedia.org/wiki/Entropy) of a distribution $p$,
and it is captured by the following equation:

$$
H[p] = \sum_j - p(j) \log p(j)
$$

One of the fundamental theorems of information theory states 
that in order to encode data drawn randomly from the distribution $p$,
we need at least $H[p]$ 'nats' to encode it. 
If you wonder what a 'nat' is, it is the equivalent of bit 
but when using a code with base $e$ rather than one with base 2. 
One nat is $\frac{1}{\log(2)} \approx 1.44$ bit. 
$H[p] / 2$ is often also called the binary entropy.


### Surprisal

You might be wondering what compression has to do with prediction. 
Imagine that we have a stream of data that we want to compress. 
If it is always easy for us to predict the next token,
then this data is easy to compress! 
Take the extreme example where every token in the stream always takes the same value.
That is a very boring data stream! 
And not only is it boring, but it is easy to predict. 
Because they are always the same, we do not have to transmit any information 
to communicate the contents of the stream.
Easy to predict, easy to compress.

However if we cannot perfectly predict every event,
then we might some times be surprised. 
Our surpise is greater when we assigned an event lower probability.
For reasons that we will elaborate in the appendix,
Claude Shannon settled on $\log(1/p(j)) = -\log p(j)$ 
to quantify one's *surprisal* at observing an event $j$
having assigned it a (subjective) probability $p(j)$.
The entropy is then the *expected surprisal* 
when one assigned the correct probabilities 
(that truly match the data-generating process).
The entropy of the data is then the least surprised 
that one can ever be (in expectation). 


### Cross-Entropy Revisited

So if entropy is level of surprise experienced 
by someone who knows the true probability, 
then you might be wondering, *what is cross-entropy?* 
The cross-entropy *from $p$ to $q$*, denoted H(p, q),
is the expected surprisal of an observer with subjective probabilities $q$
upon seeing data that was actually generated according to probabilities $p$.
The lowest possible cross-entropy is achieved when $p=q$. 
In this case, the cross-entropy from $p$ to $q$ is $H(p,p)= H(p)$.
Relating this back to our classification objective,
even if we get the best possible predictions, 
if the best possible possible, then we will never be perfect. 
Our loss is lower-bounded by the entropy given by the 
actual conditional distributions $p(\mathbf{y}|\mathbf{x})$.


### Kullback Leibler Divergence

Perhaps the most common way to measure the distance between two distributions
is to calculate the Kullback Leibler divergence $D(p\|q)$. 
This is simply the difference between the cross-entropy and the entropy,
i.e., the additional cross-entropy incurred over the irreducible minimum value it could take:
$$
D(p\|q) = H(p,q) - H[p] = \sum_j p(j) \log \frac{p(j)}{q(j)}
$$
Note that in classificatio, we do not know the true $p$,
so we cannot compute the entropy directly. 
However, because the entropy is out of our control, 
minimizing $D(p\|q)$ with respect to $q$ 
is equivalent to minimizing the cross-entropy loss.

In short, we can think of the cross-entropy classification objective 
in two ways: (i) as maximizing the likelihood of the observed data;
and (ii) as minimizing our surprise (and thus the number of bits)
required to communicate the labels. 


## Model Prediction and Evaluation

After training the softmax regression model, given any example features, 
we can predict the probability of each output category. 
Normally, we use the category with the highest predicted probability as the output category. The prediction is correct if it is consistent with the actual category (label). 
In the next part of the experiment,
we will use accuracy to evaluate the model’s performance. 
This is equal to the ratio between the number of correct predictions a
nd the total number of predictions.

## Summary

* We introduced the softmax operation which takes a vector maps it into probabilities.
* Softmax regression applies to classification problems. It uses the probability distribution of the output category in the softmax operation.
* cross-entropy is a good measure of the difference between two probability distributions. It measures the number of bits needed to encode the data given our model.

## Exercises

1. Show that the Kullback-Leibler divergence $D(p\|q)$ is nonnegative for all distributions $p$ and $q$. Hint - use Jensen's inequality, i.e., use the fact that $-\log x$ is a convex function.
1. Show that $\log \sum_j \exp(o_j)$ is a convex function in $o$.
1. We can explore the connection between exponential families and the softmax in some more depth
    * Compute the second derivative of the cross-entropy loss $l(y,\hat{y})$ for the softmax.
    * Compute the variance of the distribution given by $\mathrm{softmax}(o)$ and show that it matches the second derivative computed above.
1. Assume that we three classes which occur with equal probability, i.e., the probability vector is $(\frac{1}{3}, \frac{1}{3}, \frac{1}{3})$.
    * What is the problem if we try to design a binary code for it? Can we match the entropy lower bound on the number of bits?
    * Can you design a better code. Hint - what happens if we try to encode two independent observations? What if we encode $n$ observations jointly?
1. Softmax is a misnomer for the mapping introduced above (but everyone in deep learning uses it). The real softmax is defined as $\mathrm{RealSoftMax}(a,b) = \log (\exp(a) + \exp(b))$.
    * Prove that $\mathrm{RealSoftMax}(a,b) > \mathrm{max}(a,b)$.
    * Prove that this holds for $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b)$, provided that $\lambda > 0$.
    * Show that for $\lambda \to \infty$ we have $\lambda^{-1} \mathrm{RealSoftMax}(\lambda a, \lambda b) \to \mathrm{max}(a,b)$.
    * What does the soft-min look like?
    * Extend this to more than two numbers.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2334)

![](../img/qr_softmax-regression.svg)
