# Linear Regression
:label:`sec_linear_regression`

Regression refers to a set of methods for modeling 
the relationship between data points $\mathbf{x}$ 
and corresponding real-valued targets $y$.
In the natural sciences and social sciences,
the purpose of regression is most often to
*characterize* the relationship between the inputs and outputs.
Machine learning, on the other hand,
is most often concerned with *prediction*.

Regression problems pop up whenever we want to predict a numerical value. 
Common examples include predicting prices (of homes, stocks, etc.), 
predicting length of stay (for patients in the hospital),
demand forecasting (for retail sales), among countless others.
Not every prediction problem is a classic *regression* problem.
In subsequent sections, we will introduce classification problems,
where the goal is to predict membership among a set of categories.



## Basic Elements of Linear Regression

*Linear regression* may be both the simplest
and most popular among the standard tools to regression.
Dating back to the dawn of the 19th century,
linear regression flows from a few simple assumptions.
First, we assume that the relationship between
the *features* $\mathbf{x}$ and targets $y$ is linear,
i.e., that $y$ can be expressed as a weighted sum 
of the inputs $\textbf{x}$,
give or take some noise on the observations.
Second, we assume that any noise is well-behaved
(following a Gaussian distribution). 
To motivate the approach, let's start with a running example.
Suppose that we wish to estimate the prices of houses (in dollars) 
based on their area (in square feet) and age (in years).  

To actually fit a model for predicting house prices,
we would need to get our hands on a dataset
consisting of sales for which we know
the sale price, area and age for each home.
In the terminology of machine learning,
the dataset is called a *training data* or *training set*,
and each row (here the data corresponding to one sale)
is called an *instance* or *example*.
The thing we are trying to predict (here, the price)
is called a *target* or *label*.  
The variables (here *age* and *area*)
upon which the predictions are based
are called *features* or *covariates*.

Typically, we will use $n$ to denote 
the number of examples in our dataset.
We index the samples by $i$, denoting each input data point 
as $x^{(i)} = [x_1^{(i)}, x_2^{(i)}]$ 
and the corresponding label as $y^{(i)}$.


### Linear Model

The linearity assumption just says that the target (price)
can be expressed as a weighted sum of the features (area and age):

$$\mathrm{price} = w_{\mathrm{area}} \cdot \mathrm{area} + w_{\mathrm{age}} \cdot \mathrm{age} + b.$$

Here, $w_{\mathrm{area}}$ and $w_{\mathrm{age}}$ 
are called *weights*, and $b$ is called a *bias* 
(also called an *offset* or *intercept*).
The weights determine the influence of each feature
on our prediction and the bias just says 
what value the predicted price should take
when all of the features take value $0$.
Even if we will never see any homes with zero area,
or that are precisely zero years old,
we still need the intercept or else we will 
limit the expressivity of our linear model.

Given a dataset, our goal is to choose 
the weights $w$ and bias $b$ such that on average, 
the predictions made according our  model
best fit the true prices observed in the data.

In disciplines where it is common to focus 
on datasets with just a few features, 
explicitly expressing models long-form like this is common.
In ML, we usually work with high-dimensional datasets,
so it is more convenient to employ linear algebra notation.  
When our inputs consist of $d$ features, 
we express our prediction $\hat{y}$ as

$$\hat{y} = w_1 \cdot x_1 + ... + w_d \cdot x_d + b.$$

Collecting all features into a vector $\mathbf{x}$ 
and all weights into a vector $\mathbf{w}$, 
we can express our model compactly using a dot product:

$$\hat{y} = \mathbf{w}^T \mathbf{x} + b.$$

Here, the vector $\mathbf{x}$ corresponds to a single data point.  
We will often find it convenient 
to refer to our entire dataset via the *design matrix* $X$.
Here, $X$ contains one row for every example 
and one column for every feature.

For a collection of data points $\mathbf{X}$, 
the predictions $\hat{\mathbf{y}}$ 
can be expressed via the matrix-vector product:

$${\hat{\mathbf{y}}} = \mathbf X \mathbf{w} + b.$$

Given a training dataset $X$ 
and corresponding (known) targets $\mathbf{y}$,
the goal of linear regression is to find 
the *weight* vector $w$ and bias term $b$ 
that given some a new data point $\mathbf{x}_i$,
sampled from the same distribution as the training data
will (in expectation) predict the target $y_i$ with the lowest error.

Even if we believe that the best model for 
predicting $y$ given  $\mathbf{x}$ is linear, 
we would not expect to find real-world data where 
$y_i$ exactly equals $\mathbf{w}^T \mathbf{x}+b$ 
for all points ($\mathbf{x}, y)$. 
For example, whatever instruments we use to observe 
the features $X$ and labels $\mathbf{y}$
might suffer small amount of measurement error.
Thus, even when we are confident 
that the underlying relationship is linear,
we will incorporate a noise term to account for such errors.

Before we can go about searching for the best parameters $w$ and $b$, 
we will need two more things: 
(i) a quality measure for some given model; 
and (ii) a procedure for updating the model to improve its quality.

### Loss Function

Before we start thinking about how *to fit* our model,
we need to determine a measure of *fitness*.
The *loss function* quantifies the distance
between the *real* and *predicted* value of the target.
The loss will usually be a non-negative number
where smaller values are better
and perfect predictions incur a loss of $0$.
The most popular loss function in regression problems
is the sum of squared errors.
When our prediction for some example $i$ is $\hat{y}^{(i)}$
and the corresponding true label is $y^{(i)}$,
the squared error is given by:

$$l^{(i)}(\mathbf{w}, b) = \frac{1}{2} \left(\hat{y}^{(i)} - y^{(i)}\right)^2.$$

The constant $1/2$ makes no real difference
but will prove notationally convenient,
cancelling out when we take the derivative of the loss.
Since the training dataset is given to us, and thus out of our control,
the empirical error is only a function of the model parameters.
To make things more concrete, consider the example below
where we plot a regression problem for a one-dimensional case 
as shown in :numref:`fig_fit_linreg`. 

![Fit data with a linear model.](../img/fit_linreg.svg)
:label:`fig_fit_linreg`

Note that large differences between
estimates $\hat{y}^{(i)}$ and observations $y^{(i)}$
lead to even larger contributions to the loss,
due to the quadratic dependence.
To measure the quality of a model on the entire dataset,
we simply average (or equivalently, sum)
the losses on the training set.

$$L(\mathbf{w}, b) =\frac{1}{n}\sum_{i=1}^n l^{(i)}(\mathbf{w}, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right)^2.$$

When training the model, we want to find parameters ($\mathbf{w}^*, b^*$)
that minimize the total loss across all training samples:

$$\mathbf{w}^*, b^* = \operatorname*{argmin}_{\mathbf{w}, b}\  L(\mathbf{w}, b).$$


### Analytic Solution

Linear regression happens to be an unusually simple optimization problem.
Unlike most other models that we will encounter in this book,
linear regression can be solved analytically by applying a simple formula, 
yielding a global optimum.  
To start, we can subsume the bias $b$ into the parameter $\mathbf{w}$
by appending a column to the design matrix consisting of all $1s$.  
Then our prediction problem is to minimize $||\mathbf{y} - X\mathbf{w}||$.
Because this expression has a quadratic form, it is convex,
and so long as the problem is not degenerate 
(our features are linearly independent), it is strictly convex.

Thus there is just one critical point on the loss surface 
and it corresponds to the global minimum.
Taking the derivative of the loss with respect to $\mathbf{w}$ 
and setting it equal to $0$ yields the analytic solution:

$$\mathbf{w}^* = (\mathbf X^T \mathbf X)^{-1}\mathbf X^T y$$

While simple problems like linear regression 
may admit analytic solutions, 
you should not get used to such good fortune.
Although analytic solutions allow for nice mathematical analysis,
the requirement of an analytic solution is so restrictive 
that it would exclude all of deep learning.

### Gradient descent

Even in cases where we cannot solve the models analytically,
and even when the loss surfaces are high-dimensional and nonconvex,
it turns out that we can still train models effectively in practice.
Moreover, for many tasks, these difficult-to-optimize models
turn out to be so much better that figuring out how to train them
ends up being well worth the trouble.

The key technique for optimizing nearly any deep learning model,
and which we will call upon throughout this book,
consists of iteratively reducing the error
by updating the parameters in the direction
that incrementally lowers the loss function.
This algorithm is called *gradient descent*.
On convex loss surfaces, it will eventually converge to a global minimum, 
and while the same cannot be said for nonconvex surfaces, 
it will at least lead towards a (hopefully good) local minimum.

The most naive application of gradient descent 
consists of taking the derivative of the true loss, 
which is an average of the losses computed 
on every single example in the dataset. 
In practice, this can be extremely slow. 
We must pass over the entire dataset before making a single update.
Thus, we will often settle for sampling a random minibatch of examples 
every time we need to computer the update, 
a variant called *stochastic gradient descent*.

In each iteration, we first randomly sample a minibatch $\mathcal{B}$ 
consisting of a fixed number of training data examples.  
We then compute the derivative (gradient) of the average loss 
on the mini batch with regard to the model parameters.
Finally, we multiply the gradient by a predetermined step size $\eta > 0$ 
and subtract the resulting term from the current parameter values.

We can express the update mathematically as follows
($\partial$ denotes the partial derivative) :

$$(\mathbf{w},b) \leftarrow (\mathbf{w},b) - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{(\mathbf{w},b)} l^{(i)}(\mathbf{w},b)$$


To summarize, steps of the algorithm are the following:
(i) we initialize the values of the model parameters, typically at random;
(ii) we iteratively sample random batches from the the data (many times),
updating the parameters in the direction of the negative gradient.

For quadratic losses and linear functions,
we can write this out explicitly as follows:
Note that $\mathbf{w}$ and $\mathbf{x}$ are vectors. 
Here, the more elegant vector notation makes the math 
much more readable than expressing things in terms of coefficients, 
say $w_1, w_2, \ldots, w_d$.

$$
\begin{aligned}
\mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && =
w - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\
b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  && =
b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

In the above equation, $|\mathcal{B}|$ represents 
the number of examples in each minibatch (the *batch size*)
and $\eta$ denotes the *learning rate*. 
We emphasize that the values of the batch size and learning rate 
are manually pre-specified and not typically learned through model training. 
These parameters that are tunable but not updated 
in the training loop are called *hyper-parameters*. 
*Hyperparameter tuning* is the process by which these are chosen,
and typically requires that we adjust the hyperparameters 
based on the results of the inner (training) loop
as assessed on a separate *validation* split of the data. 

After training for some predetermined number of iterations
(or until some other stopping criteria is met),
we record the estimated model parameters,
denoted $\hat{\mathbf{w}}, \hat{b}$
(in general the "hat" symbol denotes estimates).
Note that even if our function is truly linear and noiseless,
these parameters will not be the exact minimizers of the loss
because, although the algorithm converges slowly towards a local minimum
it cannot achieve it exactly in a finite number of steps.

Linear regression happens to be a convex learning problem,
and thus there is only one (global) minimum.
However, for more complicated models, like deep networks,
the loss surfaces contain many minima.
Fortunately, for reasons that are not yet fully understood,
deep learning practitioners seldom struggle to find parameters
that minimize the loss *on training data*.
The more formidable task is to find parameters
that will achieve low loss on data
that we have not seen before,
a challenge called *generalization*.
We return to these topics throughout the book.



### Making Predictions with the Learned Model


Given the learned linear regression model 
$\hat{\mathbf{w}}^\top x + \hat{b}$,
we can now estimate the price of a new house 
(not contained in the training data) 
given its area $x_1$ and age (year) $x_2$. 
Estimating targets given features is 
commonly called *prediction* and *inference*.

We will try to stick with *prediction* because
calling this step *inference*,
despite emerging as standard jargon in deep learning,
is somewhat of a misnomer.
In statistics, *inference* more often denotes
estimating parameters based on a dataset.
This misuse of terminology is a common source of confusion 
when deep learning practitioners talk to statisticians.


### Vectorization for Speed

When training our models, we typically want to process
whole minibatches of examples simultaneously.
Doing this efficiently requires that we vectorize the calculations
and leverage fast linear algebra libraries
rather than writing costly for-loops in Python.

To illustrate why this matters so much, 
we can consider two methods for adding vectors. 
To start we instantiate two $100000$-dimensional vectors 
containing all ones.
In one method we will loop over the vectors with a Python `for` loop.
In the other method we will rely on a single call to `np`.

```{.python .input}
%matplotlib inline
import d2l
import math
from mxnet import np
import time

n = 10000
a = np.ones(n)
b = np.ones(n)
```

Since we will benchmark the running time frequently in this book,
let's define a timer (hereafter accessed via the `d2l` package
to track the running time.

```{.python .input  n=1}
# Saved in the d2l package for later use
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # Start the timer
        self.start_time = time.time()

    def stop(self):
        # Stop the timer and record the time in a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # Return the average time
        return sum(self.times)/len(self.times)

    def sum(self):
        # Return the sum of time
        return sum(self.times)

    def cumsum(self):
        # Return the accumuated times
        return np.array(self.times).cumsum().tolist()
```

Now we can benchmark the workloads.
First, we add them, one coordinate at a time,
using a `for` loop.

```{.python .input  n=2}
timer = Timer()
c = np.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
'%.5f sec' % timer.stop()
```

Alternatively, we rely on `np` to compute the elementwise sum:

```{.python .input  n=3}
timer.start()
d = a + b
'%.5f sec' % timer.stop()
```

You probably noticed that the second method 
is dramatically faster than the first. 
Vectorizing code often yields order-of-magnitude speedups. 
Moreover, we push more of the math to the library 
and need not write as many calculations ourselves,
reducing the potential for errors.

## The Normal Distribution and Squared Loss

While you can already get your hands dirty using only the information above,
in the following section we can more formally motivate the square loss objective
via assumptions about the distribution of noise.

Recall from the above that the squared loss
$l(y, \hat{y}) = \frac{1}{2} (y - \hat{y})^2$
has many convenient properties.
These include a simple derivative
$\partial_{\hat{y}} l(y, \hat{y}) = (\hat{y} - y)$.

As we mentioned earlier, linear regression was invented by Gauss in 1795,
who also discovered the normal distribution (also called the *Gaussian*).
It turns out that the connection between 
the normal distribution and linear regression
runs deeper than common parentage. 
To refresh your memory, the probability density
of a normal distribution with mean $\mu$ and variance $\sigma^2$
is given as follows:

$$p(z) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (z - \mu)^2\right)$$

Below we define a Python function to compute the normal distribution.

```{.python .input}
x = np.arange(-7, 7, 0.01)

def normal(z, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(- 0.5 / sigma**2 * (z - mu)**2)
```

We can now visualize the normal distributions.

```{.python .input  n=2}
# Mean and variance pairs
parameters = [(0, 1), (0, 2), (3, 1)]
d2l.plot(x, [normal(x, mu, sigma) for mu, sigma in parameters], xlabel='z',
         ylabel='p(z)', figsize=(4.5, 2.5),
         legend=['mean %d, var %d' % (mu, sigma) for mu, sigma in parameters])
```

As you can see, changing the mean corresponds to a shift along the *x axis*, 
and increasing the variance spreads the distribution out, lowering its peak. 

One way to motivate linear regression with the mean squared error loss function
is to formally assume that observations arise from noisy observations,
where the noise is normally distributed as follows

$$y = \mathbf{w}^\top \mathbf{x} + b + \epsilon \text{ where } \epsilon \sim \mathcal{N}(0, \sigma^2).$$

Thus, we can now write out the *likelihood* 
of seeing a particular $y$ for a given $\mathbf{x}$ via

$$p(y|\mathbf{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (y - \mathbf{w}^\top \mathbf{x} - b)^2\right)$$

Now, according to the *maximum likelihood principle*,
the best values of $b$ and $\mathbf{w}$ are those
that maximize the *likelihood* of the entire dataset:

$$P(Y\mid X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)})$$

Estimators chosen according to the *maximum likelihood principle*
are called *Maximum Likelihood Estimators* (MLE). 
While, maximizing the product of many exponential functions, 
might look difficult,
we can simplify things significantly, without changing the objective,
by maximizing the log of the likelihood instead. 
For historical reasons, optimizations are more often expressed 
as minimization rather than maximization.
So, without changing anything we can minimize the *Negative Log-Likelihood (NLL)*
$-\log p(\mathbf y|\mathbf X)$.
Working out the math gives us:

$$-\log p(\mathbf y|\mathbf X) = \sum_{i=1}^n \frac{1}{2} \log(2 \pi \sigma^2) + \frac{1}{2 \sigma^2} \left(y^{(i)} - \mathbf{w}^\top \mathbf{x}^{(i)} - b\right)^2$$

Now we just need one more assumption: that $\sigma$ is some fixed constant. 
Thus we can ignore the first term because 
it does not depend on $\mathbf{w}$ or $b$. 
Now the second term is identical to the squared error objective introduced earlier, 
but for the multiplicative constant $\frac{1}{\sigma^2}$. 
Fortunately, the solution does not depend on $\sigma$.
It follows that minimizing squared error
is equvalent to maximum likelihood estimation 
of a linear model under the assumption of additive Gaussian noise.

## From Linear Regression to Deep Networks

So far we only talked about linear functions.
While neural networks cover a much richer family of models,
we can begin thinking of the linear model
as a neural network by expressing it the language of neural networks.
To begin, let's start by rewriting things in a 'layer' notation.

### Neural Network Diagram

Deep learning practitioners like to draw diagrams 
to visualize what is happening in their models. 
In :numref:`fig_single_neuron`, 
we depict our linear model as a neural network. 
Note that these diagrams indicate the connectivity pattern
(here, each input is connected to the output)
but not the values taken by the weights or biases.

![Linear regression is a single-layer neural network. ](../img/singleneuron.svg)
:label:`fig_single_neuron`

Because there is just a single computed neuron (node) in the graph
(the input values are not computed but given),
we can think of linear models as neural networks 
consisting of just a single artificial neuron.
Since for this model, every input is connected 
to every output (in this case there is only one output!),
we can regard this transformation as a *fully-connected layer*,
also commonly called a *dense layer*.
We will talk a lot more about networks composed of such layers
in the next chapter on multilayer perceptrons. 

### Biology

Although linear regression (invented in 1795) 
predates computational neuroscience,
so it might seem anachronistic to describe
linear regression as a neural network.
To see why linear models were a natural place to begin
when the cyberneticists/neurophysiologists 
Warren McCulloch and Walter Pitts
looked when they began to develop 
models of artificial neurons,
consider the cartoonish picture
of a biological neuron in :numref:`fig_Neuron`, consisting of 
*dendrites* (input terminals), 
the *nucleus* (CPU), the *axon* (output wire),
and the *axon terminals* (output terminals),
enabling connections to other neurons via *synapses*.

![The real neuron](../img/Neuron.svg)
:label:`fig_Neuron`

Information $x_i$ arriving from other neurons 
(or environmental sensors such as the retina)
is received in the dendrites. 
In particular, that information is weighted by *synaptic weights* $w_i$ 
determining the effect of the inputs
(e.g., activation or inhibition via the product $x_i w_i$). 
The weighted inputs arriving from multiple sources
are aggregated in the nucleus as a weighted sum $y = \sum_i x_i w_i + b$, 
and this information is then sent for further processing in the axon $y$, 
typically after some nonlinear processing via $\sigma(y)$. 
From there it either reaches its destination (e.g., a muscle) 
or is fed into another neuron via its dendrites.

Certainly, the high-level idea that many such units
could be cobbled together with the right connectivity
and right learning algorithm,
to produce far more interesting and complex behavior
than any one neuron along could express
owes to our study of real biological neural systems.

At the same time, most research in deep learning today
draws little direct inspiration in neuroscience.
We invoke Stuart Russell and Peter Norvig who, 
in their classic AI text book 
[Artificial Intelligence: A Modern Approach](https://en.wikipedia.org/wiki/Artificial_Intelligence:_A_Modern_Approach),
pointed out that although airplanes might have been *inspired* by birds,
orninthology has not been the primary driver 
of aeronautics innovation for some centuries. 
Likewise, inspiration in deep learning these days
comes in equal or greater measure from mathematics, 
statistics, and computer science.

## Summary

* Key ingredients in a machine learning model are training data, a loss function, an optimization algorithm, and quite obviously, the model itself.
* Vectorizing makes everything better (mostly math) and faster (mostly code).
* Minimizing an objective function and performing maximum likelihood can mean the same thing.
* Linear models are neural networks, too.

## Exercises

1. Assume that we have some data $x_1, \ldots, x_n \in \mathbb{R}$. Our goal is to find a constant $b$ such that $\sum_i (x_i - b)^2$ is minimized.
    * Find a closed-form solution for the optimal value of $b$.
    * How does this problem and its solution relate to the normal distribution?
1. Derive the closed-form solution to the optimization problem for linear regression with squared error. To keep things simple, you can omit the bias $b$ from the problem (we can do this in principled fashion by adding one column to $X$ consisting of all ones).
    * Write out the optimization problem in matrix and vector notation (treat all the data as a single matrix, all the target values as a single vector).
    * Compute the gradient of the loss with respect to $w$.
    * Find the closed form solution by setting the gradient equal to zero and solving the matrix equation.
    * When might this be better than using stochastic gradient descent? When might this method break? 
1. Assume that the noise model governing the additive noise $\epsilon$ is the exponential distribution. That is, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    * Write out the negative log-likelihood of the data under the model $-\log P(Y \mid X)$.
    * Can you find a closed form solution?
    * Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint - what happens near the stationary point as we keep on updating the parameters). Can you fix this?

## [Discussions](https://discuss.mxnet.io/t/2331)

![](../img/qr_linear-regression.svg)
