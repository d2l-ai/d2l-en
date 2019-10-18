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
the data set is called a *training data* or *training set*,
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
cancelling out when we take the dervative of the loss.
Since the training dataset is given to us, and thus out of our control,
the empirical error is only a function of the model parameters.
To make things more concrete, consider the example below
where we plot a regression problem for a one-dimensional case:

![Fit data with a linear model.](../img/fit_linreg.svg)

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
and while the same can't be said for nonconvex surfaces, 
it will at least lead towards a (hopefully good) local minimum.

The most naive application of gradient descent 
consists of taking the derivative of the true loss, 
which is an average of the losses computed 
on every single example in the dataset. 
In practice, this can be extremely slow. 
We must pass over the entire dataset before making a single update.
Thus, we'll often settle for sampling a random mini-batch of examples 
every time we need to computer the update, 
a variant called *stochastic gradient descent*.

In each iteration, we first randomly sample a mini-batch $\mathcal{B}$ 
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
say $w_1, w_2, \ldots w_d$.

$$
\begin{aligned}
\mathbf{w} &\leftarrow \mathbf{w} -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_{\mathbf{w}} l^{(i)}(\mathbf{w}, b) && =
w - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbf{x}^{(i)} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right),\\
b &\leftarrow b -  \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \partial_b l^{(i)}(\mathbf{w}, b)  && =
b - \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left(\mathbf{w}^\top \mathbf{x}^{(i)} + b - y^{(i)}\right).
\end{aligned}
$$

In the above equation, $|\mathcal{B}|$ represents 
the number of examples in each mini-batch (the *batch size*)
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
deep learning practitioners seldom stuggle to find parameters
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

```{.python .input  n=1}
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

```{.python .input  n=2}
# Save to the d2l package.
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        """Start the timer"""
        self.start_time = time.time()

    def stop(self):
        """Stop the timer and record the time in a list"""
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        """Return the average time"""
        return sum(self.times)/len(self.times)

    def sum(self):
        """Return the sum of time"""
        return sum(self.times)

    def cumsum(self):
        """Return the accumuated times"""
        return np.array(self.times).cumsum().tolist()
```

Now we can benchmark the workloads.
First, we add them, one coordinate at a time,
using a `for` loop.

```{.python .input  n=3}
timer = Timer()
c = np.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
'%.5f sec' % timer.stop()
```

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "'3.41232 sec'"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Alternatively, we rely on `np` to compute the elementwise sum:

```{.python .input  n=4}
timer.start()
d = a + b
'%.5f sec' % timer.stop()
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "'0.00031 sec'"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
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

As we mentioned earlier, linear regression was invented by Gausss in 1795,
who also discovered the normal distribution (also called the *Gaussian*).
It turns out that the connection between 
the normal distribution and linear regression
runs deeper than common parentage. 
To refresh your memory, the probability density
of a normal distribution with mean $\mu$ and variance $\sigma^2$
is given as follows:

$$p(z) = \frac{1}{\sqrt{2 \pi \sigma^2}} \exp\left(-\frac{1}{2 \sigma^2} (z - \mu)^2\right)$$

Below we define a Python function to compute the normal distribution.

```{.python .input  n=5}
x = np.arange(-7, 7, 0.01)

def normal(z, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma**2)
    return p * np.exp(- 0.5 / sigma**2 * (z - mu)**2)
```

For convenience, we also define a `plot` function
to plot multiple curves succinctly
since we will need to visualize many curves throughout the book.

```{.python .input  n=6}
# Save to the d2l package.
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear', fmts=None,
         figsize=(3.5, 2.5), axes=None):
    """Plot multiple lines"""
    d2l.set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    
    # Return True if X has 1 dimension
    def is_1d(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if is_1d(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif is_1d(Y):
        Y = [Y]

    # Broadcast
    if len(X) != len(Y):
        X = X * len(Y)
    if not fmts:
        fmts = ['-'] * len(X)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

# Save to the d2l package.
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """A utility function to set matplotlib axes"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend: axes.legend(legend)
    axes.grid()
```

We can now visualize the normal distributions.

```{.python .input  n=7}
# Mean and variance pairs
parameters = [(0,1), (0,2), (3,1)]
plot(x, [normal(x, mu, sigma) for mu, sigma in parameters],
     xlabel='z', ylabel='p(z)', figsize=(4.5, 2.5),
     legend = ['mean %d, var %d'%(mu, sigma) for mu, sigma in parameters])
```

```{.json .output n=7}
[
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 302.08125 180.65625\" width=\"302.08125pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 302.08125 180.65625 \nL 302.08125 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 43.78125 143.1 \nL 294.88125 143.1 \nL 294.88125 7.2 \nL 43.78125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 71.511736 143.1 \nL 71.511736 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"mf747359cc8\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"71.511736\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- \u22126 -->\n      <defs>\n       <path d=\"M 10.59375 35.5 \nL 73.1875 35.5 \nL 73.1875 27.203125 \nL 10.59375 27.203125 \nz\n\" id=\"DejaVuSans-8722\"/>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(64.140643 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-8722\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 104.145436 143.1 \nL 104.145436 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"104.145436\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- \u22124 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(96.774343 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-8722\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 136.779136 143.1 \nL 136.779136 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"136.779136\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- \u22122 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(129.408042 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-8722\"/>\n       <use x=\"83.789062\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 169.412836 143.1 \nL 169.412836 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"169.412836\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(166.231586 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 202.046536 143.1 \nL 202.046536 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"202.046536\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 2 -->\n      <g transform=\"translate(198.865286 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 234.680236 143.1 \nL 234.680236 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"234.680236\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 4 -->\n      <g transform=\"translate(231.498986 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_7\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 267.313936 143.1 \nL 267.313936 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"267.313936\" xlink:href=\"#mf747359cc8\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_7\">\n      <!-- 6 -->\n      <g transform=\"translate(264.132686 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_8\">\n     <!-- z -->\n     <defs>\n      <path d=\"M 5.515625 54.6875 \nL 48.1875 54.6875 \nL 48.1875 46.484375 \nL 14.40625 7.171875 \nL 48.1875 7.171875 \nL 48.1875 0 \nL 4.296875 0 \nL 4.296875 8.203125 \nL 38.09375 47.515625 \nL 5.515625 47.515625 \nz\n\" id=\"DejaVuSans-122\"/>\n     </defs>\n     <g transform=\"translate(166.707031 171.376563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-122\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_15\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 43.78125 136.922727 \nL 294.88125 136.922727 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_16\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m0800f1c772\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m0800f1c772\" y=\"136.922727\"/>\n      </g>\n     </g>\n     <g id=\"text_9\">\n      <!-- 0.0 -->\n      <defs>\n       <path d=\"M 10.6875 12.40625 \nL 21 12.40625 \nL 21 0 \nL 10.6875 0 \nz\n\" id=\"DejaVuSans-46\"/>\n      </defs>\n      <g transform=\"translate(20.878125 140.721946)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_17\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 43.78125 105.954475 \nL 294.88125 105.954475 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m0800f1c772\" y=\"105.954475\"/>\n      </g>\n     </g>\n     <g id=\"text_10\">\n      <!-- 0.1 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n      </defs>\n      <g transform=\"translate(20.878125 109.753694)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-49\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_19\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 43.78125 74.986223 \nL 294.88125 74.986223 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_20\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m0800f1c772\" y=\"74.986223\"/>\n      </g>\n     </g>\n     <g id=\"text_11\">\n      <!-- 0.2 -->\n      <g transform=\"translate(20.878125 78.785442)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-50\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_21\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 43.78125 44.017971 \nL 294.88125 44.017971 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_22\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m0800f1c772\" y=\"44.017971\"/>\n      </g>\n     </g>\n     <g id=\"text_12\">\n      <!-- 0.3 -->\n      <defs>\n       <path d=\"M 40.578125 39.3125 \nQ 47.65625 37.796875 51.625 33 \nQ 55.609375 28.21875 55.609375 21.1875 \nQ 55.609375 10.40625 48.1875 4.484375 \nQ 40.765625 -1.421875 27.09375 -1.421875 \nQ 22.515625 -1.421875 17.65625 -0.515625 \nQ 12.796875 0.390625 7.625 2.203125 \nL 7.625 11.71875 \nQ 11.71875 9.328125 16.59375 8.109375 \nQ 21.484375 6.890625 26.8125 6.890625 \nQ 36.078125 6.890625 40.9375 10.546875 \nQ 45.796875 14.203125 45.796875 21.1875 \nQ 45.796875 27.640625 41.28125 31.265625 \nQ 36.765625 34.90625 28.71875 34.90625 \nL 20.21875 34.90625 \nL 20.21875 43.015625 \nL 29.109375 43.015625 \nQ 36.375 43.015625 40.234375 45.921875 \nQ 44.09375 48.828125 44.09375 54.296875 \nQ 44.09375 59.90625 40.109375 62.90625 \nQ 36.140625 65.921875 28.71875 65.921875 \nQ 24.65625 65.921875 20.015625 65.03125 \nQ 15.375 64.15625 9.8125 62.3125 \nL 9.8125 71.09375 \nQ 15.4375 72.65625 20.34375 73.4375 \nQ 25.25 74.21875 29.59375 74.21875 \nQ 40.828125 74.21875 47.359375 69.109375 \nQ 53.90625 64.015625 53.90625 55.328125 \nQ 53.90625 49.265625 50.4375 45.09375 \nQ 46.96875 40.921875 40.578125 39.3125 \nz\n\" id=\"DejaVuSans-51\"/>\n      </defs>\n      <g transform=\"translate(20.878125 47.81719)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-51\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_23\">\n      <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 43.78125 13.049719 \nL 294.88125 13.049719 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_24\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"43.78125\" xlink:href=\"#m0800f1c772\" y=\"13.049719\"/>\n      </g>\n     </g>\n     <g id=\"text_13\">\n      <!-- 0.4 -->\n      <g transform=\"translate(20.878125 16.848938)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-46\"/>\n       <use x=\"95.410156\" xlink:href=\"#DejaVuSans-52\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_14\">\n     <!-- p(z) -->\n     <defs>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 31 75.875 \nQ 24.46875 64.65625 21.28125 53.65625 \nQ 18.109375 42.671875 18.109375 31.390625 \nQ 18.109375 20.125 21.3125 9.0625 \nQ 24.515625 -2 31 -13.1875 \nL 23.1875 -13.1875 \nQ 15.875 -1.703125 12.234375 9.375 \nQ 8.59375 20.453125 8.59375 31.390625 \nQ 8.59375 42.28125 12.203125 53.3125 \nQ 15.828125 64.359375 23.1875 75.875 \nz\n\" id=\"DejaVuSans-40\"/>\n      <path d=\"M 8.015625 75.875 \nL 15.828125 75.875 \nQ 23.140625 64.359375 26.78125 53.3125 \nQ 30.421875 42.28125 30.421875 31.390625 \nQ 30.421875 20.453125 26.78125 9.375 \nQ 23.140625 -1.703125 15.828125 -13.1875 \nL 8.015625 -13.1875 \nQ 14.5 -2 17.703125 9.0625 \nQ 20.90625 20.125 20.90625 31.390625 \nQ 20.90625 42.671875 17.703125 53.65625 \nQ 14.5 64.65625 8.015625 75.875 \nz\n\" id=\"DejaVuSans-41\"/>\n     </defs>\n     <g transform=\"translate(14.798438 84.85)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"63.476562\" xlink:href=\"#DejaVuSans-40\"/>\n      <use x=\"102.490234\" xlink:href=\"#DejaVuSans-122\"/>\n      <use x=\"154.980469\" xlink:href=\"#DejaVuSans-41\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_25\">\n    <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 55.194886 136.922727 \nL 108.224649 136.813535 \nL 113.609208 136.566288 \nL 117.198915 136.184417 \nL 119.972781 135.668952 \nL 122.257142 135.025185 \nL 124.215162 134.257847 \nL 126.01001 133.330371 \nL 127.804866 132.138337 \nL 129.436549 130.77945 \nL 131.06824 129.113083 \nL 132.699924 127.093511 \nL 134.331607 124.674772 \nL 135.963291 121.812697 \nL 137.594974 118.467299 \nL 139.226665 114.605492 \nL 141.021513 109.733828 \nL 142.816369 104.197086 \nL 144.774389 97.412526 \nL 146.895581 89.247641 \nL 149.506275 78.224543 \nL 153.259151 61.239325 \nL 157.501535 42.275262 \nL 159.785892 33.113073 \nL 161.580748 26.820509 \nL 163.049259 22.424539 \nL 164.354614 19.173265 \nL 165.496788 16.88464 \nL 166.475798 15.362588 \nL 167.454808 14.263614 \nL 168.270654 13.679586 \nL 169.086499 13.401979 \nL 169.739173 13.401979 \nL 170.391846 13.599458 \nL 171.207684 14.122462 \nL 172.02353 14.948572 \nL 173.00254 16.33118 \nL 173.98155 18.126548 \nL 175.123732 20.717339 \nL 176.429079 24.286967 \nL 177.897598 29.000681 \nL 179.692446 35.615337 \nL 181.813638 44.367147 \nL 184.91384 58.245054 \nL 190.461572 83.161144 \nL 192.909093 93.115019 \nL 195.030286 100.899654 \nL 196.988306 107.299468 \nL 198.783169 112.473252 \nL 200.578017 116.986085 \nL 202.209708 120.534576 \nL 203.841384 123.58549 \nL 205.473075 126.176452 \nL 207.104751 128.35022 \nL 208.736442 130.152332 \nL 210.368133 131.62881 \nL 211.999809 132.824477 \nL 213.794657 133.865798 \nL 215.752693 134.73293 \nL 217.873869 135.421683 \nL 220.321406 135.972059 \nL 223.258444 136.389278 \nL 227.011312 136.67952 \nL 232.395871 136.850878 \nL 243.001818 136.917995 \nL 283.467614 136.922727 \nL 283.467614 136.922727 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_26\">\n    <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 55.194886 136.7876 \nL 65.80084 136.522951 \nL 73.14342 136.126434 \nL 79.017488 135.590287 \nL 83.912546 134.926494 \nL 88.318087 134.10523 \nL 92.234135 133.153577 \nL 95.823839 132.063161 \nL 99.250378 130.79875 \nL 102.513745 129.367712 \nL 105.77712 127.695009 \nL 109.04049 125.764112 \nL 112.303861 123.563438 \nL 115.567228 121.0879 \nL 118.993767 118.196043 \nL 122.583479 114.860884 \nL 126.499519 110.903145 \nL 131.06824 105.948686 \nL 138.247647 97.770815 \nL 144.611224 90.644946 \nL 148.527265 86.58968 \nL 151.790639 83.53067 \nL 154.564497 81.224439 \nL 157.175199 79.344213 \nL 159.459555 77.957411 \nL 161.743912 76.832369 \nL 163.865105 76.036198 \nL 165.823125 75.522599 \nL 167.781145 75.22717 \nL 169.739173 75.153087 \nL 171.697193 75.301157 \nL 173.655213 75.669778 \nL 175.613233 76.254996 \nL 177.734426 77.126086 \nL 179.855618 78.233158 \nL 182.139975 79.673624 \nL 184.587504 81.480058 \nL 187.361362 83.820856 \nL 190.461572 86.751108 \nL 194.051283 90.469341 \nL 198.783169 95.721754 \nL 211.67348 110.215095 \nL 215.752693 114.383395 \nL 219.505561 117.905417 \nL 222.9321 120.825256 \nL 226.195467 123.328266 \nL 229.458833 125.556318 \nL 232.722216 127.513772 \nL 235.985583 129.211617 \nL 239.24895 130.665968 \nL 242.675489 131.952578 \nL 246.2652 133.063568 \nL 250.18124 134.03448 \nL 254.423625 134.846714 \nL 259.155511 135.514669 \nL 264.54007 136.040363 \nL 270.903632 136.432353 \nL 279.225229 136.707949 \nL 283.467614 136.785216 \nL 283.467614 136.785216 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_27\">\n    <path clip-path=\"url(#pd8a2b2726d)\" d=\"M 55.194886 136.922727 \nL 157.175199 136.813535 \nL 162.559758 136.566288 \nL 166.149461 136.184417 \nL 168.923327 135.668953 \nL 171.207684 135.025188 \nL 173.165712 134.257847 \nL 174.96056 133.330371 \nL 176.755415 132.138337 \nL 178.387099 130.77945 \nL 180.018782 129.113093 \nL 181.650474 127.093511 \nL 183.282157 124.674772 \nL 184.91384 121.812697 \nL 186.545532 118.467281 \nL 188.177207 114.605511 \nL 189.972071 109.733805 \nL 191.766919 104.197086 \nL 193.724939 97.412526 \nL 195.846131 89.247641 \nL 198.456825 78.224543 \nL 202.209708 61.239288 \nL 206.452077 42.27529 \nL 208.736442 33.113073 \nL 210.53129 26.820546 \nL 211.999809 22.424539 \nL 213.305156 19.173284 \nL 214.447346 16.884622 \nL 215.426348 15.362588 \nL 216.405366 14.263605 \nL 217.221196 13.679596 \nL 218.037042 13.401979 \nL 218.689715 13.401979 \nL 219.342388 13.599449 \nL 220.158234 14.122462 \nL 220.97408 14.948572 \nL 221.953082 16.331171 \nL 222.9321 18.126548 \nL 224.074274 20.71733 \nL 225.379621 24.286948 \nL 226.84814 29.000654 \nL 228.643003 35.615364 \nL 230.764196 44.367184 \nL 233.86439 58.245054 \nL 239.412122 83.161144 \nL 241.859643 93.115019 \nL 243.980835 100.899654 \nL 245.938856 107.299468 \nL 247.733719 112.473252 \nL 249.528567 116.986085 \nL 251.160243 120.534544 \nL 252.791934 123.58549 \nL 254.423625 126.176452 \nL 256.055301 128.35022 \nL 257.686992 130.152332 \nL 259.318668 131.628798 \nL 260.950359 132.824477 \nL 262.745207 133.865798 \nL 264.703242 134.73293 \nL 266.824419 135.421683 \nL 269.271956 135.972059 \nL 272.208978 136.389276 \nL 275.961862 136.67952 \nL 281.346421 136.850878 \nL 283.467614 136.879593 \nL 283.467614 136.879593 \n\" style=\"fill:none;stroke:#2ca02c;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 43.78125 143.1 \nL 43.78125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 294.88125 143.1 \nL 294.88125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 43.78125 143.1 \nL 294.88125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 43.78125 7.2 \nL 294.88125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 50.78125 59.234375 \nL 152.735938 59.234375 \nQ 154.735938 59.234375 154.735938 57.234375 \nL 154.735938 14.2 \nQ 154.735938 12.2 152.735938 12.2 \nL 50.78125 12.2 \nQ 48.78125 12.2 48.78125 14.2 \nL 48.78125 57.234375 \nQ 48.78125 59.234375 50.78125 59.234375 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_28\">\n     <path d=\"M 52.78125 20.298438 \nL 72.78125 20.298438 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_29\"/>\n    <g id=\"text_15\">\n     <!-- mean 0, var 1 -->\n     <defs>\n      <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-109\"/>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n      <path id=\"DejaVuSans-32\"/>\n      <path d=\"M 11.71875 12.40625 \nL 22.015625 12.40625 \nL 22.015625 4 \nL 14.015625 -11.625 \nL 7.71875 -11.625 \nL 11.71875 4 \nz\n\" id=\"DejaVuSans-44\"/>\n      <path d=\"M 2.984375 54.6875 \nL 12.5 54.6875 \nL 29.59375 8.796875 \nL 46.6875 54.6875 \nL 56.203125 54.6875 \nL 35.6875 0 \nL 23.484375 0 \nz\n\" id=\"DejaVuSans-118\"/>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n     </defs>\n     <g transform=\"translate(80.78125 23.798438)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-109\"/>\n      <use x=\"97.412109\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"158.935547\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"220.214844\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"283.59375\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"315.380859\" xlink:href=\"#DejaVuSans-48\"/>\n      <use x=\"379.003906\" xlink:href=\"#DejaVuSans-44\"/>\n      <use x=\"410.791016\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"442.578125\" xlink:href=\"#DejaVuSans-118\"/>\n      <use x=\"501.757812\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"563.037109\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"604.150391\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"635.9375\" xlink:href=\"#DejaVuSans-49\"/>\n     </g>\n    </g>\n    <g id=\"line2d_30\">\n     <path d=\"M 52.78125 34.976562 \nL 72.78125 34.976562 \n\" style=\"fill:none;stroke:#ff7f0e;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_31\"/>\n    <g id=\"text_16\">\n     <!-- mean 0, var 2 -->\n     <g transform=\"translate(80.78125 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-109\"/>\n      <use x=\"97.412109\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"158.935547\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"220.214844\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"283.59375\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"315.380859\" xlink:href=\"#DejaVuSans-48\"/>\n      <use x=\"379.003906\" xlink:href=\"#DejaVuSans-44\"/>\n      <use x=\"410.791016\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"442.578125\" xlink:href=\"#DejaVuSans-118\"/>\n      <use x=\"501.757812\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"563.037109\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"604.150391\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"635.9375\" xlink:href=\"#DejaVuSans-50\"/>\n     </g>\n    </g>\n    <g id=\"line2d_32\">\n     <path d=\"M 52.78125 49.654688 \nL 72.78125 49.654688 \n\" style=\"fill:none;stroke:#2ca02c;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_33\"/>\n    <g id=\"text_17\">\n     <!-- mean 3, var 1 -->\n     <g transform=\"translate(80.78125 53.154688)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-109\"/>\n      <use x=\"97.412109\" xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"158.935547\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"220.214844\" xlink:href=\"#DejaVuSans-110\"/>\n      <use x=\"283.59375\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"315.380859\" xlink:href=\"#DejaVuSans-51\"/>\n      <use x=\"379.003906\" xlink:href=\"#DejaVuSans-44\"/>\n      <use x=\"410.791016\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"442.578125\" xlink:href=\"#DejaVuSans-118\"/>\n      <use x=\"501.757812\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"563.037109\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"604.150391\" xlink:href=\"#DejaVuSans-32\"/>\n      <use x=\"635.9375\" xlink:href=\"#DejaVuSans-49\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"pd8a2b2726d\">\n   <rect height=\"135.9\" width=\"251.1\" x=\"43.78125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 324x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
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

$$p(Y|X) = \prod_{i=1}^{n} p(y^{(i)}|\mathbf{x}^{(i)})$$

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
it doesn't depend on $\mathbf{w}$ or $b$. 
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
consider the following cartoonish picture
of a biological neuron, consisting of 
*dendrites* (input terminals), 
the *nucleus* (CPU), the *axon* (output wire),
and the *axon terminals* (output terminals),
enabling connections to other neurons via *synapses*.

![The real neuron](../img/Neuron.svg)

Information $x_i$ arriving from other neurons 
(or environmental sensors such as the retina)
is received in the dendrites. 
In particular, that information is weighted by *synaptic weights* $w_i$ 
determining the effect of the inputs
(e.g. activation or inhibition via the product $x_i w_i$). 
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

At the same time, most reserach in deep learning today
draws little direct inspiration in neuroscience.
We invoke Stuart Russell and Peter Norvig who, 
in their classic AI text book 
[Artificial Intelligence: A Modern Approach](https://en.wikipedia.org/wiki/Artificial_Intelligence:_A_Modern_Approach),
pointed out that although airplanes might have been *inspired* by birds,
orninthology has not been the primary driver 
of aeronautics innovaton for some centuries. 
Likewise, inspiration in deep learning these days
comes in equal or greater measure from mathematics, 
statistics, and computer science.

## Summary

* Key ingredients in a machine learning model are training data, a loss function, an optimization algorithm, and quite obviously, the model itself.
* Vectorizing makes everything better (mostly math) and faster (mostly code).
* Minimizing an objective function and performing maximum likelihood can mean the same thing.
* Linear models are neural networks, too.

## Exercises

1. Assume that we have some data $x_1, \ldots x_n \in \mathbb{R}$. Our goal is to find a constant $b$ such that $\sum_i (x_i - b)^2$ is minimized.
    * Find a closed-form solution for the optimal value of $b$.
    * How does this problem and its solution relate to the normal distribution?
1. Derive the closed-form solution to the optimization problem for linear regression with squared error. To keep things simple, you can omit the bias $b$ from the problem (we can do this in principled fashion by adding one column to $X$ consisting of all ones).
    * Write out the optimization problem in matrix and vector notation (treat all the data as a single matrix, all the target values as a single vector).
    * Compute the gradient of the loss with respect to $w$.
    * Find the closed form solution by setting the gradient equal to zero and solving the matrix equation.
    * When might this be better than using stochastic gradient descent? When might this method break? 
1. Assume that the noise model governing the additive noise $\epsilon$ is the exponential distribution. That is, $p(\epsilon) = \frac{1}{2} \exp(-|\epsilon|)$.
    * Write out the negative log-likelihood of the data under the model $-\log p(Y|X)$.
    * Can you find a closed form solution?
    * Suggest a stochastic gradient descent algorithm to solve this problem. What could possibly go wrong (hint - what happens near the stationary point as we keep on updating the parameters). Can you fix this?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2331)

![](../img/qr_linear-regression.svg)
