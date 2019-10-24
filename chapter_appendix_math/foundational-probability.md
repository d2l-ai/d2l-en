# Intermediate Probability Theory
:label:`foundational-probability`

In :numref:`chapter_prob` we saw the basics of how to work with discrete random variables, which in our case refer to those random variables which take either a finite set of possible values, or the integers.  In this appendix, we investigate applications of probability to ML via Naive Bayes, learn about continuous random variables, and then learn about common distributions we encounter.  

## Naive Bayes

Before we worry about complex optimization algorithms or GPUs, we can already deploy our first classifier, relying only on simple statistical estimators and our understanding of conditional independence. Learning is all about making assumptions. If we want to classify a new data point that we have never seen before we have to make some assumptions about which data points are similar to each other. The naive Bayes classifier, a popular and remarkably simple algorithm, assumes all features are independent of each other to simplify the computation. In this chapter, we will apply this model to recognize characters in images.

Let us first import libraries and modules. Especially, we import `d2l`, which now contains the function `use_svg_display` we defined in :numref:`chapter_prob`.

```{.python .input  n=72}
%matplotlib inline
import d2l
import math
from mxnet import np, npx, gluon
npx.set_np()
d2l.use_svg_display()
```

### Optical Character Recognition

MNIST :cite:`LeCun.Bottou.Bengio.ea.1998` is one of widely used datasets. It contains 60,000 images for training and 10,000 images for validation. We will formally introduce training data in :numref:`chapter_linear_regression` and validation data in :numref:`chapter_model_selection` later, here we just simply remember we will train the naive Bayes model in the training data and then test its quality on the validation data. Each image contains a handwritten digit from 0 to 9. The task is classifying each image into the corresponding digit.

Gluon, MXNet's high-level interface for implementing neural networks, provides a `MNIST` class in the `data.vision` module to 
automatically retrieve the dataset via our Internet connection.
Subsequently, Gluon will use the already-downloaded local copy.
We specify whether we are requesting the training set or the test set
by setting the value of the parameter `train` to `True` or `False`, respectively.
Each image is a grayscale image with both width and height of 28 with shape $(28,28,1)$. We use a customized transformation to remove the last channel dimension. In addition, each pixel is presented by a unsigned 8-bit integer, we quantize them into binary features to simplify the problem.

```{.python .input}
np.floor?
```

```{.python .input}
def transform(data, label):
    return np.floor(data.astype('float32')/128).squeeze(axis=-1), label

mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test  = gluon.data.vision.MNIST(train=False, transform=transform)
```

We can access a particular example, which contains the image and the corresponding label.

```{.python .input}
image, label = mnist_train[2]
image.shape, label
```

Our example, stored here in the variable `image` corresponds to an image with a height and width of 28 pixels. Each pixel is an 8-bit unsigned integer (uint8) with values between 0 and 255. It is stored in a 3D ndarray, whose last dimension is the number of channels. Since the data set is a grayscale image, the number of channels is 1. When we encounter color, images, we will have 3 channels for red, green, and blue. To keep things simple, we will record the shape of the image with the height and width of $h$ and $w$ pixels, respectively, as $h \times w$ or `(h, w)`.

```{.python .input}
image.shape, image.dtype
```

The label of each image is represented as a scalar in NumPy. Its type is a 32-bit integer.

```{.python .input}
label, type(label), label.dtype
```

We can also access multiple examples at the same time.

```{.python .input}
images, labels = mnist_train[10:38]
images.shape, labels.shape
```

Now Let us create a function to visualize these examples.

```{.python .input}
# Save to the d2l package. 
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """Plot a list of images."""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img.asnumpy())
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

show_images(images, 2, 9);
```

### The Probabilistic Model for Classification

In a classification task, we map an example into a category. Here an example is a grayscale $28\times 28$ image, and a category is a digit. (Refer to :numref:`chapter_softmax` for a more detailed explanation.) 
One natural way to express the classification task is via the probabilistic question: what is the most likely label given the features (i.e. image pixels)? Denote by $\mathbf x\in\mathbb R^d$ the features of the example and $y\in\mathbb R$ the label. Here features are image pixels, where we can reshape a 2-dimensional image to a vector so that $d=28^2=784$, and labels are digits. We will formally define general features and labels in :numref:`chapter_linear_regression`. The $p(y | \mathbf{x})$ is the probability of the label given the features. If we are able to compute these probabilities, which are $p(y | \mathbf{x})$ for $y=0,\ldots,9$ in our example, then the classifier will output the prediction  $\hat{y}$ given by the expression:

$$\hat{y} = \operatorname*{argmax} \> p(y | \mathbf{x}).$$

Unfortunately, this requires that we estimate $p(y | \mathbf{x})$ for every value of $\mathbf{x} = x_1, ..., x_d$. Imagine that each feature could take one of $2$ values. For example, the feature $x_1 = 1$ might signify that the word apple appears in a given document and $x_1 = 0$ would signify that it does not. If we had $30$ such binary features, that would mean that we need to be prepared to classify any of $2^{30}$ (over 1 billion!) possible values of the input vector $\mathbf{x}$.

Moreover, where is the learning? If we need to see every single possible example in order to predict the corresponding label then we are not really learning a pattern but just memorizing the dataset.

## The Naive Bayes Classifier

Fortunately, by making some assumptions about conditional independence, we can introduce some inductive bias and build a model capable of generalizing from a comparatively modest selection of training examples. To begin, Let us use Bayes Theorem, to express the classifier as

$$\hat{y} = \operatorname*{argmax}_y \> p(y | \mathbf{x}) = \operatorname*{argmax}_y \> \frac{p( \mathbf{x} | y) p(y)}{p(\mathbf{x})}.$$

Note that the denominator is the normalizing term $p(\mathbf{x})$ which does not depend on the value of the label $y$. As a result, we only need to worry about comparing the numerator across different values of $y$. Even if calculating the denominator turned out to be intractable, we could get away with ignoring it, so long as we could evaluate the numerator. Fortunately, however, even if we wanted to recover the normalizing constant, we could, since we know that $\sum_y p(y | \mathbf{x}) = 1$, hence we can always recover the normalization term.

Now, Let us focus on $p( \mathbf{x} | y)$. Using the chain rule of probability, we can express the term $p( \mathbf{x} | y)$ as

$$p(x_1 |y) \cdot p(x_2 | x_1, y) \cdot ... \cdot p( x_d | x_1, ..., x_{d-1}, y)$$

By itself, this expression does not get us any further. We still must estimate roughly $2^d$ parameters. However, if we assume that *the features are conditionally independent of each other, given the label*, then suddenly we are in much better shape, as this term simplifies to $\prod_i p(x_i | y)$, giving us the predictor

$$ \hat{y} = \operatorname*{argmax}_y \> \prod_{i=1}^d p(x_i | y) p(y).$$

If we can estimate $\prod_i p(x_i=1 | y)$ for every $i$ and $y$, and save its value in $P_{xy}[i,y]$, here $P_{xy}$ is a $d\times n$ matrix with $n$ being the number of classes and $y\in\{1,\ldots,n\}$. In addition, we estimate $p(y)$ for every $y$ and save it in $P_y[y]$, with $P_y$ a $n$-length vector. Then for any new example $\mathbf x$, we could compute

$$ \hat{y} = \operatorname*{argmax}_y \> \prod_{i=1}^d P_{xy}[x_i, y]P_y[y],$$
:eqlabel:`eq_naive_bayes_estimation`

for any $y$. So our assumption of conditional independence has taken the complexity of our model from an exponential dependence on the number of features $O(2^dn)$ to a linear dependence, which is $O(dn)$.


### Training

The problem now is that we do not actually know $P_{xy}$ and $P_y$. So we need to estimate their values given some training data first. This is what is called *training* the model. Estimating $P_y$ is not too hard. Since we are only dealing with $10$ classes, this is pretty easy - simply count the number of occurrences $n_y$ for each of the digits and divide it by the total amount of data $n$. For instance, if digit 8 occurs $n_8 = 5,800$ times and we have a total of $n = 60,000$ images, the probability estimate is $p(y=8) = 0.0967$.

```{.python .input  n=50}
X, Y = mnist_train[:]  # all training examples

n_y = np.zeros((10))
for y in range(10):
    n_y[y] = (Y==y).sum()
P_y = n_y / n_y.sum()
P_y
```

Now on to slightly more difficult things $P_{xy}$. Since we picked black and white images, $p(x_i | y)$ denotes the probability that pixel $i$ is switched on for class $y$. Just like before we can go and count the number of times $n_{iy}$ such that an event occurs and divide it by the total number of occurrences of $y$, i.e. $n_y$. But there is something slightly troubling: certain pixels may never be black (e.g. for very well cropped images the corner pixels might always be white). A convenient way for statisticians to deal with this problem is to add pseudo counts to all occurrences. Hence, rather than $n_{iy}$ we use $n_{iy}+1$ and instead of $n_y$ we use $n_{y} + 1$. This is also called [Laplace Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing).

```{.python .input  n=66}
n_x = np.zeros((10, 28, 28))
for y in range(10):
    n_x[y] = np.array(X.asnumpy()[Y.asnumpy()==y].sum(axis=0))
P_xy = (n_x+1) / (n_y+1).reshape(10,1,1)

show_images(P_xy, 2, 5);
```

By visualizing these $10\times 28\times 28$ probabilities (for each pixel for each class) we could get some mean looking digits.  ...

Now we can use :eqref:`eq_naive_bayes_estimation` to predict a new image. Given $\mathbf x$, the following functions computes $p(\mathbf x|y)p(y)$ for every $y$.

```{.python .input}
np.expand_dims?
```

```{.python .input}
def bayes_pred(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = P_xy * x + (1-P_xy)*(1-x)
    p_xy = p_xy.reshape(10,-1).prod(axis=1) # p(x|y)
    return np.array(p_xy) * P_y

image, label = mnist_test[0]
bayes_pred(image)
```

This went horribly wrong! To find out why, Let us look at the per pixel probabilities. They are typically numbers between $0.001$ and $1$. We are multiplying $784$ of them. At this point it is worth mentioning that we are calculating these numbers on a computer, hence with a fixed range for the exponent. What happens is that we experience *numerical underflow*, i.e. multiplying all the small numbers leads to something even smaller until it is rounded down to zero.

To fix this we use the fact that $\log a b = \log a + \log b$, i.e. we switch to summing logarithms.
Even if both $a$ and $b$ are small numbers, the logarithm values should be in a proper range.

```{.python .input}
a = 0.1
print('underflow:', a**784)
print('logrithm is normal:', 784*math.log(a))
```

Since the logarithm is an increasing function, so we can rewrite :eqref:`eq_naive_bayes_estimation` as

$$ \hat{y} = \operatorname*{argmax}_y \> \sum_{i=1}^d \log P_{xy}[x_i, y] + \log P_y[y].$$

We can implement the following stable version:

```{.python .input}
log_P_xy = np.log(P_xy)
log_P_xy_neg = np.log(1-P_xy)
log_P_y = np.log(P_y)

def bayes_pred_stable(x):
    x = np.expand_dims(x, axis=0)  # (28, 28) -> (1, 28, 28)
    p_xy = log_P_xy * x + log_P_xy_neg * (1-x)
    p_xy = p_xy.reshape(10,-1).sum(axis=1) # p(x|y)
    return p_xy + log_P_y

py = bayes_pred_stable(image)
py
```

Check if the prediction is correct.

```{.python .input}
# convert label which is a scalar tensor of int32 dtype
# to a Python scalar integer for comparison
py.argmax(axis=0) == int(label)
```

Now predict a few validation examples, we can see the the Bayes
classifier works pretty well except for the 9th 16th digits.

```{.python .input}
def predict(X):
    return [bayes_pred_stable(x).argmax(axis=0).astype(np.int32) for x in X]

X, y = mnist_test[:18]
preds = predict(X)
show_images(X, 2, 9, titles=[str(d) for d in preds]);
```

Finally, Let us compute the overall accuracy of the classifier.

```{.python .input}
X, y = mnist_test[:]
preds = np.array(predict(X), dtype=np.int32)
'Validation accuracy', float((preds == y).sum()) / len(y)
```

Modern deep networks achieve error rates of less than 0.01. The poor performance is due to the incorrect statistical assumptions that we made in our model: we assumed that each and every pixel are *independently* generated, depending only on the label. This is clearly not how humans write digits, and this wrong assumption led to the downfall of our overly naive (Bayes) classifier.

## Continuous Random Variables

All of the probability we have discussed so far has dealt with discrete random variables.  This is sufficient to understand the core concepts we encounter in machine learning, however some applications require understanding how to deal with continuous random variables, that is random variables which take on a any value value in $\mathbb{R}$.  

### From Discrete to Continuous

To understand the additional technical challenges encountered when working with continuous random variables, Let us perform a thought experiment.  Suppose we are throwing a dart at the dart board, and we want to know the probability that it hits exactly $2 \text{cm}$ from the center of the board.

To start with, we imagine measuring to a single digit of accuracy, that is to say with bins for $0 \text{cm}$, $1 \text{cm}$, $2 \text{cm}$, and so on.  We throw say $100$ darts at the dart board, and if $20$ of them fall into the bin for $2\text{cm}$ we conclude that $20\%$ of the darts we throw hit the board $2 \text{cm}$ away from the center.

However, when we look closer, this does not really match our question!  We wanted exactly, whereas these bins hold all that fell between say $1.5\text{cm}$ and $2.5\text{cm}$.  

Undeterred, we continue further.  We measure even more precisely, say $1.9\text{cm}$, $2.0\text{cm}$, $2.1\text{cm}$, and now see that perhaps $3$ of the $100$ darts hit the board in the $2.0\text{cm}$ bucket.  Thus we conclude the probability is $3\%$.

However, this does not actually solve anything!  we have just pushed the issue down one digit further.  Indeed, if we abstract a bit, Let us imagine we know the probability for the first $k$ digits matching with $2.00000\ldots$ and we want to know the probability it matches for the first $k+1$, it is fairly reasonable to assume that the $k+1$-st digit is essentially a random choice from the set $\{0,1,2,\ldots,9\}$.  At least, I cannot conceive of a physically meaningful process which would force the number of micrometers away form the center to prefer to end in a $7$ vs a $3$.  

What this means is that in essence each additional digit of accuracy we require should decrease probability of matching by a factor of $10$.  Or put another way, we would expect that 
$$
P(\text{distance is } 2.00\ldots \text{ to } k \text{ digits} ) \approx p\cdot10^{-k}.
$$
The value $p$ essentially encodes what happens with the first few digits, and the $10^{-k}$ handles the rest.

Notice that if we know the position accurate to $k=4$ digits after the decimal. that means we know the value falls within the interval say $[(1.99995,2.00005]$ which is an interval of length $2.00005-1.99995 = 10^{-4}$.  Thus, if we call the length of this interval $\epsilon$, we can say
$$
P(\text{distance is in an } \epsilon\text{-sized interval around } 2 ) \approx \epsilon \cdot p.
$$

Let us take this one final step further.  We have been thinking about the point $2$ the entire time, but never thinking about other points.  Of course, nothing is different there fundamentally, but it is the case that the value $p$ will likely depend on the point.  We would at least hope that a dart thrower was more likely to hit a point near the center, like $2\text{cm}$ rather than $20\text{cm}$.  Thus, the value $p$ is not fixed, but rather should depend on the point $x$.  This tells us that we should expect
$$
P(\text{distance is in an } \epsilon \text{-sized interval around } x ) \approx \epsilon \cdot p(x).
$$

Indeed, this is precisely what the *probability density function* is.  It is a function $p(x)$ which encodes the relative probability of hitting near one point versus another.

??? NEED FIGURES HERE ???

### Probability Density Functions

Let us now investigate this further.  We have already seen what a probability density function is intuitively, namely for a random variable $X$, the density function is a function $p(x)$ so that 
$$
P(X \text{ is in an } \epsilon \text{-sized interval around } x ) \approx \epsilon \cdot p(x).
$$
but what does this imply for the properties of $p(x)$?

First, probabilities are never negative, thus we should expect that $p(x) \ge 0$ as well.  

Second, Let us imagine that we slice up the $\mathbb{R}$ into an infinite number of slices which are $\epsilon$ wide, say the slice from $(\epsilon\cdot i, \epsilon \cdot (i+1)]$.  For each of these, we know the probability is approximately
$$
P(X \text{is in an } \epsilon\text{-sized interval around } x ) \approx \epsilon \cdot p(\epsilon \cdot i),
$$
so summed over all of them it should be
$$
P(X\in\mathbb{R}) \approx \sum_i \epsilon \cdot p(\epsilon\cdot i).
$$
This is nothing more than the approximation of an integral discussed in ???REF???, thus we can say that
$$
P(X\in\mathbb{R}) = \int_{-\infty}^{\infty} p(x) \; dx.
$$
Since we know that $P(X\in\mathbb{R}) = 1$, since the random variable must take on *some* number, we can conclude that for any density
$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$
Indeed, digging into this further shows that for any $a,b$, we see that
$$
P(X\in(a,b]) = \int _ {a}^{b} p(x) \; dx.
$$

It turns out that these two properties describe exactly the space of possible probability density functions (or *p.d.f.*'s for the commonly encountered abbreviation).  They are non-negative functions $p(x) \ge 0$ such that 
$$
\int_{-\infty}^{\infty} p(x) \; dx = 1.
$$

We interpret this function to mean that if we want to know the probability our random variable is in a specific interval we can get that by integration:
$$
P(X\in(a,b]) = \int _ {a}^{b} p(x) \; dx.
$$

In ???REF??? we will see a number of common distributions, but Let us continue working in the abstract for a little longer.

### Cumulative Distribution Functions

In the previous section, we saw the notion of the p.d.f.  In practice, this is a commonly encountered method to discuss continuous random variables, but it has one significant pitfall: that the values of the p.d.f. are not themselves probabilities, but rather a function that must be integrated to yield probabilities.  There is nothing wrong with a density being larger than $10$, as long as it is not larger than $10$ for more than an interval of length $1/10$!  This can be counter-intuitive, so people often also think in terms of the *cumulative distribution function*, or c.d.f., which *is* a probability.

In particular, the c.d.f. for a random variable $X$ with density $p(x)$ is defined as
$$
F(x) = \int _ {-\infty}^{x} p(x) \; dx = P(X \le x).
$$
The c.d.f. is the function which tells us the probability that the random variable is less than or equal to $x$.  

Let us observe a few properties.

* $F(x) \rightarrow 0$ as $x\rightarrow -\infty$.
* $F(x) \rightarrow 1$ as $x\rightarrow \infty$.
* $F(x)$ is non-decreasing ($y > x \implies F(y) \ge F(x)$).
* $F(x)$ is continuous (has no jumps) if $X$ is a continuous random variable.

With the fourth bullet point, note that this would not be true if $X$ were discrete, say taking the values $0$ and $1$ both with probability $1/2$.  In that case
$$
F(x) = \begin{cases}
0 & x < 0 \\
\frac{1}{2} & x < 1 \\
1 & x \ge 1.
\end{cases}
$$

In this example, we see one of the benefits of working with the c.d.f., the ability to deal with continuous or discrete random variables in the same framework, or indeed mixtures of the two (flip a coin: if heads return the roll of a die, if tails return the distance of a dart throw from the center of a dart board).  

### Means, Variances

Suppose we are dealing with a random variables $X$.  The distribution itself can be hard to interpret.  It is often useful to be able to summarize the behavior of a random variable concisely.  Numbers that help we capture the behavior of a random variable are called *summary statistics*.  The most commonly encountered ones are the *mean*, the *variance*, and the *standard deviation*.

The *mean* encodes the average value of a random variable.  If we have a discrete random variable $X$, which takes the values $x_i$ with probabilities $p_i$, then the mean is simply given by the weighted average: sum the values times the probability that the random variable takes on that value.
$$
\mu_X = E[X] = \sum_i x_i p_i.
$$

The way we should interpret the mean (albeit with caution) is that it tells we essentially where the random variable tends to be located.  

As a simple example that we will examine throughout this section, Let us take $X$ to be the random variable which takes the value $a-2$ with probability $p$, $a+2$ with probability $p$ and $a$ with probability $1-2p$.  We can compute that, for any possible choice of $a$ and $p$, the mean is
$$
mu_X = E[X] = \sum_i x_i p_i. = (a-2)p + a(1-2p) + (a+2)p = a.
$$
Thus we see that $a$ is the same as the mean.  This matches the intuition since $a$ is the location around which we centered our random variable.

Because they are helpful, Let us summarize a few properties:

* For any random variable $X$ and numbers $a$ and $b$, we have that $\mu_{aX+b} = a\mu_X + b$
* If we have two random variables $X$ and $Y$, we have $\mu_{X+Y} = \mu_X+\mu_Y$.

Means are extremely useful to understand the average behavior of a random variable, however the mean is not sufficient to even have an intuitive understanding.  As a simple example, making a profit of $\$10 \pm \$1$ per sale is very different from making $\$10 \pm \$15$ per sale, despite having the same average value.  The second one has a much larger degree of fluctuation, and thus represents a much larger risk.  Thus, to understand the behavior of a random variable, we will need at minimum one more measure: some measure of how widely a random variable fluctuates.

This leads us to consider the variance of a random variable.  This is a quantitative measure of how far a random variable deviates from the mean.  Consider the expression $X - \mu_X$.  This is the deviation of the random variable from its mean.  This value can be positive or negative, so we need to do something to make it positive so that we are measuring the magnitude of the deviation.  A reasonable thing to try is to look at $\left|X-\mu_X\right|$, and indeed this leads to a useful quantity called the *mean absolute deviation*, however due to connections with other areas of mathematics and statistics, people often look at a different solution.  In particular, they look at $(X-\mu_X)^2.$  If we look at the typical size of this quantity by taking the mean, we arrive at the variance
$$
\sigma_X^2 = \mathrm{var}(X) = E\left[(X-\mu_X)^2\right] = E[X^2] - \mu_X^2,
$$
where the last equation holds by expanding out the definition in the middle, and applying the properties of expectation listed above.

Let us look at in in our example where $X$ is the random variable which takes the value $a-2$ with probability $p$, $a+2$ with probability $p$ and $a$ with probability $1-2p$.  In this case $\mu_X = a$.  So all we need to compute is $E\left[X^2\right]$.  This can readily be done:
$$
E\left[X^2\right] = (a-2)^2p + a^2(1-2p) + (a+2)p = a^2 + 8p.
$$
Thus, we see that our variance is
$$
\sigma_X^2 = \mathrm{var}(X) = E[X^2] - \mu_X^2 = a^2 + 8p - a^2 = 8p.
$$

This result again makes sense.  The largest $p$ can be is $1/2$ which corresponds to picking $a-2$ or $a+2$ with a coinflip.  The variance of this being $4$ corresponds to the fact that both $a-2$ and $a+2$ are $2$ units away from the mean, and $2^2 = 4$.  On the other end of the spectrum, if $p=0$, this random variable always takes the value $0$ and so it has no variance at all.

We will list a few simple properties of variances below.

* For any random variable $X$, $\mathrm{var}(X) \ge 0$.
* For any random variable $X$ and numbers $a$ and $b$, we have that $\mathrm{var}(aX+b) = a^2\mathrm{var}(X)$
* If we have two *independent* random variables $X$ and $Y$, we have $\mathrm{var}(X+Y) = \mathrm{var}(X) + \mathrm{var}(Y)$.


When interpreting these values, there can be a bit of a hiccup.  In particular, Let us try imagining what happens if we keep track of units through this computation.  Suppose we are working with the star rating assigned to a product on the web page.  Then $a$, $a-2$, and $a+2$ are all measured in units of stars.  Similarly, the mean $\mu_X$ is then also measured in stars (being a weighted average).  However, if we get to the variance, we immediately encounter an issue, which is we want to look at $(X-\mu_X)^2$, which is in units of *squared stars*.  This means that the variance itself is not comparable to the original measurements.  To make it interpretable, we will need to return to our original units.

This can be easily fixed: take the square root!  Thus we define
$$
\sigma_X = \mathrm{sd}(X) = \sqrt{\mathrm{var}(X)}.
$$
In our example, this means we now have the standard deviation is $\sigma_X = 2\sqrt{2p}$.  If we are dealing with units of stars for our review example, $\sigma_X$ is again in units of stars.

The properties we had for variances can be restated for standard deviations.

* For any random variable $X$, $\mathrm{sd}(X) \ge 0$.
* For any random variable $X$ and numbers $a$ and $b$, we have that $\mathrm{sd}(aX+b) = |a|\mathrm{sd}(X)$
* If we have two *independent* random variables $X$ and $Y$, we have $\mathrm{sd}(X+Y) = \sqrt{\mathrm{sd}(X)^2 + \mathrm{sd}(Y)^2}$.

It is very natural at this moment to ask, "If the standard deviation is in the units of our original random variable, does it represent something I can draw with regards to that random variable?"  The answer is a resounding yes!  Indeed much like the mean told we the typical location of our random variable, the standard deviation gives the typical range of variation of that random variable.  We can make this rigorous with what is known as Chebychev's inequality:

$$
P\left(X \not\in [\mu_X - a\sigma_X, \mu_X + a\sigma_X]\right) \le \frac{1}{a^2}.
$$

Or to state it verbally in the case of $a=10$: $99\%$ of the samples from any random variable fall within $10$ standard deviations of the mean.  This gives an immediate interpretation to our standard summary statistics.

??? Add application to the running example

??? FIGURE ???

This has all been in terms of discrete random variables, but the case of continuous random variables is similar.  To intuitively understand how this works, imagine that we split the real number line into intervals of length $\epsilon$ given by $(\epsilon i, \epsilon (i+1)]$.  Once we do this, our continuous random variable has been made discrete and we can say that
$$
\begin{aligned}
\mu_X & \approx \sum_{i} (\epsilon i)P(X \in (\epsilon i, \epsilon (i+1)]) \\
& \approx \sum_{i} (\epsilon i)p_X(\epsilon i)\epsilon, \\
\end{aligned}
$$
where $p_X$ is the density of $X$.  This is an approximation to the integral of $xp_X(x)$, so we can conclude that
$$
\mu_X = \int_{-\infty}^\infty xp_X(x) \; dx.
$$

Similarly, the variance can be written as
$$
\sigma^2_X = E[X^2] - \mu_X^2 = \int_{-\infty}^\infty x^2p_X(x) \; dx - \left(\int_{-\infty}^\infty xp_X(x) \; dx\right)^2.
$$

Everything stated above about the mean, variance, and standard deviation above still apply in this case.  For instance, if we consider the random variable with density 
$$
p(x) = \begin{cases}
1 & x \in [0,1] \\
0 & \text{otherwise}.
\end{cases}
$$ we can compute
$$
\mu_X = \int_{-\infty}^\infty xp(x) \; dx = \int_0^1 x \; dx = \frac{1}{2}.
$$
and
$$
\sigma_X^2 = \int_{-\infty}^\infty x^2p(x) \; dx - \left(\frac{1}{2}\right)^2 = \frac{1}{3} - \frac{1}{4} = \frac{1}{12}.
$$

As a warning, Let us examine one more example, known as the Cauchy distribution.  This is the distribution with p.d.f. given by
$$
p(x) = \frac{1}{1+x^2}.
$$

??? PICTURE ???

This function looks quite innocent, and indeed consulting a table of integrals will show it has area one under it, and thus it defines a continuous random variable.

To see what goes astray, Let us try to compute the variance of this.  This would involve computing
$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx.
$$
The function on the inside looks like this

??? Draw the function that is essentially the constant 1 with a dip in the middle ???

This function clearly has infinite area under it, and indeed one can see that
$$
\int_{-\infty}^\infty \frac{x^2}{1+x^2}\; dx = \infty.
$$
This means it does not have a well-defined finite variance.

However, looking deeper shows an even more disturbing result.  Let us try to compute the mean.  Using the change of variables formula, we see
$$
\mu_X = \int_{-\infty}^{\infty} \frac{x}{1+x^2} \; dx = \frac{1}{2}\int_1^\infty \frac{1}{u} \; du. 
$$
The integral inside is the definition of the logarithm, so this is in essence $\log(\infty) = \infty$, so there is no well defined average value either!  

Machine learning scientists define their models so that we do not need to deal with these issues, and will in the vast majority of cases deal with random variables with well defined means and variances.  However, every so often random variables with "heavy tails" (that is those random variables where the probabilities of getting large values are large enough to make things like the mean or variance undefined) are helpful in modeling physical systems, thus it is worth knowing that they exist.

### Joint Density Functions

The above work all assumes we are working with a single real valued random variable.  But what if we are dealing with two or more potentially highly correlated random variables?  This circumstance is the norm in machine learning: imagine random variables like $R_{i,j}$ which encode the red value of the pixel at the $(i,j)$ coordinate in an image, or $P_t$ which is a random variable given by a stock price at time $t$.  Nearby pixels tend to have similar color, and nearby times tend to have similar prices.  We cannot treat them as separate random variables, and expect to create a successful model (recall ???REF??? where we saw naive Bayes failing for just such an assumption).  We need to develop the mathematical language to handle these correlated continuous random variables.

Thankfully, with the multiple integrals in ???REF??? we can develop such a language.  Suppose we have, for simplicity, two random variables $X,Y$ which can be correlated.  Then, similar to the case of a single variable, we can ask the question,
$$
P(X \text{ is in an } \epsilon \text{-sized interval around } x \text{ and }Y \text{ is in an } \epsilon \text{-sized interval around } y ).
$$
Similar reasoning to the single variable case shows that this should be approximately
$$
P(X \text{ is in an } \epsilon \text{-sized interval around } x \text{ and }Y \text{ is in an } \epsilon \text{-sized interval around } y ) \approx \epsilon^{2}p(x,y),
$$
for some function $p(x,y)$.  This is referred to as the joint density of $X$ and $Y$.  Similar properties are true for this as we saw in the single variable case. Namely:

* $p(x,y) \ge 0$
* $\int _ {\mathbb{R}^2} p(x,y) \;dx \;dy = 1$
* $P((X,Y) \in \mathcal{D}) = \int _ {\mathcal{D}} p(x,y) \;dx \;dy$.

In this way, we can deal with multiple, potentially correlated random variables.  If we wish to work with more than two random variables, we can extend the multivariate density to as many coordinates as desired by considering $p(\mathbf{x}) = p(x_1, \ldots, x_n)$.  The same properties of being non-negative, and having total integral of one still hold.


### Marginal Distributions
When dealing with multiple variables, we often times want to be able to ignore the relationships and ask, "how is this one variable distributed?"  Such a distribution is called a *marginal distribution*.  

To be concrete, let us suppose we have two random variables $X,Y$ with joint density given by $p _ {X,Y}(x,y)$.  I will be using the subscript to indicate what random variables the density is for.  The question of finding the marginal distribution is taking this function, and using it to find $p _ X(x)$.

As with most things, it is best to return to the intuitive picture to figure out what should be true.  Recall that the density is the function $p _ X$ so that
$$
P(X \in [x,x+\epsilon]) \approx \epsilon \cdot p _ X(x).
$$
This currently has no mention of $Y$, but if all we are given is $p _{X,Y}$, we need to include $Y$ somehow. We can first observe that this is the same as
$$
P(X \in [x,x+\epsilon] \text{, and } Y \in \mathbb{R}) \approx \epsilon \cdot p _ X(x).
$$
Our density does not directly tell us about what happens in this case, we need to split into small intervals in $y$ as well, so we can write this as
$$
\begin{aligned}
\epsilon \cdot p _ X(x) & \approx \sum _ {i} P(X \in [x, x+\epsilon] \text{, and } Y \in [\epsilon \cdot i, \epsilon \cdot (i+1)]) \\
& \approx \sum _ {i} \epsilon^{2} p _ {X,Y}(x, \epsilon\cdot i).
\end{aligned}
$$

??? FIGURE HERE ???

This simply tells us to add up the value of the density along a series of squares in a line.  Indeed, after canceling one factor of epsilon from both sides, and recognizing the sum on the right is the integral over $y$, we can conclude that
$$
\begin{aligned}
 p _ X(x) &  \approx \sum _ {i} \epsilon p _ {X,Y}(x, \epsilon\cdot i) \\
 & \approx \int_{-\infty}^\infty p_{X,Y}(x,y) \; dy.
\end{aligned}
$$
Thus we see
$$
p _ X(x) = \int_{-\infty}^\infty p_{X,Y}(x,y) \; dy.
$$

This tells us that to get a marginal distribution, we integrate over the variables we do not care about.  This process is often referred to as *integrating out* or *marginalized out* the unneeded variables.

### Covariance and correlation

When dealing with multiple random variables, there is one additional summary statistic which is extremely helpful to know: the *covariance*.  This measures the degree that two random variable fluctuate together.

Suppose we have two random variables $X$ and $Y$, to begin with, Let us suppose they are discrete, taking on values $(x_i, y_j)$ with probability $p_{ij}$.  In this case, the covariance is defined as
$$
\sigma_{XY} = \mathrm{cov}(X,Y) = \sum_{i,j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij}.
$$

To think about this intuitively: consider the following pair of random variables.  Suppose $X$ takes the values $1$ and $3$, and $Y$ takes the values $-1$ and $3$.  Suppose we have the following probabilities
$$
\begin{aligned}
P(X = 1, \text{ and} Y = -1) & = \frac{p}{2} \\
P(X = 1, \text{ and} Y = 3) & = \frac{1-p}{2} \\
P(X = 3, \text{ and} Y = -1) & = \frac{1-p}{2} \\
P(X = 3, \text{ and} Y = 3) & = \frac{p}{2} 
\end{aligned}
$$
where $p$ is a parameter in $[0,1]$ we get to pick.  Notice that if $p=1$ then they are both always their minimum or maximum values simultaneously, and if $p=0$ they are guaranteed to take their flipped values simultaneously (one is large when the other is small and vice versa).  If $p=1/2$, then the four possibilities are all equally likely, and neither should be related.  Let us compute the covariance.  First, note $\mu_X = 2$ and $\mu_Y = 1$, so we may compute:
$$
\begin{aligned}
\mathrm{cov}(X,Y) & = \sum_{i,j} (x_i - \mu_X) (y_j-\mu_Y) p_{ij} \\
& = (1-2)(-1-1)\frac{p}{2} + (1-2)(3-1)\frac{1-p}{2} + (3-2)(-1-1)\frac{1-p}{2} + (3-2)(3-1)\frac{p}{2} \\
& = 4p-2
\end{aligned}
$$

When $p=1$ (the case where the are both maximally positive or negative at the same time) has a covariance of $2$. When $p=0$ (the case where they are flipped) the covariance is $-2$.  Finally, when $p=1/2$ (the case where they are unrelated), the covariance is $0$.  Thus we see that the covariance measures how these two random variables are related.

A quick note on the covariance is that it only measures these simple linear relationships.  More complex relationships like $X = Y^2$ where $Y$ is $-2,-1,0,1,2$ with equal probability can be missed.  Indeed a quick computation shows that these random variables have covariance zero, despite one being a deterministic function of the other.

For continuous random variables, much the same story holds.  At this point, we are pretty comfortable with doing the transition between discrete and continuous, so we will provide the definition without any derivation.  

$$
\sigma_{XY} = \mathrm{cov}(X,Y) = \int_{\mathbb{R}^2} xyp(x,y) \;dx \;dy.
$$

Let us see some properties of covariances:

* For any random variable $X$, $\mathrm{cov}(X,X) = \mathrm{var}(X)$.
* For any random variables $X,Y$ and numbers $a$ and $b$, $\mathrm{cov}(aX+b,Y) = \mathrm{cov}(X,aY+b) = a\mathrm{cov}(X,Y)$.
* If $X$ and $Y$ are independent then $\mathrm{cov}(X,Y) = 0$

In addition, we can use the covariance to expand a relationship we saw before.  Recall that is $X$ and $Y$ are two independent random variables then
$$
\mathrm{var}(X+Y) = \mathrm{var}(X) + \mathrm{var}(Y).
$$

With knowledge of covariances, we can expand this relationship.  Indeed, some algebra can show that in general, 
$$
\mathrm{var}(X+Y) = \mathrm{var}(X) + \mathrm{var}(Y) + 2\mathrm{cov}(X,Y).
$$
This allows us to generalize the variance summation rule for correlated random variables.

As we did in the case of means and variances, Let us now consider units.  If $X$ is measured in one unit (say inches), and $Y$ is measured in another (say dollars), the the covariance is measured in the product of these two units $\text{inches}\cdot\text{dollars}$.  These units can be hard to interpret.  What we will often want in this case is a unit-less measurement of relatedness.  Indeed, often we do not care about exact quantitative correlation, but rather ask if the correlation is in the same direction, and how strong the relationship is.  

To see what makes sense, Let us perform a thought experiment.  Suppose we convert our random variables in inches and dollars to be in inches and cents.  In this case the random variable $Y$ is simply multiplied by $100$.  If we work through the definition, this means that $\mathrm{cov}(X,Y)$ will be multiplied by $100$.  Thus we see that in this case a simple change of units change the covariance by a factor of $100$.  Thus, to find our unit-invariant measure of correlation, we will need to divide by something else that also gets scaled by $100$.  Indeed we have a clear candidate, the standard deviation!  Indeed if we define the *correlation coefficient* to be
$$
\mathrm{cor}(X,Y) = \frac{\mathrm{cov}(X,Y)}{\mathrm{sd}(X)\mathrm{sd{Y}}}
$$
we see that this is a unit-less value.  A little mathematics can show that this number is between $-1$ and $1$ with $1$ meaning maximally positively correlated, whereas $-1$ means maximally negatively correlated.

Returning to our explicit discrete example above, we can see that $\sigma_X = 1$ and $\sigma_Y = 2$, so the correlation between the two random variables is
$$
\mathrm{cor}(X,Y) = \frac{4p-2}{1\cdot 2} = 2p-1.
$$
This now ranges between $-1$ and $1$ with the expected behavior of $1$ meaning most correlated, and $-1$ meaning minimally correlated.

As a final example, consider $X$ as any random variable, and $Y=aX+b$ as any linear deterministic function of $X$.  Then, one can compute that
$$
\begin{aligned}
\mathrm{sd}(Y) & = \mathrm{sd}(aX+b) = |a|\mathrm{sd(X)} \\
\mathrm{cov}(X,Y) &= \mathrm{cov}(X,aX+b) = a\mathrm{cov}(X,X) = a\mathrm{var}(X),
\end{aligned}
$$
and thus that
$$
\mathrm{cor}(X,Y) = \frac{a\mathrm{var}(X)}{|a|\mathrm{sd}(X)^2} = \frac{a}{|a|} = \mathrm{sign}(a).
$$
Thus we see that the correlation is $+1$ for any $a > 0$, and $-1$ for any $a < 0$ illustrating that correlation measures the degree and directionality the two random variables are related, not the scale that the variation takes.

Let us list a few properties of correlation:

* For any random variable $X$, $\mathrm{cor}(X,X) = 1$.
* For any random variables $X,Y$ and numbers $a$ and $b$, $\mathrm{cor}(aX+b,Y) = \mathrm{cor}(X,aY+b) = \mathrm{cor}(X,Y)$.
* If $X$ and $Y$ are independent with non-zero variance then $\mathrm{cor}(X,Y) = 0$.

As a final note, the eagle-eyed amongst we may recognize these formulae.  Indeed, if we expand everything out assuming that $\mu_X = \mu_Y = 0$, we see that this is
$$
\mathrm{cor}(X,Y) = \frac{\sum_{i,j} x_iy_ip_{ij}}{\sqrt{\sum_{i,j}x_i^2 p_{ij}}\sqrt{\sum_{i,j}y_j^2 p_{ij}}}.
$$
This looks like a sum of a product of terms divided by the square root of sums of terms.  This is exactly the formula for the cosine of the angle between two vectors $\mathbf{v},\mathbf{w}$ with the different coordinates weighted by $p_{ij}$:
$$
\cos(\theta) = \frac{\mathbf{v}\cdot \mathbf{w}}{\|\mathbf{v}\|\|\mathbf{w}\|} = \frac{\sum_{i} v_iw_i}{\sqrt{\sum_{i}v_i^2}\sqrt{\sum_{i}w_i^2}}.
$$

Indeed if we think of norms as being related to standard deviations, and correlations as being cosines of angles, much of the intuition we have from geometry can be applied to thinking about random variables.

## Maximum Likelihood

One of the most commonly encounter way of thinking in machine learning is the maximum likelihood point of view.  This is the concept that when working with a probabilistic model with unknown parameters, the parameters which make the data have the highest probability are the most likely ones.

This has a Bayesian which can be helpful to think about.  Suppose we have a model with parameters $\boldsymbol{\theta}$ and a collection of data points $X$.  For concreteness, we can imagine that $\boldsymbol{\theta}$ is a single value representing the probability that a coin comes up heads when flipped, and $X$ is a sequence of independent coin flips.  We will look at this example in depth later.

If we want to find the most likely value for the parameters of our model, that means we want to find
$$
\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).
$$
By Bayes' rule, this is the same thing as
$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$
The expression $P(X)$, a parameter agnostic probability of generating the data, does not depend on $\boldsymbol{\theta}$ at all, and so can be dropped without changing the best choice of $\boldsymbol{\theta}$.  Similarly, we may now posit that we have no prior assumption on which set of parameters are better than any others, so we may declare that $P(\boldsymbol{\theta})$ does not depend on theta either!  This, for instance, makes sense in our coin flipping example where the probability it comes up heads could be any value in $[0,1]$ without any prior belief it is fair or not (often referred to as an *uninformative prior*).  Thus we see that our application of Bayes' rule shows that our best choice of $\boldsymbol{\theta}$ is the maximum likelihood estimate for $\boldsymbol{\theta}$:
$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

As a matter of common terminology, the probability of the data given the parameters ($P(X \mid \boldsymbol{\theta})$) is referred to as the *likelihood*.

Let us see how this works in the concrete example from before.  Suppose we have a single parameter $\theta$ representing the probability that a coin flip is heads.  Then the probability of getting a tails is $1-\theta$, and so if our observed data $X$ is a sequence with $n_H$ heads and $n_T$ tails, we can use the fact that independent probabilities multiply to see that 
$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

If we flip $13$ coins and get the sequence "HHHTHTTHHHHHT", which has $n_H = 9$ and $n_T = 4$, we see that this is
$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

One nice thing about this example will be that we know the answer going in.  Indeed, if I told we verbally, "I flipped 13 coins, and 9 came up heads, what is our best guess for the probability that the coin comes us heads?," I believe that everyone would correctly guess $9/13$.  What this maximum likelihood method will give us is a way to get that number from first principals, and in a way that will generalize to vastly more complex situations.

In any case, for our example, the plot of $P(X \mid \theta)$ is as follows

???PLOT THE LIKELIHOOD???

This has its maximum value somewhere near our expected $9/13 \approx 0.69\ldots$.  To see if it is exactly there, we can turn to calculus.  Notice that at the maximum, the function is flat.  Indeed if the slope was not zero there, then we could shift the input to make it larger (think about the process of gradient descent).  Thus, we could find the maximum likelihood estimate by finding the values of $\theta$ where the derivative is zero, and finding the one that gives the highest probability.  We compute:
$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

This has three solutions: $0$, $1$ and $9/13$.  The first two are clearly minima, not maxima as they assign probability $0$ to our sequence.  The final one does *not* assign zero probability to our sequence, and thus must be the maximum likelihood estimate $\hat \theta = 9/13$, matching our intuition.

### Numerical Optimization and the $-\log$-Likelihood
This story is nice, but what is we have billions of parameters and data points.  What do we do then?

First notice that, if we make the assumption that all the data points are independents, we can no longer consider the likelihood itself as it is a product of many probabilities.  Indeed, each probability is in $[0,1]$, say typically of size about $1/2$, and the product of $(1/2)^{1000000000}$ is far below machine precision.  We cannot work with that directly.  

However, recall that the logarithm turns products to sums, in which case 
$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$
This number fits perfectly within even a single precision 32-bit float.  Thus, we are lead to consider the $\log$-likelihood, which is
$$
\log(P(X \mid \boldsymbol{\theta})).
$$
Since the function $x \mapsto \log(x)$ is increasing, maximizing the likelihood is the same thing as maximizing the $\log$-likelihood.  Indeed in ???REF??? we saw this reasoning applied when working with the specific example of the Naive Bayes classifier.

As mentioned in ???REF???, we often work with loss functions, where we wish to minimize the loss.  We may turn this into a minimization problem by taking $-\log(P(X \mid \boldsymbol{\theta}))$, which is the $-\log$-Likelihood.

To illustrate this, consider the coin flipping problem from before, and pretend that we do not know the closed form solution.  The we may compute that
$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta))
$$
This can be easily written into code, even for billions of coin flips, and freely optimized with gradient descent.

```{.python .input}
from mxnet import autograd, np, npx
npx.set_np()

### Set up our data ###
n_H = 8675309
n_T = 25624

### Initialize our paramteres ###
theta = np.array(0.5); theta.attach_grad()

### Perform gradient descent ###
lr = 0.00000000001
for iter in range(10) :
    with autograd.record():
        loss = -(n_H*np.log(theta) + n_T*np.log(1-theta))
    loss.backward()
    theta -= lr*theta.grad

### Check Output ###
print(theta)
print(n_H/(n_H+n_T))
```

Numerical convenience is only one reason people like to use $-\log$-likelihoods.  Indeed, there are a number of reasons that it can be preferable.

* **Simplification of Calculus rules.** As discussed above, due to independence assumptions, most probabilities we encounter in machine learning are products of individual probabilities.
$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$
This means that if we directly apply the product rule to compute a derivative we get
$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right)
\end{aligned}
$$
This requires $n(n-1)$ multiplications, along with $(n-1)$ additions, so it is total of quadratic time in the inputs!  Sufficient cleverness in grouping terms will reduce this to linear time, but it requires some thought.  For the $-\log$-likelihood we have instead
$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta}))
$$
which then gives
$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$
This requires only $n$ divides and $n-1$ sums, and thus is linear time in the inputs.

* **Relationships with information theory**.  In ???REF??? we will discuss a concept known as information theory.  This is a rigorous mathematical theory which gives a way to measure the degree of randomness in a random variable.  The key object of study in that field is the entropy which is 
$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$
which measures the randomness of a source in bits. Notice that this is nothing more than the average $-\log$ probability, and thus if we take our $-\log$-likelihood and divide by the number of data points, we get a relative of entropy (known as cross-entropy) that measures how random our model thinks our data is.  This theoretical interpretation alone would be sufficiently compelling to motivate reporting the average $-\log$-likelihood over the dataset as a way of measuring model performance.

### Maximum likelihood for continuous variables

Everything that we have done so far assumes we are working with discrete random variables, but what if we want to work with continuous ones?  Indeed, in applications, continuous random variables are as likely to be encountered as discrete, and so we need to extend this story line to this case.

The short summary is that nothing at all changes, except we replace all the instances of the probability with the probability density.  Recalling that we write densities with lower case $p$, this means that for example we now say
$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

The question becomes, "Why is this OK?"  After all, the reason we introduced densities was because probabilities of getting specific outcomes themselves was zero, and thus is not the probability of generating our data for any set of parameters zero?

Indeed, this is the case, and understanding why we can shift to densities is an exercise in tracing what happens to the epsilons.

Let us first re-define our goal.  Let us suppose that for continuous random variables we no longer want to compute the probability of getting exactly the right value, but instead matching to say the first four digits.  For simplicity, Let us assume our data is repeated observations $x_1, \ldots, x_N$ of identically distributed random variables $X_1, \ldots, X_N$.  As we have seen previously ???REF???, this can be written as
$$
P(X_1 \in [x_1,x_1+\epsilon], X_2 \in [x_2,x_2+\epsilon], \ldots, X_N \in [x_N,x_N+\epsilon]\mid\boldsymbol{\theta}) \approx \epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}),
$$
with $\epsilon = 10^{-4}$. Thus, if we take negative logarithms of this we obtain
$$
-\log(P(X_1 \in [x_1,x_1+\epsilon], X_2 \in [x_2,x_2+\epsilon], \ldots, X_N \in [x_N,x_N+\epsilon]\mid\boldsymbol{\theta})) \approx -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

If we examine this expression, the only place that the $\epsilon$ occurs is in the additive constant $-N\log(\epsilon)$.  This does not depend on the parameters $\boldsymbol{\theta}$ at all, so the optimal choice of $\boldsymbol{\theta}$ does not depend on our choice of $\epsilon$!  If we demand four digits or four-hundred, the best choice of $\boldsymbol{\theta}$ remains the same, thus we may freely drop the epsilon to see that what we want to optimize is
$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})),
$$
as was claimed at the beginning of this section.  Thus we see that the maximum likelihood point of view can operate with continuous random variables as easily as with discrete ones by simply replacing the probabilities with probability densities.

## Common Distributions

Now that we have learned about how to work with probability theory in both discrete and continuous setting, lets get to know some of the common random distributions encountered.  Depending on the area of machine learning we are working in, we may potentially need to be familiar with vastly more of these, or for some areas of deep learning potentially none at all.  This is, however, a good basic list to be familiar with.

### Bernoulli

This is the simplest random variable usually encountered.  This is the random variable that encodes a coin flip which comes up $1$ with probability $p$ and $0$ with probability $1-p$.  If we have a random variable with this distribution, we will write
$$
X \sim \mathrm{Bernoulli}(p).
$$

The cumulative distribution function is 
$$
F(x) = \begin{cases}
0 & x < 0, \\
1-p & 0 \le x < 1, \\
1 & x >= 1 .
\end{cases}
$$

Let us plot the probability mass function and cumulative distribution function.

```{.python .input}
p = 0.3

d2l.plt.stem([0,1],[1-p,p])
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
x = np.arange(-1,2,0.01)
F = lambda x: 0 if x < 0 else 1 if x > 1 else 1-p

d2l.plot(x,np.array([F(y) for y in x]),'x','c.d.f.')
```

If $X \sim \mathrm{Bernoulli}(p)$, then:

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

We can sample an array of arbitrary shape from a Bernoulli random variable in numpy as follows. 

```
1*(np.random.rand(10,10) < p)
```

### Discrete Uniform

The next simplest random variable encountered is a discrete uniform distribution.  For our discussion here, we will assume that it is on the integers $\{1,2,\ldots, n\}$, however any other set of values can be freely chosen.  The meaning of the word *uniform* in this context is that every possible value is equally likely.  The the probability for each value $i \in \{1,2,3,\ldots,n\}$ is $p_i = \frac{1}{n}$.  We will denote this relationship as
$$
X \sim \mathrm{Uniform}(n).
$$

The cumulative distribution function is 
$$
F(x) = \begin{cases}
0 & x < 1, \\
\frac{k}{n} & k \le x < k+1 \text{ with } 1 \le k < n, \\
1 & x >= n .
\end{cases}
$$

Let us plot the probability mass function and cumulative distribution function.

```{.python .input}
n = 5

d2l.plt.stem([i+1 for i in range(n)],n*[1/n])
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
x = np.arange(-1,6,0.01)
F = lambda x: 0 if x < 1 else 1 if x > n else np.floor(x)/n

d2l.plot(x,np.array([F(y) for y in x]),'x','c.d.f.')
```

If $X \sim \mathrm{Uniform}(n)$, then:

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

We can an array of arbitrary shape from a discrete uniform random variable in numpy as follows.  Note that the range 

```
np.random.random_integers(1, n, size=(10,10))
```

### Continuous Uniform

The next simplest random variable encountered is the continuous uniform distribution. The idea behind this random variable is that if we increase the $n$ in the previous distribution, and then scale it to fit within the interval $[a,b]$, we will approach a continuous random variable that just picks an arbitrary value in $[a,b]$ all with equal probability.  We will denote this distribution as
$$
X \sim \mathrm{Uniform}([a,b]).
$$

The probability density function is 
$$
p(x) = \begin{cases}
\frac{1}{b-a} & x \in [a,b], \\
0 & x \not\in [a,b].
\end{cases}
$$

The cumulative distribution function is 
$$
F(x) = \begin{cases}
0 & x < a, \\
\frac{x-a}{b-a} & x \in [a,b], \\
1 & x >= b .
\end{cases}
$$

Let us plot the probability density function and cumulative distribution function.
```{.python .input}
a = 1; b = 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
F = lambda x: 0 if x < a else 1 if x > b else (x-a)/(b-a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

If $X \sim \mathrm{Uniform}([a,b])$, then:

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

We can an array of arbitrary shape from a uniform random variable in numpy as follows.  Note that it by default samples from a $\mathrm{Uniform}([a,b])$, so if we want a different range we need to scale it.

```
(b - a) * np.random.rand(10, 10) + a
```

### Binomial

The first non-trivial random variable we will discuss is the binomial random variable.  This random variable originates from performing a sequence of $n$ independent experiments, each of which have probability $p$ of succeeding, and asking how many successes we expect to see.

Let us express this mathematically.  Each experiment is an independent random variable $X_i$ where we will use $1$ to encode success, and $0$ to encode failure.  Since each is an independent coin flip which is successful with probability $p$, we can say that $X_i \sim \mathrm{Bernoulli}(p)$.  Then, the binomial random variable is
$$
X = \sum_{i=1}^n X_i.
$$
In this case, we will write
$$
X \sim \mathrm{Binomial}(n,p).
$$

To get the cumulative distribution function, we need to notice that getting exactly $k$ successes can occur in $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ ways each of which has a probability of $p^m(1-p)^{n-m}$ of occuring.  Thus the cumulative distribution function is
$$
F(x) = \begin{cases}
0 & x < 0, \\
\sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ with } 0 \le k < n, \\
1 & x >= n .
\end{cases}
$$

Let us plot the probability density function and cumulative distribution function.

```{.python .input}
n = 10
p = 0.2

# Compute Binomial Coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i*(1-p)**(n-i)*binom(n,i) for i in range(n+1)])

d2l.plt.stem([i for i in range(n+1)],pmf)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)
F = lambda x: 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

While this result is not simple, the means and variances are.  If $X \sim \mathrm{Binomial}(n,p)$, then:

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

We may easily sample from this distribution in numpy.

```
np.random.binomial(n, p, size = (10,10))
```

### Poisson
Let us now perform a thought experiment.  Let us say we are standing at a bus stop and we want to know how many buses will arrive in the next minute.  Lets start by considering $X^{(1)} \sim \mathrm{Bernoulli}(p)$ Which is simply the probability that a bus arrives in the the one minute window.  For bus stops far from an urban center, this might be a pretty good approximation since we will never see more than one bus at a time.

However, if we are in a busy area, it is possible or even likely that two buses will arrive.  We can model this by splitting our random variable into two parts for the first 30 seconds, or the second 30 seconds.  In this case we can write
$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2
$$
where $X^{(2)}$ is the total sum, and $X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$.  The total distribution is then $X^{(2)} \sim \mathrm{Binomial}(2,p/2)$.

Why stop here?  Let us continue to split that minute into $n$ parts.  By the same reasoning as above, we see that
$$
X^{(n)} \sim \mathrm{Binomial}(n,p/n).
$$

Let us consider these random variables.  By the previous section, we know that this has mean $\mu_{X^{(n)}} = n(p/n) = p$, and variance $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$.  If we take $n \rightarrow \infty$, we can see that these numbers actually stabilize to $\mu_{X^{(\infty)}} = p$, and variance $\sigma_{X^{(\infty)}}^2 = p$!  What this indicates is that there could be some random variable we can define which is well defined in this infinite subdivision limit.  

This should not come as too much of a surprise, since in the real world we can just count the number of bus arrivals, however it is nice to see that our mathematical model is well defined.  This result is known as the *law of rare events*.

Following through this reasoning carefully, we can arrive at the following model.  We will say that $X \sim \mathrm{Poisson}(\lambda)$ if it is a random variable which takes the values $\{0,1,2,\ldots}\}$ with probability
$$
p_k = \frac{\lambda^ke^{-\lambda}}{k!}
$$
The value $\lambda > 0$ is known as the *rate*, and denotes the average number of arrivals we expect in one unit of time (note that we above restricted our rate to be less than zero, but that was only to simplify the explanation).  

We may sum this probability mass function to get the cumulative distribution function.
$$
F(x) = \begin{cases}
0 & x < 0, \\
e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ with } 0 \le k.
\end{cases}
$$

Let us plot the probabilty mass function and cumulative distibution function.

```{.python .input}
lambda = 5.0

xs = [i for i in range(20)]
pmf = [np.exp(-lambda)*lambda^k/np.factorial(k) for k in xs]

d2l.plt.stem(xs,pmf)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
F = lambda x: 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

As we saw above, the means and variances are particularly simple.  If $X \sim \mathrm{Poisson}(\lambda)$, then:

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

This can be sampled in numpy as follows.
```{.python .input}
np.random.poisson(lambda,size=(10,10))
```

### Gaussian
Now Let us try a different, but related experiment.  Let us say we again are performing $n$ independent $\mathrm{Bernoulli}(p)$ measurements $X_i$.  The distribution of the sum of these is $X^{(n)} \sim \mathrm{Binomial}(n,p)$.  Rather than taking a limit as $n$ increases and $p$ decreases, Let us fix $p$, and then send $n \rightarrow \infty$.  In this case $\mu_{X^{(n)}} = np \rightarrow \infty$ and $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, so there is no reason to think this limit should be well defined.

However, not all hope is lost!  Let us just make the mean and variance be well behaved by defining
$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$
This can be seen to have mean zero and variance one, and so it is plausible to believe that it will converge to some limiting distribution.

??? Add picture of different Bernoulli distributions with $p=1/2$ ???

One thing to note: compared to the Poisson case, we are now diving by the standard deviation which means that we are squeezing the possible outcomes into smaller and smaller areas.  This is an indication that our limit will no longer be discrete, but rather a continuous distribution.

A derivation of what occurs is well beyond the scope of this document, but the *central limit theorem* states that as $n \rightarrow \infty$, this will yield the Gaussian Distribution (or sometimes Normal distribution).  More explicitly, for any $a,b$:
$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a,b]) = P(\mathcal{N}(0,1) \in [a,b]),
$$
where we say a random variable is normally distributed with given mean $\mu$ and variance $\sigma^2$, written $X \sim \mathcal{N}(\mu,\sigma^2)$ if $X$ has density
$$
p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.
$$

Let us plot the probability density function and cumulative distribution function.
```{.python .input}
mu = 1, sigma = 0.5

x = np.arange(-2,2, 0.01)
p = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-(x**2)/(2*sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
from math import erf
def phi(x):
    return (1.0 + math.erf((x-mu) / (sigma*sqrt(2)))) / 2.0

d2l.plot(x, phi(x), 'x', 'c.d.f.')
```

Keen-eyed readers will recognize some of these terms.  Indeed, we encountered this integral we encountered in :numref:`appendix-calculus`.  Indeed we need exactly that computation to see that this $p_X(x)$ has total area one and is thus a valid density.

Our choice of working with coin flips made it easy to identify what we were working with, but nothing about that choice was needed.  Indeed, if we take any collection of independent identically distributed random variables $X_i$, and form
$$
X^{(N)} = \sum_{i=1}^N X_i.
$$
then
$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}},
$$
will be approximately Gaussian.

This is the reason that the Gaussian is so central to probability, statistics, and machine learning.  Whenever we can say that something we measured is a sum of many small independent contributions, we can safely assume that the thing being measured will be close to Gaussian.

There are many more fascinating properties of Gaussians than we can get into at this point.  In particular, the Gaussian is what is known as a *maximum entropy distribution*.  We will get into entropy more deeply in ???REF???, however all we need to know at this point is that it is a measure of randomness.  In a rigorous mathematical sense, we can think of the Gaussian as the *most* random choice of random variable with fixed mean and variance.  Thus, if we know that our random variable has some mean and variance, the Gaussian is in a sense the most conservative choice of distribution we can make.

To close the section, Let us recall that if $X \sim \mathcal{N}(\mu,\sigma^2)$, then:

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

We can sample from the Gaussian (or normal) using numpy.
```{.python .input}
np.random.normal(mu,sigma,size=(10,10))
```

## Summary
* Using Bayes' rule, a simple classifier can be made.  This classifier was the gold standard for decades.
* Continuous random variables are random variables that can take on a continuum of values.  They have some technical difficulties that make them more challenging to work with compared to discrete random variables.
* The probability density function allows us to work with continuous rnadom variables by giving a function where the area under the curve on some interval gives the probability of finding a sample point in that interval.
* The cumulative distribution function is the probability of observing the random variable to be less than a given threshold.  It can provide a useful alternate viewpoint which unifies discrete and continuous variables.
* The mean is the average value of a random variable.
* The variance is the expected square of the difference between the random variable and its mean.
* The standard deviation is the square root of the variance.  It can be thought of as measuring the range of values the random variable may take.
* Chebychev's inequalty allows us to make this intution rigorous by giving an explicit interval that contains the random variable most of the time.
* Joint densities allow us to work with correlated random variables.  We may marginalize joint densities by integrating over unwanted random variables to get the distribution of the desired random varible.
* The covariance and correlation coefficient provide a way to measure any linear relationship between two correlated random variables.
* The maximum likelihood principle tells us that the best fit model for a given dataset is the one that generates the data with the highest probability.

## Exercises
