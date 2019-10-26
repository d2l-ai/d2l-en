# D2L Appendix - Statistics Inference
:label:`sec_statistics`


“What is the difference between machine learning and statistics?” You may often hear similar questions and eager to know the answers. Fundamentally, statistics theories and algorithms focus on the inference problems. This type of problems include modeling relationships between the variables and conducting hypothesis testing to determine the statistically significance of the variables. In contrast, machine learning algorithms emphasis on making accurate predictions, without explicitly programming.

Undoubtedly, the ability to train state-of-the-art models and make precise predictions is crucial in both academia and industry. On the other hand, the statistical understanding mindset behind the models will strengthen your ability to deal with different probings, such as “why your model works”. 

*Statistical inference* deduces the characteristics of a population from the observed data (*samples*) that are sampled from that population. In statistics, the *population* denotes the total set of observations which we can get samples from, while the *sample* set is a set of data that are collected from the given population group. Contrary to descriptive statistics, which only cares about the characteristics of the observed data, but not the larger population.

In the section, we will introduce three types of statistics inference methods: evaluating and comparing estimators , constructing confidence intervals, and conducting hypothesis tests. These methods can help us infer the characteristics of a given population, i.e., the true parameter $\theta$. From now on, for brevity, we assume that the true parameter $\theta$ of a given population is a scale value. It is also straightforward to extend $\theta$ to a vector or a tensor, and evaluate its estimators using the same techniques below. 


## Evaluating and comparing estimators

In statistics, an [estimator](https://en.wikipedia.org/wiki/Estimator) is a function of given samples for calculating the true parameter $\theta$. We estimate $\theta$ with an estimator $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ as the best guess of $\theta$ after modeling through the training samples {$x_1, x_2, \ldots, x_n$}. 

A better illustration of what we are going to discuss about is plotted in the following graph, where we are estimating the true parameter $\theta$ from the population by the sample estimator $\hat{\theta}_n$. 



```{.python .input  n=118}
%matplotlib inline

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# sample
theta_population, theta_sample = 0, 1
sigma_population, sigma_sample = 1, 1.2
x_population = np.linspace(theta_population - 3*sigma_population, 
                           theta_population + 3*sigma_population, 1000)
y_population = stats.norm.pdf(x_population, theta_population, sigma_population)
x_sample = np.linspace(theta_sample - 3*sigma_sample, 
                       theta_sample + 3*sigma_sample, 10)
y_sample = stats.norm.pdf(x_sample, theta_sample, sigma_sample)


# plot
fig, ax = plt.subplots()
ax.set(frame_on=False)
ax.plot(x_population, y_population, label='population', color='#66BFFF')
ax.plot(x_sample, y_sample, label='sample', color='#B2D9FF')

# annotation
ax.axvline(x=theta_population, linewidth=1, linestyle='--', color='#66BFFF')
ax.text(x=theta_population-2, y=0.41, s='Population\nParameter\n'+r'$\theta$', 
        multialignment='center', fontsize=12, color='#66BFFF')
ax.axvline(x=theta_sample, linewidth=1, linestyle='--', color='#B2D9FF')
ax.text(x=theta_sample, y=0.4, s='Sample\nEstimator\n'+r'$\hat{\theta_n}$',
        multialignment='center', fontsize=12, color='#B2D9FF')

plt.show()
```

```{.json .output n=118}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

There are a lot functions to model an estimator $\hat{\theta_n}$, such as mean or median. You may be curious about which estimator is the best one? Here, we will introduce three of the most common methods to evaluate and compare estimators: statistical bias, standard deviation, and mean square error. 

### Definition

#### Statistical Bias

First, the [*statistical bias*](https://en.wikipedia.org/wiki/Bias_of_an_estimator) of an estimator is the difference between this estimator’s expected value and the true parameter value, i.e.,

$$bias (\hat{\theta}_n) = E(\hat{\theta}_n) - \theta.$$

Note that when $bias(\hat{\theta}_n) = 0$, that is when the expectation of an estimator $\hat{ \theta}_n$ is equal to the value of the true estimator, then we say $\hat{\theta}_n$ is an unbiased estimator. Other things being equal, an unbiased estimator is more desirable than a biased estimator. However, biased estimators are frequently used in practice, especially when unbiased estimators does not exist without further assumptions.


#### Standard Deviation


?????? [refer to Brent section link]

Next, another widely used evaluating method, the *standard deviation* (or *standard error*) is defined as the squared root of the varianve of the estimator, i.e.,

$$se(\hat{\theta}_n) = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$

The standard deviation measures the average distance of each value from the mean of the distribution. As a result, the larger the standard deviation, the larger the variability that each data point is from their mean.


#### Mean Squared Error

Last but not the least, the *mean squared error (MSE)* (or *l2 loss*) of the estimator is defined as the expectation of the square difference between $\hat{\theta}_n$ and the true parameter $\theta$, i.e.,

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$


MSE is always non-negative. It is the most commonly used regression loss function. As a measure to evaluate an estimator, the more its value closer to zero, the better the estimator is.

### Applications

Interestingly, the mean squared error can be explain as the sum of squared bias and variance as

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[\hat{\theta}_n - E(\hat{\theta}_n) + E(\hat{\theta}_n)) - \theta)^2] \\
 &= E[(\hat{\theta}_n - E(\hat{\theta}_n))^2] + E[(E(\hat{\theta}_n)) - \theta)^2] \\
 &= \mathrm{Var} (\hat{\theta}_n) + [{bias} (\hat{\theta}_n)]^2.\\
\end{aligned}
$$

We refer the above formula as *bias-variance trade-off*. To be specific, the mean squared error can be divided into two error sourses: the high bias error and the variance. The bias error is commonly seen in the too simple model, which cannot extract relevant relations between the features and the outputs. If a model suffers from high bias error, we often say it is *underfitting* and lack of *generalization* as introduced in (:numref:sec_model_selection). On the flip side, the high variance usually results from the too complex model, which overfits the training data and is sensitive to small fluctuations in the data. If a model suffers from  high variance, we often say it is *overfitting* and lack of *flexibility* as introduced in (:numref:sec_model_selection).


### Example

Let us implement the statistical bias, the standard deviation, and the mean squared error in MXNet. First of all, let us import the math packages that we need in this section.

```{.python .input  n=39}
from mxnet import np, npx
npx.set_np()

## statistical bias
def stat_bias(ture_theta, est_theta):
    return(np.mean(est_theta) - ture_theta)

## mean squared error
def mse(data, est_theta):
    return(np.mean(np.square(data - est_theta)))

```

To illustrate the equation of bias-variance trade-off, let us simulate of normal distribution $\mathcal{N}(\theta, \sigma^2)$ with $10,000$ samples. Here, we use a $\theta = 1$ and $\sigma$ = 4. As the estimator is a function of the given samples, here we use the mean of the samples as an estimator for true $\theta$ in this normal distribution $\mathcal{N}(\theta, \sigma^2)$ .

```{.python .input  n=45}
theta_true = 1
sigma = 4
sample_length = 1000000
samples = np.random.normal(theta_true, 4, sample_length)
theta_est = np.mean(samples)
theta_est
```

```{.json .output n=45}
[
 {
  "data": {
   "text/plain": "array(1.003481)"
  },
  "execution_count": 45,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Let us validate the trade-off equation by calculating the summation of the squared bias and the variance of our estimator. First, calculate the MSE of our estimator.

```{.python .input  n=46}
mse(samples, theta_est)
```

```{.json .output n=46}
[
 {
  "data": {
   "text/plain": "array(15.984866)"
  },
  "execution_count": 46,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

Next, we calculate $\mathrm{Var} (\hat{\theta}_n) + [{bias} (\hat{\theta}_n)]^2$ as below. As you can see, it is pretty closed to the value of the above MSE.

```{.python .input  n=43}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

```{.json .output n=43}
[
 {
  "data": {
   "text/plain": "array(15.987916)"
  },
  "execution_count": 43,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Constructing confidence intervals

To estimate the value of the true parameter $\theta$, we cannot limit ourselves to some point estimators. Besides, we can find an interval so that the true parameter $\theta$ may be located in. Certainly, we are more confident that the wider intervals are more likely to contain the true parameter $\theta$ than the shorter intervals. At the other extreme, a super high confidence level will lead to a too wide confidence interval that is meaningless in real world applications. Hence, it is critical to determine the confidence level of each inteval when locating $\theta$.


### Definition

A *confidence interval* is an estimated range of a true population parameter that we can construct by given the samples. Mathematically, a *confidence interval* for the true parameter $\theta$ is an interval $C_n$ that computed from the sample data such that 

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta. $$

Here $\alpha \in (0,1)$, and $1 - \alpha$ is called the *coverage* of the interval. Note that the above probability statement is about $C_n$, not $\theta$, which is fixed. To emphasize this, we write $P_{\theta} (C_n \ni \theta)$ rather than $P_{\theta} (C_n \in \theta)$.
 

Suppose that $\hat{\theta}_n \sim N(\theta, \hat{\sigma}_n^2)$, where $\hat{\sigma}_n^2$ is the standard deviation of $n$ samples. Then we can form an approximate $1-\alpha$ confidence interval for $\theta$ of
 
 $$C_n = [\hat{\theta}_n - z_{\alpha/2} \hat{\sigma}_n, \ \hat{\theta}_n + z_{\alpha/2} \hat{\sigma}_n]$$
 
where $z_{\alpha/2}$ is chosen from the [t-distribution table](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values) such that $P(Z > z_{\alpha/2}) = \alpha/2$ for $Z \sim N(0,1)$. 
 
 
### Example

For example, if $\alpha = 0.05$, then we will have a 95% confidence interval. If we refer to the [t-distribution table](https://en.wikipedia.org/wiki/Student%27s_t-distribution#Table_of_selected_values), the last row will give us the $z_{\alpha/2} = 1.96$ for a two-sided test. 

Let us get a 95% two-sided confidence interval for the estimator $\theta_n$ of previous samples.

```{.python .input  n=49}
sample_std = samples.std()
(theta_est - 1.96 * sample_std, theta_est + 1.96 * sample_std)
```

```{.json .output n=49}
[
 {
  "data": {
   "text/plain": "(array(-6.832811), array(8.839773))"
  },
  "execution_count": 49,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

## Conducting hypothesis tests

A *hypothesis test* is a way of evaluating evidence against some default statements about a population. We refer the default statement as the *null hypothesis* $H_0$, which we try to reject using our sample data. Here, we use $H_0$ as a starting point for the statistical significance testing. A great hypotheses is often stated in a declarative form which posits a relationship between variables. It should reflect the brief as explicit as possible, and be testable by statistics theory. 
 
There are two commomn hypothesis tests: the z-tests and the t-tests. The *z-test* is helpful if we are comparing the characteristic between a sample and the given population, based on the characteristic's standard deviation of the given population. On the other hand, the *t-test* is applicable when need to determine the characteristic's difference between two independent sample groups.


* through the test of the research hypothesis, we find that the likelihood of an event that occurred is somewhat extreme, then the research hypothesis is a more attractive explanation than is the null. So, if we find a z score that is extreme (How extreme? Having less than a 5% chance of occurring.), we like to say that the reason for the extreme score is something to do with treatments or relationships and not just chance. 



### Definitions

Before walking through the general steps of the hypothesis testing, let us start with a few of its key definitions.


#### statistical significance 

The *statistical significance* can be explained as the level of risk we are willing to take so that we will reject a null hypothesis when it is actually true. It is also refered as *type I error* and *false positive*. Notice that this $\alpha$ is the same one to use when calculating the confidence interval as we talked about in ??????


#### test statistic 

A *test statistic* $T(x)$ is a scalar which summarizes the characteristics of the sample data, which is then used to compared with the expected value under the null hypothesis. Assuming that the null hypothesis is corrected, we can summarize the characteristics of the population, which is denoted as $T(X)$. The $T(X)$ is often follow a common probability distribution such as a normal distribution (for z-test) or a [Student's t-distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution) (for t-test). With both $T(X)$ and $T(x)$, we can calculate the *p-value*, which leads to the decision of whether rejecting the null hypothesis or not. 


#### p-value

Assuming that the null hypothesis is correct, the *p-value* (or *probability value*) is the probability of $T(X)$ happen at least as extreme as the test statistics $T(x)$, i.e., 

$$ p-value = P(T(X) \geq T(x)).$$

If the p-value is less than or equal to a pre-defined and fixed statistical significance $\alpha$, we will conclude that the null hypothesis can be rejected. Otherwise, we will conclude that we cannot reject the null hypothesis, rather than the null hypothesis is true. 



### General Steps of Hypothesis Testing

After you get familar with the above concepts, let us go through the general steps of hypothesis testing.

1. Establishing a null hypotheses;
2. Setting the level of statistical significance;
3. Selecting and calculating the test statistic;
4. Determin the critical value (or p-value ??????);
5. Making the decision to keep or reject the null hypothesis based on the p-value and the critical value.


In practice, to conduct a hypothesis test, we start by initail defining a null hypothesis and a level of risk that we are willing to take. Then we calculate the test statistic of the sample, taking an “extreme” value of the test statistic as evidence against the null hypothesis. If the test statistic falls within the reject region, we knows that the null hypothesis may not be a convincible statement given the observed samples. In contrast, the null hypothesis is a favored statement.

Hypothesis testing is quite applicable is a variety of scenarios such as clinical trails and A/B testing. In the next section, we will show you how to implement a few functions of A/B testing.



```{.python .input}

```
