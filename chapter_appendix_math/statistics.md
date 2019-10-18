# D2L Appendix - Statistics Inference
:label:`sec_statistics`


“What is the difference between machine learning and statistics?” You may often hear similar questions and eager to know the answers. Fundamentally, statistics theories and algorithms focus on the inference problems. This type of problems include modeling relationships between the variables and conducting hypothesis testing to determine the statistically significance of the variables. In contrast, machine learning algorithms emphasis on making accurate predictions, without explicitly programming.

Undoubtedly, the ability to train state-of-the-art models and make precise predictions is crucial in both academia and industry. On the other hand, the statistical understanding mindset behind the models will strengthen your ability to deal with different probings, such as “why your model works”. 

*Statistical inference* deduces the characteristics of a population from the observed data (*samples*) that are sampled from that population. In statistics, the *population* denotes the total set of observations which we can get samples from, while the *sample* set is a set of data that are collected from the given population group. Contrary to descriptive statistics, which only cares about the characteristics of the observed data, but not the larger population.

In the section, we will introduce three types of statistics inference methods: evaluating and comparing estimators , constructing confidence intervals, and conducting hypothesis tests. These methods can help us infer the characteristics of a given population, i.e., the true parameter $\theta$. From now on, for brevity, we assume that the true parameter $\theta$ of a given population is a scale value. It is also straightforward to extend $\theta$ to a vector or a tensor, and evaluate its estimators using the same techniques below. 


## Evaluating and comparing estimators

In statistics, an [estimator](https://en.wikipedia.org/wiki/Estimator) is a function of given samples for calculating the true parameter $\theta$. We estimate $\theta$ with an estimator $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ as the best guess of $\theta$ after modeling through the training samples {$x_1, x_2, \ldots, x_n$}. There are many ways to evaluate and compare estimators. Here, we introduce three of the most common methods: statistical bias, standard deviation, and mean square error. 


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

Let us implement the statistical bias, the standard deviation, and the mean squared error in MXNet. First of all, let us import the packages that we need in this section.



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


* To sum up, for a hypothesis test,  we construct a function of the sample called *test statistic* and consider its sampling distribution, taking an “extreme” value of the test statistic as evidence against the null hypothesis.




### General Steps of Hypothesis Testing

* P233
* (P235) Comparing Obtained Values With Critical Values and Making Decisions About Rejecting or Accepting the Null Hypothesis.



* In sum (and in practice), the researcher defines a level of risk that he or she is willing to take. If the results fall within the region that says, "This could not have occurred by chance alone-something else is going on," the researcher knows that the null hypothesis (which states an equality) is not the most auractive explanation for the observed outcomes. Instead, the research hypothesis (that there is an inequality or a difference) is the favored explanation.







### model evaluation metric
https://scikit-learn.org/stable/modules/model_evaluation.html
