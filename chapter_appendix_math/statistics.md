# Statistics 
:label:`sec_statistics`

Undoubtedly, to be a top deep learning practitioner, the ability to train the state-of-the-art and high accurate models is crucial. Beyond that, understanding the mathematical mindset behind the models will strengthen your ability to deal with different probings, no matter in the interviews or the customer questioning such as “why should we trust the key parameters of your model?” As a result, in this section, we will empower you to explain the mathematical reasoning behind the models by introducing some fundamental concepts of statistics, which are the backbone of the deep learning algorithms.


The earliest reference of *statistics* can be traced back to an Arab scholar Al-Kindi in the 9th-century, who gave a detailed description of how to use statistics and frequency analysis to decipher encrypted messages. His work laid the foundations for statistical and cryptanalysis. After 800 years, the modern statistics arose from Germany in 1700s, when the researchers focused on the systematic collection of demographic and economic data. Today, statistics is the science subject that concerns the collection, processing, analysis, interpretation and visualization of data. It has been widely used in government, business, sciences, etc.  


More specifically, statistics can be divided to *descriptive statistics* and *statistical inference*. The former is a summary statistic that quantitatively describes or summarizes features of a collection of observed data, which is referred to as a *sample*. The sample is drawn from a *population*, denotes the total set of similar individuals, items, or events of our experiment interests. Contrary to descriptive statistics, *Statistical inference* further deduces the characteristics of a population from *samples*, based on the assumptions that the sample distribution can replicate the population distribution at some degree.


You may wonder: “What is the essential difference between deep learning and statistics?” Fundamentally speaking, statistics focuses on inference problems. This type of problems includes modeling the relationship between the variables, such as causal inference, and testing the statistically significance of model parameters, such as A/B testing. In contrast, deep learning emphasizes on making accurate predictions, without explicitly programming and understanding each parameter's functionality. 
 

In the section, we will introduce three types of statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals. These methods can help us infer the characteristics of a given population, i.e., the true parameter $\theta$. For brevity, we assume that the true parameter $\theta$ of a given population is a scale value. It is also straightforward to extend $\theta$ to a vector or a tensor, and evaluate its estimators using the same techniques below. 


## Evaluating and Comparing Estimators

In statistics, an *estimator* is a function of given samples for calculating the true parameter $\theta$. We estimate $\theta$ with an estimator $\hat{\theta}_n = \hat{f}(x_1, \ldots, x_n)$ as the best guess of $\theta$ after modeling through the training samples {$x_1, x_2, \ldots, x_n$}. 

A better illustration of what we are going to discuss about is plotted in :numref:`fig_comparing_estimators`, where we are estimating the true parameter $\theta$ from the population by the sample estimator $\hat{\theta}_n$.

![Population parameter and a sample parameter.](../img/comparing_estimators.svg)
:label:`fig_comparing_estimators`


There are a lot functions to model an estimator $\hat{\theta_n}$, such as mean or median. You may be curious about which estimator is the best one? Here, we introduce three most common methods to evaluate and compare estimators: statistical bias, standard deviation, and mean square error. 


### Statistical Bias

First, the *statistical bias* of an estimator is the difference between this estimator’s expected value and the true parameter value of $\theta$, i.e.,

$$bias (\hat{\theta}_n) = E(\hat{\theta}_n) - \theta.$$

Note that when $bias(\hat{\theta}_n) = 0$, that is when the expectation of an estimator $\hat{ \theta}_n$ is equal to the value of the true estimator, then we say $\hat{\theta}_n$ is an unbiased estimator. Other things being equal, an unbiased estimator is more desirable than a biased estimator. However, biased estimators are frequently used in practice, especially when unbiased estimators does not exist without further assumptions.


### Standard Deviation


Recall in :numref:`sec_probabillity`, the *standard deviation* (or *standard error*) is defined as the squared root of the variance of an estimator, i.e.,

$$se(\hat{\theta}_n) = \sqrt{\mathrm{Var} (\hat{\theta}_n )} = \sqrt{E[(\hat{\theta}_n - E(\hat{\theta}_n))^2]}.$$

The standard deviation measures the average distance of each value from the mean of the distribution. As a result, the larger the standard deviation, the larger the variability that each data point is from their mean.


### Mean Squared Error

Another widely used evaluating method, the *mean squared error (MSE)* (or *$l_2$ loss*) of an estimator is defined as the expectation of the square difference between $\hat{\theta}_n$ and the true parameter $\theta$, i.e.,

$$\mathrm{MSE} (\hat{\theta}_n, \theta) = E[(\hat{\theta}_n - \theta)^2].$$


MSE is always non-negative. It is the most commonly used regression loss function. As a measure to evaluate an estimator, the more its value closer to zero, the more the estimator is close to the true parameter $\theta$.


### Applications

The above three estimators seem to be unrelated to each other. However, if we scrutinize their definitions in details, we can conclude that the mean squared error can be explain as the sum of squared bias and variance as follows

$$
\begin{aligned}
\mathrm{MSE} (\hat{\theta}_n, \theta) &= E[(\hat{\theta}_n - E(\hat{\theta}_n) + E(\hat{\theta}_n) - \theta)^2] \\
 &= E[(\hat{\theta}_n - E(\hat{\theta}_n))^2] + E[(E(\hat{\theta}_n) - \theta)^2] \\
 &= \mathrm{Var} (\hat{\theta}_n) + [{bias} (\hat{\theta}_n)]^2.\\
\end{aligned}
$$

We refer the above formula as *bias-variance trade-off*. To be specific, the mean squared error can be divided into two error sources: the high bias error and the variance. On the one hand, the bias error is commonly seen in a simple model (such as a linear regression model), which cannot extract high dimensional relations between the features and the outputs. If a model suffers from high bias error, we often say it is *underfitting* or lack of *generalization* as introduced in (:numref:`sec_model_selection`). On the flip side, the other error source - high variance usually results from a too complex model, which overfits the training data. As a result, an *overfitting* model is sensitive to small fluctuations in the data. If a model suffers from high variance, we often say it is *overfitting* and lack of *flexibility* as introduced in (:numref:`sec_model_selection`).


### Implement Estimators from Scratch

Since the standard deviation of an estimator has been implementing in MXNet by simply calling `a.std()` for a ndarray "a", we will skip it but implement the statistical bias and the mean squared error in MXNet. First of all, let us import the packages that we will need in this section.

```{.python .input  n=6}
from mxnet import np, npx
npx.set_np()

## Statistical bias
def stat_bias(ture_theta, est_theta):
    return(np.mean(est_theta) - ture_theta)

## Mean squared error
def mse(data, est_theta):
    return(np.mean(np.square(data - est_theta)))

```

To illustrate the equation of the bias-variance trade-off, let us simulate of normal distribution $\mathcal{N}(\theta, \sigma^2)$ with $10,000$ samples. Here, we use a $\theta = 1$ and $\sigma$ = 4. As the estimator is a function of the given samples, here we use the mean of the samples as an estimator for true $\theta$ in this normal distribution $\mathcal{N}(\theta, \sigma^2)$ .

```{.python .input  n=10}
theta_true = 1
sigma = 4
sample_length = 10000
samples = np.random.normal(theta_true, sigma, sample_length)
theta_est = np.mean(samples)
theta_est
```

Let us validate the trade-off equation by calculating the summation of the squared bias and the variance of our estimator. First, calculate the MSE of our estimator.

```{.python .input  n=11}
mse(samples, theta_est)
```

Next, we calculate $\mathrm{Var} (\hat{\theta}_n) + [{bias} (\hat{\theta}_n)]^2$ as below. As you can see, it is pretty closed to the value of the above MSE.

```{.python .input  n=12}
bias = stat_bias(theta_true, theta_est)
np.square(samples.std()) + np.square(bias)
```

## Conducting Hypothesis Tests


Another statistical inference method is hypothesis testing. While hypothesis testing was popularized in the early 20th century, the first use can be traced back to John Arbuthnot in the 1700s. By examining birth records in London for each of the 82 years from 1629 to 1710 and the human sex ratio at birth, John concluded that the number of males born in London exceeded the number of females in every year. 

Following that, the modern significance testing is the intelligence heritage by Karl Pearson who invented $p$-value and Pearson's chi-squared test), William Gosset who is the father of Student's t-distribution, and Ronald Fisher who initialed the null hypothesis and the significance test. 


A *hypothesis test* is a way of evaluating some evidence against the default statement about a population. We refer the default statement as the *null hypothesis* $H_0$, which we try to reject using the observed data. Here, we use $H_0$ as a starting point for the statistical significance testing. The *alternative hypothesis* $H_A$ (or $H_1$) is a statement that is contrary to the null hypothesis. A great null hypothesis is often stated in a declarative form which posits a relationship between variables. It should reflect the brief as explicit as possible, and be testable by statistics theory. 


Imagine yourself as a chemist. After spending thousands of hours in the lab, you develop a new medicine which can dramatically improve one's ability to understand math. To show its magic power, you need to test it out. Naturally, you may need some volunteers to take the medicine and see whether the pills can help them learn math better. So how do you get started? First, you will need carefully random selected two groups of volunteers, so that there is no difference between their math understanding ability measured by some metrics. The two groups are commonly referred to as the test group and the control group. The *test group* (or *treatment group*) is a group of individuals who will experience the medicine, while the *control group* represents the group of users who are set aside as a benchmark, i.e., identical environment setups except taking this medicine. In this way, the influence of all the variables are minimized, except the impact of the independent variable in the treatment. Next, after a period of taking the medicine, you will need to measure the two groups' math understanding by the same metrics, such as letting the volunteers do the same tests after learning a new math formula. Then, you can collect their performance and compare the results. Really simple, isn't it? However, there are many details you have to think of carefully. For example, what is the suitable metrics to test their math understanding ability? How many volunteers for your test so you can be confident to claim the effectiveness of your medicine? How long should you run the test? And so on.


That is how hypothesis test coming in, where we can solve the above example followed a formalized statistically procedure. Let us start with a few definitions and see how does hypothesis testing empower us to make a decision. Before walking through the general steps of the hypothesis testing, let us start with a few of its key definitions.


### Statistical Significance 

The *statistical significance* measures the probability of erroneously reject the null hypothesis, $H_0$, when it should not be rejected, i.e.,

$$ \text{statistical significance }= 1 - \alpha = P(\text{reject } H_0 \mid H_0 \text{ is true} ).$$

It is also referred to as *type I error* or *false positive*. The $\alpha$, is called as the *significance level* and its commonly used value is $5\%$, i.e., $1-\alpha = 95\%$. Statistical significance can be explained as the level of risk that we are willing to take, when we reject a true null hypothesis. 

:numref:`fig_statistical_significance` shows the the observations' values and probability of a given normal distribution in a two-sample hypothesis test. if the observation data point is located outsides the $95\%$ threshold, it will be a very unlikely observation under the null hypothesis assumption. Hence, there might be something wrong with the null hypothesis and we will reject it. 


![Statistical significance.](../img/statistical_significance.svg)
:label:`fig_statistical_significance`

### Statistical Power 

The *statistical power* (or *sensitivity*) measures the ability to identify a true difference as it states in $H_0$, i.e., 

$$ \text{statistical power }= P(\text{reject } H_0  \mid H_0 \text{ is false} ).$$

Recall that a *type I error* is the rejection of a true null hypothesis, while a *type II error* is referred to as the non-rejection of a false null hypothesis. A type II error is usually denoted as $\beta$, and hence the corresponding statistical power is $1-\beta$. 

Intuitively, statistical power can be interpreted as how likely our test will detect a real discrepancy of some minimum magnitude at a desired statistical significance level. $80\%$ is a commonly used statistical power threshold. The higher the statistical power, the more likely we can catch the true difference. We can imagine the power as the fishing net as shown in :numref:`fig_statistical_power`. In this analogy, a high power hypothesis test is like a high graduality fishing net and a smaller discrepancy is like a smaller fish. If the fishing net is not of enough high graduality, then the smaller fish may easily escape from the gaps, and hence the fishing net cannot catch the fish. Similarly, if the statistical power is not of enough high power, then the test may not catch the smaller discrepancy.

![Statistical significance.](../img/statistical_power.svg)
:label:`fig_statistical_power`

### Test Statistic 

There are two common statistical hypothesis tests: the z-tests and the t-tests. The *z-test* is helpful if we are comparing the characteristic between a sample and the given population, based on the characteristic's standard deviation of the given population. On the other hand, the *t-test* is applicable when comparing the characteristic's difference between two independent sample groups. Both the z-tests and the t-tests will return us a $p$-value, which will talk more about it later.

Assume that the null hypothesis is corrected, we can summarize the characteristics of the population, which is denoted as $T(X)$. Similarly, a *test statistic* $T(x)$ is a scalar which summarizes the characteristics of the sample data, which is then used to compared with the expected value under the null hypothesis. The $T(X)$ is often following a common probability distribution such as a normal distribution (for z-test) or a Student's t-distribution (for t-test). With both $T(X)$ and $T(x)$, we can calculate the $p$-value, which leads to the decision of whether rejecting the null hypothesis or not. 


### $p$-value

Assume that the null hypothesis is correct, the *$p$-value* (or the *probability value*) is the probability of $T(X)$ happen at least as extreme as the test statistics $T(x)$ threshold, i.e., 

$$ p\text{-value} = P(T(X) \geq T(x)).$$

If the $p$-value is smaller than or equal to a pre-defined and fixed statistical significance level $\alpha$, we will conclude to reject the null hypothesis. Otherwise, we will conclude that we do not have enough evidence to reject the null hypothesis, rather than the null hypothesis is true. For a given population distribution, the *region of rejection* will be the interval contained of points which has a $p$-value smaller than the statistical significance level $\alpha$.


### One-side Test and Two-sided Test

Normally there are two kinds of significance test: the one-sided test and the two-sided test. The *one-sided test* (or *one-tailed test*) is appropriate when the region of rejection is on only one side of the sampling distribution. For example, the null hypothesis may state that the true parameter $\theta$ is less than or equal to a value $c$. The alternative hypothesis would be that $\theta$ is greater than $a$. On the other hand, the *two-sided test* (or *two-tailed test*) is appropriate when the region of rejection is on both sides of the sampling distribution. An example in this case may have a null hypothesis state that the true parameter $\theta$ is equal to a value $c$. The alternative hypothesis would be that $\theta$ is not equal to $a$.



### General Steps of Hypothesis Testing

After getting familiar with the above concepts, let us go through the general steps of hypothesis testing.

1. State the question and establish a null hypotheses $H_0$.
2. Set the statistical significance level $\alpha$ and a statistical power ($1 - \beta$).
3. Take samples through experiments.
4. Calculate the test statistic and the $p$-value.
5. Make the decision to keep or reject the null hypothesis based on the $p$-value and the  statistical significance level $\alpha$.


To sum up, to conduct a hypothesis test, we start by defining a null hypothesis and a level of risk that we are willing to take. Then we calculate the test statistic of the sample, taking an “extreme” value of the test statistic as evidence against the null hypothesis. If the test statistic falls within the reject region, we knows that the null hypothesis may not be a convincible statement given the observed samples. In contrast, the null hypothesis is a favored statement.

Hypothesis testing is quite applicable in a variety of scenarios such as the clinical trails and the A/B testing (which we will show you more details in the next section).


## Constructing Confidence Intervals


To estimate the value of the true parameter $\theta$, one should not limit to some point estimators. Rather, one can expand a point to a range by finding an interval so that the true parameter $\theta$ may be located in. If you have a similar idea and if you were born about a century ago, then you would be on the same wavelength with Jerzy Neyman, who first introduced the modern concept of confidence interval in 1937.


Certainly, we are more confident that the wider intervals are more likely to contain the true parameter $\theta$ than the shorter intervals. At the other extreme, a super high confidence level may be too wide to be meaningful in real world applications. Hence, it is critical to determine the length of an interval when locating $\theta$ based on the confidence level. Let us see how to derive it together!


### Definition

A *confidence interval* is an estimated range of a true population parameter that we can construct by given the samples. Mathematically, a *confidence interval* for the true parameter $\theta$ is an interval $C_n$ that computed from the sample data such that 

$$P_{\theta} (C_n \ni \theta) \geq 1 - \alpha, \forall \theta. $$

Here $\alpha \in (0,1)$, and $1 - \alpha$ is called the *confidence level* or *coverage* of the interval. This is the same $\alpha$ as the significance level as we discussed about above.
Note that the above probability statement is about variable $C_n$, not about the fixed $\theta$. To emphasize this, we write $P_{\theta} (C_n \ni \theta)$ rather than $P_{\theta} (C_n \in \theta)$.
 

Suppose that $\hat{\theta}_n \sim N(\theta, \hat{\sigma}_n^2)$, where $\hat{\sigma}_n^2$ is the standard deviation of $n$ samples. For a two-sided test, we can form an approximate $1-\alpha$ confidence interval for the true parameter $\theta$ as follows:
 
 $$C_n = [\hat{\theta}_n - z_{\alpha/2} \hat{\sigma}_n, \ \hat{\theta}_n + z_{\alpha/2} \hat{\sigma}_n]$$
 
where $z_{\alpha/2}$ is calculated by the infinity case ($\infty$) in the last row of the t-distribution table such that $P(Z > z_{\alpha/2}) = \alpha/2$ for $Z \sim N(0,1)$. 
 

### Interpretation

For a single experiment, a $1-\alpha$ (e.g., $95\%$) confidence interval can be interpreted as: there is $1-\alpha$ probability that the calculated confidence interval contains the true population parameter. Note this is a probability statement about the confidence interval, not about the population parameter $\theta$. 

Notice that the confidence interval is not a precise range of plausible values for the sample parameter, rather it is an estimate of plausible values to locate for the population parameter. Furthermore, a particular $1-\alpha$ confidence level calculated from an experiment cannot conclude that there is $1-\alpha$ probability of a sample estimator of a new experiment falling within this interval.


### Example

For example, if $\alpha = 0.05$, then we will have a $95\%$ confidence interval. If we refer to the last row of the t-distribution table, we will have a $z$-score $z_{\alpha/2} = 1.96$ for a two-sided test. 

Let us calculate a $95\%$ two-sided confidence interval for the estimator $\theta_n$ of previous samples.

```{.python .input  n=6}
sample_std = samples.std()
(theta_est - 1.96 * sample_std, theta_est + 1.96 * sample_std)
```

## Summary

* Statistics focuses on inference problems, whereas deep learning emphasizes on making accurate predictions without explicitly programming and understanding. 
* There are three common statistics inference methods: evaluating and comparing estimators, conducting hypothesis tests, and constructing confidence intervals.
* There are three most common estimators: statistical bias, standard deviation, and mean square error. 
* A confidence interval is an estimated range of a true population parameter that we can construct by given the samples.
* Hypothesis testing is a way of evaluating some evidence against the default statement about a population.


## Exercises

1. For our chemist example in introduction, can you derive the 5 steps to conduct a two-sided hypothesis testing? Given the statistical significance level $\alpha = 0.05$ and the statistical power $1 - \beta = 0.8$.
