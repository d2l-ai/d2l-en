# AB Testing 
:label:`sec_ab_testing`


![AB Testing - Green beans linked to acne.](../img/ab_testing.svg)
:label:`fig_ab_testing`


In the 21th century of data explosion, internet connects everything and everybody together. As a result, billions of data is generated, analyzed, processed, and re-used everyday to promote almost every aspect of our life. Behind the scenes of the NCAA games, you may say hundreds of statisticians busily analyze each palyer's performance and provide the audience real-time statistics. Again, behind the screen of the social media APPs, a myriad of data scientists and engineers dig deeply to users' behavior to elevate the customers' experience. And so on! 
As we can see, smart companies pay a lot attention to better utilize users' data and optimize the products. One of the widely accepted way is *A/B Testing*, which is an application of the [two-sided hypothesis test](https://en.wikipedia.org/wiki/Two-sample_hypothesis_testing) as we introduced in :numref:`sec_statistics`,.

A/B Testing is quite applicable to test the influence of a new product design on the users' behavior, expecially in the technology companies who continuously collect tons of data. The new product design can vary from the design of website layouts to personalized recommendation features. The goal of A/B testing is to design an experiment and decide whether there is a dramatic behavior difference between the test group and the control group. The experiment of A/B testing is usually robust and reproducible for a given user population within a time span. As a result, A/B testing can help us make a data-informed decision on whether or not to launch a new product design. In this section, we will dive deep into the details of A/B testing in practice.


## A/B Testing General Steps

Similarly to the 5 steps in hypothesis testing introduced in :numref:`sec_statistics`, a common A/B testing experiment also needs the following 5 steps:

1. State a null hypotheses, $H_0$, about the effect of a new product design through one or more measurement metrics.
2. Set a practical statistical significance level $\alpha$ and a statistical power ($1 - \beta$), so that we are willing to change the given product design if the test is statistically significant.
3. Calculate the required sample size for both control group and the test group. Then, carefully take samples.
4. Run the test and collect data.
5. Calculate the test statistic and determine the $p$-value. Then decide keep or reject the null hypothesis, and hence make the decision of whether to launch the new product design or recycle the test.

Let us walk through the details of each step together!


### Step 1: Stating A Null Hypotheses by Choosing A Metrics

In A/B testing, the null hypothesis usually states that there is no discrepancy of a characteristics metrics (such as the conversion rate or the website traffic) between the control group and the test group. So how to measure whether a new product design makes a difference or not? First, choosing the corrected metrics will navigate us to the right destination. Here are some common choices of metrics in practice:

* Click-through-rate (CTR): the ratio of all the clicks of a specific link to the number of total views of the webpage (such as an email or an advertisement) containing that link, i.e., $$ CTR = \frac{\text{# of clicks of a link}}{\text{ # of page views }}.$$ 
* Click-through-probability (CTP): the probability of all the unique users who click a specific link to the number of total unique users who view the webpage containing that link, i.e., $$ CTP = \frac{\text{# unique users who click a link}}{\text{# unique users who view a page}}.$$
* DAU/MAU: DAU means "daily active users" and MAU means "monthly active users". DAU/MAU is a popular metric for user engagement expressed as a percentage. Usually a product that has over $20 \%$ are said to be good, and over $50 \%$ is of the world class.


No matter what metric we choose, it should not varied a lot between a well-defined control and test group, which we usually refer to as metric invariance through a A/A test. A *A/A test* is a sanity check test before the A/B test, where we measure the same metric on both the control group and the treatment group without any product design change. Therefore, the A/A test can help us eliminate the uncertainty of unexpected sampling bias between the two groups.

Ultimately, the metric that we choosed is used for evaluation of a A/B testing. As a result, we prefer a metric of high *sensitivity*, i.e., accurately reflecting the change that we care about. At the same time, we need the metric to be *robust* - invariance to the irrelevant changes. With a well-defined metric, we can then state our null hypothesis by measuring whether the metric has a significant change from the new product design.



### Step 2: Setting Statistical Significance Level and Statistical Power 

As introduced in :numref:`sec_statistics`, a common used statistical significance level $\alpha$ is $5 \%$. Even though $1-\alpha = 95 \%$ looks like a pretty high confidence, we need to be aware that a "significant" result is more likely to occur by chance rather than reflect the truth. This scenario happens a lot especially when we simulate a similar test multiple times while end up with unconsistant test results. One funny example by [xkcd](https://xkcd.com/882/)shown in :numref:`fig_significance`, which explained why we need to be skeptical of the headlines like this.

![Significance - Green beans linked to acne.](../img/significant_green_bean.png)
:label:`fig_significance`

As we can see, even if there is no real discrepancy, a $95 \%$ statistical significance would still be observed by chance at 1 out of 20 times: "We found a link between green jelly beans and acne". Hence, it is critical to understand what may result in a statistically significance from the given sample. Practically, it can be explained as one of the following:

1. There is a real discrepancy between the control group and the test group.
1. There is no discrepancy between the control group and the test group, but it happened to be a rare observation.
1. The data distribution is not well captured by the model, and hence the test accidentally lead to a statistical significance.


What is more, another important preset parameter for A/B testing is statistical power. As mentioned in :numref:`sec_statistics`, $0.8$ or above are widely used thresdholds for statistical power. The choice of statistical power are based on the following factors:

* The potential losses of missing a real discrepancy. The greater the statistical power, the higher such losses are expected to be for a given sample size, .
* The cost of involving more users into a test. If adding more users is relatively cost-efftive, then it is worth to endeavor to increase the statistical power to lower the chance of missing a true discrepancy.
* The limit of time to run the test. The time limit is closely related to the required sample size for a test. If there is a hard deadline to launch (for instance because of an upcoming event, an special holiday such as Black Friday or Christmas, or a particular season), then we cannot wait too long to collect all the test outcomes. As a result, a smaller sample size may lead to a lower statistical power.

To sum up, a higher statistical significance or a higher statistical power will both draw a more convicible test conclusion, at the cost of collecting more data.


### Step 3: Calculating Proper Sample Size

Recall from :numref:`sec_statistics`, the *population* denotes the total set of users from which we can sample our control group and test group. Deciding the proper sample size of A/B testing is crucial. If we overestimated the sample size, we would waste time and effort to collect data. Whereas if we underestimated it, no convincible decision could be made. To avoid underestimating or overestimating the sample size, it is critical to understand the other relavent factors, such as statistical significance, statistical power, and the level of discrepancy that we want to capture from the new product design.


#### Sample Size Formula

Before introducing the formula, let us get familiar with the notations and hypothesis. Suppose that we have data from a control group $A$ as {$x_1, \ldots, x_n$} and an test group $B$ as {$y_1, \ldots, y_n$}, with the following notations for our A/B test:
* $n$: the sample size for each group; 
* $\mu_A, \mu_B$: the true mean of group $A$ and $B$ respectively;
* $\bar{x}_n, \bar{y}_n$: the sample mean of group $A$ and $B$ respectively, i.e.,
$$\bar{x}_n = \sum_{i=1}^{n} x_i \text{ and } \bar{y}_n = \sum_{j=1}^{n} y_j;$$
* $s$: the pooled standard deviation of both groups. Mathematically, it can be calculated as

$$s^2 = \frac{\sum_{i=1}^{n} (x_i - \bar{x}_n )^2 + \sum_{j=1}^{n} (y_j - \bar{y}_n)^2}{2n};$$

* $1 - \alpha$: the statistical significance;
* $1 - \beta$: the statistical power;
* $\Phi^{-1}(p)$: the inverse distribution function of x. To be specific, if the [cumulative distribution function](https://en.wikipedia.org/wiki/Cumulative_distribution_function) of a random variable $x$ is denoted by $\Phi$, then $\Phi^{-1}(x) = \inf \{ x \in \mathbb{R}: \Phi (x) \geq p \},$ where $p \in (0, 1)$.

Furthermore, assume that we want to check whether the discrepancy of users' behavior between the control group $A$ and the text group $B$ is $0$ or not. With the above notations, we will formulize a null hypothesis

$$H_0 : \mu_A - \mu_B = 0, $$

versus an alternative hypothesis 

$$H_A : \mu_A - \mu_B \neq 0.$$


Following the above setups, the minumum sample size of the A/B test can be calculated as 

$$n = \frac{2 s^2 (\Phi^{-1}(\beta) + \Phi^{-1}(\alpha/2))^2}{(\mu_A - \mu_B)^2}.$$



#### Sample Size Formula Interpretation

If you are curious about how we deduce this formula, here are the intuitions. For a A/B testing with large enough sample size, test statistics $T_n$ is often following a standard normal distribution. That is, $\bar{x}_n -\bar{y}_n  \sim \mathcal{N} (\mu_A-\mu_B, s^2)$, as $n \to \infty$, and hence the test statistics 

$$T_n = \frac{\bar{x}_n - \bar{y}_n}{\sqrt{\frac{s^2}{n} + \frac{s^2}{n}}} = \frac{\bar{x}_n - \bar{y}_n}{s \sqrt{\frac{2}{n}}} \sim \mathcal{N} (0, 1).$$


Now we can start the deduction from the definition of statistical significance. The null hypothesis $H_0$ will be rejected if 

$$ |T_n| \geq \Phi^{-1}(1-\alpha).$$

As a two-sided hypothesis test, the above formula is equivalent to 

$$ T_n \geq \Phi^{-1}(1-\frac{\alpha}{2}).$$


Next, by the definition of statistical power, we need the power 

$$P(\text{reject } H_0  \mid H_A \text{ is true}) = 1 - \beta,$$ i.e.,

$$ P \left[T_n \geq \Phi^{-1}(1-\frac{\alpha}{2}) \mid H_A \text{ is true}\right] = 1 - \beta.$$



Under the assumption that the alternative hypothesis $H_A$ is true, we have $\tilde{T}_n = T_n - \frac{\mu_A - \mu_B}{s \sqrt{\frac{2}{n}}} \sim \mathcal{N} (0,1)$. Replace $T_n$ with $\tilde{T}_n$, we get 
$$P \left[ \tilde{T}_n \geq \Phi^{-1}(1-\frac{\alpha}{2}) - \frac{\mu_A - \mu_B}{s \sqrt{\frac{2}{n}}} \mid H_A \text{ is true} \right] = 1-\beta.$$


$$P \left[ \tilde{T}_n \leq \Phi^{-1}(1-\frac{\alpha}{2}) - \frac{\mu_A - \mu_B}{s \sqrt{\frac{2}{n}}} \mid H_A \text{ is true} \right] = \beta.$$


$$ \Phi^{-1}(1-\frac{\alpha}{2}) - \frac{\mu_A - \mu_B}{s \sqrt{\frac{2}{n}}} \geq  \Phi^{-1}(\beta).$$


This is equivalent to 

$$n \geq \frac{2 s^2 (\Phi^{-1}(\beta) + \Phi^{-1}(\alpha/2))^2}{(\mu_A - \mu_B)^2}.$$


#### Sample Size Formula from Scratch

Now let us implement the sample size formula from scratch together!

```{.python .input  n=1}
from scipy.stats import norm

def min_sample_size(mu_A, mu_B, sd, beta=0.2, alpha=0.05):
    Z_beta = norm.ppf(1-beta)
    Z_alpha = norm.ppf(1-alpha/2)
    min_N = (2 * sd**2 * (Z_beta + Z_alpha)**2/ (mu_A - mu_B)**2)
    return(min_N)


s1 = min_sample_size(1, 2, 100, beta=0.2, alpha=0.05)
s2 = min_sample_size(1, 20, 100, beta=0.2, alpha=0.05)
s3 = min_sample_size(1, 2, 100, beta=0.2, alpha=0.5)
s4 = min_sample_size(1, 2, 100, beta=0.5, alpha=0.05)

s1, s2, s3, s4
```

From the above experiments, you may have some idea on how to reduce the minimum sample size? Here are some quick tips. In order to lower the minimum sample size of a two-sided A/B test, we can modify the value of one of the following parameters in the null hypothesis:

* Increase the accepted level of discrepancy, $\mu_A - \mu_B$ (by comparing $s_1$ with $s_2$).
* Increase the statistical significance level, $\alpha$ (by comparing $s_1$ with $s_3$).
* Increase the statistical power level, $\beta$ (by comparing $s_1$ with $s_4$).


### Step 4:  Running The Test

This step differentiates A/B testing from the traditional hypothesis testing as the users online have some unique characteristics. If we ignored the users perferences or other objective scenarios but blindly ran the test, we may not conclude a convincible decision. Hence, we need to carefully think of the other factors that may impact the variability of the results. Here are a few things to keep in mind before running the test:

* Time duration. Based on your new feature design, think of what time of a day, a week, a month, or even a year may be sentitive to the design. For instance, a online shopping website may have tons of traffic on black friday, and hence we can easily get enough samples on a single day. However, we may question whether the user behavior on that single day will represent the general user behavior in a regular day.

* Geographical limitation. Be alert of the geographical differences may add some variability to the results. As a result, if you are running a A/B test globally, you may need to cautiously take proportional samples from different regions.

* Learning effect. When testing a new product design, at the start, users may either use the new feature a lot or do not touch it as all. After a while, the users behavior becomes stable, which is called the *plateau stage*. The key thing to dilute the learning effect is time, but in reality we may not wait so long to make a decision. In this scenario, a A/A test is pretty useful here, when we can test the learning effect on the same user group at "pre-period" and "post-period".



### Step 5: Making The Decision

Last but not the least, we need to make a decision by calculating the test statistics as introduced in :numref:`sec_statistics`. Recall that the test statistics $T(x)$ is a function about the characteristics calculated from the sample. Since the null hypothesis states that there is no discrepancy of a characteristics between the control group and the test group, we can apply the two-sided hypothesis test and calculate the $p$-value = $P(T(X) \geq T(x))$. For detailed formulation and interpretations, please check :numref:`sec_statistics`.


However, no matter whether the $p$-value is smaller or larger than the significan level, we cannot simply conclude a rejection or non-rejection before the following checks:

* The sanity check: to check if the invariant metrics have changed or not. If the sanity check failed, we cannot make any conclusion. Rather, we need to analyze why it failed (such as doing retrospective analysis or thinking of the learning effect).
* Cross checking. In case of no significance, then the experiment setup may be incorrect, or the change affects different subgroups of users differently. In this case, it is helpful to break down into different subgroups of users, or differnt days of a week. For example, the [Simpson’s paradox](https://en.wikipedia.org/wiki/Simpson%27s_paradox) is a phenomenon in which a trend appears in different groups of data but disappears or reverses when these groups are combined).
* Triple (or more) checks. As we alluded in step 2, a statistically significance result may happen by chance. Hence, bootstrap and run the same experiment multiple times to see you can reach the same conclusion. In addition, control the [false discovery rate](https://en.wikipedia.org/wiki/False_discovery_rate) would be helpful as well.



## Summary

* The goal of A/B testing is to design an hypothesis testing experiment and decide whether there is a dramatic behavior difference between the test group and the control group.
* The 5 key steps of A/B testing: 
    1. State a null hypotheses. 
    2. Set the statistical significance level and the statistical power.
    3. Calculate the sample size.
    4. Run The Test.
    5. Calculate the test statistic and make the decision.
* Triple check the statistically significance before making the decision, since it may happen by chance.


## Exercise
1. Think of suitable scenarios as many as possible where you can run a A/B testing.
1. Beside the common choices of metrics we introduced, can you think or search for any other metrics of A/B testing?
