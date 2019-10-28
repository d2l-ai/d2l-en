# AB Testing 
:label:`sec_ab_tesing`


![AB Testing - Green beans linked to acne.](../img/ab_testing.svg)
:label:`fig_ab_testing`


In the "internet century", it is not unimaginable to collect millions of data everyday. The application of hypothesis testing as we introduced in :numref:`sec_statistics` is refered to as *A/B Testing* in the world of internet.

A/B Testing is quite useful in the internet companies who are testing the influence on users' behaviors of a new product design (such website layouts and personalized recommendations). The goal of A/B testing is to design an experiment and decide whether there is a dramatic behavior difference between the experiment user group and the control user group. This experiment is usually robust and reproducible for a given user population within a time span. As a result, A/B tesing can help us make a data-informed decision on whether or not launch this new product design.

In this section, we will dive into the details of A/B testing in practice.

* So we'll need to compare the proportion of clicks estimated on the control side, with the proportion estimated on the experiment side. Then the quantitative task tells us whether it's likely that the results we got, the difference we observed, could have occurred by chance, or if it would be extremely unlikely to have occurred if the two sides were actually the same.


## A/B Testing General Steps

Similarly to the 5 steps we introduced in hypothesis testing in :numref:`sec_statistics`, a common A/B testing experiment also needs the following 5 steps:

1. State a null hypotheses, $H_0$, about the new product design by choosing a metrics to measure the effect.
2. Set a practical level of significance ($\alpha$) and a statistical power ($1 - \beta$), so that we are willing to change the given product design if the test is statistically significant.
3. Calculate the required sample size for both control group and the experiment group and take samples.
4. Calculate the test statistic and determine the critical value (or p-value ??????).
5. Keep or reject the null hypothesis based on the p-value and the critical value, and hence make the decision of whether to launch the new product design.

Let us walk through the details of each step together!


### Step 1: Stating A Null Hypotheses by Choosing A Metrics

How to measure a given product design change make a difference or not? First, choosing the corrected metrics will navigate us to a rigorous destination. Here are some common choice of metrics in practice:

* Click-through-rate (CTR): the ratio of all the clicks of a specific link to the number of total views of a webpage (such as an email or an advertisement) containing the link, i.e., $$ CTR = \frac{\text{# of clicks of a link}}{\text{ # of page views }}.$$ 
* Click-through-probability (CTP): the probability of all the unique users who click a specific link to the number of total unique users who view a webpage containing the link, i.e., $$ CTP = \frac{\text{# unique users who click a link}}{\text{# unique users who view a page}}.$$
* DAU/MAU: DAU means "daily active users" and MAU means "monthly active users". DAU/MAU is a popular metric for user engagement expressed as a percentage. Usually a product that has over 20% are said to be good, and over $50%$ is of world class.

No matter what metric we choose, it should not varied a lot between a well-defined control and treatment group, which we usually refer to as metric invariance through a A/A test. A *A/A test* is a sanity check before A/B testing, where we measure the same metric on both the control group and the treatment group without any product design change. Therefore, the A/A test can help us eliminate the worries of unexpected sampling bias between the two groups.

Ultemately, the metric who choosed is used for evaluation of a A/B testing. As a result, we prefer a metric of high *sensitivity*, i.e., accurately reflecting the change that we care about. At the same time, we need the metric to be *robust* - invariance to the irrelevant changes. With a well-defined metric, we can then state our null hypothesis by measuring whether the metric has a significant change from the new product design.


### Step 2: Setting Statistical Significance Level and Statistical Power 

As we introduced in :numref:`sec_statistics`, a common used statistical significance level is $95%$ and statistical power is $0.8$.

* Practical statistical significance level varies depends on each individual tests, it tells you how much change the test detects that makes you really want to launch the change. You may not want to launch a change even if the test is statistically significant because you need to consider the business impact of the change, whether it is worthwhile to launch considering the engineering cost, customer support or sales issue, and opportunity costs.



### Step 3: Calculating Proper Sample Size

Recall from :numref:`sec_statistics`, the *population* denotes the total set of users where we can sample our control group and experiment group. It is critical to decide the size of sample for our A/B testing, so that it is not overestimated or underestimated. Specifically, if we overestimated the sample size, we would waste time and effort to collect data, whereas if we underestimated it, we would not able to make a convincible product design decision.


Calculate the sample size

??????


* How to reduce the size of an experiment to get it done faster? You can increase signiffcance level alpha, or reduce power (1-beta) which means increase beta, or change the unit of diversion if originally it is not the same with unit of analysis (unit of analysis: denominator of your evaluation metric) .



* Several things to keep in mind:
• Duration: What’s the best time to run it? Students going back to college? Holidays? Weekend vs. weekdays?
• Exposure: What fraction of traffic you want to expose the experiment to? Suggestion is take a small fraction, run multiple tests at the same time (different days: weekend, weekday, holiday).
• Learning effect: When there’s a new change, in the beginning users may against the change or use the change a lot. But overtime, user behavior becomes stable, which is called plateau stage. The key thing to measure learning effect is time, but in reality you don’t have that much luxury of taking that much time to make a decision. Suggestion: run on a smaller group of users, for a longer period of time.

### Step 4: Calculating the test statistic and the critical value



### Step 5: Making the decision

When making the decision, be aware that the fact that a "significant" result is more likely to occur by chance alone.  One funny example by [xkcd](https://xkcd.com/882/) explained why we need to be skeptical when seeing headlines like :numref:`fig_significance` below:

![Significance - Green beans linked to acne.](../img/significant_green_bean.png)
:label:`fig_significance`



Because we have two samples, we'll need to choose a standard error that gives us a good comparison of both. The simplest thing we can do is calculate what is a called a *pooled
standard error*. Recall that we'll measure the users who click in each group, which we'll call x control and x experiment, as well as the total number of users in each group, which we'll call n control and n experiment.


Now, the first thing we'll calculate will be what's called the *pooled probability of a click*. And I'm using a hat here because this is an estimated probability. And the pooled probability is the total probability of a click across groups, that is, the total number of users who clicked divided by the total number of users. Then we'll calculate the pooled standard error, which is given by this formula. Now recall that we're going to estimate the difference between p experiment and p control, and I'll call this difference d hat for difference.

```{.python .input}

```
