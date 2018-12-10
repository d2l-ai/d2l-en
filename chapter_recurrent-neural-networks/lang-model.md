# Language Models

Text is an important example of sequence data. In fact, we will use natural language models as the basis for many of the examples in this chapter. Given that, it's worth while discussing some things in a bit more detail. In the following we will view words (or sequences of characters) as a time series of discrete observations. 
Assuming the words in a text of length $T$ are in turn $w_1, w_2, \ldots, w_T$, then, in the discrete time series, $w_t$($1 \leq t \leq T$) can be considered as the output or label of time step $t$. Given such a sequence, the goal of a language model is to estimate the probability 

$$p(w_1, w_2, \ldots, w_T).$$

Language models are incredibly useful. For instance, an ideal language model would be able to generate natural text just on its own, simply by drawing one word at a time $w_t \sim p(w_t|w_{t-1}, \ldots w_1)$. Quite unlike the monkey using a typewriter, all text emerging from such a model would pass as natural language, e.g. English text. Furthermore, it would be sufficient for generating a meaningful dialog, simply by conditioning the text on previous dialog fragments. Clearly we are still very far from designing such a system, since it would need to *understand* the text rather than just generate grammatically sensible content. 

Nonetheless language models are of great service even in their limited form. For instance, the phrases *'to recognize speech'* and *'to wreck a nice beach'* sound very similar. This can cause ambiguity in speech recognition, ambiguity that is easily resolved through a language model which rejects the second translation as outlandish. Likewise, in a document summarization algorithm it's worth while knowing that *'dog bites man'* is much more frequent than *'man bites dog'*, or that *'let's eat grandma'* is a rather disturbing statement, whereas *'let's eat, grandma'* is much more benign. 

## Estimating a language model

The obvious question is how we should model a document, or even a sequence of words. We can take recourse to the analysis we applied to sequence models in the previous section. Let's start by applying basic probability rules:

$$p(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T p(w_t | w_1, \ldots, w_{t-1}).$$

For example, the probability of a text sequence containing four tokens consisting of words and punctuation would be given as:

$$p(\mathrm{Statistics}, \mathrm{is},  \mathrm{fun}, \mathrm{.}) =  p(\mathrm{Statistics}) p(\mathrm{is} | \mathrm{Statistics}) p(\mathrm{fun} | \mathrm{Statistics}, \mathrm{is}) p(\mathrm{.} | \mathrm{Statistics}, \mathrm{is}, \mathrm{fun}).$$

In order to compute the language model, we need to calculate the
probability of words and the conditional probability of a word given
the previous few words, i.e. language model parameters. Here, we
assume that the training data set is a large text corpus, such as all
Wikipedia entries, Project Gutenberg, or all text posted online on the
web. The probability of words can be calculated from the relative word
frequency of a given word in the training data set.

For example, $p(\mathrm{Statistics})$ can be calculated as the
probability of any sentence starting with the word 'statistics'. A
slightly less accurate approach would be to count all occurrences of
the word 'statistics' and divide it by the total number of words in
the corpus. This works fairly well, particularly for frequent
words. Moving on, we could attempt to estimate

$$\hat{p}(\mathrm{is}|\mathrm{Statistics}) = \frac{n(\mathrm{Statistics~is})}{n(\mathrm{Statistics})}.$$

Here $n(w)$ and $n(w, w')$ are the number of occurrences of singletons
and pairs of words respectively. Unfortunately, estimating the
probability of a word pair is somewhat more difficult, since the
occurrences of *'Statistics is'* are a lot less frequent. In
particular, for some unusual word combinations it may be tricky to
find enough occurrences to get accurate estimates. Things take a turn for the worse for 3 word combinations and beyond. There will be many plausible 3-word combinations that we likely won't see in our dataset. Unless we provide some solution to give such word combinations nonzero weight we will not be able to use these as a language model. If the dataset is small or if the words are very rare, we might not find even a single one of them. 

A common strategy is to perform some form of Laplace smoothing. We already ecountered this in our discussion of [Naive Bayes](../chapter_crashcourse/naive-bayes.md) where the solution was to add a small constant to all counts. This helps with singletons, e.g. via

$$\begin{aligned}
	\hat{p}(w) & = \frac{n(w) + \epsilon_1/m}{n + \epsilon_1} \\
	\hat{p}(w'|w) & = \frac{n(w,w') + \epsilon_2 \hat{p}(w')}{n(w) + \epsilon_2} \\
	\hat{p}(w''|w',w) & = \frac{n(w,w',w'') + \epsilon_3 \hat{p}(w',w'')}{n(w,w') + \epsilon_3} 
\end{aligned}$$

Here the coefficients $\epsilon_i > 0$ determine how much we use the
estimate for a shorter sequence as a fill-in for longer
ones. Moreover, $m$ is the total number of words we encounter. The
above is a rather primitive variant of what is Kneser-Ney smoothing
and Bayesian Nonparametrics can accomplish. See e.g. the Sequence
Memoizer of Wood et al., 2012 for more details of how to accomplish
this. Unfortunately models like this get unwieldy rather quickly:
first off, we need to store all counts and secondly, this entirely
ingores the meaning of the words. For instance, *'cat'* and *'feline'*
should occur in related contexts. Deep learning based language models
are well suited to take this into account. This, it is quite difficult
to adjust such models to additional context. Lastly, long word
sequences are almost certain to be novel, hence a model that simply
counts the frequency of previously seen word sequences is bound to
perform poorly there. 


## Markov Models and $n$-grams

Before we discuss solutions involving deep learning we need some more terminology and concepts. Recall our discussion of Markov Models in the previous section. Let's apply this to language modeling. A distribution over sequences satisfies the Markov property of first order if $p(w_{t+1}|w_t, \ldots w_1) = p(w_{t+1}|w_t)$. Higher orders correspond to longer dependencies. This leads to a number of approximations that we could apply to model a sequence:

$$
\begin{aligned}
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2) p(w_3) p(w_4)\\
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2 | w_1) p(w_3 | w_2) p(w_4 | w_3)\\
p(w_1, w_2, w_3, w_4) &=  p(w_1) p(w_2 | w_1) p(w_3 | w_1, w_2) p(w_4 | w_2, w_3) 
\end{aligned}
$$

Since they involve one, two or three terms, these are typically referred to as unigram, bigram and trigram models. In the following we will learn how to design better models. 

## Summary

* Language models are an important technology for natural language processing.
* $n$-grams provide a convenient model for dealing with long sequences by truncating the dependence. 
* Long sequences suffer from the problem that they occur very rarely or never. This requires smoothing, e.g. via Bayesian Nonparametrics or alternatively via deep learning.

## Problems

1. Suppose there are 100,000 words in the training data set. How many word frequencies and multi-word adjacent frequencies does a four-gram need to store?
1. Review the smoothed probability estimates. Why are they not accurate? Hint - we are dealing with a contiguous sequence rather than singletons.
1. How would you model a dialogue?

## Discuss on our Forum

<div id="discuss" topic_id="2361"></div>
