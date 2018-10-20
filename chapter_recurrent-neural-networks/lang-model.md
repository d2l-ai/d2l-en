# Language Models

Language models are an important technique used in natural language processing. The most common data in natural language processing is text data. In fact, we can consider a natural language text as a discrete time series. Assuming the words in a text of length $T$ are in turn $w_1, w_2, \ldots, w_T$, then, in the discrete time series, $w_t$($1 \leq t \leq T$) can be considered as the output or label of time step $t$. Given a sequence of words of length $T$: $w_1, w_2, \ldots, w_T$, the language model will calculate the probability of the sequence:

$$\mathbb{P}(w_1, w_2, \ldots, w_T).$$


Language models can be used to improve the performance of speech recognition and machine translation. For example, in speech recognition, given the speech segment "the kitchen runs out of cooking oil", there are two possible text outputs, "the kitchen runs out of cooking oil" and "the kitchen runs out of petroleum", as the words "cooking oil" and "petroleum" are homophones in Chinese. If the language model determines that the probability of the former is greater than the probability of the latter, we can output the text sequence of "the kitchen runs out of cooking oil" even though the speech segment is the same for both text outputs. In machine translation, if the English words "you go first" are translated into Chinese word by word, you may get the text sequence "你走先" (you go first), "你先走" (you first go), or another arrangement. If the language model determines that the probability of "你先走" is greater than the probability of other permutations, we can translate "you go first" into "你先走".


## Language model calculation


Now we see that the language model is useful, but how should we compute it? Assuming each word in the sequence $w_1, w_2, \ldots, w_T$ is generated in turn, we have

$$\mathbb{P}(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T \mathbb{P}(w_t \mid w_1, \ldots, w_{t-1}) .$$

For example, the probability of a text sequence containing four words would be given as:

$$\mathbb{P}(w_1, w_2, w_3, w_4) =  \mathbb{P}(w_1) \mathbb{P}(w_2 \mid w_1) \mathbb{P}(w_3 \mid w_1, w_2) \mathbb{P}(w_4 \mid w_1, w_2, w_3) .$$

In order to compute the language model, we need to calculate the probability of words and the conditional probability of a word given the previous few words, i.e. language model parameters. Here, we assume that the training data set is a large text corpus, such as all Wikipedia entries. The probability of words can be calculated from the relative word frequency of a given word in the training data set. For example, $\mathbb{P}(w_1)$ can be calculated as the ratio of the word frequency (the number of occurrences of a given word) of $w_1$ in the training data set to the total number of words in the training data set. Therefore, according to the conditional probability definition, the conditional probability of a word given the previous few words can also be calculated by the relative word frequency in the training data set. For example, $\mathbb{P}(w_2 \mid w_1)$ can be calculated as the ratio of the frequency of two words $w_1, w_2$ being adjacent to the word frequency of $w_1$, since this is the ratio of $\mathbb{P}(w_1, W_2)$ to $\mathbb{P}(w_1)$. In the same way, $\mathbb{P}(w_3 \mid w_1, w_2)$ can be calculated as the ratio of the frequency of three words $w_1, w_2, w_3$ being adjacent and the frequency of two words $w_1, w_2$ being adjacent. This same process can be used for longer strings of words.


## $N$-grams

As the length of the word sequence increases, the probabilities of multiple words appearing together that are calculated and stored increase exponentially. The $N$-grams simplifies the calculation of the language model through the Markov assumption (although this assumption is not necessarily true). Here, the Markov assumption holds that the appearance of a word is only associated with the previous $n$ words, namely the Markov chain of order $n$. If $n=1$, then we have $\mathbb{P}(w_3 \mid w_1, w_2) = \mathbb{P}(w_3 \mid w_2)$. Based on a Markov chain of $n-1$ order, we can rewrite the language model to

$$\mathbb{P}(w_1, w_2, \ldots, w_T) \approx \prod_{t=1}^T \mathbb{P}(w_t \mid w_{t-(n-1)}, \ldots, w_{t-1}) .$$


The above is also called $n$-grams. It is a probabilistic language model based on the Markov chain of $n-1$ order. When $n$ is 1, 2, and 3, respectively, we call them unigram, bigram, and trigram. For example, the probability of the sequence $w_1, w_2, w_3, w_4$ with length 4 in unigram, bigram and trigram is:

$$
\begin{aligned}
\mathbb{P}(w_1, w_2, w_3, w_4) &=  \mathbb{P}(w_1) \mathbb{P}(w_2) \mathbb{P}(w_3) \mathbb{P}(w_4) ,\\
\mathbb{P}(w_1, w_2, w_3, w_4) &=  \mathbb{P}(w_1) \mathbb{P}(w_2 \mid w_1) \mathbb{P}(w_3 \mid w_2) \mathbb{P}(w_4 \mid w_3) ,\\
\mathbb{P}(w_1, w_2, w_3, w_4) &=  \mathbb{P}(w_1) \mathbb{P}(w_2 \mid w_1) \mathbb{P}(w_3 \mid w_1, w_2) \mathbb{P}(w_4 \mid w_2, w_3) .
\end{aligned}
$$

When $n$ is small, the $n$-grams is often inaccurate. For example, in unigram, the probability of the three-word sentences "you go first" and "you first go" is the same. However, when $n$ is large, the $n$-grams needs to calculate and store a large number of word frequencies and multi-word adjacent frequencies.

So, is there a way to better balance these two features in the language model? We will explore such methods in this chapter.

## Summary

* Language models are an important technology for natural language processing.
* $N$-grams provide a probabilistic language model based on the Markov chain of $n-1$ order, where $n$ balances computational complexity and model accuracy.


## exercise

* Suppose there are 100,000 words in the training data set. How many word frequencies and multi-word adjacent frequencies does a four-gram need to store?
* What other applications of language models can you think of?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/6650)

![](../img/qr_lang-model.svg)
