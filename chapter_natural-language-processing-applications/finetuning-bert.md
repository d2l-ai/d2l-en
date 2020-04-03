# Fine-Tuning BERT for Sequence-Level and Token-Level Applications
:label:`sec_finetuning-bert`

In the previous sections of this chapter,
we have designed different models for natural language processing applications,
such as based on RNNs, CNNs, attention, and MLPs.
These models are helpful when there is space or time constraint,
however,
crafting a specific model for every natural language processing task
is practically infeasible.
In :numref:`sec_bert`,
we introduced a pretraining model, BERT,
that requires minimal architecture changes
for a wide range of natural language processing tasks.
One one hand,
at the time of its proposal,
BERT improved the state of the art on various natural language processing tasks.
On the other hand,
as noted in :numref:`sec_bert-pretraining`,
the two versions of the original BERT model
come with 110 million and 340 million parameters.
Thus, when there are sufficient computational resources,
we may consider
fine-tuning BERT for downstream natural language processing applications.

In the following,
we generalize a subset of natural language processing applications
as sequence-level and token-level.
On the sequence level,
we introduce how to fine-tune BERT for single text classification and text pair classification, such as sentiment analysis and natural language inference,
which we have already examined.
On the token level, we will briefly introduce new applications
such as text tagging and question answering
and shed light on how BERT can fit in their problem settings.
During fine-tuning,
the "minimal architecture changes" required by BERT across different applications
are the extra fully-connected layers.


## Single Text Classification

Single text classification takes a single text sequence as the input and outputs its classification result.
Besides sentiment analysis that we have studied in this chapter,
the Corpus of Linguistic Acceptability (CoLA)
is also a dataset for single text classification,
judging whether a given sentence is grammatically acceptable or not :cite:`Warstadt.Singh.Bowman.2019`.
For instance, "I should study." is acceptable but "I should studying." is not.

![Fine-tuning BERT for single text classification applications, such as sentiment analysis and testing linguistic acceptability. Suppose that the input single text has six tokens.](../img/bert-one-seq.svg)
:label:`fig_bert-one-seq`

:numref:`sec_bert` describes the input representation of BERT.
The BERT input sequence unambiguously represents both single text and text pairs,
where the special classification token 
“&lt;cls&gt;” is used for sequence classification and 
the special classification token 
“&lt;sep&gt;” marks the end of single text or separates a pair of text.
As shown in :numref:`fig_bert-one-seq`,
in single text classification applications,
the BERT representation of the special classification token 
“&lt;cls&gt;” encodes the information of the entire input text sequence.
As the representation of the input single text,
it will be fed into a small MLP made of fully-connected (dense) layers
to output the distribution of all the discrete label values.


## Text Pair Classification or Regression

We have also examined natural language inference in this chapter.
It belongs to *text pair classification*,
a type of application classifying a pair of text.

Taking a pair of text as the input but outputting a continuous value,
*semantic textual similarity* is a popular *text pair regression* task.
This task measures semantic similarity of sentences.
For instance, in the Semantic Textual Similarity Benchmark dataset,
the similarity score of a pair of sentences
is an ordinal scale ranging from 0 (no meaning overlap) to 5 (meaning equivalence) :cite:`Cer.Diab.Agirre.ea.2017`.
The goal is to predict these scores.

![Fine-tuning BERT for text pair classification or regression applications, such as natural language inference and semantic textual similarity. Suppose that the input text pair has two and three tokens.](../img/bert-two-seqs.svg)
:label:`fig_bert-two-seqs`

Comparing with single text classification in :numref:`fig_bert-one-seq`,
fine-tuning BERT for text pair classification in :numref:`fig_bert-two-seqs` 
is different in the input representation.
For text pair regression tasks such as semantic textual similarity,
trivial changes can be applied such as outputting a continuous label value
and using the mean squared loss: they are common for regression.


## Text Tagging

![Fine-tuning BERT for text tagging applications, such as part-of-speech tagging. Suppose that the input single text has six tokens.](../img/bert-tagging.svg)
:label:`fig_bert-tagging`


## Question Answering

![Fine-tuning BERT for question answering. Suppose that the input text pair has two and three tokens.](../img/bert-qa.svg)
:label:`fig_bert-qa`


## Summary

* Fine-tune BERT.

## Exercises

1. Let us design a search engine algorithm for news articles. When the system receives an query (e.g., "oil industry during the coronavirus outbreak"), it should return a ranked list of news articles that are most relevant to the query. Suppose that we have a huge pool of news articles and a large number of queries. To simplify the problem, suppose that the most relevant article has been labeled for each query. How can we apply negative sampling (see :numref:`subsec_negative-sampling`) and BERT in the algorithm design?
1. How can we leverage BERT in text generation tasks such as machine translation?



## [Discussions](https://discuss.mxnet.io/t/5882)

![](../img/qr_finetuning-bert.svg)
