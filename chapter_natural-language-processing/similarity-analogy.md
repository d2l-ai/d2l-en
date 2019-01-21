# Finding Synonyms and Analogies

In the ["Implementation of Word2vec"](./word2vec-gluon.md) section, we trained a word2vec word embedding model on a small-scale data set and searched for synonyms using the cosine similarity of word vectors. In practice, word vectors pre-trained on a large-scale corpus can often be applied to downstream natural language processing tasks. This section will demonstrate how to use these pre-trained word vectors to find synonyms and analogies. We will continue to apply pre-trained word vectors in subsequent sections.

## Using Pre-trained Word Vectors

MXNet's `contrib.text` package provides functions and classes related to natural language processing (see the GluonNLP tool package[1] for more details). Next, let us check out names of the provided pre-trained word embeddings.

```{.python .input}
from mxnet import nd
from mxnet.contrib import text

text.embedding.get_pretrained_file_names().keys()
```

Given the name of the word embedding, we can see which pre-trained models are provided by the word embedding. The word vector dimensions of each model may be different or obtained by pre-training on different data sets.

```{.python .input  n=35}
print(text.embedding.get_pretrained_file_names('glove'))
```

The general naming conventions for pre-trained GloVe models are "model.(data set.)number of words in data set.word vector dimension.txt". For more information, please refer to the GloVe and fastText project sites [2,3]. Below, we use a 50-dimensional GloVe word vector based on Wikipedia subset pre-training. The corresponding word vector is automatically downloaded the first time we create a pre-trained word vector instance.

```{.python .input  n=11}
glove_6b50d = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.50d.txt')
```

Print the dictionary size. The dictionary contains 400,000 words and a special unknown token.

```{.python .input}
len(glove_6b50d)
```

We can use a word to get its index in the dictionary, or we can get the word from its index.

```{.python .input  n=12}
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Applying Pre-trained Word Vectors

Below, we demonstrate the application of pre-trained word vectors, using GloVe as an example.

### Finding Synonyms

Here, we re-implement the algorithm used to search for synonyms by cosine similarity introduced in the ["Implementation of Word2vec"](./word2vec-gluon.md) section. In order to reuse the logic for seeking the $k$ nearest neighbors when seeking analogies, we encapsulate this part of the logic separately in the `knn` ($k$-nearest neighbors) function.

```{.python .input}
def knn(W, x, k):
    # The added 1e-9 is for numerical stability.
    cos = nd.dot(W, x.reshape((-1,))) / (
        (nd.sum(W * W, axis=1) + 1e-9).sqrt() * nd.sum(x * x).sqrt())
    topk = nd.topk(cos, k=k, ret_typ='indices').asnumpy().astype('int32')
    return topk, [cos[i].asscalar() for i in topk]
```

Then, we search for synonyms by pre-training the word vector instance `embed`.

```{.python .input}
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec,
                    embed.get_vecs_by_tokens([query_token]), k+1)
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words.
        print('cosine sim=%.3f: %s' % (c, (embed.idx_to_token[i])))
```

The dictionary of pre-trained word vector instance `glove_6b50d` already created contains 400,000 words and a special unknown token. Excluding input words and unknown words, we search for the three words that are the most similar in meaning to "chip".

```{.python .input}
get_similar_tokens('chip', 3, glove_6b50d)
```

Next, we search for the synonyms of "baby" and "beautiful".

```{.python .input}
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input}
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Finding Analogies

In addition to seeking synonyms, we can also use the pre-trained word vector to seek the analogies between words. For example, “man”:“woman”::“son”:“daughter” is an example of analogy, “man” is to “woman” as “son” is to “daughter”. The problem of seeking analogies can be defined as follows: for four words in the analogical relationship $a : b :: c : d$, given the first three words, $a$, $b$ and $c$, we want to find $d$. Assume the word vector for the word $w$ is $\text{vec}(w)$. To solve the analogy problem, we need to find the word vector that is most similar to the result vector of $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input}
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed.get_vecs_by_tokens([token_a, token_b, token_c])
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[topk[0]]  # Remove unknown words.
```

Verify the "male-female" analogy.

```{.python .input  n=18}
get_analogy('man', 'woman', 'son', glove_6b50d)
```

“Capital-country” analogy: "beijing" is to "china" as "tokyo" is to what? The answer should be "japan".

```{.python .input  n=19}
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

"Adjective-superlative adjective" analogy: "bad" is to "worst" as "big" is to what? The answer should be "biggest".

```{.python .input  n=20}
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

"Present tense verb-past tense verb" analogy: "do" is to "did" as "go" is to what? The answer should be "went".

```{.python .input  n=21}
get_analogy('do', 'did', 'go', glove_6b50d)
```

## Summary

* Word vectors pre-trained on a large-scale corpus can often be applied to downstream natural language processing tasks.
* We can use pre-trained word vectors to seek synonyms and analogies.


## Problems

* Test the fastText results.
* If the dictionary is extremely large, how can we accelerate finding synonyms and analogies?




## Reference

[1] GluonNLP tool package. https://gluon-nlp.mxnet.io/

[2] GloVe project website. https://nlp.stanford.edu/projects/glove/

[3] fastText project website. https://fasttext.cc/

## Discuss on our Forum

<div id="discuss" topic_id="2390"></div>
