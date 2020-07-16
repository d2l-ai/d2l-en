# Finding Synonyms and Analogies
:label:`sec_synonyms`

In :numref:`sec_word2vec_gluon` we trained a word2vec word embedding model
on a small-scale dataset and searched for synonyms using the cosine similarity
of word vectors. In practice, word vectors pretrained on a large-scale corpus
can often be applied to downstream natural language processing tasks. This
section will demonstrate how to use these pretrained word vectors to find
synonyms and analogies. We will continue to apply pretrained word vectors in
subsequent sections.

```{.python .input  n=1}
from d2l import mxnet as d2l
import matplotlib.pyplot as plt
from mxnet import np, npx
import numpy
import os
from sklearn.decomposition import PCA

npx.set_np()
```

## Using Pretrained Word Vectors

Below lists pretrained GloVe embeddings of dimensions 50, 100, and 300,
which can be downloaded from the [GloVe website](https://nlp.stanford.edu/projects/glove/).
The pretrained fastText embeddings are available in multiple languages.
Here we consider one English version (300-dimensional "wiki.en") that can be downloaded from the
[fastText website](https://fasttext.cc/).

```{.python .input  n=2}
#@save
d2l.DATA_HUB['glove.6b.50d'] = (d2l.DATA_URL + 'glove.6B.50d.zip',
                                '0b8703943ccdb6eb788e6f091b8946e82231bc4d')

#@save
d2l.DATA_HUB['glove.6b.100d'] = (d2l.DATA_URL + 'glove.6B.100d.zip',
                                 'cd43bfb07e44e6f27cbcc7bc9ae3d80284fdaf5a')

#@save
d2l.DATA_HUB['glove.42b.300d'] = (d2l.DATA_URL + 'glove.42B.300d.zip',
                                  'b5116e234e9eb9076672cfeabf5469f3eec904fa')

#@save
d2l.DATA_HUB['wiki.en'] = (d2l.DATA_URL + 'wiki.en.zip',
                           'c1816da3821ae9f43899be655002f6c723e91b88')
```

We define the following `TokenEmbedding` class to load the above pretrained Glove and fastText embeddings.

```{.python .input  n=3}
#@save
class TokenEmbedding:
    """Token Embedding."""
    def __init__(self, embedding_name):
        self.idx_to_token, self.idx_to_vec = self._load_embedding(
            embedding_name)
        self.unknown_idx = 0
        self.token_to_idx = {token: idx for idx, token in 
                             enumerate(self.idx_to_token)}

    def _load_embedding(self, embedding_name):
        idx_to_token, idx_to_vec = ['<unk>'], []
        data_dir = d2l.download_extract(embedding_name)
        # GloVe website: https://nlp.stanford.edu/projects/glove/
        # fastText website: https://fasttext.cc/
        with open(os.path.join(data_dir, 'vec.txt'), 'r') as f:
            for line in f:
                elems = line.rstrip().split(' ')
                token, elems = elems[0], [float(elem) for elem in elems[1:]]
                # Skip header information, such as the top row in fastText
                if len(elems) > 1:
                    idx_to_token.append(token)
                    idx_to_vec.append(elems)
        idx_to_vec = [[0] * len(idx_to_vec[0])] + idx_to_vec
        return idx_to_token, np.array(idx_to_vec)

    def __getitem__(self, tokens):
        indices = [self.token_to_idx.get(token, self.unknown_idx)
                   for token in tokens]
        vecs = self.idx_to_vec[np.array(indices)]
        return vecs

    def __len__(self):
        return len(self.idx_to_token)
```

Next, we use 50-dimensional GloVe embeddings pretrained on a subset of the Wikipedia. The corresponding word embedding is automatically downloaded the first time we create a pretrained word embedding instance.

```{.python .input  n=4}
glove_6b50d = TokenEmbedding('glove.6b.50d')
```

Output the dictionary size. The dictionary contains $400,000$ words and a special unknown token.

```{.python .input  n=5}
len(glove_6b50d)
```

We can use a word to get its index in the dictionary, or we can get the word from its index.

```{.python .input  n=6}
glove_6b50d.token_to_idx['beautiful'], glove_6b50d.idx_to_token[3367]
```

## Applying Pretrained Word Vectors

Below, we demonstrate the application of pretrained word vectors, using GloVe as an example.

### Finding Synonyms

Here, we re-implement the algorithm used to search for synonyms by cosine
similarity introduced in :numref:`sec_word2vec`

In order to reuse the logic for seeking the $k$ nearest neighbors when
seeking analogies, we encapsulate this part of the logic separately in the `knn`
($k$-nearest neighbors) function.

```{.python .input  n=7}
def knn(W, x, k):
    # The added 1e-9 is for numerical stability
    cos = np.dot(W, x.reshape(-1,)) / (
        np.sqrt(np.sum(W * W, axis=1) + 1e-9) * np.sqrt((x * x).sum()))
    topk = npx.topk(cos, k=k, ret_typ='indices')
    return topk, [cos[int(i)] for i in topk]
```

Then, we search for synonyms by pre-training the word vector instance `embed`.

```{.python .input  n=8}
def get_similar_tokens(query_token, k, embed):
    topk, cos = knn(embed.idx_to_vec, embed[[query_token]], k + 1)
    for i, c in zip(topk[1:], cos[1:]):  # Remove input words
        print(f'cosine sim={float(c):.3f}: {embed.idx_to_token[int(i)]}')
```

The dictionary of pretrained word vector instance `glove_6b50d` already created contains 400,000 words and a special unknown token. Excluding input words and unknown words, we search for the three words that are the most similar in meaning to "chip".

```{.python .input  n=9}
get_similar_tokens('chip', 3, glove_6b50d)
```

Next, we search for the synonyms of "baby" and "beautiful".

```{.python .input  n=10}
get_similar_tokens('baby', 3, glove_6b50d)
```

```{.python .input  n=11}
get_similar_tokens('beautiful', 3, glove_6b50d)
```

### Finding Analogies

In addition to seeking synonyms, we can also use the pretrained word vector to seek the analogies between words. For example, “man”:“woman”::“son”:“daughter” is an example of analogy, “man” is to “woman” as “son” is to “daughter”. The problem of seeking analogies can be defined as follows: for four words in the analogical relationship $a : b :: c : d$, given the first three words, $a$, $b$ and $c$, we want to find $d$. Assume the word vector for the word $w$ is $\text{vec}(w)$. To solve the analogy problem, we need to find the word vector that is most similar to the result vector of $\text{vec}(c)+\text{vec}(b)-\text{vec}(a)$.

```{.python .input  n=12}
def get_analogy(token_a, token_b, token_c, embed):
    vecs = embed[[token_a, token_b, token_c]]
    x = vecs[1] - vecs[0] + vecs[2]
    topk, cos = knn(embed.idx_to_vec, x, 1)
    return embed.idx_to_token[int(topk[0])]  # Remove unknown words
```

Verify the "male-female" analogy.

```{.python .input  n=13}
get_analogy('man', 'woman', 'son', glove_6b50d)
```

“Capital-country” analogy: "beijing" is to "china" as "tokyo" is to what? The answer should be "japan".

```{.python .input  n=14}
get_analogy('beijing', 'china', 'tokyo', glove_6b50d)
```

"Adjective-superlative adjective" analogy: "bad" is to "worst" as "big" is to what? The answer should be "biggest".

```{.python .input  n=15}
get_analogy('bad', 'worst', 'big', glove_6b50d)
```

"Present tense verb-past tense verb" analogy: "do" is to "did" as "go" is to what? The answer should be "went".

```{.python .input  n=16}
get_analogy('do', 'did', 'go', glove_6b50d)
```

```{.python .input  n=51}
def visualization(token_pairs, embed):
    plt.figure(figsize=(7, 5))
    vecs = np.concatenate([embed[pair] for pair in token_pairs])
    vecs_pca = PCA(n_components=2).fit_transform(numpy.array(vecs))
    for i, pair in enumerate(token_pairs):
        x1, y1 = vecs_pca[2 * i]
        x2, y2 = vecs_pca[2 * i + 1]
        plt.scatter(x1, y1)
        plt.scatter(x2, y2)
        plt.annotate(pair[0], xy=(x1, y1))
        plt.annotate(pair[1], xy=(x2, y2))
        plt.plot([x1, x2], [y1, y2])
    plt.show()
```

```{.python .input  n=57}
token_pairs = [['man', 'woman'], ['son', 'daughter'], ['king', 'queen'],
              ['uncle', 'aunt'], ['sir', 'madam'], ['sister', 'brother']]
visualization(token_pairs, glove_6b50d)
```

## Summary

* Word vectors pre-trained on a large-scale corpus can often be applied to downstream natural language processing tasks.
* We can use pre-trained word vectors to seek synonyms and analogies.


## Exercises

1. Test the fastText results using `TokenEmbedding('wiki.en')`.
1. If the dictionary is extremely large, how can we accelerate finding synonyms and analogies?


## [Discussions](https://discuss.mxnet.io/t/2390)

![](../img/qr_similarity-analogy.svg)
