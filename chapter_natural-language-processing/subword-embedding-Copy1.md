# Subword Embedding
:label:`sec_fasttext`

English words usually have internal structures and formation methods. For example, we can deduce the relationship between "dog", "dogs", and "dogcatcher" by their spelling. All these words have the same root, "dog", but they use different suffixes to change the meaning of the word. Moreover, this association can be extended to other words. For example, the relationship between "dog" and "dogs" is just like the relationship between "cat" and "cats". The relationship between "boy" and "boyfriend" is just like the relationship between "girl" and "girlfriend". This characteristic is not unique to English. In French and Spanish, a lot of verbs can have more than 40 different forms depending on the context. In Finnish, a noun may have more than 15 forms. In fact, morphology, which is an important branch of linguistics, studies the internal structure and formation of words.


## fastText

In word2vec, we did not directly use morphology information.  In both the
skip-gram model and continuous bag-of-words model, we use different vectors to
represent words with different forms. For example, "dog" and "dogs" are
represented by two different vectors, while the relationship between these two
vectors is not directly represented in the model. In view of this, fastText :cite:`Bojanowski.Grave.Joulin.ea.2017`
proposes the method of subword embedding, thereby attempting to introduce
morphological information in the skip-gram model in word2vec.

In fastText, each central word is represented as a collection of subwords. Below we use the word "where" as an example to understand how subwords are formed. First, we add the special characters “&lt;” and “&gt;” at the beginning and end of the word to distinguish the subwords used as prefixes and suffixes. Then, we treat the word as a sequence of characters to extract the $n$-grams. For example, when $n=3$, we can get all subwords with a length of $3$:

$$\textrm{"<wh"}, \ \textrm{"whe"}, \ \textrm{"her"}, \ \textrm{"ere"}, \ \textrm{"re>"},$$

and the special subword $\textrm{"<where>"}$.

In fastText, for a word $w$, we record the union of all its subwords with length of $3$ to $6$ and special subwords as $\mathcal{G}_w$. Thus, the dictionary is the union of the collection of subwords of all words. Assume the vector of the subword $g$ in the dictionary is $\mathbf{z}_g$. Then, the central word vector $\mathbf{u}_w$ for the word $w$ in the skip-gram model can be expressed as

$$\mathbf{u}_w = \sum_{g\in\mathcal{G}_w} \mathbf{z}_g.$$

The rest of the fastText process is consistent with the skip-gram model, so it is not repeated here. As we can see, compared with the skip-gram model, the dictionary in fastText is larger, resulting in more model parameters. Also, the vector of one word requires the summation of all subword vectors, which results in higher computation complexity. However, we can obtain better vectors for more uncommon complex words, even words not existing in the dictionary, by looking at other words with similar structures.


## Byte Pair Encoding

In fastText, all the extracted subwords have to be of the specified lengths, such as $3$ to $6$, thus the vocabulary size cannot be predefined.
To allow for variable-length subwords in a fixed-size vocabulary,
we can apply a compression algorithm
called *byte pair encoding* (BPE) to extract subwords :cite:`Sennrich.Haddow.Birch.2015`.

```{.python .input  n=1}
import re, collections

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out

vocab = {'l o w </w>' : 5, 'l o w e r </w>' : 2,
         'n e w e s t </w>':6, 'w i d e s t </w>':3}
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
```

```{.json .output n=1}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "('e', 's')\n('es', 't')\n('est', '</w>')\n('l', 'o')\n('lo', 'w')\n('n', 'e')\n('ne', 'w')\n('new', 'est</w>')\n('low', '</w>')\n('w', 'i')\n"
 }
]
```

```{.python .input  n=2}
original_words = {'low_': 5, 'lower_': 2, 'newest_': 6, 'widest_': 3}
words = {}
for word, freq in original_words.items():
    new_word = ' '.join(list(word))
    words[new_word] = original_words[word]
words
```

```{.json .output n=2}
[
 {
  "data": {
   "text/plain": "{'l o w _': 5, 'l o w e r _': 2, 'n e w e s t _': 6, 'w i d e s t _': 3}"
  },
  "execution_count": 2,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=3}
def get_max_freq_pair(words):
    pairs = collections.defaultdict(int)
    for word, freq in words.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # Key of pairs is a tuple composed of two adjacent units
            pairs[symbols[i], symbols[i + 1]] += freq
    return max(pairs, key=pairs.get)  # Key of pairs with the max value
```

```{.python .input  n=4}
def merge_vocab(max_freq_pair, words, symbols):
    bigram = ' '.join(max_freq_pair)
    symbols.append(''.join(max_freq_pair))
    words_out = {}
    for word, freq in words.items():
        new_word = word.replace(bigram, ''.join(max_freq_pair))
        words_out[new_word] = words[word]
    return words_out
```

```{.python .input  n=5}
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(words)
    words = merge_vocab(max_freq_pair, words, symbols)
    print("Merge #%d:" % (i + 1), max_freq_pair)
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Merge #1: ('e', 's')\nMerge #2: ('es', 't')\nMerge #3: ('est', '_')\nMerge #4: ('l', 'o')\nMerge #5: ('lo', 'w')\nMerge #6: ('n', 'e')\nMerge #7: ('ne', 'w')\nMerge #8: ('new', 'est_')\nMerge #9: ('low', '_')\nMerge #10: ('w', 'i')\n"
 }
]
```

```{.python .input  n=6}
print("Words:", list(original_words.keys()))
print("Wordpieces:", list(words.keys()))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Words: ['low_', 'lower_', 'newest_', 'widest_']\nWordpieces: ['low_', 'low e r _', 'newest_', 'wi d est_']\n"
 }
]
```

```{.python .input  n=7}
print("Symbols:", symbols)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Symbols: ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '[UNK]', 'es', 'est', 'est_', 'lo', 'low', 'ne', 'new', 'newest_', 'low_', 'wi']\n"
 }
]
```

```{.python .input  n=8}
inputs = ['slow_', 'slowest_']
outputs = []
for word in inputs:
    start, end = 0, len(word)
    cur_output = []
    while start < len(word) and start < end:
        if word[start : end] in symbols:
            cur_output.append(word[start : end])
            start = end
            end = len(word)
        else:
            end -= 1
    if start < len(word):
        cur_output.append('[UNK]')
    outputs.append(' '.join(cur_output))
print('Words:', inputs)
print('Wordpieces:', outputs)
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Words: ['slow_', 'slowest_']\nWordpieces: ['s low_', 's low est_']\n"
 }
]
```

## Summary

* FastText proposes a subword embedding method. Based on the skip-gram model in word2vec, it represents the central word vector as the sum of the subword vectors of the word.
* Subword embedding utilizes the principles of morphology, which usually improves the quality of representations of uncommon words.


## Exercises

1. When there are too many subwords (for example, 6 words in English result in about $3\times 10^8$ combinations), what problems arise? Can you think of any methods to solve them? Hint: Refer to the end of section 3.2 of the fastText paper[1].
1. How can you design a subword embedding model based on the continuous bag-of-words model?



## [Discussions](https://discuss.mxnet.io/t/2388)

![](../img/qr_subword-embedding.svg)
