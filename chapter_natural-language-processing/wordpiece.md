# Byte-Pair Embedding (WordPiece)

...

```{.python .input  n=1}
import collections

vocabs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
          'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '_', '[UNK]']
```

...

```{.python .input  n=2}
original_words = {'low_' : 5, 'lower_' : 2, 'newest_' : 6, 'widest_' : 3}
words = {}
for word, freq in original_words.items():
    new_word = ' '.join(list(word))
    words[new_word] = original_words[word]
```

...

```{.python .input  n=3}
def get_max_freq_pair(words):
    pairs = collections.defaultdict(int)
    for word, freq in words.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            # Occurrence of each adjacent unit
            pairs[symbols[i], symbols[i + 1]] += freq
    max_freq_pair = max(pairs, key = pairs.get)
    return max_freq_pair
```

...

```{.python .input  n=4}
def merge_vocab(max_freq_pair, words, vocabs):
    bigram = ' '.join(max_freq_pair)
    vocabs.append(''.join(max_freq_pair))
    words_out = {}
    for word, freq in words.items():
        new_word = word.replace(bigram, ''.join(max_freq_pair))
        words_out[new_word] = words[word]
    return words_out
```

...

```{.python .input  n=5}
num_merges = 10
for i in range(num_merges):
    max_freq_pair = get_max_freq_pair(words)
    words = merge_vocab(max_freq_pair, words, vocabs)
    print("Merge #%d:" % (i + 1), max_freq_pair)
```

...

```{.python .input  n=6}
print("Words:", list(original_words.keys()))
print("Wordpieces:", list(words.keys()))
```

...

```{.python .input  n=7}
print("Vocabs:", vocabs)
```

...

```{.python .input  n=8}
inputs = ['slow_', 'slowest_']
outputs = []
for word in inputs:
    start, end = 0, len(word)
    cur_output = []
    while start < len(word) and start < end:
        if word[start : end] in vocabs:
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

## Summary
...

## References

[1] Wu, Y., Schuster, M., Chen, Z., Le, Q. V., Norouzi, M., Macherey, W., ... & Klingner, J. (2016). Google's neural machine translation system: Bridging the gap between human and machine translation. arXiv preprint arXiv:1609.08144.

[2] Sennrich, R., Haddow, B., & Birch, A. (2015). Neural machine translation of rare words with subword units. arXiv preprint arXiv:1508.07909.
