# Text Classification and Data Sets
:label:`chapter_sentiment`

Text classification is a common task in natural language processing, which transforms a sequence of text of indefinite length into a category of text. It's similar to the image classification, the most frequently used application in this book, e.g. :numref:`chapter_naive_bayes`. The only difference is that, rather than an image, text classification's example is a text sentence. 

This section will focus on loading data for one of the sub-questions in this field: using text sentiment classification to analyze the emotions of the text's author. This problem is also called sentiment analysis and has a wide range of applications. For example, we can analyze user reviews of products to obtain user satisfaction statistics, or analyze user sentiments about market conditions and use it to predict future trends.

```{.python .input  n=2}
import d2l
from mxnet import gluon, nd
import os
import tarfile
```

## Text Sentiment Classification Data

We use Stanford's Large Movie Review Dataset as the data set for text sentiment classification[1]. This data set is divided into two data sets for training and testing purposes, each containing 25,000 movie reviews downloaded from IMDb. In each data set, the number of comments labeled as "positive" and "negative" is equal.

###  Reading Data

We first download this data set to the "../data" path and extract it to "../data/aclImdb".

```{.python .input  n=23}
# Save to the d2l package.
def download_imdb(data_dir='../data'):
    url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
    fname = gluon.utils.download(url, data_dir)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)
        
download_imdb()
```

Next, read the training and test data sets. Each example is a review and its corresponding label: 1 indicates "positive" and 0 indicates "negative".

```{.python .input  n=24}
# Save to the d2l package.
def read_imdb(folder='train', data_dir='../data'):
    data, labels = [], []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_dir, 'aclImdb', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data = read_imdb('train')
print('# trainings:', len(train_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

### Tokenization and Vocabulary

We use a word as a token, and then create a dictionary based on the training data set.

```{.python .input  n=28}
train_tokens = d2l.tokenize(train_data[0], token='word')
vocab = d2l.Vocab(train_tokens, min_freq=5)

d2l.set_figsize((3.5, 2.5))
d2l.plt.hist([len(line) for line in train_tokens], bins=range(0,1000,50));
```

### Padding to the Same Length

Because the reviews have different lengths, so they cannot be directly combined into mini-batches. Here we fix the length of each comment to 500 by truncating or adding "&lt;unk&gt;" indices.

```{.python .input  n=44}
num_steps = 500  # sequence length
train_features = nd.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk) 
                          for line in train_tokens])
train_features.shape
```

### Create Data Iterator

Now, we will create a data iterator. Each iteration will return a mini-batch of data.

```{.python .input}
train_iter = d2l.load_array((train_features, train_data[1]), 64)

for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'# batches:', len(train_iter)
```

## Put All Things Together

Lastly, we will save a function `load_data_imdb` into `d2l`, which returns the vocabulary and data iterators.

```{.python .input}
# Save to the d2l package.
def load_data_imdb(batch_size, num_steps=500):
    download_imdb()
    train_data, test_data = read_imdb('train'), read_imdb('test')
    train_tokens = d2l.tokenize(train_data[0], token='word')
    test_tokens = d2l.tokenize(test_data[0], token='word')
    vocab = d2l.Vocab(train_tokens, min_freq=5)
    train_features = nd.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk) 
                               for line in train_tokens])
    test_features = nd.array([d2l.trim_pad(vocab[line], num_steps, vocab.unk) 
                               for line in test_tokens])
    train_iter = d2l.load_array((train_features, train_data[1]), batch_size)
    test_iter = d2l.load_array((test_features, test_data[1]), batch_size, 
                               is_train=False)
    return train_iter, test_iter, vocab
```

## Summary
