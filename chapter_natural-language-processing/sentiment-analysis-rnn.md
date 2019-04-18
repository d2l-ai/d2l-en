# Text Sentiment Classification: Using Recurrent Neural Networks

Text classification is a common task in natural language processing, which transforms a sequence of text of indefinite length into a category of text. This section will focus on one of the sub-questions in this field: using text sentiment classification to analyze the emotions of the text's author. This problem is also called sentiment analysis and has a wide range of applications. For example, we can analyze user reviews of products to obtain user satisfaction statistics, or analyze user sentiments about market conditions and use it to predict future trends.

Similar to search synonyms and analogies, text classification is also a downstream application of word embedding. In this section, we will apply pre-trained word vectors and bidirectional recurrent neural networks with multiple hidden layers. We will use them to determine whether a text sequence of indefinite length contains positive or negative emotion. Import the required package or module before starting the experiment.

```{.python .input  n=2}
import sys
sys.path.insert(0, '..')

import d2l
from mxnet import gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
from mxnet.contrib import text
import os
import tarfile
```

## Text Sentiment Classification Data

We use Stanford's Large Movie Review Dataset as the data set for text sentiment classification[1]. This data set is divided into two data sets for training and testing purposes, each containing 25,000 movie reviews downloaded from IMDb. In each data set, the number of comments labeled as "positive" and "negative" is equal.

###  Reading Data

We first download this data set to the "../data" path and extract it to "../data/aclImdb".

```{.python .input  n=23}
data_dir = './'
url = 'http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'
fname = gutils.download(url, data_dir)
with tarfile.open(fname, 'r') as f:
    f.extractall(data_dir)
```

Next, read the training and test data sets. Each example is a review and its corresponding label: 1 indicates "positive" and 0 indicates "negative".

```{.python .input  n=24}
def read_imdb(folder='train'):
    data, labels = [], []
    for label in ['pos', 'neg']:
        folder_name = os.path.join(data_dir, 'aclImdb', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '')
                data.append(review)
                labels.append(1 if label == 'pos' else 0)
    return data, labels

train_data, test_data = read_imdb('train'), read_imdb('test')
print('# trainings:', len(train_data[0]), '\n# tests:', len(test_data[0]))
for x, y in zip(train_data[0][:3], train_data[1][:3]):
    print('label:', y, 'review:', x[0:60])
```

### Tokenization and Vocabulary 

We use a word as a token, which can be split based on spaces.

```{.python .input  n=28}
def tokenize(sentences):
    return [line.split(' ') for line in sentences]

train_tokens = tokenize(train_data[0])
test_tokens = tokenize(test_data[0])
```

Then we can create a dictionary based on the training data set with the words segmented. 
Here, we have filtered out words that appear less than 5 times.

```{.python .input}
vocab = d2l.Vocab([tk for line in train_tokens for tk in line], min_freq=5)
```

### Padding to the Same Length

Because the reviews have different lengths, so they cannot be directly combined into mini-batches. Here we fix the length of each comment to 500 by truncating or adding "&lt;unk&gt;" indices.

```{.python .input  n=44}
max_len = 500

def pad(x):
    if len(x) > max_len:        
        return x[:max_len]
    else:
        return x + [vocab.unk] * (max_len - len(x))
    
train_features = nd.array([pad(vocab[line]) for line in train_tokens])
test_features = nd.array([pad(vocab[line]) for line in test_tokens])
```

### Create Data Iterator

Now, we will create a data iterator. Each iteration will return a mini-batch of data.

```{.python .input}
batch_size = 64
train_set = gdata.ArrayDataset(train_features, train_data[1])
test_set = gdata.ArrayDataset(test_features, test_data[1])
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)
```

Print the shape of the first mini-batch of data and the number of mini-batches in the training set.

```{.python .input}
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'# batches:', len(train_iter)
```

Lastly, we will save a function `get_data_imdb` into `d2l`, which returns the vocabulary and data iterators. 

## Use a Recurrent Neural Network Model

In this model, each word first obtains a feature vector from the embedding layer. Then, we further encode the feature sequence using a bidirectional recurrent neural network to obtain sequence information. Finally, we transform the encoded sequence information to output through the fully connected layer. Specifically, we can concatenate hidden states of bidirectional long-short term memory in the initial time step and final time step and pass it to the output layer classification as encoded feature sequence information. In the `BiRNN` class implemented below, the `Embedding` instance is the embedding layer, the `LSTM` instance is the hidden layer for sequence encoding, and the `Dense` instance is the output layer for generated classification results.

```{.python .input  n=46}
class BiRNN(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # Set Bidirectional to True to get a bidirectional recurrent neural
        # network
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of inputs is (batch size, number of words). Because LSTM
        # needs to use sequence as the first dimension, the input is
        # transformed and the word feature is then extracted. The output shape
        # is (number of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # The shape of states is (number of words, batch size, 2 * number of
        # hidden units).
        states = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final
        # time step to use as the input of the fully connected layer. Its
        # shape is (batch size, 4 * number of hidden units)
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs
```

Create a bidirectional recurrent neural network with two hidden layers.

```{.python .input}
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, d2l.try_all_gpus()
net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)
```

### Load Pre-trained Word Vectors

Because the training data set for sentiment classification is not very large, in order to deal with overfitting, we will directly use word vectors pre-trained on a larger corpus as the feature vectors of all words. Here, we load a 100-dimensional GloVe word vector for each word in the dictionary `vocab`.

```{.python .input}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt')
```

Query the word vectors that in our vocabulary.

```{.python .input}
embeds = glove_embedding.get_vecs_by_tokens(vocab.idx_to_token)
embeds.shape
```

Then, we will use these word vectors as feature vectors for each word in the reviews. Note that the dimensions of the pre-trained word vectors need to be consistent with the embedding layer output size `embed_size` in the created model. In addition, we no longer update these word vectors during training.

```{.python .input  n=47}
net.embedding.weight.set_data(embeds)
net.embedding.collect_params().setattr('grad_req', 'null')
```

### Train and Evaluate the Model

Now, we can start training.

```{.python .input  n=48}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
d2l.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

Finally, define the prediction function.

```{.python .input  n=49}
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab[sentence.split()], ctx=d2l.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'
```

Then, use the trained model to classify the sentiments of two simple sentences.

```{.python .input  n=50}
predict_sentiment(net, vocab, 'this movie is so great')
```

```{.python .input}
predict_sentiment(net, vocab, 'this movie is so bad')
```

## Summary

* Text classification transforms a sequence of text of indefinite length into a category of text. This is a downstream application of word embedding.
* We can apply pre-trained word vectors and recurrent neural networks to classify the emotions in a text.


## Exercises

* Increase the number of epochs. What accuracy rate can you achieve on the training and testing data sets? What about trying to re-tune other hyper-parameters?

* Will using larger pre-trained word vectors, such as 300-dimensional GloVe word vectors, improve classification accuracy?

* Can we improve the classification accuracy by using the spaCy word tokenization tool? You need to install spaCy: `pip install spacy` and install the English package: `python -m spacy download en`. In the code, first import spacy: `import spacy`. Then, load the spacy English package: `spacy_en = spacy.load('en')`. Finally, define the function `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` and replace the original `tokenizer` function. It should be noted that GloVe's word vector uses "-" to connect each word when storing noun phrases. For example, the phrase "new york" is represented as "new-york" in GloVe. After using spaCy tokenization, "new york" may be stored as "new york".





## Reference

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2391)

![](../img/qr_sentiment-analysis-rnn.svg)
