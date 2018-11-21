# Text Sentiment Classification: Using Recurrent Neural Networks

Text classification is a common task in natural language processing, which transforms a sequence of text of indefinite length into a category of text. This section will focus on one of the sub-questions in this field: using text sentiment classification to analyze the emotions of the text's author. This problem is also called sentiment analysis and has a wide range of applications. For example, we can analyze user reviews of products to obtain user satisfaction statistics, or analyze user sentiments about market conditions and use it to predict future trends.

Similar to search synonyms and analogies, text classification is also a downstream application of word embedding. In this section, we will apply pre-trained word vectors and bidirectional recurrent neural networks with multiple hidden layers. We will use them to determine whether a text sequence of indefinite length contains positive or negative emotion. Import the required package or module before starting the experiment.

```{.python .input  n=2}
import collections
import gluonbook as gb
from mxnet import gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn, utils as gutils
import os
import random
import tarfile
```

## Text Sentiment Classification Data

We use Stanford's Large Movie Review Dataset as the data set for text sentiment classification[1]. This data set is divided into two data sets for training and testing purposes, each containing 25,000 movie reviews downloaded from IMDb. In each data set, the number of comments labeled as "positive" and "negative" is equal.

###  Reading Data

We first download this data set to the "../data" path and extract it to "../data/aclImdb".

```{.python .input  n=3}
# This function is saved in the gluonbook package for future use.
def download_imdb(data_dir='../data'):
    url = ('http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
    sha1 = '01ada507287d82875905620988597833ad4e0903'
    fname = gutils.download(url, data_dir, sha1_hash=sha1)
    with tarfile.open(fname, 'r') as f:
        f.extractall(data_dir)

download_imdb()
```

Next, read the training and test data sets. Each example is a review and its corresponding label: 1 indicates "positive" and 0 indicates "negative".

```{.python .input  n=13}
def read_imdb(folder='train'):  # This function is saved in the gluonbook package for future use.
    data = []
    for label in ['pos', 'neg']:
        folder_name = os.path.join('../data/aclImdb/', folder, label)
        for file in os.listdir(folder_name):
            with open(os.path.join(folder_name, file), 'rb') as f:
                review = f.read().decode('utf-8').replace('\n', '').lower()
                data.append([review, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data

train_data, test_data = read_imdb('train'), read_imdb('test')
```

### Data Preprocessing

We need to segment each review to get a review with segmented words. The `get_tokenized_imdb` function defined here uses the easiest method: word tokenization based on spaces.

```{.python .input  n=14}
def get_tokenized_imdb(data):  # This function is saved in the gluonbook package for future use.
    def tokenizer(text):
        return [tok.lower() for tok in text.split(' ')]
    return [tokenizer(review) for review, _ in data]
```

Now, we can create a dictionary based on the training data set with the words segmented. Here, we have filtered out words that appear less than 5 times.

```{.python .input  n=28}
def get_vocab_imdb(data):  # This function is saved in the gluonbook package for future use.
    tokenized_data = get_tokenized_imdb(data)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    return text.vocab.Vocabulary(counter, min_freq=5)

vocab = get_vocab_imdb(train_data)
'# Words in vocab:', len(vocab)
```

Because the reviews have different lengths, so they cannot be directly combined into mini-batches, we define the `preprocess_imdb` function to segment each comment, convert it into a word index through a dictionary, and then fix the length of each comment to 500 by truncating or adding 0s.

```{.python .input  n=44}
def preprocess_imdb(data, vocab):  # This function is saved in the gluonbook package for future use.
    max_l = 500  # Make the length of each comment 500 by truncating or adding 0s.

    def pad(x):
        return x[:max_l] if len(x) > max_l else x + [0] * (max_l - len(x))

    tokenized_data = get_tokenized_imdb(data)
    features = nd.array([pad(vocab.to_indices(x)) for x in tokenized_data])
    labels = nd.array([score for _, score in data])
    return features, labels
```

### Create Data Iterator

Now, we will create a data iterator. Each iteration will return a mini-batch of data.

```{.python .input}
batch_size = 64
train_set = gdata.ArrayDataset(*preprocess_imdb(train_data, vocab))
test_set = gdata.ArrayDataset(*preprocess_imdb(test_data, vocab))
train_iter = gdata.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gdata.DataLoader(test_set, batch_size)
```

Print the shape of the first mini-batch of data and the number of mini-batches in the training set.

```{.python .input}
for X, y in train_iter:
    print('X', X.shape, 'y', y.shape)
    break
'#batches:', len(train_iter)
```

## Use a Recurrent Neural Network Model

In this model, each word first obtains a feature vector from the embedding layer. Then, we further encode the feature sequence using a bidirectional recurrent neural network to obtain sequence information. Finally, we transform the encoded sequence information to output through the fully connected layer. Specifically, we can concatenate hidden states of bidirectional long-short term memory in the initial time step and final time step and pass it to the output layer classification as encoded feature sequence information. In the `BiRNN` class implemented below, the `Embedding` instance is the embedding layer, the `LSTM` instance is the hidden layer for sequence encoding, and the `Dense` instance is the output layer for generated classification results.

```{.python .input  n=46}
class BiRNN(nn.Block):
    def __init__(self, vocab, embed_size, num_hiddens, num_layers, **kwargs):
        super(BiRNN, self).__init__(**kwargs)
        self.embedding = nn.Embedding(len(vocab), embed_size)
        # Set Bidirectional to True to get a bidirectional recurrent neural network.
        self.encoder = rnn.LSTM(num_hiddens, num_layers=num_layers,
                                bidirectional=True, input_size=embed_size)
        self.decoder = nn.Dense(2)

    def forward(self, inputs):
        # The shape of inputs is (batch size, number of words). Because LSTM needs to use sequence as the first dimension, the input is transformed
        # and the word feature is then extracted. The output shape is (number of words, batch size, word vector dimension).
        embeddings = self.embedding(inputs.T)
        # The shape of states is (number of words, batch size, 2 * number of hidden units).
        states = self.encoder(embeddings)
        # Concatenate the hidden states of the initial time step and final time step to use as the input of the fully connected layer. Its shape is (batch size,
        # 4 * number of hidden units).
        encoding = nd.concat(states[0], states[-1])
        outputs = self.decoder(encoding)
        return outputs
```

Create a bidirectional recurrent neural network with two hidden layers.

```{.python .input}
embed_size, num_hiddens, num_layers, ctx = 100, 100, 2, gb.try_all_gpus()
net = BiRNN(vocab, embed_size, num_hiddens, num_layers)
net.initialize(init.Xavier(), ctx=ctx)
```

### Load Pre-trained Word Vectors

Because the training data set for sentiment classification is not very large, in order to deal with overfitting, we will directly use word vectors pre-trained on a larger corpus as the feature vectors of all words. Here, we load a 100-dimensional GloVe word vector for each word in the dictionary `vocab`.

```{.python .input  n=45}
glove_embedding = text.embedding.create(
    'glove', pretrained_file_name='glove.6B.100d.txt', vocabulary=vocab)
```

Then, we will use these word vectors as feature vectors for each word in the reviews. Note that the dimensions of the pre-trained word vectors need to be consistent with the embedding layer output size `embed_size` in the created model. In addition, we no longer update these word vectors during training.

```{.python .input  n=47}
net.embedding.weight.set_data(glove_embedding.idx_to_vec)
net.embedding.collect_params().setattr('grad_req', 'null')
```

### Train and Evaluate the Model

Now, we can start training.

```{.python .input  n=48}
lr, num_epochs = 0.01, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gloss.SoftmaxCrossEntropyLoss()
gb.train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs)
```

Finally, define the prediction function.

```{.python .input  n=49}
# This function is saved in the gluonbook package for future use.
def predict_sentiment(net, vocab, sentence):
    sentence = nd.array(vocab.to_indices(sentence), ctx=gb.try_gpu())
    label = nd.argmax(net(sentence.reshape((1, -1))), axis=1)
    return 'positive' if label.asscalar() == 1 else 'negative'
```

Then, use the trained model to classify the sentiments of two simple sentences.

```{.python .input  n=50}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'great'])
```

```{.python .input}
predict_sentiment(net, vocab, ['this', 'movie', 'is', 'so', 'bad'])
```

## Summary

* Text classification transforms a sequence of text of indefinite length into a category of text. This is a downstream application of word embedding.
* We can apply pre-trained word vectors and recurrent neural networks to classify the emotions in a text.


## exercise

* Increase the number of epochs. What accuracy rate can you achieve on the training and testing data sets? What about trying to re-tune other hyper-parameters?

* Will using larger pre-trained word vectors, such as 300-dimensional GloVe word vectors, improve classification accuracy?

* Can we improve the classification accuracy by using the spaCy word tokenization tool? You need to install spaCy: `pip install spacy` and install the English package: `python -m spacy download en`. In the code, first import spacy: `import spacy`. Then, load the spacy English package: `spacy_en = spacy.load('en')`. Finally, define the function `def tokenizer(text): return [tok.text for tok in spacy_en.tokenizer(text)]` and replace the original `tokenizer` function. It should be noted that GloVe's word vector uses "-" to connect each word when storing noun phrases. For example, the phrase "new york" is represented as "new-york" in GloVe. After using spaCy tokenization, "new york" may be stored as "new york".


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/6155)


![](../img/qr_sentiment-analysis.svg)


## Reference

[1] Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.
