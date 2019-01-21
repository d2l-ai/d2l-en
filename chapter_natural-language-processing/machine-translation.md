# Machine Translation

Machine translation refers to the automatic translation of a segment of text from one language to another. Because a sequence of texts does not necessarily retain the same length in different languages, we use machine translation as an example to introduce the applications of the encoder-decoder and attention mechanism.

## Read and Pre-process Data

We will define some special symbols first. The “&lt;pad&gt;” (padding) symbol is added after a shorter sequence until each sequence is equal in length and the “&lt;bos&gt;” and “&lt;eos&gt;” symbols indicate the beginning and end of the sequence.

```{.python .input  n=2}
import collections
import io
import math
from mxnet import autograd, gluon, init, nd
from mxnet.contrib import text
from mxnet.gluon import data as gdata, loss as gloss, nn, rnn

PAD, BOS, EOS = '<pad>', '<bos>', '<eos>'
```

Then, we define two auxiliary functions to preprocess the data to be read later.

```{.python .input}
# For a sequence, we record all the words in all_tokens in order to subsequently construct the dictionary, then we add PAD after the sequence, until
# the length becomes max_seq_len. Then, we record the sequence in all_seqs.
def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    all_tokens.extend(seq_tokens)
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)
    all_seqs.append(seq_tokens)

# Use all the words to construct a dictionary. Construct an NDArray instance after transforming the words in all sequences into a word index.
def build_data(all_tokens, all_seqs):
    vocab = text.vocab.Vocabulary(collections.Counter(all_tokens),
                                  reserved_tokens=[PAD, BOS, EOS])
    indices = [vocab.to_indices(seq) for seq in all_seqs]
    return vocab, nd.array(indices)
```

For simplicity, we use a very small French-English data set here. In this data set, each line is a French sentence and its corresponding English sentence, separated by `'\t'`. When reading data, we attach the “&lt;eos&gt;” symbol at the end of the sentence, and if necessary, make the length of each sequence `max_seq_len` by adding the “&lt;pad&gt;” symbol. We create separate dictionaries for French and English words. The index of French words and the index of the English words are independent of each other.

```{.python .input  n=31}
def read_data(max_seq_len):
    # In and out are the abbreviations of input and output, respectively.
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('../data/fr-en-small.txt') as f:
        lines = f.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:
            continue  # If a sequence is longer than the max_seq_len after adding EOS, this example will be ignored.
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, gdata.ArrayDataset(in_data, out_data)
```

Set the maximum length of the sequence to 7, then review the first example read. The example contains a French word index sequence and an English word index sequence.

```{.python .input  n=181}
max_seq_len = 7
in_vocab, out_vocab, dataset = read_data(max_seq_len)
dataset[0]
```

## Encoder-Decoder with Attention Mechanism

We will use an encoder-decoder with an attention mechanism to translate a short French paragraph into English. Next, we will show how to implement the model.

### Encoder

In the encoder, we use the word embedding layer to obtain a feature index from the word index of the input language and then input it into a multi-level gated recurrent unit. As we mentioned in the ["Gluon implementation of the recurrent neural network"](../chapter_recurrent-neural-networks/rnn-gluon.md) section, Gluon's `rnn.GRU` instance also returns the multi-layer hidden states of the output and final time steps after forward calculation. Here, the output refers to the hidden state of the hidden layer of the last layer at each time step, and it does not involve output layer calculation. The attention mechanism uses these output as key items and value items.

```{.python .input  n=165}
class Encoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 drop_prob=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        # The input shape is (batch size, number of time steps). Change the example dimension and time step dimension of the output.
        embedding = self.embedding(inputs).swapaxes(0, 1)
        return self.rnn(embedding, state)

    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```

Next, we will create a mini-batch sequence input with a batch size of 4 and 7 time steps. We assume the number of hidden layers of the gated recurrent unit is 2 and the number of hidden units is 16. The output shape returned by the encoder after performing forward calculation on the input is (number of time steps, batch size, number of hidden units). The shape of the multi-layer hidden state of the gated recurrent unit in the final time step is (number of hidden layers, batch size, number of hidden units). For the gated recurrent unit, the `state` list contains only one element, which is the hidden state. If long short-term memory is used, the `state` list will also contain another element, which is the memory cell.

```{.python .input  n=166}
encoder = Encoder(vocab_size=10, embed_size=8, num_hiddens=16, num_layers=2)
encoder.initialize()
output, state = encoder(nd.zeros((4, 7)), encoder.begin_state(batch_size=4))
output.shape, state[0].shape
```

### Attention Mechanism

Before we introduce how to implement vectorization calculation for the attention mechanism, we will take a look at the `flatten` option for a `Dense` instance. When the input dimension is greater than 2, by default, the `Dense` instance will treat all dimensions other than the first dimension (example dimension) as feature dimensions that require affine transformation, and will automatically convert the input into a two-dimensional matrix with rows of behavioral examples and columns of features. After calculation, the shape of the output matrix is (number of examples, number of outputs). If we want the fully connected layer to only perform affine transformation on the last dimension of the input while keeping the shapes of the other dimensions unchanged, we need to set the `flatten` option of the `Dense` instance to `False`. In the following example, the fully connected layer only performs affine transformation on the last dimension of the input, therefore, only the last dimension of the output shape becomes the number of outputs of the fully connected layer, i.e. 2.

```{.python .input}
dense = nn.Dense(2, flatten=False)
dense.initialize()
dense(nd.zeros((3, 5, 7))).shape
```

We will implement the function $a$ defined in the ["Attention Mechanism"](./attention.md) section to transform the concatenated input through a multilayer perceptron with a single hidden layer. The input of the hidden layer is a one-to-one concatenation between the hidden state of the decoder and the hidden state of the encoder on all time steps, which uses tanh as the activation function. The number of outputs of the output layer is 1. Neither of the 2 `Dense` instances use a bias or flatten. Here, the length of the vector $\boldsymbol{v}$ in the $a$ function definition is a hyper-parameter, i.e. `attention_size`.

```{.python .input  n=167}
def attention_model(attention_size):
    model = nn.Sequential()
    model.add(nn.Dense(attention_size, activation='tanh', use_bias=False,
                       flatten=False),
              nn.Dense(1, use_bias=False, flatten=False))
    return model
```

The inputs of the attention model include query items, key items, and value items. Assume the encoder and decoder have the same number of hidden units. The query item here is the hidden state of the decoder in the previous time step, with a shape of (batch size, number of hidden units); the key and the value items are the hidden states of the encoder at all time steps, with a shape of (number of time steps, batch size, number of hidden units). The attention model returns the context variable of the current time step, and the shape is (batch size, number of hidden units).

```{.python .input  n=168}
def attention_forward(model, enc_states, dec_state):
    # Broadcast the decoder hidden state to the same shape as the encoder hidden state and then perform concatenation.
    dec_states = nd.broadcast_axis(
        dec_state.expand_dims(0), axis=0, size=enc_states.shape[0])
    enc_and_dec_states = nd.concat(enc_states, dec_states, dim=2)
    e = model(enc_and_dec_states)  # The shape is (number of time steps, batch size, 1).
    alpha = nd.softmax(e, axis=0)  # Perform the softmax operation on the time step dimension.
    return (alpha * enc_states).sum(axis=0)  # This returns the context variable.
```

In the example below, the encoder has 10 time steps and a batch size of 4. Both the encoder and the decoder have 8 hidden units. The attention model returns a mini-batch of context vectors, and the length of each context vector is equal to the number of hidden units of the encoder. Therefore, the output shape is (4, 8).

```{.python .input  n=169}
seq_len, batch_size, num_hiddens = 10, 4, 8
model = attention_model(10)
model.initialize()
enc_states = nd.zeros((seq_len, batch_size, num_hiddens))
dec_state = nd.zeros((batch_size, num_hiddens))
attention_forward(model, enc_states, dec_state).shape
```

### Decoder with Attention Mechanism

We directly use the hidden state of the encoder in the final time step as the initial hidden state of the decoder. This requires that the encoder and decoder RNNs have the same numbers of layers and hidden units.

In forward calculation of the decoder, we first calculate and obtain the context vector of the current time step by using the attention model introduced above. Since the input of the decoder comes from the word index of the output language, we obtain the feature expression of the input through the word embedding layer, and then concatenate the context vector in the feature dimension. We calculate the output and hidden state of the current time step through the gated recurrent unit, using the concatenated results and the hidden state of the previous time step. Finally, we use the fully connected layer to transform the output into predictions for each output word, with the shape of (batch size, output dictionary size).

```{.python .input  n=170}
class Decoder(nn.Block):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 attention_size, drop_prob=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = attention_model(attention_size)
        self.rnn = rnn.GRU(num_hiddens, num_layers, dropout=drop_prob)
        self.out = nn.Dense(vocab_size, flatten=False)

    def forward(self, cur_input, state, enc_states):
        # Use the attention mechanism to calculate the context vector.
        c = attention_forward(self.attention, enc_states, state[0][-1])
        # The embedded input and the context vector are concatenated in the feature dimension.
        input_and_c = nd.concat(self.embedding(cur_input), c, dim=1)
        # Add a time step dimension, with 1 time step, for the concatenation of the input and the context vector.
        output, state = self.rnn(input_and_c.expand_dims(0), state)
        # Remove the time step dimension, so the output shape is (batch size, output dictionary size).
        output = self.out(output).squeeze(axis=0)
        return output, state

    def begin_state(self, enc_state):
        # Directly use the hidden state of the final time step of the encoder as the initial hidden state of the decoder.
        return enc_state
```

## Training

We first implement the `batch_loss` function to calculate the loss of a mini-batch. The input of the decoder in the initial time step is the special character `BOS`. After that, the input of the decoder in a given time step is the word from the example output sequence in the previous time step, that is, teacher forcing. Also, just as in the implementation in the ["Implementation of Word2vec" ](word2vec-gluon.md) section, we also use mask variables here to avoid the impact of padding on loss function calculations.

```{.python .input}
def batch_loss(encoder, decoder, X, Y, loss):
    batch_size = X.shape[0]
    enc_state = encoder.begin_state(batch_size=batch_size)
    enc_outputs, enc_state = encoder(X, enc_state)
    # Initialize the hidden state of the decoder.
    dec_state = decoder.begin_state(enc_state)
    # The input of decoder at the initial time step is BOS.
    dec_input = nd.array([out_vocab.token_to_idx[BOS]] * batch_size)
    # We will use the mask variable to ignore the loss when the label is PAD.
    mask, num_not_pad_tokens = nd.ones(shape=(batch_size,)), 0
    l = nd.array([0])
    for y in Y.T:
        dec_output, dec_state = decoder(dec_input, dec_state, enc_outputs)
        l = l + (mask * loss(dec_output, y)).sum()
        dec_input = y  # Use teacher forcing.
        num_not_pad_tokens += mask.sum().asscalar()
        # When we encounter EOS, words after the sequence will all be PAD and the mask for the corresponding position is set to 0.
        mask = mask * (y != out_vocab.token_to_idx[EOS])
    return l / num_not_pad_tokens
```

In the training function, we need to update the model parameters of the encoder and the decoder at the same time.

```{.python .input  n=188}
def train(encoder, decoder, dataset, lr, batch_size, num_epochs):
    encoder.initialize(init.Xavier(), force_reinit=True)
    decoder.initialize(init.Xavier(), force_reinit=True)
    enc_trainer = gluon.Trainer(encoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    dec_trainer = gluon.Trainer(decoder.collect_params(), 'adam',
                                {'learning_rate': lr})
    loss = gloss.SoftmaxCrossEntropyLoss()
    data_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)
    for epoch in range(num_epochs):
        l_sum = 0.0
        for X, Y in data_iter:
            with autograd.record():
                l = batch_loss(encoder, decoder, X, Y, loss)
            l.backward()
            enc_trainer.step(1)
            dec_trainer.step(1)
            l_sum += l.asscalar()
        if (epoch + 1) % 10 == 0:
            print("epoch %d, loss %.3f" % (epoch + 1, l_sum / len(data_iter)))
```

Next, we create a model instance and set hyper-parameters. Then, we can train the model.

```{.python .input}
embed_size, num_hiddens, num_layers = 64, 64, 2
attention_size, drop_prob, lr, batch_size, num_epochs = 10, 0.5, 0.01, 2, 50
encoder = Encoder(len(in_vocab), embed_size, num_hiddens, num_layers,
                  drop_prob)
decoder = Decoder(len(out_vocab), embed_size, num_hiddens, num_layers,
                  attention_size, drop_prob)
train(encoder, decoder, dataset, lr, batch_size, num_epochs)
```

## PREDICTION

We introduced three methods to generate the output of the decoder at each time step in the ["Beam Search"](beam-search.md) section. Here we implement the simplest method, greedy search.

```{.python .input  n=177}
def translate(encoder, decoder, input_seq, max_seq_len):
    in_tokens = input_seq.split(' ')
    in_tokens += [EOS] + [PAD] * (max_seq_len - len(in_tokens) - 1)
    enc_input = nd.array([in_vocab.to_indices(in_tokens)])
    enc_state = encoder.begin_state(batch_size=1)
    enc_output, enc_state = encoder(enc_input, enc_state)
    dec_input = nd.array([out_vocab.token_to_idx[BOS]])
    dec_state = decoder.begin_state(enc_state)
    output_tokens = []
    for _ in range(max_seq_len):
        dec_output, dec_state = decoder(dec_input, dec_state, enc_output)
        pred = dec_output.argmax(axis=1)
        pred_token = out_vocab.idx_to_token[int(pred.asscalar())]
        if pred_token == EOS:  # When an EOS symbol is found at any time step, the output sequence is complete.
            break
        else:
            output_tokens.append(pred_token)
            dec_input = pred
    return output_tokens
```

Simply test the model. Enter the French sentence "ils regardent.". The translated English sentence should be "they are watching."

```{.python .input}
input_seq = 'ils regardent .'
translate(encoder, decoder, input_seq, max_seq_len)
```

## Evaluation of Translation Results

BLEU (Bilingual Evaluation Understudy) is often used to evaluate machine translation results[1]. For any subsequence in the model prediction sequence, BLEU evaluates whether this subsequence appears in the label sequence.

Specifically, the precision of the subsequence with $n$ words is $p_n$. It is the ratio of the number of subsequences with $n$ matched words for the prediction sequence and label sequence to the number of subsequences with $n$ words in the prediction sequence. For example, assume the label sequence is $A$, $B$, $C$, $D$, $E$, $F$, and the prediction sequence is $A$, $B$, $B$, $C$, $D$. Then $p_1 = 4/5, \ p_2 = 3/4, \ p_3 = 1/3, and \ p_4 = 0$. Assume $len_{\text{label}}$ and $len_{\text{pred}}$ are the numbers of words in the label sequence and the prediction sequence. Then, BLEU is defined as

$$ \exp\left(\min\left(0, 1 - \frac{len_{\text{label}}}{len_{\text{pred}}}\right)\right) \prod_{n=1}^k p_n^{1/2^n},$$

Here, $k$ is the maximum number of words in the subsequence we wish to match. It can be seen that the BLEU is 1 when the prediction sequence and the label sequence are identical.

Because matching longer subsequences is more difficult than matching shorter subsequences, BLEU gives greater weight to the precision of longer subsequence matches. For example, when $p_n$ is fixed at 0.5, as $n$ increases, $0.5^{1/2} \approx 0.7, 0.5^{1/4} \approx 0.84, 0.5^{1/8} \approx 0.92, and 0.5^{1/16} \approx 0.96$. In addition, the prediction of shorter sequences by the model tends to obtain higher $p_n$ values. Therefore, the coefficient before the multiplication term in the above equation is a penalty to the shorter output. For example, when $k=2$, we assume the label sequence is $A$, $B$, $C$, $D$, $E$, $F$ and the prediction sequence is $A$, $B$. Although $p_1 = p_2 = 1$, the penalty factor is $\exp(1-6/2) \approx 0.14$, so the BLEU is also close to 0.14.

Next, we calculate the BLEU

```{.python .input}
def bleu(pred_tokens, label_tokens, k):
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):
        num_matches = 0
        for i in range(len_pred - n + 1):
            if ' '.join(pred_tokens[i: i + n]) in ' '.join(label_tokens):
                num_matches += 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score
```

and define an auxiliary printing function.

```{.python .input}
def score(input_seq, label_seq, k):
    pred_tokens = translate(encoder, decoder, input_seq, max_seq_len)
    label_tokens = label_seq.split(' ')
    print('bleu %.3f, predict: %s' % (bleu(pred_tokens, label_tokens, k),
                                      ' '.join(pred_tokens)))
```

A correct prediction receives a score of 1.

```{.python .input}
score('ils regardent .', 'they are watching .', k=2)
```

Test an example that is not in the training set.

```{.python .input}
score('ils sont canadiens .', 'they are canadian .', k=2)
```

## Summary

* We can apply encoder-decoder and attention mechanisms to machine translation.
* BLEU can be used to evaluate translation results.

## Problems

* If the encoder and decoder have different number of hidden units or layers, how can we improve the decoder's hidden state initialization method?
* During training, we experiment by replacing "teacher forcing" with the output of the decoder at the previous time step as the input of the decoder at the current time step. Has the result changed?
* Try to train the model with larger translation data sets, such as WMT[2] and Tatoeba Project[3].



## Reference

[1] Papineni, K., Roukos, S., Ward, T., & Zhu, W. J. (2002, July). BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting on association for computational linguistics (pp. 311-318). Association for Computational Linguistics.

[2] WMT. http://www.statmt.org/wmt14/translation-task.html

[3] Tatoeba Project. http://www.manythings.org/anki/

## Discuss on our Forum

<div id="discuss" topic_id="2396"></div>
