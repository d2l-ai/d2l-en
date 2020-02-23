# BERT

```{.python .input  n=3}
import d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

...
![输入表示](../img/bert_inputs.svg)

...

```{.python .input  n=3}
# Saved in the d2l package for later use
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, embed_size, pw_num_hiddens, num_heads,
                 num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = nn.Embedding(2, embed_size)
        self.pos_encoding = d2l.PositionalEncoding(embed_size, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                embed_size, pw_num_hiddens, num_heads, dropout))

    def forward(self, tokens, segments, valid_length):
        # Shape of X remains unchanged in the following code snippet:
        # (batch size, max sequence length, embed_size)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, valid_length)
        return X
```

...

```{.python .input  n=4}
vocab_size, embed_size, pw_num_hiddens = 10000, 768, 1024
num_heads, num_layers, dropout = 4, 2, 0.1
encoder = BERTEncoder(vocab_size, embed_size, pw_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
tokens = np.random.randint(0, 10000, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1, 1, 1]])
encodings = encoder(tokens, segments, None)
encodings.shape
```

...

![双向语言模型](../img/biLM_Leakage.svg)

...
![遮蔽语言模型](../img/bert_mlm.svg)

...

```{.python .input  n=29}
# Saved in the d2l package for later use
class MaskLMDecoder(nn.Block):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLMDecoder, self).__init__(**kwargs)
        self.decoder = nn.Sequential()
        self.decoder.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.decoder.add(nn.LayerNorm())
        self.decoder.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, masked_positions):
        num_masked_positions = masked_positions.shape[1]
        masked_positions = masked_positions.reshape((1, -1))
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)   
        batch_idx = np.repeat(batch_idx, num_masked_positions)
        batch_idx = np.expand_dims(batch_idx, axis=0)
        encoded = X[batch_idx, masked_positions]
        encoded = encoded.reshape((batch_size, num_masked_positions, X.shape[-1]))
        pred = self.decoder(encoded)
        return pred
```

...

```{.python .input  n=30}
mlm_decoder = MaskLMDecoder(vocab_size, embed_size)
mlm_decoder.initialize()

mlm_positions = np.array([[0, 1], [4, 8]])
mlm_label = np.array([[100, 200], [100, 200]])
mlm_pred = mlm_decoder(encodings, mlm_positions)
mlm_pred.shape
```



```{.python .input}
mlm_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_loss = mlm_loss_fn(mlm_pred, mlm_label)
mlm_loss.shape
```

...
![下一句预测](../img/bert_nsp.svg)

```{.python .input  n=13}
# Saved in the d2l package for later use
class NextSentenceClassifier(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(NextSentenceClassifier, self).__init__(**kwargs)
        self.classifier = nn.Sequential()
        self.classifier.add(nn.Dense(num_hiddens, flatten=False,
                                     activation='tanh'))
        self.classifier.add(nn.Dense(2, flatten=False))

    def forward(self, X):
        X = X[:, 0, :]
        return self.classifier(X)
```

...

```{.python .input  n=14}
ns_classifier = NextSentenceClassifier(embed_size)
ns_classifier.initialize()

ns_pred = ns_classifier(encodings)
ns_label = np.array([0, 1])
ns_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
ns_loss = ns_loss_fn(ns_pred, ns_label)
print(ns_pred.shape, ns_loss.shape)
```

...

```{.python .input  n=10}
# Saved in the d2l package for later use
class BERTModel(nn.Block):
    def __init__(self, vocab_size, embed_size, pw_num_hiddens, num_heads,
                 num_layers, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, embed_size, pw_num_hiddens,
                                   num_heads, num_layers, dropout)
        self.ns_classifier = NextSentenceClassifier(embed_size)
        self.mlm_decoder = MaskLMDecoder(vocab_size, embed_size)

    def forward(self, inputs, token_types, valid_length=None, masked_positions=None):
        seq_out = self.encoder(inputs, token_types, valid_length)
        next_sentence_classifier_out = self.ns_classifier(seq_out)
        if not masked_positions is None:
            mlm_decoder_out = self.mlm_decoder(seq_out, masked_positions)
        else:
            mlm_decoder_out = None
        return seq_out, next_sentence_classifier_out, mlm_decoder_out
```

...
