# BERT

```{.python .input  n=1}
import d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

...
![输入表示](../img/bert_inputs.svg)

...

```{.python .input  n=2}
# Saved in the d2l package for later use
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, embed_size, pw_num_hiddens, num_heads,
                 num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = gluon.nn.Embedding(vocab_size, embed_size)
        self.segment_embedding = gluon.nn.Embedding(2, embed_size)
        self.pos_encoding = d2l.PositionalEncoding(embed_size, dropout)
        self.blks = gluon.nn.Sequential()
        for i in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                embed_size, pw_num_hiddens, num_heads, dropout))

    def forward(self, tokens, segments, mask):
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, mask)
        return X
```

...

```{.python .input  n=3}
encoder = BERTEncoder(vocab_size=10000, embed_size=768, pw_num_hiddens=1024,
                      num_heads=4, num_layers=2, dropout=0.1)
encoder.initialize()
num_samples, num_tokens = 2, 8
tokens = np.random.randint(0, 10000, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1, 1, 1]])
encodings = encoder(tokens, segments, None)
print(encodings.shape)
```

...

![双向语言模型](../img/biLM_Leakage.svg)

...
![遮蔽语言模型](../img/bert_mlm.svg)

...

```{.python .input  n=4}
# Saved in the d2l package for later use
class MaskLMDecoder(nn.Block):
    def __init__(self, vocab_size, units, **kwargs):
        super(MaskLMDecoder, self).__init__(**kwargs)
        self.decoder = gluon.nn.Sequential()
        self.decoder.add(gluon.nn.Dense(units, flatten=False, activation='relu'))
        self.decoder.add(gluon.nn.LayerNorm())
        self.decoder.add(gluon.nn.Dense(vocab_size, flatten=False))

    def forward(self, X, masked_positions):
        ctx = masked_positions.context
        dtype = masked_positions.dtype
        num_masked_positions = masked_positions.shape[1]
        masked_positions = masked_positions.reshape((1, -1))
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        batch_idx = np.repeat(batch_idx, num_masked_positions)
        batch_idx = batch_idx.reshape((1, -1))
        encoded = X[batch_idx, masked_positions]
        encoded = encoded.reshape((batch_size, num_masked_positions, X.shape[-1]))
        pred = self.decoder(encoded)
        return pred
```

...

```{.python .input  n=5}
mlm_decoder = MaskLMDecoder(vocab_size=10000, units=768)
mlm_decoder.initialize()

mlm_positions = np.array([[0, 1], [4, 8]])
mlm_label = np.array([[100, 200], [100, 200]])
mlm_pred = mlm_decoder(encodings, mlm_positions)
mlm_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_loss = mlm_loss_fn(mlm_pred, mlm_label)
print(mlm_pred.shape, mlm_loss.shape)
```

...
![下一句预测](../img/bert_nsp.svg)

```{.python .input  n=6}
# Saved in the d2l package for later use
class NextSentenceClassifier(nn.Block):
    def __init__(self, units=768, **kwargs):
        super(NextSentenceClassifier, self).__init__(**kwargs)
        self.classifier = gluon.nn.Sequential()
        self.classifier.add(gluon.nn.Dense(units=units, flatten=False,
                                           activation='tanh'))
        self.classifier.add(gluon.nn.Dense(units=2, flatten=False))

    def forward(self, X):
        X = X[:, 0, :]
        return self.classifier(X)
```

...

```{.python .input  n=7}
ns_classifier = NextSentenceClassifier()
ns_classifier.initialize()

ns_pred = ns_classifier(encodings)
ns_label = np.array([0, 1])
ns_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
ns_loss = ns_loss_fn(ns_pred, ns_label)
print(ns_pred.shape, ns_loss.shape)
```

...

```{.python .input  n=8}
# Saved in the d2l package for later use
class BERTModel(nn.Block):
    def __init__(self, vocab_size=None, embed_size=128, pw_num_hiddens=512,
                 num_heads=2, num_layers=4, dropout=0.1):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size=vocab_size, embed_size=embed_size, pw_num_hiddens=pw_num_hiddens,
                                   num_heads=num_heads, num_layers=num_layers, dropout=dropout)
        self.ns_classifier = NextSentenceClassifier()
        self.mlm_decoder = MaskLMDecoder(vocab_size=vocab_size, units=embed_size)

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
