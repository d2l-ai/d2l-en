# BERT

*This section is under construction.*

```{.python .input  n=3}
import d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

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

    def forward(self, tokens, segments, valid_len):
        # Shape of X remains unchanged in the following code snippet:
        # (batch size, max sequence length, embed_size)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, valid_len)
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
X = encoder(tokens, segments, None)
X.shape
```

...

```{.python .input  n=29}
# Saved in the d2l package for later use
class MaskLM(nn.Block):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, masked_positions):
        num_masked_positions = masked_positions.shape[1]
        masked_positions = masked_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that batch_size = 2, num_masked_positions = 3,
        # batch_idx = np.array([0, 0, 0, 1, 1, 1])
        batch_idx = np.repeat(batch_idx, num_masked_positions)
        masked_X = X[batch_idx, masked_positions]
        masked_X = masked_X.reshape((batch_size, num_masked_positions, -1))
        masked_preds = self.mlp(masked_X)
        return masked_preds
```

...

```{.python .input  n=30}
mlm = MaskLM(vocab_size, embed_size)
mlm.initialize()
mlm_positions = np.array([[0, 2, 1], [6, 5, 7]])
mlm_preds = mlm(X, mlm_positions)
mlm_preds.shape
```

```{.python .input}
mlm_labels = np.array([[1, 3, 5], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_preds, mlm_labels)
# The value on the batch axis is the average of loss at each masked position
mlm_l.shape
```

...

```{.python .input  n=13}
# Saved in the d2l package for later use
class NextSentencePred(nn.Block):
    def __init__(self, num_hiddens, **kwargs):
        super(NextSentencePred, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(nn.Dense(num_hiddens, activation='tanh'))
        self.mlp.add(nn.Dense(2))

    def forward(self, X):
        # 0 is the index of the CLS token
        X = X[:, 0, :]
        # X shape: (batch size, embed_size)
        return self.mlp(X)
```

...

```{.python .input  n=14}
nsp = NextSentencePred(embed_size)
nsp.initialize()
ns_pred = nsp(X)
ns_pred.shape
```

```{.python .input}
ns_label = np.array([0, 1])
ns_loss = loss(ns_pred, ns_label)
ns_loss.shape
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
        self.nsp = NextSentencePred(embed_size)
        self.mlm = MaskLM(vocab_size, embed_size)

    def forward(self, tokens, segments, valid_len=None,
                masked_positions=None):
        X = self.encoder(tokens, segments, valid_len)
        if masked_positions is not None:
            mlm_Y = self.mlm(X, masked_positions)
        else:
            mlm_Y = None
        nsp_Y = self.nsp(X)
        return X, mlm_Y, nsp_Y
```

...
