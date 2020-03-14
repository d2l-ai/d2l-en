# BERT

We have introduced several word embedding models for natural language understanding.
After pretraining, the output can be thought of as a matrix
where each row is a vector that represents a word of a predefined vocabulary.
In fact, these word embedding models are all *context-independent*.
Let's begin by illustrating this property.


## From Context-Independent to Context-Sensitive

Recall the experiments in :numref:`sec_word2vec_gluon` and :numref:`sec_synonyms`.
For instance, word2vec and GloVe both assign the same pretrained vector to the same word regardless of the context of the word (if any).
Given the abundance of polysemy and complex semantics in natural languages,
context-independent representations have obvious limitations.
For instance, the word "crane" in contexts
"a crane is flying" and "a crane driver came" has completely different meanings;
thus, the same word may be assigned different representations depending on contexts.

This motivates the development of *context-sensitive* word representations,
where the representation of words depend on their contexts.
For example, by taking the entire sequence as the input,
ELMo (Embeddings from Language Models) is a function that assigns a representation to each word from the input sequence :cite:`Peters.Neumann.Iyyer.ea.2018`.


Specifically, ELMo combines all the intermediate layer representations from pretrained bidirectional LSTM as the output representation.
Then the ELMo representation will be added to a downstream task's existing supervised model
as additional features, such as by concatenating ELMo representation and the original representation (e.g., GloVe) of tokens in the existing model.
On one hand,
all the weights in the pretrained bidirectional LSTM model are frozen after ELMo representations are added.
On the other hand,
the existing supervised model is task specific.
Leveraging different best models for different tasks at that time,
adding ELMo improved the state of the art across $6$ NLP tasks.


## From Task-Specific to Task-Agnostic

Although ELMo has significantly improved solutions to diverse NLP tasks,
each solution is still customized to each task.
However, it is practically non-trivial to craft a specific architecture for every NLP task.
The GPT (Generative Pre-Training) model represents an effort in designing
a general task-agnostic model for context-sensitive representations :cite:`Radford.Narasimhan.Salimans.ea.2018`.

```{.python .input  n=1}
import d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn

npx.set_np()
```

...

```{.python .input  n=2}
# Saved in the d2l package for later use
class BERTEncoder(nn.Block):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout, **kwargs):
        super(BERTEncoder, self).__init__(**kwargs)
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segment_embedding = nn.Embedding(2, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_layers):
            self.blks.add(d2l.EncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout))

    def forward(self, tokens, segments, valid_lens):
        # Shape of X remains unchanged in the following code snippet:
        # (batch size, max sequence length, num_hiddens)
        X = self.token_embedding(tokens) + self.segment_embedding(segments)
        X = self.pos_encoding(X)
        for blk in self.blks:
            X = blk(X, valid_lens)
        return X
```

...

```{.python .input  n=3}
vocab_size, num_hiddens, ffn_num_hiddens = 10000, 768, 1024
num_heads, num_layers, dropout = 4, 2, 0.1
encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                      num_layers, dropout)
encoder.initialize()
tokens = np.random.randint(0, 10000, (2, 8))
segments = np.array([[0, 0, 0, 0, 1, 1, 1, 1],
                     [0, 0, 0, 1, 1, 1, 1, 1]])
encoded_X = encoder(tokens, segments, None)
encoded_X.shape
```

...

```{.python .input  n=4}
# Saved in the d2l package for later use
class MaskLM(nn.Block):
    def __init__(self, vocab_size, num_hiddens, **kwargs):
        super(MaskLM, self).__init__(**kwargs)
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(num_hiddens, flatten=False, activation='relu'))
        self.mlp.add(nn.LayerNorm())
        self.mlp.add(nn.Dense(vocab_size, flatten=False))

    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = np.arange(0, batch_size)
        # Suppose that batch_size = 2, num_pred_positions = 3,
        # batch_idx = np.array([0, 0, 0, 1, 1, 1])
        batch_idx = np.repeat(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.mlp(masked_X)
        return mlm_Y_hat
```

...

```{.python .input  n=5}
mlm = MaskLM(vocab_size, num_hiddens)
mlm.initialize()
mlm_positions = np.array([[0, 2, 1], [6, 5, 7]])
mlm_Y_hat = mlm(encoded_X, mlm_positions)
mlm_Y_hat.shape
```

```{.python .input  n=6}
mlm_Y = np.array([[1, 3, 5], [10, 20, 30]])
loss = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_l = loss(mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y.reshape(-1))
mlm_l.shape
```

...

```{.python .input  n=7}
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
        # X shape: (batch size, num_hiddens)
        return self.mlp(X)
```

...

```{.python .input  n=8}
nsp = NextSentencePred(num_hiddens)
nsp.initialize()
nsp_Y_hat = nsp(encoded_X)
nsp_Y_hat.shape
```

```{.python .input  n=9}
nsp_y = np.array([0, 1])
nsp_l = loss(nsp_Y_hat, nsp_y)
nsp_l.shape
```

...

```{.python .input  n=10}
# Saved in the d2l package for later use
class BERTModel(nn.Block):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_layers, dropout):
        super(BERTModel, self).__init__()
        self.encoder = BERTEncoder(vocab_size, num_hiddens, ffn_num_hiddens,
                                   num_heads, num_layers, dropout)
        self.nsp = NextSentencePred(num_hiddens)
        self.mlm = MaskLM(vocab_size, num_hiddens)

    def forward(self, tokens, segments, valid_lens=None,
                pred_positions=None):
        encoded_X = self.encoder(tokens, segments, valid_lens)
        if pred_positions is not None:
            mlm_Y_hat = self.mlm(encoded_X, pred_positions)
        else:
            mlm_Y_hat = None
        nsp_Y_hat = self.nsp(encoded_X)
        return encoded_X, mlm_Y_hat, nsp_Y_hat
```

...
