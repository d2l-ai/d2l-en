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

```{.python .input  n=3}
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

```{.json .output n=3}
[
 {
  "data": {
   "text/plain": "(2, 8, 768)"
  },
  "execution_count": 3,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

...

![双向语言模型](../img/biLM_Leakage.svg)

...
![遮蔽语言模型](../img/bert_mlm.svg)

...

```{.python .input  n=4}
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

```{.python .input  n=5}
print(encodings)

mlm_decoder = MaskLMDecoder(vocab_size, embed_size)
mlm_decoder.initialize()

mlm_positions = np.array([[0, 1], [4, 8]])
mlm_label = np.array([[100, 200], [100, 200]])
mlm_pred = mlm_decoder(encodings, mlm_positions)
mlm_pred.shape
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "[[[ 1.1954827   1.2913651   0.10027605 ...  0.1350488   0.75628126\n   -0.0414779 ]\n  [ 1.3979956   1.0108925   0.25892806 ... -0.05029266  0.8459386\n   -0.10251292]\n  [ 1.4573783   0.5411349   0.27132586 ... -0.18223734  0.94110787\n   -0.13618408]\n  ...\n  [ 0.4399233   0.39078018  0.05209133 ... -0.2943      0.9758663\n   -0.3676404 ]\n  [ 0.6068444   0.5813842   0.3420519  ... -0.22070077  0.9723146\n   -0.5000176 ]\n  [ 0.8938725   0.5913224   0.5731718  ... -0.00974521  1.0499434\n   -0.49452758]]\n\n [[ 1.2279879   1.3020388   0.1130784  ...  0.15336953  0.7338273\n   -0.08130451]\n  [ 1.4514403   0.9618422   0.26130652 ...  0.04136745  0.86831087\n   -0.14407705]\n  [ 1.3674632   0.561611    0.29686394 ... -0.20095065  0.9411951\n   -0.27811623]\n  ...\n  [ 0.5336177   0.30347967  0.08665802 ... -0.28749332  0.9696437\n   -0.36487216]\n  [ 0.64547795  0.5173523   0.37236992 ... -0.15298648  1.0318404\n   -0.53901917]\n  [ 0.9145003   0.5290929   0.6179654  ...  0.01441563  0.98607665\n   -0.51857775]]]\nmasked_positions [[0. 1. 4. 8.]]\nbatch_idx [0. 1.]\nbatch_idx [0. 0. 1. 1.]\nbatch_idx [[0. 0. 1. 1.]]\nencoded [[[ 1.19548273e+00  1.29136515e+00  1.00276046e-01 ...  1.35048807e-01\n    7.56281257e-01 -4.14779000e-02]\n  [ 1.39799559e+00  1.01089251e+00  2.58928061e-01 ... -5.02926633e-02\n    8.45938623e-01 -1.02512918e-01]\n  [ 6.53404891e-01  7.54371285e-02  6.93439767e-02 ... -3.67644310e-01\n    9.53921556e-01 -3.40112358e-01]\n  [ 6.88990429e-41  0.00000000e+00  4.33421615e-42 ...  0.00000000e+00\n    0.00000000e+00  0.00000000e+00]]]\nencoded [[[ 1.19548273e+00  1.29136515e+00  1.00276046e-01 ...  1.35048807e-01\n    7.56281257e-01 -4.14779000e-02]\n  [ 1.39799559e+00  1.01089251e+00  2.58928061e-01 ... -5.02926633e-02\n    8.45938623e-01 -1.02512918e-01]]\n\n [[ 6.53404891e-01  7.54371285e-02  6.93439767e-02 ... -3.67644310e-01\n    9.53921556e-01 -3.40112358e-01]\n  [ 6.88990429e-41  0.00000000e+00  4.33421615e-42 ...  0.00000000e+00\n    0.00000000e+00  0.00000000e+00]]]\npred [[[ 2.0093503e+00 -1.4084423e+00  1.3939968e-01 ... -6.5847826e-01\n    2.5387597e-01 -2.0788229e-01]\n  [ 2.1118746e+00 -1.4440082e+00  2.1123180e-01 ... -5.3184783e-01\n    2.8383601e-01 -2.1656406e-01]]\n\n [[ 2.5205467e+00 -1.2264202e+00  7.2730100e-01 ... -1.0135853e+00\n    2.4946362e-01 -5.2340269e-02]\n  [ 4.2662532e-41  7.6231197e-40 -1.3354795e-40 ...  1.1522275e-39\n    1.1825978e-40  7.2022397e-40]]]\n"
 },
 {
  "data": {
   "text/plain": "(2, 2, 10000)"
  },
  "execution_count": 5,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

```{.python .input  n=6}
a = bx
```

```{.json .output n=6}
[
 {
  "ename": "NameError",
  "evalue": "name 'bx' is not defined",
  "output_type": "error",
  "traceback": [
   "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
   "\u001b[0;32m<ipython-input-6-c56086892aad>\u001b[0m in \u001b[0;36m<module>\u001b[0;34m\u001b[0m\n\u001b[0;32m----> 1\u001b[0;31m \u001b[0ma\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mbx\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m",
   "\u001b[0;31mNameError\u001b[0m: name 'bx' is not defined"
  ]
 }
]
```

```{.python .input}
mlm_loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
mlm_loss = mlm_loss_fn(mlm_pred, mlm_label)
mlm_loss.shape
```

...
![下一句预测](../img/bert_nsp.svg)

```{.python .input  n=6}
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

```{.python .input  n=7}
ns_classifier = NextSentenceClassifier(embed_size)
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
