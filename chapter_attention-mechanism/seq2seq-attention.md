# Sequence to Sequence with Attention Mechanism
:label:`sec_seq2seq_attention`

In this section, we add the attention mechanism to the sequence to sequence
model introduced in :numref:`sec_seq2seq`
to explicitly select state. :numref:`fig_s2s_attention` shows the model
architecture for encoding and decoding at the timestep $t$. Here, the memory of the
attention layer consists of all the information that the encoder has 
seen---the encoder outputs at each timestep. 
During the decoding, the decoder output from the previous timestep $t-1$ is used as the query.
While the output of the attention model is viewed as the context information, 
which is then concatenated with the decode input $D_t$. Finally, we feed the concatenation into the decoder.

![The second timestep in decoding for the sequence to sequence model with attention mechanism.](../img/seq2seq_attention.svg)
:label:`fig_s2s_attention`


What is more, to better visualize the overall architecture of seq2seq with attention model, the layer structure of its encoder and decoder is shown in :numref:`fig_s2s_attention_details`.

![The layers in the sequence to sequence model with attention mechanism.](../img/seq2seq-attention-details.svg)
:label:`fig_s2s_attention_details`

```{.python .input  n=1}
import d2l
from mxnet import np, npx
from mxnet.gluon import rnn, nn
npx.set_np()
```

## Decoder

Since the encoder of seq2seq with attention mechanism is the same as `Seq2SeqEncoder` in :numref:`sec_seq2seq`, we will dive directly into the decoder. We add a MLP attention layer (`MLPAttention`) which has the same hidden size as the LSTM layer in the decoder. Then we initialize the state of the decoder by passing three items from the encoder:

- **the encoder outputs of all timesteps**, which are used as the attention layer's memory with identical keys and values;

- **the hidden state of the encoder's last timestep** that is used as the initial decoder's hidden state;

- **the encoder valid lengths**, so the attention layer will not consider the padding tokens with in the encoder outputs.

In each timestep of the decoding, we use the output of the decoder's last RNN layer as the query for the attention layer. The attention model's output is then concatenated with the input embedding vector to feed into the RNN layer. Despite the RNN layer hidden state also contains history information from decoder, the attention output explicitly selects the encoder outputs based on the `enc_valid_len`, so that the attention output suspends other non-correlated information.

Let's implement the `Seq2SeqAttentionDecoder` together, and see how it differs from the decoder in seq2seq model from :numref:`sec_seq2seq_decoder`.

```{.python .input  n=2}
class Seq2SeqAttentionDecoder(d2l.Decoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention_cell = d2l.MLPAttention(num_hiddens, dropout)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = rnn.LSTM(num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Dense(vocab_size, flatten=False)

    def init_state(self, enc_outputs, enc_valid_len, *args):
        outputs, hidden_state = enc_outputs
        # Transpose outputs to (batch_size, seq_len, hidden_size)
        return (outputs.swapaxes(0, 1), hidden_state, enc_valid_len)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        X = self.embedding(X).swapaxes(0, 1)
        outputs = []
        for x in X:
            # query shape: (batch_size, 1, hidden_size)
            query = np.expand_dims(hidden_state[0][-1], axis=1)
            # context has same shape as query
            context = self.attention_cell(
                query, enc_outputs, enc_outputs, enc_valid_len)
            # Concatenate on the feature dimension
            x = np.concatenate((context, np.expand_dims(x, axis=1)), axis=-1)
            # Reshape x to (1, batch_size, embed_size+hidden_size)
            out, hidden_state = self.rnn(x.swapaxes(0, 1), hidden_state)
            outputs.append(out)
        outputs = self.dense(np.concatenate(outputs, axis=0))
        return outputs.swapaxes(0, 1), [enc_outputs, hidden_state,
                                        enc_valid_len]
```

Now it is the time to validate our seq2seq with attention model. To be consistant with the seq2seq without attention in :numref:`sec_seq2seq`, we use the same hyper-parameters for `vocab_size`, `embed_size`, `num_hiddens`, and `num_layers`. As a result, we get the same decoder output shape, but the state structure is changed.

```{.python .input  n=3}
encoder = d2l.Seq2SeqEncoder(vocab_size=10, embed_size=8,
                             num_hiddens=16, num_layers=2)
encoder.initialize()
decoder = Seq2SeqAttentionDecoder(vocab_size=10, embed_size=8,
                                  num_hiddens=16, num_layers=2)
decoder.initialize()
X = np.zeros((4, 7))
state = decoder.init_state(encoder(X), None)
out, state = decoder(X, state)
out.shape, len(state), state[0].shape, len(state[1]), state[1][0].shape
```

## Training

Similar to :numref:`sec_seq2seq_training`, we try a toy model by applying
the same training hyper-parameters and the same training loss.
As we can see from the result, since the
sequences in the training dataset are relative short. The additional attention
layer does not lead to a significant improvement. But due to the computational 
overhead of both the encoder's and the decoder's attention layers, this model
is much slower than the seq2seq model without attention.

```{.python .input  n=5}
embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.0
batch_size, num_steps = 64, 10
lr, num_epochs, ctx = 0.005, 200, d2l.try_gpu()

src_vocab, tgt_vocab, train_iter = d2l.load_data_nmt(batch_size, num_steps)
encoder = d2l.Seq2SeqEncoder(
    len(src_vocab), embed_size, num_hiddens, num_layers, dropout)
decoder = Seq2SeqAttentionDecoder(
    len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout)
model = d2l.EncoderDecoder(encoder, decoder)
d2l.train_s2s_ch9(model, train_iter, lr, num_epochs, ctx)
```

Last, we predict several sample examples.

```{.python .input  n=6}
for sentence in ['Go .', 'Wow !', "I'm OK .", 'I won !']:
    print(sentence + ' => ' + d2l.predict_s2s_ch9(
        model, sentence, src_vocab, tgt_vocab, num_steps, ctx))
```

## Summary

* The seq2seq with attention adds an additional attention layer to the standard `Seq2seqDecoder`.
* The decoder of the seq2seq with attention model passes three items from the encoder: the encoder outputs of all timesteps, the hidden state of the encoder's last timestep, and the encoder valid lengths.

## Exercises

* Compare the `Seq2SeqAttentionDecoder` versus the standard `Seq2seqDecoder`. Use the same parameters and compare their losses.
* Can you think of any scenarios that the `Seq2SeqAttentionDecoder` will outperform the `Seq2seqDecoder`?


## [Discussions](https://discuss.mxnet.io/t/seq2seq-attention/4345)

![](../img/qr_seq2seq-attention.svg)
