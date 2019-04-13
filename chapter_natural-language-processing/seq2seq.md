# Encoder-Decoder (seq2seq)

We have processed and analyzed variable-length input sequences in the previous chapters. However, in many applications, both the input and output can be variable-length sequences. For instance, in the case of machine translation, the input can be a variable-length sequence of English text, and the output can be a variable-length sequence of French text.

> English input: "They", "are", "watching", "."

> French output: "Ils", "regardent", "."

When the input and output are both variable-length sequences, we can use the encoder-decoder[1] or the seq2seq model[2]. Both models use two recurrent neural networks (RNNs) named encoders and decoders. The encoder is used to analyze the input sequence and the decoder is used to generate the output sequence.

Figure 11.8 depicts a method for translating the English sentences above into French sentences using an encoder-decoder. In the training data set, we can attach a special symbol "&lt;eos&gt;" (end of sequence) after each sentence to indicate the termination of the sequence. For each time step, the encoder generates inputs following the order of words, punctuation, and special symbols "&lt;eos&gt;" in the English sentence. Figure 11.8 uses the hidden state of the encoder at the final time step as the encoding information for the input sentence. The decoder uses the encoding information of the input sentence, the output of the last time step, and the hidden state as inputs for each time step.
We hope that the decoder can correctly output the translated French words, punctuation, and special symbols "&lt;eos&gt;" at each time step.
It should be noted that the input of the decoder at the initial time step uses a special symbol "&lt;bos&gt;" to indicate the beginning of the sequence.

![Use an encoder-decoder to translate this sentence from English to French.  The encoder and decoder are each recurrent neural networks. ](../img/seq2seq.svg)

Next, we are going to introduce the definitions of the encoder and decoder individually.

## Encoder

The role of the encoder is to transform an input sequence of variable length into a fixed-length context variable $\boldsymbol{c}$, and encode the input sequence information in that context variable. The most commonly used encoder is an RNN.

We will consider a time-series data instance with a batch size of 1. We assume that the input sequence is $x_1, \ldots, x_T$, such that $x_i$ is the $i$th word in the input sentence. At time step $t$, the RNN will enter feature vector $\boldsymbol{x}_t$ for $x_t$ and hidden state $\boldsymbol{h} _{t-1}$ from the previous time step will be transformed into the current hidden state $\boldsymbol{h}_t$. We can use function $f$ to express the transformation of the RNN's hidden layer:

$$\boldsymbol{h}_t = f(\boldsymbol{x}_t, \boldsymbol{h}_{t-1}). $$

Next, the encoder transforms the hidden state of each time step into context variables through custom function $q$.

$$\boldsymbol{c} =  q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T).$$

For example, when we select $q(\boldsymbol{h}_1, \ldots, \boldsymbol{h}_T) = \boldsymbol{h}_T$, the context variable is the hidden state of input sequence $\boldsymbol{h}_T$ for the final time step.

The encoder discussed above is a unidirectional RNN, and the hidden state of each time step depends only on itself and the input subsequences from previous time steps. We can also construct encoders using bidirectional RNNs. In this case, the hidden state from each time step of the encoder depends on the subsequence before and after the time step (including the input of the current time step), which encodes the information of the entire sequence.


## Decoder

As we just mentioned, the context variable $\boldsymbol{c}$ of the encoder's output encodes the entire input sequence $x_1, \ldots, x_T$. Given the output sequence $y_1, y_2, \ldots, y_{T'}$ in the training example, for each time step $t'$ (the symbol differs from the input sequence and the encoder's time step $t$), the conditional probability of decoder output $y_{t'}$ will be based on the previous output sequence $y_1, \ldots, y_{t'-1}$ and context variable $\boldsymbol{c}$, i.e. $\mathbb{P }(y_{t'} \mid y_1, \ldots, y_{t'-1}, \boldsymbol{c})$.

Therefore, we can use another RNN as a decoder.
At time step $t^\prime$ of the output sequence, the decoder uses the output $y_{t^\prime-1}$ from the previous time step and context variable $\boldsymbol{c}$ as its input and transforms their hidden state $\boldsymbol{s}_{t^\prime-1}$ from the previous time step into hidden state $\boldsymbol{s}_{t^\prime}$ of the current time step.  Therefore, we can use function $f$ to express the transformation of the decoder's hidden layer:

$$\boldsymbol{s}_{t^\prime} = g(y_{t^\prime-1}, \boldsymbol{c}, \boldsymbol{s}_{t^\prime-1}).$$

After obtaining the hidden state of the decoder, we can use a custom output layer and the softmax operation to compute $\mathbb{P}(y_{t^\prime} \mid y_1, \ldots, y_{t^\prime-1}, \boldsymbol{c})$. For example, using hidden state $\boldsymbol{s}_{t^\prime}$ based on the current time step of the decoder, the output $y_{t^\prime-1}$ from the previous time step, and the context variable $\boldsymbol{c}$ to compute the probability distribution of output $y_{t^\prime}$ from the current time step.


## Model Training

According to the maximum likelihood estimation, we can maximize the conditional probability of the output sequence based on the input sequence

$$
\begin{aligned}
\mathbb{P}(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T)
&= \prod_{t'=1}^{T'} \mathbb{P}(y_{t'} \mid y_1, \ldots, y_{t'-1}, x_1, \ldots, x_T)\\
&= \prod_{t'=1}^{T'} \mathbb{P}(y_{t'} \mid y_1, \ldots, y_{t'-1}, \boldsymbol{c}),
\end{aligned}
$$

to get the loss of the output sequence

$$- \log\mathbb{P}(y_1, \ldots, y_{T'} \mid x_1, \ldots, x_T) = -\sum_{t'=1}^{T'} \log \mathbb{P}(y_{t'} \mid y_1, \ldots, y_{t'-1}, \boldsymbol{c}),$$

In model training, the mean of losses for all the output sequences is usually used as a loss function that needs to be minimized. In the model prediction discussed in Figure 11.8, we need to use the output of the decoder from the previous time step as the input to the current time step. In contrast, in training, we can also use the label of the label sequence from the previous time step as the input of the decoder for the current time step. This is called teacher forcing. 

## Summary

* The encoder-decoder (seq2seq) model can input and output a sequence of variable length.
* The encoder-decoder uses two RNNs.
* In encoder-decoder training, we can use teacher forcing.


## Exercises

* In addition to machine translation, what other applications can you think of for encoder-decoder?
* What methods can be used to design the output layer of the decoder?




## Reference

[1] Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., & Bengio, Y. (2014). Learning phrase representations using RNN encoder-decoder for statistical machine translation. arXiv preprint arXiv:1406.1078.

[2] Sutskever, I., Vinyals, O., & Le, Q. V. (2014). Sequence to sequence learning with neural networks. In Advances in neural information processing systems (pp. 3104-3112).

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2393)

![](../img/qr_seq2seq.svg)
