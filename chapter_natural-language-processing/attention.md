# Attention Mechanism

As we learned in the ["Encoder-Decoder(seq2seq)"](seq2seq.md) section, the decoder relies on the same context variable at each time step to obtain input sequence information.  When the encoder is an RNN, the context variable will be from the hidden state of its final time step.

Now, let us take another look at the translation example mentioned in that section: the input is an English sequence "They", "are", "watching", ".", and the output is a French sequence "Ils", "regardent", ".". It is not hard to see that the decoder only needs to use partial information from the input sequence to generate each word in the output sequence. For example, at time step 1 of the output sequence, the decoder can mainly rely on the information of "They" and "are" to generate "Ils". At time step 2, mainly the encoded information from "watching" is used to generate "regardent". Finally, at time step 3, the period "." is mapped directly.  At each time step, it looks like the decoder is assigning different attentions to the encoded information of different time steps in the input sequence. This is the source of the attention mechanism[1].

Here, we will continue to use RNN as an example. The attention mechanism obtains the context variable by weighting the hidden state of all time steps of the encoder. The decoder adjusts these weights, i.e., attention weights, at each time step so that different portions of the input sequence can be focused on at different time steps and encoded into context variables of the corresponding time step. In this section, we will discuss how the attention mechanism works.


In the ["Encoder-Decoder(seq2seq)"](seq2seq.md) section, we were able to distinguish between the input sequence/encoder index $t$ and the output sequence/decoder index $t'$. In that section, $\boldsymbol{s}_{t'} = g(\boldsymbol{y}_{t'-1}, \boldsymbol{c}, \boldsymbol{s}_{t'-1})$ is the hidden state of the decoder at time step $t'$.Here, $\boldsymbol{y}_{t'-1}$ is the feature representation of output $y_{t'-1}$ from the previous time step $t'-1$, and any time step $t'$ uses the same context variable $\boldsymbol{c}$. However, in the attention mechanism, each time step of the decoder will use variable context variables. If $\boldsymbol{c}_{t'}$ is the context variable of the decoder at time step $t'$, then the hidden state of the decoder at that time step can be rewritten as

$$\boldsymbol{s}_{t'} = g(\boldsymbol{y}_{t'-1}, \boldsymbol{c}_{t'}, \boldsymbol{s}_{t'-1}).$$

The key here is to figure out how to compute the context variable $\boldsymbol{c}_{t'}$ and use it to update the hidden state $\boldsymbol{s}_{t'}$. Below, we will introduce these two key points separately.


## Compute the Context Variable

Figure 10.12 depicts how the attention mechanism computes the context variable for the decoder at time step 2. First, function $a$ will compute the input of the softmax operation based on the hidden state of the decoder at time step 1 and the hidden states of the encoder at each time step. The Softmax operation outputs a probability distribution and weights the hidden state of each time step of the encoder to obtain a context variable.

![Attention mechanism based on the encoder-decoder. ](../img/attention.svg)


Specifically, if we know that the hidden state of the encoder at time step $t$ is $\boldsymbol{h}_t$ and the total number of time steps is $T$, then the context variable of the decoder at time step $t'$ is the weighted average on all the hidden states of the encoder:

$$\boldsymbol{c}_{t'} = \sum_{t=1}^T \alpha_{t' t} \boldsymbol{h}_t,$$

When $t'$ is given, the value of weight $\alpha_{t't}$ at $t=1, \ldots,T$ is a probability distribution. In order to obtain the probability distribution, we are going to use the softmax operation:

$$\alpha_{t' t} = \frac{\exp(e_{t' t})}{ \sum_{k=1}^T \exp(e_{t' k}) },\quad t=1,\ldots,T.$$

Now we need to define how to compute input $e_{t' t}$ of the softmax operation in the formula above. Since $e_{t' t}$ depends on both the decoder's time step $t'$ and the encoder's time step $t$, we might as well use the decoder's hidden state $\boldsymbol{s}_{t' - 1}$ at that time step $t'-1$ and the encoder's hidden state $\boldsymbol{h}_t$ at that time step as the input and compute $e_{t' t}$ with function $a$.

$$e_{t' t} = a(\boldsymbol{s}_{t' - 1}, \boldsymbol{h}_t).$$


Here, we have several options for function $a$. If the two input vectors are of the same length, a simple choice is to compute their inner product $a(\boldsymbol{s}, \boldsymbol{h})=\boldsymbol{s}^\top \boldsymbol{h}$. In the paper that first introduced the attention mechanism, the authors transformed the concatenated input through a multilayer perceptron with a single hidden layer[1].

$$a(\boldsymbol{s}, \boldsymbol{h}) = \boldsymbol{v}^\top \tanh(\boldsymbol{W}_s \boldsymbol{s} + \boldsymbol{W}_h \boldsymbol{h}),$$

Here, $\boldsymbol{v}$, $\boldsymbol{W}_s$, and $\boldsymbol{W}_h$ are all model parameters that can be learned.

### Vectorization

We can also use vectorization to compute more efficiently within the attention mechanism. Generally speaking, the input of the attention model consists of query entries, key entries, and value entries. There is also a one-to-one correspondence between the key entries and value entries. Here, the value entry is a set of entries that requires a weighted average. In the weighted average, the weight of the value entry is obtained by computing the query entry and the key entry corresponding to the value entry.

In the example above, the query entry is the hidden state of the decoder, and the key entry and value entry are hidden states of the encoder.
Now, we will look at a common simple case where the encoder and decoder have $h$ hidden units and we have the function $a(\boldsymbol{s}, \boldsymbol{h})=\boldsymbol{s}^ \top \boldsymbol{h}$. Assume that we want to compute the context vector $\boldsymbol{c}_{t'}\in \mathbb{R}^{h}$ based on the single hidden state of the decoder $\boldsymbol{s}_{t' - 1} \in \mathbb{R}^{h}$ and all the hidden states of the encoder $\boldsymbol{h}_t \in \mathbb{R}^{h}, t = 1,\ldots,T$.
We can let the query entry matrix $\boldsymbol{Q} \in \mathbb{R}^{1 \times h}$ be $\boldsymbol{s}_{t' - 1}^\top$ and the key entry matrix $\boldsymbol{K} \in \mathbb{R}^{T \times h}$ have the same value as the entry matrix $\boldsymbol{V} \in \mathbb{R}^{T \times h}$, with all the values in row $t$ set to $\boldsymbol{h}_t^\top$. Now, we only need to use vectorization

$$\text{softmax}(\boldsymbol{Q}\boldsymbol{K}^\top)\boldsymbol{V}$$

to compute the transposed context vector $\boldsymbol{c}_{t'}^\top$. When the query entry matrix $\boldsymbol{Q}$ has $n$ rows, the formula above will be able to obtain the output matrix of row $n$. The output matrix and the query entry matrix correspond one-to-one on the same row.



## Update the Hidden State

Using the gated recurrent unit (GRU) as an example, we can modify the design of the GRU slightly in the decoder[1]. The decoder's hidden state at time step $t'$ will be

$$\boldsymbol{s}_{t'} = \boldsymbol{z}_{t'} \odot \boldsymbol{s}_{t'-1}  + (1 - \boldsymbol{z}_{t'}) \odot \tilde{\boldsymbol{s}}_{t'},$$

Here, the candidate implied states of the reset gate and update gate are


$$
\begin{aligned}
\boldsymbol{r}_{t'} &= \sigma(\boldsymbol{W}_{yr} \boldsymbol{y}_{t'-1} + \boldsymbol{W}_{sr} \boldsymbol{s}_{t' - 1} + \boldsymbol{W}_{cr} \boldsymbol{c}_{t'} + \boldsymbol{b}_r),\\
\boldsymbol{z}_{t'} &= \sigma(\boldsymbol{W}_{yz} \boldsymbol{y}_{t'-1} + \boldsymbol{W}_{sz} \boldsymbol{s}_{t' - 1} + \boldsymbol{W}_{cz} \boldsymbol{c}_{t'} + \boldsymbol{b}_z),\\
\tilde{\boldsymbol{s}}_{t'} &= \text{tanh}(\boldsymbol{W}_{ys} \boldsymbol{y}_{t'-1} + \boldsymbol{W}_{ss} (\boldsymbol{s}_{t' - 1} \odot \boldsymbol{r}_{t'}) + \boldsymbol{W}_{cs} \boldsymbol{c}_{t'} + \boldsymbol{b}_s),
\end{aligned}
$$

Here, $\boldsymbol{W}$ and $\boldsymbol{b}$ with subscripts are the weight parameters and bias parameters of the GRU.

## Summary

* We can use different context variables at each time step of the decoder and assign different attentions to the information encoded in different time steps of the input sequence.
* Generally speaking, the input of the attention model consists of query entries, key entries, and value entries. There is also a one-to-one correspondence between the key entries and value entries.
* With the attention mechanism, we can adopt vectorization for higher efficiency.


## Problems

* Based on the model design in this section, why can't we concatenate hidden state $\boldsymbol{s}_{t' - 1}^\top \in \mathbb{R}^{1 \times h}, t' \in 1, \ldots, T'$ from different time steps of the decoder to create the query entry matrix $\boldsymbol{Q} \in \mathbb{R}^{T' \times h}$ to compute context variable $\boldsymbol{c}_{t'}^\top, t' \in 1, \ldots, T'$ of the attention mechanism at different time steps simultaneously?

* Without modifying the function `gru` from the ["Gated Recurrent Unit (GRU)"](../chapter_recurrent-neural-networks/gru.md) section, how can we use it to implement the decoder introduced in this section?

* In addition to natural language processing, where else can the attention mechanism be applied?

## Reference

[1] Bahdanau, D., Cho, K., & Bengio, Y. (2014). Neural machine translation by jointly learning to align and translate. arXiv preprint arXiv:1409.0473.

## Discuss on our Forum

<div id="discuss" topic_id="2395"></div>
