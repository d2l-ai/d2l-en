# Recurrent Neural Networks
:label:`chapter_plain_rnn`


In the previous section we introduced $n$-gram models, where the conditional probability of word $w_t$ at position $t$ only depends on the $n-1$ previous words. If we want to check the possible effect of words earlier than $t-(n-1)$ on $w_t$, we need to increase $n$. However, the number of model parameters would also increase exponentially with it, as we need to store $|V|^n$ numbers for a vocabulary $V$. Hence, rather than modeling $p(w_t|w_{t-1}, \ldots w_{t-n+1})$ it is preferable to use a latent variable model in which we have

$$p(w_t|w_{t-1}, \ldots w_1) \approx p(w_t|h_t(w_{t-1}, h_{t-1})).$$

For a sufficiently powerful function $h_t$ this is not an approximation. After
all, $h_t$ could simply store all the data it observed so far. We discussed this
in :numref:`chapter_sequence`. Let's see why
building such models is a bit more tricky than simple autoregressive models
where

$$p(w_t|w_{t-1}, \ldots w_1) \approx p(w_t|f(w_{t-1}, \ldots w_{t-n+1})).$$

As a warmup we will review the latter for discrete outputs and $n=2$, i.e. for Markov model of first order. To simplify things further we use a single layer in the design of the RNN. Later on we will see how to add more expressivity efficiently across items.

## Recurrent Networks Without Hidden States

Let us take a look at a multilayer perceptron with a single hidden layer. Given a mini-batch of instances $\mathbf{X} \in \mathbb{R}^{n \times d}$ with sample size $n$ and $d$ inputs (features or feature vector dimensions). Let the hidden layer's activation function be $\phi$. Hence the hidden layer's output $\mathbf{H} \in \mathbb{R}^{n \times h}$ is calculated as

$$\mathbf{H} = \phi(\mathbf{X} \mathbf{W}_{xh} + \mathbf{b}_h).$$

Here, we have the weight parameter $\mathbf{W}_{xh} \in \mathbb{R}^{d \times
h}$, bias parameter $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$, and the number
of hidden units $h$, for the hidden layer. Recall that $\mathbf{b}_h$ is just a
vector - its values are replicated using the broadcasting mechanism (:numref:`chapter_ndarray`) to match those of the matrix-matrix product.

Also note that hidden *state* and hidden *layer* refer to two very different concepts. Hidden layers are, as explained, layers that are hidden from view on the path from input to output. Hidden states are technically speaking *inputs* to whatever we do at a given step. Instead, they can only be computed by looking at data at previous iterations. In this sense they have much in common with latent variable models in statistics, such as clustering or topic models where e.g. the cluster ID affects the output but cannot be directly observed.

The hidden variable $\mathbf{H}$ is used as the input of the output layer. For classification purposes, such as predicting the next character, the output dimensionality $q$ might e.g. match the number of categories in the classification problem. Lastly the output layer is given by

$$\mathbf{O} = \mathbf{H} \mathbf{W}_{hq} + \mathbf{b}_q.$$

Here, $\mathbf{O} \in \mathbb{R}^{n \times q}$ is the output variable,
$\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ is the weight parameter, and
$\mathbf{b}_q \in \mathbb{R}^{1 \times q}$ is the bias parameter of the output
layer.  If it is a classification problem, we can use
$\text{softmax}(\mathbf{O})$ to compute the probability distribution of the
output category. This is entirely analogous to the regression problem we solved
previously in :numref:`chapter_sequence`, hence we omit details. Suffice it to say that we can
pick $(w_t, w_{t-1})$ pairs at random and estimate the parameters $\mathbf{W}$
and $\mathbf{b}$ of our network via autograd and stochastic gradient descent.

## Recurrent Networks with Hidden States

Matters are entirely different when we have hidden states. Let's look at the structure in some more detail. Assume that $\mathbf{X}_t \in \mathbb{R}^{n \times d}$ is the mini-batch input and $\mathbf{H}_t  \in \mathbb{R}^{n \times h}$ is the hidden layer variable of time step $t$ from the sequence.  Unlike the multilayer perceptron, here we save the hidden variable $\mathbf{H}_{t-1}$ from the previous time step and introduce a new weight parameter $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$, to describe how to use the hidden variable of the previous time step in the current time step. Specifically, the calculation of the hidden variable of the current time step is determined by the input of the current time step together with the hidden variable of the previous time step:

$$\mathbf{H}_t = \phi(\mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}  + \mathbf{b}_h).$$

Compared with the multilayer perceptron, we added one more $\mathbf{H}_{t-1} \mathbf{W}_{hh}$ here. From the relationship between hidden variables $\mathbf{H}_t$ and $\mathbf{H}_{t-1}$ of adjacent time steps, we know that those variables captured and retained the sequence's historical information up to the current time step, just like the state or memory of the neural network's current time step. Therefore, such a hidden variable is also called a hidden state. Since the hidden state uses the same definition of the previous time step in the current time step, the computation of the equation above is recurrent, hence the name recurrent neural network (RNN).

There are many different RNN construction methods.  RNNs with a hidden state defined by the equation above are very common. For time step $t$, the output of the output layer is similar to the computation in the multilayer perceptron:

$$\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq} + \mathbf{b}_q$$

RNN parameters include the weight $\mathbf{W}_{xh} \in \mathbb{R}^{d \times h}, \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ of the hidden layer with the bias $\mathbf{b}_h \in \mathbb{R}^{1 \times h}$, and the weight $\mathbf{W}_{hq} \in \mathbb{R}^{h \times q}$ of the output layer with the bias $\mathbf{b}_q \in \mathbb{R}^{1 \times q}$. It is worth mentioning that RNNs always use these model parameters, even for different time steps. Therefore, the number of RNN model parameters does not grow as the number of time steps increases.

The figure below shows the computational logic of an RNN at three adjacent time steps. In time step $t$, the computation of the hidden state can be treated as an entry of a fully connected layer with the activation function $\phi$ after concatenating the input $\mathbf{X}_t$ with the hidden state $\mathbf{H}_{t-1}$ of the previous time step.  The output of the fully connected layer is the hidden state of the current time step $\mathbf{H}_t$. Its model parameter is the concatenation of $\mathbf{W}_{xh}$ and $\mathbf{W}_{hh}$, with a bias of $\mathbf{b}_h$. The hidden state of the current time step $t$ $\mathbf{H}_t$ will participate in computing the hidden state $\mathbf{H}_{t+1}$ of the next time step $t+1$, the result of which will become the input for the fully connected output layer of the current time step.

![An RNN with a hidden state. ](../img/rnn.svg)

As discussed, the computation in the hidden state uses $\mathbf{H}_t = \mathbf{X}_t \mathbf{W}_{xh} + \mathbf{H}_{t-1} \mathbf{W}_{hh}$ to generate an object matching $\mathbf{H}_{t-1}$ in dimensionality. Moreover, we use $\mathbf{H}_t$ to generate the output $\mathbf{O}_t = \mathbf{H}_t \mathbf{W}_{hq}$.

```{.python .input  n=1}
from mxnet import nd

# Data X and hidden state H
X = nd.random.normal(shape=(3, 1))
H = nd.random.normal(shape=(3, 2))

# Weights
W_xh = nd.random.normal(shape=(1, 2))
W_hh = nd.random.normal(shape=(2, 2))
W_hq = nd.random.normal(shape=(2, 3))

def net(X, H):
    H = nd.relu(nd.dot(X, W_xh) + nd.dot(H, W_hh))
    O = nd.relu(nd.dot(H, W_hq))
    return H, O
```

The recurrent network defined above takes observations `X` and a hidden state
`H` as arguments and uses them to update the hidden state and emit an output
`O`. Since this chain could go on for a very long time, training the model with
backprop is out of the question (at least without some approximation). After
all, this leads to a very long chain of dependencies that would be prohibitive
to solve exactly: books typically have more than 100,000 characters and it is
unreasonable to assume that the later text relies indiscriminately on all
occurrences that happened, say, 10,000 characters in the past. Truncation
methods such as BPTT (:numref:`chapter_bptt`) and long short term memory (:numref:`chapter_lstm`) are useful to address this in a more principled manner. For now, let's see how a state update works.

```{.python .input}
(H, O) = net(X,H)
print(H, O)
```

## Steps in a Language Model

We conclude this section by illustrating how RNNs can be used to build a language model. For simplicity of illustration we use words rather than characters, since the former are easier to comprehend. Let the number of mini-batch examples be 1, and the sequence of the text be the beginning of our dataset, i.e. "the time machine by h. g. wells". The figure below illustrates how to estimate the next character based on the present and previous characters. During the training process, we run a softmax operation on the output from the output layer for each time step, and then use the cross-entropy loss function to compute the error between the result and the label. Due to the recurrent computation of the hidden state in the hidden layer, the output of time step 3 $\mathbf{O}_3$ is determined by the text sequence "the", "time", "machine".  Since the next word of the sequence in the training data is "by", the loss of time step 3 will depend on the probability distribution of the next word generated based on the sequence "the", "time", "machine" and the label "by" of this time step.

![Word-level RNN language model. The input and label sequences are `The Time Machine by H.` and `Time Machine by H. G.` respectively. ](../img/rnn-train.svg)

The number of words is huge compared to the number of characters. This is why quite often (such as in the subsequent sections) we will use a character-level RNN instead. In the next few sections, we will introduce its implementation.


## Summary

* A network that uses recurrent computation is called a recurrent neural network (RNN).
* The hidden state of the RNN can capture historical information of the sequence up to the current time step.
* The number of RNN model parameters does not grow as the number of time steps increases.
* We can create language models using a character-level RNN.

## Exercises

1. If we use an RNN to predict the next character in a text sequence, how many output dimensions do we need?
1. Can you design a mapping for which an RNN with hidden states is exact? Hint - what about a finite number of words?
1. What happens to the gradient if you backpropagate through a long sequence?
1. What are some of the problems associated with the simple sequence model described above?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2362)

![](../img/qr_rnn.svg)
