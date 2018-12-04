# Recurrent Neural Networks

In the last section, we introduced the $n$-gram, in which the conditional probability of word $w_t$ for time step $t$ based on all previous words only takes $n-1$ number of words from the last time step into account.  If we want to check the possible effect of words earlier than $t-(n-1)$ on $w_t$, we need to increase $n$. However, the number of model parameters would also increase exponentially with it (see the exercises in the last section).

In this section, we will discuss recurrent neural networks (RNNs).  Instead of rigidly remembering all fixed-length sequences, RNNs use hidden states to store information from previous time steps. First, recall the multilayer perceptron introduced earlier, and then discuss how to add a hidden state to turn it into an RNN.


## Neural Networks Without Hidden States

Let us take a look at a multilayer perceptron with a single hidden layer.  Given a mini-batch data instance $\boldsymbol{X} \in \mathbb{R}^{n \times d}$ with sample size $n$ and $d$ inputs (features or feature vector dimensions). Let the hidden layer's activation function be $\phi$, so the hidden layer's output $\boldsymbol{H} \in \mathbb{R}^{n \times h}$ is calculated as

$$\boldsymbol{H} = \phi(\boldsymbol{X} \boldsymbol{W}_{xh} + \boldsymbol{b}_h),$$

Here, we have the weight parameter $\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}$, bias parameter $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$, and the number of hidden units $h$, for the hidden layer. The two equations above have different shapes, so they will be added through the broadcasting mechanism (see section["Data Operation"](../chapter_prerequisite/ndarray.md)). The hidden variable $\boldsymbol{H}$ is used as the input of the output layer. We assume the output number is $q$ (like the number of categories in the classification problem), and the output of the output layer is

$$\boldsymbol{O} = \boldsymbol{H} \boldsymbol{W}_{hq} + \boldsymbol{b}_q,$$

Here, $\boldsymbol{O} \in \mathbb{R}^{n \times q}$ is the output variable, $\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$ is the weight parameter, and $\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$ is the bias parameter of the output layer.  If it is a classification problem, we can use $\text{softmax}(\boldsymbol{O})$ to compute the probability distribution of the output category.


## RNNs with Hidden States

Now we move on to the case where the input data has temporal correlation. Assume that $\boldsymbol{X}_t \in \mathbb{R}^{n \times d}$ is the mini-batch input and $\boldsymbol{H}_t  \in \mathbb{R}^{n \times h}$ is the hidden layer variable of time step $t$ from the sequence.  Unlike the multilayer perceptron, here we save the hidden variable $\boldsymbol{H}_{t-1}$ from the previous time step and introduce a new weight parameter $\boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$, to describe how to use the hidden variable of the previous time step in the current time step. Specifically, the calculation of the hidden variable of the current time step is determined by the input of the current time step together with the hidden variable of the previous time step:

$$\boldsymbol{H}_t = \phi(\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}  + \boldsymbol{b}_h).$$

Compared with the multilayer perceptron, we added one more $\boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$ here. From the relationship between hidden variables $\boldsymbol{H}_t$ and $\boldsymbol{H}_{t-1}$ of adjacent time steps, we know that those variables captured and retained the sequence's historical information up to the current time step, just like the state or memory of the neural network's current time step. Therefore, such a hidden variable is also called a hidden state. Since the hidden state uses the same definition of the previous time step in the current time step, the computation of the equation above is recurrent.  A network that uses such recurrent computation is called a recurrent neural network (RNN).

There are many different RNN construction methods.  RNNs with a hidden state defined by the equation above are very common. Unless otherwise stated, all the RNNs in this chapter are based on the recurrent computation of the hidden state in the equation above. For time step $t$, the output of the output layer is similar to the computation in the multilayer perceptron:

$$\boldsymbol{O}_t = \boldsymbol{H}_t \boldsymbol{W}_{hq} + \boldsymbol{b}_q.$$

RNN parameters include the weight $\boldsymbol{W}_{xh} \in \mathbb{R}^{d \times h}, \boldsymbol{W}_{hh} \in \mathbb{R}^{h \times h}$ of the hidden layer with the bias $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$, and the weight $\boldsymbol{W}_{hq} \in \mathbb{R}^{h \times q}$ of the output layer with the bias $\boldsymbol{b}_q \in \mathbb{R}^{1 \times q}$. It is worth mentioning that RNNs always use these model parameters, even for different time steps. Therefore, the number of RNN model parameters does not grow as the number of time steps increases.

Figure 6.1 shows the computational logic of an RNN at three adjacent time steps. In time step $t$, the computation of the hidden state can be treated as an entry of a fully connected layer with the activation function $\phi$ after concatenating the input $\boldsymbol{X}_t$ with the hidden state $\boldsymbol{H}_{t-1}$ of the previous time step.  The output of the fully connected layer is the hidden state of the current time step $\boldsymbol{H}_t$. Its model parameter is the concatenation of $\boldsymbol{W}_{xh}$ and $\boldsymbol{W}_{hh}$, with a bias of $\boldsymbol{b}_h$. The hidden state of the current time step $t$ $\boldsymbol{H}_t$ will participate in computing the hidden state $\boldsymbol{H}_{t+1}$ of the next time step $t+1$, the result of which will become the input for the fully connected output layer of the current time step.

![An RNN with a hidden state. ](../img/rnn.svg)

As we just mentioned, the computation in hidden state $\boldsymbol{X}_t \boldsymbol{W}_{xh} + \boldsymbol{H}_{t-1} \boldsymbol{W}_{hh}$ equals the result from the concatenated matrix of $\boldsymbol{X}_t$ and $\boldsymbol{H}_{t-1}$ multiplied by the concatenated matrix of $\boldsymbol{W}_{xh}$ and $\boldsymbol{W}_{hh}$.  Next, we will use an example to verify this. First of all, we construct the matrixes `X`, `W_xh`, `H`, and `W_hh`, with the shapes (3,1), (1,4), (3,2), and (2,4), respectively. Multiply `X` by `W_xh`, and `H` by `W_hh`, then add the results from the multiplication to obtain a matrix with the shape (3,4).

```{.python .input  n=1}
from mxnet import nd

X, W_xh = nd.random.normal(shape=(3, 1)), nd.random.normal(shape=(1, 4))
H, W_hh = nd.random.normal(shape=(3, 2)), nd.random.normal(shape=(2, 4))
nd.dot(X, W_xh) + nd.dot(H, W_hh)
```

Concatenate matrices `X` and `H` vertically (dimension 1). The shape of the concatenated matrix is (3,3). It can be seen that the length of the concatenated matrix on dimension 1 is the sum of lengths ($1+2$) from the matrices `X` and `H` on dimension 1.  Then, concatenate matrixes `W_xh` and `W_hh` horizontally (dimension 0). The resulting matrix shape will be (3,4). Lastly, multiply the two concatenated matrices to obtain a matrix that has the same shape (3, 4) as the above code output.

```{.python .input  n=2}
nd.dot(nd.concat(X, H, dim=1), nd.concat(W_xh, W_hh, dim=0))
```

## Application: Character-level RNN Language Model

In the last step, we are going to explain how to use RNN to build a language model. Let the number of mini-batch examples be 1, and the sequence of the text be "想", "要", "有", "直", "升", "机". Figure 6.2 demonstrates how we can use RNN to predict the next character based on the present and previous characters. During the training process, we run a softmax operation on the output from the output layer for each time step, and then use the cross-entropy loss function to compute the error between the result and the label. In Figure 6.2, due to the recurrent computation of the hidden state in the hidden layer, the output of time step 3 $\boldsymbol{O}_3$ is determined by the text sequence "想", "要", "有".  Since the next word of the sequence in the training data is "直", the loss of time step 3 will depend on the probability distribution of the next word generated based on the sequence "想", "要", "有" and the label "直" of this time step.

![Character-level RNN language model. The input and label sequences are "想", "要", "有", "直", "升" and "要", "有", "直", "升", "机", respectively. ](../img/rnn-train.svg)

Since each word entered is a character, this model is called a "character-level recurrent neural network". Since the number of different characters is much smaller than the number of different words (especially for English), the computation of character-level RNNs is usually much simpler. In the next few sections, we will introduce its implementation.


## Summary

* A network that uses recurrent computation is called a recurrent neural network (RNN).
* The hidden state of the RNN can capture historical information of the sequence up to the current time step.
* The number of RNN model parameters does not grow as the number of time steps increases.
* We can create language models using a character-level RNN.

## Problems

* If we use an RNN to predict the next word in a text sequence, how many outputs should be set?
* How can an RNN be used to express the word of a time step based on the conditional probability of all the previous words in the text sequence?

## Discuss on our Forum

<div id="discuss" topic_id="2362"></div>
