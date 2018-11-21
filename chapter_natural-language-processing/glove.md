# Word Embedding For Global Vectors (GloVe)

First, we should review the skip-gram model in word2vec.  The conditional probability $\mathbb{P}(w_j\mid w_i)$ expressed in the skip-gram model using the softmax operation will be recorded as $q_{ij}$, that is:

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$

其中$\mathbf{v}_i$和$\mathbf{u}_i$分别是索引为$i$的词$w_i$作为中心词和背景词时的向量表示，$\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$为词典索引集。

对于词$w_i$，它在数据集中可能多次出现。我们将每一次以它作为中心词的所有背景词全部汇总并保留重复元素，记作多重集（multiset）$\mathcal{C}_i$。一个元素在多重集中的个数称为该元素的重数（multiplicity）。举例来说，假设词$w_i$在数据集中出现2次：文本序列中以这2个$w_i$作为中心词的背景窗口分别包含背景词索引$2,1,5,2$和$2,3,2,1$。那么多重集$\mathcal{C}_i = \{1,1,2,2,2,2,3,5\}$，其中元素1的重数为2，元素2的重数为4，元素3和5的重数均为1。将多重集$\mathcal{C}_i$中元素$j$的重数记作$x_{ij}$：它表示了整个数据集中所有以$w_i$为中心词的背景窗口中词$w_j$的个数。那么，跳字模型的损失函数还可以用另一种方式表达：

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$

We add up the number of all the context words for the central target word $w_i$ to get $x_i$, and record the conditional probability $x_{ij}/x_i$ for generating context word $w_j$ based on central target word $w_i$ as $p_{ij}$. We can rewrite the loss function of the skip-gram model as

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$

In the formula above, $\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ computes the conditional probability distribution $p_{ij}$ for context word generation based on the central target word $w_i$ and the cross-entropy of conditional probability distribution $q_{ij}$ predicted by the model.  The loss function is weighted using the sum of the number of context words with the central target word $w_i$.  If we minimize the loss function from the formula above, we will be able to allow the predicted conditional probability distribution to approach as close as possible to the true conditional probability distribution.

However, although the most common type of loss function, the cross-entropy loss function is sometimes not a good choice. On the one hand, as we mentioned in the ["Approximate Training"](approx-training.md) section, the cost of letting the model prediction $q_{ij}$ become the legal probability distribution has the sum of all items in the entire dictionary in its denominator. This can easily lead to excessive computational overhead. On the other hand, there are often a lot of uncommon words in the dictionary, and they appear rarely in the data set. In the cross-entropy loss function, the final prediction of the conditional probability distribution on a large number of uncommon words is likely to be inaccurate.



## The GloVe Model

To address this, GloVe, a word embedding model that came after word2vec, adopts square loss and makes three changes to the skip-gram model based on this loss[1].

1. Here, we use the non-probability distribution variables $p'_{ij}=x_{ij}$ and $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ and take their logs. Therefore, we get the square loss $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. We add two scalar model parameters for each word $w_i$: the bias terms $b_i$ (for central target words) and $c_i$( for context words).
3. Replace the weight of each loss with the function $h(x_{ij})$. The weight function $h(x)$ is a monotone increasing function with the range $[0,1]. 

Therefore, the goal of GloVe is to minimize the loss function.

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$

Here, we have a suggestion for the choice of weight function $h(x)$: when $x < c$ (e.g $c = 100$), make $h(x) = (x/c) ^\alpha$ (e.g $\alpha = 0.75$), otherwise make $h(x) = 1$. Because $h(0)=0$, the squared loss term for $x_{ij}=0$ can be simply ignored. When we use mini-batch SGD for training, we conduct random sampling to get a non-zero mini-batch $x_{ij}$ from each time step and compute the gradient to update the model parameters. These non-zero $x_{ij}$ are computed in advance based on the entire data set and they contain global statistics for the data set. Therefore, the name GloVe is taken from "Global Vectors".

Notice that if word $w_i$ appears in the context window of word $w_j$, then word $w_j$ will also appear in the context window of word $w_i$. Therefore, $x_{ij}=x_{ji}$. Unlike word2vec, GloVe fits the symmetric $\log\, x_{ij}$ in lieu of the asymmetric conditional probability $p_{ij}$. Therefore, the central target word vector and context word vector of any word are equivalent in GloVe. However, the two sets of word vectors that are learned by the same word may be different in the end due to different initialization values. After learning all the word vectors, GloVe will use the sum of the central target word vector and the context word vector as the final word vector for the word.


## Understanding GloVe from Conditional Probability Ratios

We can also try to understand GloVe word embedding from another perspective. We will continue the use of symbols from earlier in this section, $\mathbb{P}(w_j \mid w_i)$ represents the conditional probability of generating context word $w_j$ with central target word $w_i$ in the data set, and it will be recorded as $p_{ij}$. From a real example from a large corpus, here we have the following two sets of conditional probabilities with "ice" and "steam" as the central target words and the ratio between them[1]:

|$w_k$=|“solid”|“gas”|“water”|“fashion”|
|--:|:-:|:-:|:-:|
|$p_1=\mathbb{P}(w_k\mid\text{"ice"})$|0.00019|0.000066|0.003|0.000017|
|$p_2=\mathbb{P}(w_k\mid\text{"steam"})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|

We will be able to observe phenomena such as:

* For a word $w_k$ that is related to "ice" but not to "steam", such as $w_k=$"solid", we would expect a larger conditional probability ratio, like the value 8.9 in the last row of the table above.
* For a word $w_k$ that is related to "steam" but not to "ice", such as $w_k=$"gas", we would expect a smaller conditional probability ratio, like the value 0.085 in the last row of the table above.
* For a word $w_k$ that is related to both "ice" and "steam", such as $w_k=$"water", we would expect a conditional probability ratio close to 1, like the value 1.36 in the last row of the table above.
* For a word $w_k$ that is related to neither "ice" or "steam", such as $w_k=$"fashion", we would expect a conditional probability ratio close to 1, like the value 0.96 in the last row of the table above.

We can see that the conditional probability ratio can represent the relationship between different words more intuitively. We can construct a word vector function to fit the conditional probability ratio more effectively. As we know, to obtain any ratio of this type requires three words $w_i$, $w_j$, and $w_k$. The conditional probability ratio with $w_i$ as the central target word is ${p_{ij}}/{p_{ik}}$. We can find a function that uses word vectors to fit this conditional probability ratio.

$$f(\boldsymbol{u}_j, \boldsymbol{u}_k, {\boldsymbol{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$

The possible design of function $f$ here will not be unique. We only need to consider a more reasonable possibility. Notice that the conditional probability ratio is a scalar, we can limit $f$ to be a scalar function: $f(\boldsymbol{u}_j, \boldsymbol{u}_k, {\boldsymbol{v}}_i) = f\left((\boldsymbol{u}_j - \boldsymbol{u}_k)^\top {\boldsymbol{v}}_i\right)$. After exchanging index $j$ with $k$, we will be able to see that function $f$ satisfies the condition $f(x)f(-x)=1$, so one possibility could be $f(x)=\exp(x)$. Thus:

$$f(\boldsymbol{u}_j, \boldsymbol{u}_k, {\boldsymbol{v}}_i) = \frac{\exp\left(\boldsymbol{u}_j^\top {\boldsymbol{v}}_i\right)}{\exp\left(\boldsymbol{u}_k^\top {\boldsymbol{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$

One possibility that satisfies the right side of the approximation sign is $\exp\left(\boldsymbol{u}_j^\top {\boldsymbol{v}}_i\right) \approx \alpha p_{ij}$, where $\alpha$ is a constant. Considering that $p_{ij}=x_{ij}/x_i$, after taking the logarithm we get $\boldsymbol{u}_j^\top {\boldsymbol{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$. We use additional bias terms to fit $- \log\, \alpha + \log\, x_i$, such as the central target word bias term $b_i$ and context word bias term $c_j$:

$$\boldsymbol{u}_j^\top \boldsymbol{v}_i + b_i + c_j \approx \log(x_{ij}).$$

By taking the square error and weighting the left and right sides of the formula above, we can get the loss function of GloVe.


## Summary

* In some cases, the cross-entropy loss function may have a disadvantage. GloVe uses squared loss and the word vector to fit global statistics computed in advance based on the entire data set.
* The central target word vector and context word vector of any word are equivalent in GloVe.


## exercise

* If a word appears in the context window of another word, how can we use the distance between them in the text sequence to redesign the method for computing the conditional probability $p_{ij}$? Hint: See section 4.2 from the paper GloVe[1].
* For any word, will its central target word bias term and context word bias term be equivalent to each other in GloVe? Why? 

## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/4372)

![](../img/qr_glove.svg)

## Reference

[1] Pennington, J., Socher, R., & Manning, C. (2014). Glove: Global vectors for word representation. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP) (pp. 1532-1543).
