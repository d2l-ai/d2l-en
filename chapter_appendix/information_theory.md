# Information Theory
:label:`chapter_information_theory`

Machine learning, which studies how computer systems can use data to improve performance, is aimed to extract critical and interesting signals from data. While, information theory, which studies encoding, decoding, transmitting and manipulating information (aka data), can help us measure and compare how much information are presented in different signals. As a result, information theory provides a strong backbone to the fundamental of machine learning. For example, “cross entropy loss” is used as an objective function in many machine learning algorithms. This is a direct application of an information theory to measure the loss. In addition, in the rule-based learning algorithm (such as the decision tree algorithm), we can use entropy as a unit to quantify the amount of information gained from the best split at each node. In this section, we will talk about the most fundamental concept of information theory and its applications via several examples.

## Information
### Motivation

"Information" is conveyed or represented by a particular arrangement or sequence of things. It cares about the abstract possibility rather than the "knowledge" within the entity. Suppose there is a file on your laptop and you are curious about how much information it contains. No matter whether it is a song, a movie or a text document, you may find the amount of information is measured in a unit of *bit*, for the size of the file. 

So what is a "bit" and why do we use it to measure information? Historically, an antique transmitter can only send or receive two kinds of code: "0" and "1". In this way, any information symbol is encoded by a series of "0" and "1" combinations, and each digit represents 1 *bit*. 

But since information is intereted in the "abstract possibility", how do we map the possibility to the number of bits? Recall that for any series of codes, each "0" or "1" occurs with a probability of $\frac{1}{2}$. Hence, a series of codes with length $n$ occurs with a probability of $\frac{1}{2^n}$. At the same times, this series contains $n$ bits of information. Therefore, we can define *self-information (or surprisal)*: 

$$I(X) = - \log_2 (p(x)) = - \log (p(x)).$$

as the *bits* of information we have received for this code $X$. Note, in this section, we will always use base-2 logarithms, *i.e.*, $log (\cdot) = log_2 (\cdot)$. You may also hear of the term *nats*, which is another measurement unit using base-$e$ logarithms.

For example, the code "0010" has a self-information:
$$I(\text{"0010"}) = - \log_2 (p(\text{"0010"})) = - \log (p(\text{"0010"})) = - \log (\frac{1}{2^4}) = 4 \text{ bits}.$$

## Entropy 
### Definition

As self-information only measures the information of a single discrete propbability, we need a more generalized measure for any random variable with either discrete or continuous distribution. Here, for any random variable $X$ that follows a probability distribution $P$ with a p.d.f. or a p.m.f. $p(x)$, we measure the expected amount of information through *entropy (or Shannon entropy)*:

$$H(X) = - E_{x \sim P} [\log p(x)].$$

* If $X$ is discrete, $H(X) = - \sum_i p_i \cdot \log p_i$, where $p_i = P(X_i)$ ;
* If $X$ is continuous, then we also refer entropy as *differential entropy*, $H(X) = - \int_x p(x) \log p(x) dx$.


### Example
Suppose you are a movie director and you want to collect some feedbacks of your new released movie. There are three online movie review vendors, all of whom provide both positive and negative reviews, but with a different degree of polarity:

* The dataset from vendor $A$ contains $\frac{1}{6}$ of positive and $\frac{5}{6}$ reviews;
* The dataset from vendor $B$ contains $\frac{1}{3}$ of positive and $\frac{2}{3}$ reviews;
* The dataset from vendor $C$ contains half of positive and half of negative reviews.

Without loss of generality, assume any review is of the similar length and at the same price. Therefore, each review possesses the same amount of information per dollar. With the limited budget, you need to choose one vendor to partner with. Intuitively, the more balance the dataset is, the more comprehensive feedback we can obtain. Therefore, vendor $C$ may be the best one to coorperate with. But can we make the decision by more convincible math? Well, information theory can help.

Let us compute the entropy of each dataset:

* The entropy of movie reviews from vendor $A$'s dataset:
$$H(A) = - E_{x \sim P_{A}} [\log p_{A}(x)] = - [\frac{1}{6} \times \log(\frac{1}{6}) +\frac{5}{6} \times \log(\frac{5}{6})] = 0.1957;$$

* The entropy of movie reviews from vendor $B$'s dataset:
$$H(B) = - E_{x \sim P_{B}} [\log p_{B}(x)] = - [\frac{1}{3} \times \log(\frac{1}{3}) +\frac{2}{3} \times \log(\frac{2}{3})] = 0.2764;$$

* The entropy of movie reviews from vendor $C$'s dataset:
$$H(C) = - E_{x \sim P_{C}} [\log p_{C}(x)] = - [\frac{1}{2} \times \log(\frac{1}{2}) +\frac{1}{2} \times \log(\frac{1}{2})] = 0.3010.$$


As a result, the vendor $C$'s dataset has the largest entropy of movie reviews and hence its reviews may be more comprehensive to be analyzed.



### Interpretations

You may be curious about why using an expectation of a negative logarithm in the entropy function. Here are some intuitions:

* Suppose the $p(x) = f_1(x) f_2(x) ... f_n(x)$, where $f_i(x)$ functions are independent, hence each $f_i(x)$ contributes individually to the information we obtain from $p(x)$. As a result, we need the entropy formula to possess the "additive" functionality over mixed independent distributions. Therefore, the logarithm function $\log$ can naturally turn the product of different probability distributions to the summation of their information.

* The more commonly encountered events should contain less information than less likely events, since we often learn more from the unusual case than the regular one. However, because of $\log$ monotonically increasing with the probabilities, we need a "negative" sign to reverse their relationship.

* Since the random variable $X$ follows a given probability distribution, we can interpret entropy as the average amount of surprise from observing an event, or the average number of bits needed to communicate the information. For example, imagine that a slot machine emits statistical independently symbols ${s_1,...s_k}$ with probabilities ${p_1,...p_k}$ respectively, the average amount of self-information in observing the output is the entropy of this system:
$$H(S) = \sum_i {p_i \cdot I(s_i)} = - \sum_i {p_i \cdot \log p_i}.$$


### Properties of Entropy

From the above examples and interpretations, we can derive the following properties of entropy:

1. Entropy is non-negative: $H(X) \geq 0, \forall X$.
1. If $X \sim P$ with a p.d.f. or a p.m.f. $p(x)$, and we try to estimate $P$ by a new probability distribution $Q$ with a p.d.f. or a p.m.f. $q(x)$, then : $$H(X) = - E_{x \sim P} [\log p(x)] <=  - E_{x \sim P} [\log q(x)], \text{ with equality if and only if } P = Q.$$ Alternatively, $H(X)$ gives a lower bound on the number of bits needed on average to encode symbols drawn from a distribution $P$.
1. If $X \sim P$, then $x$ convey the maximum amount of information if it is evenly spread among all possible outcomes. 
    a. If $P$ is discrete, then $H(X) \leq \log(k)$, with equality if and only if $p_i = \frac{1}{k}, \forall x_i$.
    a. If $P$ is continuous, then the further $P$ is from uniform, the lower the entropy is.



## Mutual Information

![](../img/mutual_information.png)


Last section we defined entropy for a single random variable, $X$, now let us extend it to a pair, $(X,Y)$. For the following discussion, we always use $(X,Y)$ as a pair of random variables that follows a joint probability distribution $P$ with a p.d.f. or a p.m.f. $p(x,y)$.


### Joint Entropy 

Similarly to entropy, the *joint entropy* $H(X,Y)$ is defined as
$$H(X,Y) = −E_{(x,y) \sim P} [\log p(x, y)]. $$

* If $(X,Y)$ is a pair of discrete random variables, then $$H(X,Y) = - \sum_{x} \sum_{y} p(x,y) \log p(x,y);$$

* If $(X,Y)$ is a pair of continuous random variables, then the differential joint entropy is defined as $$H(X,Y) = - \int_{x, y} p(x,y) \ \log p(x,y) \ dx dy.$$


### Conditional Entropy

The conditional entropy  $H(Y \mid X)$ is defined as

\begin{align}
H(Y \mid X) &= - E_{(x,y) \sim P} [\log p(y \mid x)]. \\
\end{align}

* If $(X,Y)$ is a pair of discrete random variables, then $$H(Y \mid X) = - \sum_{x} \sum_{y} p(x,y) \log p(y \mid x);$$

* If $(X,Y)$ is a pair of continuous random variables, then the differential joint entropy is defined as $$H(Y \mid X) = - \int_x \int_y p(x,y) \ \log p(y \mid x) \ dx dy.$$


You may wonder: “Why do we need to know joint entropy and conditional entropy?” The naturalness of the definition alludes that the entropy of a pair of random variables is the entropy of one plus the conditional entropy of the other: 

$$H(X,Y) = H(X)+H(Y \mid X).$$



### Mutual Information I(X,Y)

Suppose that (X, Y) follows the probability distribution $p_{X,Y}(x, y)$, while X and Y follow probability distribution $p_X(x)$ and $p_Y(y)$, respectively. It is reasonable to ask: "How much information do X and Y share?" Well, *mutual information* of $(X,Y)$ can tell us:  

$$I(X,Y) = −E_{x} E_{y} \left\{ p_{X,Y}(x, y) \log\frac{p_{X,Y}(x, y)}{p_X(x) p_X(x)} \right\}. $$


As shown in the earlier image, if we know H(X) (*i.e.*, the entropy X) and H(X \mid Y) (*i.e.*, the conditional entropy of X given Y), mutual information tells us the average reduction in uncertainty about X that results from learning the value of Y.

\begin{align}
I(X,Y) &= H(X) − H(X \mid Y) \\
&= H(Y) − H(Y \mid X) \\
&= H(X) + H(Y) − H(X,Y) \\
&= H(X,Y) - H(Y \mid X) − H(X \mid Y). \\
\end{align}


### Properties of Mutual Information

A few notable notations to keep in mind:

1. Symmetric: $I(X,Y) = I(Y,X)$;
2. Non-negative: $I(X,Y) \geq 0$;
3. $I(X,Y) = 0$ if and only if $X$ and $Y$ are independent. For example, if $X$ and $Y$ are independent, then knowing $Y$ does not give any information about $X$ and vice versa, so their mutual information is zero;
4. Alternatively, if $X$ is a deterministic function of $Y$, then all information conveyed by $Y$ is shared with $X$: knowing $Y$ determines the value of $X$ and vice versa;





## Kullback–Leibler Divergence

### Motivation

Norm is used to measure the distance between two points in any dimension space, but can we measure the distance between two probability distributions, $P$ and $Q$? The answer points out to *Kullback–Leibler Divergence*, which is extremely useful in many machine learning problems when we are estimating a true probability distribution $P$ with another probability distribution $Q$, and computing the error.


### Definition

Given a random variable $X$ that follows the true probability distribution $P$ with a p.d.f. or a p.m.f. $p(x)$, and we estimate $P$ by another probability distribution $Q$ with a p.d.f. or a p.m.f. $q(x)$, the *Kullback–Leibler(KL) Divergence* (or *Relative Entropy*) between $P$ and $Q$ is:

$$D_{\mathrm{KL}}(P\|Q) = E_{x \sim P} [\log \frac{p(x)}{q(x)}] = E_{x \sim P} [\log p(x) - \log q(x)].$$


### Example: Implementation of KL Divergence from Scratch

Firstly, let us build a toy model: generating and sorting three 1,000,000 length ndarrays:

* an objective ndarray, $p$, follows a normal distribution $N(0,1)$; 
* two candidate ndarrays, $q_1$ and $q_2$, follow normal distributions $N(-1,1)$ and $N(1,1)$ respectively.


```{.python .input  n=2}
shape = 1000000
p = np.random.normal(loc=0.0, scale=1.0, size=(shape, 1))
q1 = np.random.normal(loc=-1.0, scale=1.0, size=(shape, 1))
q2 = np.random.normal(loc=1.0, scale=1.0, size=(shape, 1))

p = mxnet.np.array(sorted(p.asnumpy()))
q1 = mxnet.np.array(sorted(q1.asnumpy()))
q2 = mxnet.np.array(sorted(q2.asnumpy()))
```


Next, define the above KL Divergence formula in MXNet.

```{.python .input  n=1}
from mxnet import np
from mxnet.ndarray import nansum

def KLDivergence(p, q):
    kl = p * np.log(p / q)
    out = nansum(kl.reshape(-1).as_nd_ndarray())
    return out
```


Since $q_1$ and $q_2$ are symmetric about the y-axis, we expect a similar absolute value of KL divergence between $D_{\mathrm{KL}}(p\|q_1)$ and $D_{\mathrm{KL}}(p\|q_2)$. 


```{.python .input  n=2}
kl_pq1 = KLDivergence(p,q1)
print('KL Divergence between p and q1 is '.format(kl_pq1))

kl_pq2 = KLDivergence(p,q2)
print('KL Divergence between p and q2 is '.format(kl_pq2))

kl_q2p = KLDivergence(q2,p)
print('KL Divergence between q2 and p is '.format(kl_q2p))
```

On the flip side, you may find $D_{\mathrm{KL}}(q_2 \|p)$ and $D_{\mathrm{KL}}(p \| q_2)$ are off a lot in absolute value, and that comes to the following properties of KL Divergence.



### KL Divergence Properties

From the above example, we can conclude the follwoing properties:

1. Non-symmetric: $D_{\mathrm{KL}}(P\|Q) \neq D_{\mathrm{KL}}(Q\|P)$, if $P \neq Q$;
1. Non-negative: $D_{\mathrm{KL}}(P\|Q) \geq 0$, with equality holds only when $P = Q$;
1. Infinite case: If there exists an $x$ such that $p(x) > 0$ and $q(x) = 0$, then $D_{\mathrm{KL}}(P\|Q) = \infty$.
1. Relationship with mutual information: 

\begin{align}
I(X,Y) &= D_{\mathrm{KL}}(P(X, Y)  \ \| \ P(X)P(Y)) \\
  &= E_Y \{ D_{\mathrm{KL}}(P(X \mid Y) \ \| \ P(X)) \}\\
  &= E_X \{ D_{\mathrm{KL}}(P(Y \mid X) \ \| \ P(Y)) \}.\\
\end{align}


### Application: Variational Inequality

A renowned application of KL divergence is *Variational Inequality*:

$$\log E_{x \sim P} (X) = \sup_Q \{E_{x \sim Q} (\log(x)) - D_{\mathrm{KL}} (Q\|P) \}. $$


Even though the equation looks a bit formidable, it is convenient to use it in approximating a distribution, especially in *Generative Adversarial Networks(GAN)*. Here is the essence of the algorithm: for a given unknown true distribution $P$, we can apply Variational Inequality to estimate $P$ with a new probability distribution $Q$. Then we can calculate the estimation error within the current batch of data, update the weights, move to the next batch and apply the same algorithm.



## Cross Entropy

### Motivation

Say now we have a binary classification problem. Assume that we encode "1" and "0" as the positive and negative class label, respectively, and our neural network is parameterized by $\theta$. Using the maximum log-likelihood approach, for true labels $y_i$ and predictions $\hat{y_i}= p_{\theta}(y_i \mid x_i)$, the probability of being classified as positive is $\pi_i= p_{\theta}(y_i \mid x_i)$. Hence, the likelihood function would be:

\begin{align}
\mathcal{L}(\theta) &= \prod_{i=1}^n \pi_i^{y_i} (1 - \pi_i)^{1 - y_i} \\
  &= \prod_{i=1}^n  p_{\theta}(y_i \mid x_i)^{y_i}  (1 - p_{\theta}(y_i \mid x_i))^{1 - y_i}.\\
\end{align}

And the log-likelihood function would be:

\begin{align}
\mathcal{l}(\theta) &= \sum_{i=1}^n y_i \log(\pi_i) + (1 - y_i) \log (1 - \pi_i) \\
  &= \sum_{i=1}^n {y_i} \log(p_{\theta}(y_i \mid x_i))  + {(1 - y_i)} \log(1 - p_{\theta}(y_i \mid x_i)).\\
\end{align}


Maximizing the log-likelihood function $\mathcal{l}(\theta)$ is identical to minimizing $- \mathcal{l}(\theta)$. We also called $-\mathcal{l}(\theta)$ the *Cross Entropy loss* $\mathcal{CE}(y, \hat{y})$, where $y$ follows the true distribution $P$ and $\hat{y}$ follows the estimating distribution $Q$.



### Definition

Like KL Distance, for a random variable $X$, we can also measure the divergence between estimating distribution $Q$ and true distribution $P$ with *Cross Entropy $H(P,Q)$*:

$$H(P,Q) = - E_{x \sim P} [\log(q(x))].$$

It can also be interpreted as the summation of the entropy $H(P)$ and the KL Divergence between $P$ and $Q$, *i.e.*,

$$H(P, Q) = H(P) + D_{\mathrm{KL}}(P\|Q).$$

As widely accepted, cross entropy can be used to define a loss function in the optimization problem, where the true distribution, $P$, is the true label, and the estimating distribution, $Q$, is the predicted value of the current model. As we allude in the motivation, it turns out that the following are equivalent:

1. Maximizing Predictive Probability of $Q$ for distribution $P$, (i.e., $E_{x 
\sim P} [\log (q(x))]$);
2. Minimizing Cross Entropy $H(P;Q)$;
3. Minimizing the distance $D_{\mathrm{KL}}(P\|Q)$.


### Implementation of Cross Entropy from Scratch

Firstly, let us define the cross entropy formula in MXNet.

```{.python .input  n=5}
from mxnet import np

def cross_entropy(y_hat, y):
    ce = - np.log(y_hat[range(len(y_hat)), y])
    return ce.mean()
```

Now define two ndarrays for the labels and predictions, and the cross entropy would be:

```{.python .input  n=5}
labels = np.array([0, 2])
preds = np.array([[0.3, 0.6, 0.1], [0.2, 0.3, 0.5]])

cross_entropy(preds, labels)
```



### Cross Entropy as An Objective Function of Multi-class Classification

If we dive deep into the classification objective function with cross entropy loss $\mathcal{CE}$, you may find minimizing $\mathcal{CE}$ is equivalent to maximizing the log-likelihood function $\mathcal{L}$.


Let us represent any $k$-class labels by *one-hot encoding* method: setting the component corresponding to its category to be 1, and all other components to be 0, i.e. 

\begin{equation}
  \mathbf{y_i} = (y_{i1}, ..., y_{ik}) \text{, where } y_{ij} = \begin{cases}
    1, & \text{if data point $i$ belongs to category $j$ ;}\\
    0, & \text{otherwise}.
  \end{cases}
\end{equation}


For instance, if a multi-class classfication problem contains three classes A, B and C, then the labels $\mathbf{y_i}$ can be encoded in {A: (1, 0, 0); B: (0, 1, 0); C: (0, 0, 1)}.


Assume that there are $k$ classes and our neural network is parameterized by $\theta$. For true label vectors $\mathbf{y_i}$ and predictions $\hat{\mathbf{y_i}}= p_{\theta}(\mathbf{y_i} \mid \mathbf{x_i})$, the *cross entropy (CE) loss* would be:


\begin{align}
\mathcal{CE}(\mathbf{y}, \hat{\mathbf{y}}) &= - \sum_{i=1}^n \mathbf{y_i} \log \hat{\mathbf{y_i}} \\
  &= - \sum_{i=1}^n \mathbf{y_i} \log{p_{\theta} (\mathbf{y_i}  \mid  \mathbf{x_i})} \\
  &= - \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x_i})}.\\
\end{align}


At the same time, we can also demystify the myth through probabilistic approach. Since each data point, $\mathbf{y_i}$, is following a Multinoulli distribution with probabilities $\boldsymbol{\pi} =$ ($\pi_{1}$, ..., $\pi_{k}$), then the joint probability mass function of $\mathbf{y_i}$ is:

$$\boldsymbol{\pi_i}^\mathbf{y_i} = \prod_{j=1}^k \pi_{ij}^{y_{ij}} = \prod_{j=1}^k p_{\theta}(y_{ij} \mid \mathbf{x_i})^{y_{ij}}.$$


Hence, the likelihood function would be:

\begin{align}
\mathcal{L}(\theta) = \prod_{i=1}^n \boldsymbol{\pi_i}^{\mathbf{y_i}}
 = \prod_{i=1}^n \prod_{j=1}^k \pi_{ij}^{y_{ij}}
 = \prod_{i=1}^n \prod_{j=1}^k p_{\theta}(y_{ij} \mid \mathbf{x_i})^{y_{ij}}.\\
\end{align}

And the log-likelihood function would be:


\begin{align}
\mathcal{l}(\theta) = \log p_{\theta}(\mathbf{y} \mid \mathbf{x}) = \sum_{i=1}^n \sum_{j=1}^k y_{ij} \log{p_{\theta} (y_{ij}  \mid  \mathbf{x_i})}.\\
\end{align}


Therefore, for any multi-class classification, maximizing the above log-likelihood function $\mathcal{l}(\theta)$ is equivalent to minimizing the CE loss $\mathcal{CE}(y, \hat{y})$.


To verify the above proof by algorithm, let us apply the built-in metric `NegativeLogLikelihood` in MXNet. Using the same "labels" and "preds" as the earlier example, we will get the same numerical loss:

```{.python .input  n=5}
from mxnet.metric import NegativeLogLikelihood
nll_loss = NegativeLogLikelihood()
nll_loss.update(labels.as_nd_ndarray(), pred.as_nd_ndarray())
print(nll_loss.get())
```