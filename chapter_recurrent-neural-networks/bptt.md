# Backpropagation Through Time

So far we repeatedly alluded to things like *exploding gradients*,
*vanishing gradients*, truncating backprop, and the need for
*detaching the computational graph*. None of this was really fully
explained, in the interest of being able to build a model quickly and
to see how it works. In this section we will delve a bit more deeply
into the details of backpropagation for sequence models and why (and
how) the math works. For a more detailed discussion, e.g. about
randomization and backprop also see the paper by
[Tallec and Ollivier, 2017](https://arxiv.org/abs/1705.08209).

We encountered some of the effects of gradient explosion when we first
[implemented recurrent neural networks](rnn-scratch.md). In
particular, if you solved the problems in the problem set, you would
have seen that gradient clipping is vital to ensure proper
convergence. To provide a better understanding of this issue, this
section will review how gradients are computed for sequences. Note
that there is nothing conceptually new in how it works, but it is
worth while reviewing the chain rule and
[backpropagation](../chapter_deep-learning-basics/backprop.md) for
another time.

Forward propagation in a recurrent neural network is relatively
straightforward. Back-propagation through time is actually a specific
application of back propagation in recurrent neural networks. It
requires us to expand the recurrent neural network by time step to
obtain the dependencies between model variables and parameters. Then,
based on the chain rule, we apply back propagation to compute and
store gradients.


## Recurrent Model

We start with a simple and abstract definition of how the RNN might work:

$$h_t = f(x_t, h_{t-1}, w) \text{ and } o_t = g(h_t, w)$$

In other words, we have a chain of values that depend on each other via recursive computation. The forward pass is fairly straightforward. All we need is to loop through the $(x_t, h_t, o_t)$ triples one step at a time. This is then evaluated by an objective function

$$L(x,y, w) = \sum_{t=1}^T l(y_t, o_t).$$

For backpropagation matters are a bit more tricky. Let's compute the gradients with regard to the parameters $w$ of the objective function $L$. We get that

$$\begin{aligned}
\partial_{w} L & = \sum_{t=1}^T \partial_w l(y_t, o_t) \\
	& = \sum_{t=1}^T \partial_{o_t} l(y_t, o_t) \partial_w g(h_t, w) \partial_w h_t
\end{aligned}$$

The first part of the derivative is easy to compute (this is after all the instantaneous loss gradient at time $t$). The second part is where things get tricky, since we need to compute the effect of the parameters on $h_t$. For each term we have the recursion:

$$\begin{aligned}
	\partial_w h_t & = \partial_w f(x_t, h_{t-1}, w) + \partial_h f(x_t, h_{t-1}, w) \partial_w h_{t-1} \\
	& = \sum_{i=t}^1 \left[\prod_{j=t}^i \partial_h f(x_j, h_{j-1}, w) \right] \partial_w f(x_{i}, h_{i-1}, w)
\end{aligned}$$

It is evident that this chain can get *very* long, whenever $t$ is large. While we can use the chain rule to compute $\partial_w h_t$ recursively, we might as well try to truncate the sum earlier. There are a number of strategies:

**Compute the full sum.** This is very slow and gradients can blow up,
since subtle changes in the initial conditions can potentially affect
the outcome a lot. That is, we could see things similar to the
butterfly effect. This is actually quite undesirable in our situation
since we want to obtain robust estimators that generalize well. Hence
it isn't used in practice.
  
**Truncate the sum after $\tau$ steps.** This is what we've been
discussing so far. This leads to an *approximation* of the true
gradient, simply by terminating the sum above at $\partial_w
h_{t-\tau}$. The approximation error is thus given by $\partial_h
f(x_t, h_{t-1}, w) \partial_w h_{t-1}$ (multiplied by a product of
gradients involving $\partial_h f$). In practice this works quite
well. It is what is commonly referred to as truncated BPTT
(backpropgation through time). One of the consequences of this is that
the model focuses primarily on short-term influence rather than
long-term consequences. This is actually *desirable*, since it biases
the estimate towards simpler models.

**Randomized Truncation.** Lastly we can replace $\partial_w h_t$ by a
random variable which is correct in expectation but which truncates
the sequence. This is achieved by using a sequence of $\xi_t$ where
$\mathbf{E}[\xi_t] = 1$ and $\Pr(\xi_t = 0) = 1-\pi$ and furthermore
$\Pr(\xi_t = \pi^{-1}) = \pi$. We use this to replace the gradient:

$$z_t  = \partial_w f(x_t, h_{t-1}, w) + \xi_t \partial_h f(x_t, h_{t-1}, w) \partial_w h_{t-1}$$

It follows from the definition of $\xi_t$ that $\mathbf{E}[z_t] = \partial_w h_t$. Whenever $\xi_t = 0$ the expansion terminates at that point. This leads to a weighted sum of sequences of varying lengths. [Tallec and Ollivier, 2017](https://arxiv.org/abs/1705.08209) proposed this in their paper and carried out experiments regarding the efficacy of the model.

Unfortunately, while appealing in theory, the model does not lead to a significantly improved result, most likely due to a number of factors. Firstly, the effect of an observation after a number of backpropagation steps into the past is quite sufficient to capture dependencies in practice. Secondly, the increased variance counteracts the fact that the gradient is more accurate.

![From top to bottom: randomized BPTT, regularly truncated BPTT and full BPTT](../img/truncated-bptt.svg)

To keep things simple, we consider an unbiased recurrent neural network, with the activation function set to identity mapping ($\phi(x)=x$). We set the input of the time step $t$ to a single example $\mathbf{x}_t \in \mathbb{R}^d$ and use the label $y_t$, so the calculation expression for the hidden state $\mathbf{h}_t \in \mathbb{R}^h$ is:

$$\mathbf{h}_t = \mathbf{W}_{hx} \mathbf{x}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1},$$

Here, $\mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ and $\mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$ are the weight parameters of the hidden layer. Assuming the output layer weight parameter is $\mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$, the output layer variable $\mathbf{o}_t \in \mathbb{R}^q$ for time step $t$ can be calculated as follows:

$$\mathbf{o}_t = \mathbf{W}_{qh} \mathbf{h}_{t}.$$

Let the loss at time step $t$ be defined as $\ell(\mathbf{o}_t, y_t)$. Thus, the loss function $L$ for $T$ time steps is defined as:

$$L = \frac{1}{T} \sum_{t=1}^T \ell (\mathbf{o}_t, y_t).$$

In what follows, we will refer to $L$ as the "objective function" for the data instance of a given time step.


## Model Computational Graph

In order to visualize the dependencies between model variables and parameters during computation in a recurrent neural network, we can draw a computational graph for the model, as shown in Figure 6.3. For example, the computation of the hidden states of time step 3 $\mathbf{h}_3$ depends on the model parameters $\mathbf{W}_{hx} and \mathbf{W}_{hh}$, the hidden state of the last time step $\mathbf{h}_2$, and the input of the current time step $\mathbf{x}_3$.


![ Computational dependencies for a recurrent neural network model with three time steps. Boxes represent variables (not shaded) or parameters (shaded) and circles represent operators. ](../img/rnn-bptt.svg)

## Back-propagation Through Time

As just mentioned, the model parameters in Figure 6.3 are $\mathbf{W}_{hx}$, $\mathbf{W}_{hh}$, and $\mathbf{W}_{qh}$. Similar to the ["Forward Propagation, Back Propagation, and Computational Graphs"](../chapter_deep-learning-basics/backprop.md) section, model training generally requires the model parameter gradients $\partial L/\partial \mathbf{W}_{hx}$, $\partial L/\partial \mathbf{W}_{hh}$, and $\partial L/\partial \mathbf{W}_{qh}$.
According to the dependencies shown in Figure 6.3, we can calculate and store the gradients in turn going in the opposite direction of the arrows in the figure. To simplify the explanation, we continue to use the chain rule operator "prod" from the ["Forward Propagation, Back Propagation, and Computational Graphs"](../chapter_deep-learning-basics/backprop.md) section.

First, calculate the output layer variable gradient of the objective function with respect to each time step $\partial L/\partial \mathbf{o}_t \in \mathbb{R}^q$ using the following formula:

$$\frac{\partial L}{\partial \mathbf{o}_t} =  \frac{\partial \ell (\mathbf{o}_t, y_t)}{T \cdot \partial \mathbf{o}_t}.$$

Now, we can calculate the gradient for the objective function model parameter $\mathbf{W}_{qh}$: $\partial L/\partial \mathbf{W}_{qh} \in \mathbb{R}^{q \times h}$. Based on Figure 6.3, $L$ depends on $\mathbf{W}_{qh}$ through $\mathbf{o}_1, \ldots, \mathbf{o}_T$. Applying the chain rule,

$$
\frac{\partial L}{\partial \mathbf{W}_{qh}}
= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{W}_{qh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{o}_t} \mathbf{h}_t^\top.
$$


Next, we must note that there are also dependencies between hidden states.
In Figure 6.3, $L$ depends on the hidden state $\mathbf{h}_T$ of the final time step $T$ only through $\mathbf{o}_T$. Therefore, we first calculate the objective function gradient with respect to the hidden state of the final time step: $\partial L/\partial \mathbf{h}_T \in \mathbb{R}^h$. According to the chain rule, we get

$$
\frac{\partial L}{\partial \mathbf{h}_T} = \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_T}, \frac{\partial \mathbf{o}_T}{\partial \mathbf{h}_T} \right) = \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_T}.
$$



Then, for the time steps $t < T$,
$L$ depends on $\mathbf{h}_t$ through $\mathbf{h}_{t+1}$ and $\mathbf{o}_t$. Applying the chain rule,
the objective function gradient with respect to the hidden states of the time steps $t < T$ ($\partial L/\partial \mathbf{h}_t \in \mathbb{R}^h$) must be calculated for each time step in turn from large to small:


$$
\frac{\partial L}{\partial \mathbf{h}_t}
= \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_{t+1}}, \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)
+ \text{prod}\left(\frac{\partial L}{\partial \mathbf{o}_t}, \frac{\partial \mathbf{o}_t}{\partial \mathbf{h}_t} \right)
= \mathbf{W}_{hh}^\top \frac{\partial L}{\partial \mathbf{h}_{t+1}} + \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_t}.
$$

Expanding the recursive formula above, we can find a general formula for the objective function hidden state gradient for any time step $1 \leq t \leq T$.

$$
\frac{\partial L}{\partial \mathbf{h}_t}
= \sum_{i=t}^T {\left(\mathbf{W}_{hh}^\top\right)}^{T-i} \mathbf{W}_{qh}^\top \frac{\partial L}{\partial \mathbf{o}_{T+t-i}}.
$$

From the exponential term in the above formula, we can see that, when the number of time steps $T$ is large or the time step $t$ is small, the hidden state gradient of the objective function is prone to vanishing and explosion. This will also affect other gradients that contain the term $\partial L / \partial \mathbf{h}_t$, such as the gradients of model parameters in the hidden layer $\partial L / \partial \mathbf{W}_{hx} \in \mathbb{R}^{h \times d}$ and $\partial L / \partial \mathbf{W}_{hh} \in \mathbb{R}^{h \times h}$.
In Figure 6.3, $L$ depends on these model parameters through $\mathbf{h}_1, \ldots, \mathbf{h}_T$.
According to the chain rule, we get

$$
\begin{aligned}
\frac{\partial L}{\partial \mathbf{W}_{hx}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hx}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{x}_t^\top,\\
\frac{\partial L}{\partial \mathbf{W}_{hh}}
&= \sum_{t=1}^T \text{prod}\left(\frac{\partial L}{\partial \mathbf{h}_t}, \frac{\partial \mathbf{h}_t}{\partial \mathbf{W}_{hh}}\right)
= \sum_{t=1}^T \frac{\partial L}{\partial \mathbf{h}_t} \mathbf{h}_{t-1}^\top.
\end{aligned}
$$


As we already explained in the ["Forward Propagation, Back Propagation, and Computational Graphs"](../chapter_deep-learning-basics/backprop.md) section, after we calculate the above gradients in turn for each iteration, we save them to avoid the need for repeat calculation. For example, after calculating and storing the hidden state gradient $\partial L/\partial \mathbf{h}_t$, subsequent calculations of the model parameter gradients $\partial L/\partial  \mathbf{W}_{hx}$ and $\partial L/\partial \mathbf{W}_{hh}$ can directly read the value of $\partial L/\partial \mathbf{h}_t$, so they do not need to be re-calculated.
In addition, gradient calculation in back propagation may depend on the current values of variables. These are calculated using forward propagation.
To give an example, the calculation of the parameter gradient $\partial L/\partial \mathbf{W}_{hh}$ must depend on the current hidden state value at the time step $t = 0, \ldots, T-1$: $\mathbf{h}_t$ ($\mathbf{h}_0$ is obtained during initialization). These values are obtained by calculation and storage via forward propagation from the input layer to the output layer.


## Summary

* Back-propagation through time is a specific application of back propagation in recurrent neural networks.
* When the number of time steps is large or the time step is small, the gradients in recurrent neural networks are prone to vanishing or explosion.


## Exercises

* Besides gradient clipping, can you think of any other methods to cope with gradient explosion in recurrent neural networks?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2366)

![](../img/qr_bptt.svg)
