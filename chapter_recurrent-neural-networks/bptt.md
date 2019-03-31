# Backpropagation Through Time

So far we repeatedly alluded to things like *exploding gradients*,
*vanishing gradients*, *truncating backprop*, and the need to
*detach the computational graph*. For instance, in the previous 
section we invoked `s.detach()` on the sequence. None of this was really fully
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
that there is nothing conceptually new in how it works. After all, we are still merely applying the chain rule to compute gradients. Nonetheless it is 
worth while reviewing [backpropagation](../chapter_deep-learning-basics/backprop.md) for
another time.

Forward propagation in a recurrent neural network is relatively
straightforward. Back-propagation through time is actually a specific
application of back propagation in recurrent neural networks. It
requires us to expand the recurrent neural network one time step at a time to
obtain the dependencies between model variables and parameters. Then,
based on the chain rule, we apply back propagation to compute and
store gradients. Since sequences can be rather long this means that the dependency can be rather lengthy. E.g. for a sequence of 1000 characters the first symbol could potentially have significant influence on the symbol at position 1000. This is not really computationally feasible (it takes too long and requires too much memory) and it requires over 1000 matrix-vector products before we would arrive at that very elusive gradient. This is a process fraught with computational and statistical uncertainty. In the following we will address what happens and how to address this in practice.

