
# Imitation Learning
:label:`sec_imit`

So far, we have discussed how to obtain the value function, or the action-value function of an MDP (Markov decision process) using full knowledge of the MDP. There are many situations in real problems where we do not know the MDP completely. This often happens when we do not know the transition function $P(s' \mid s, a)$ at all states and actions, e.g., when driving on an icy asphalt, the next state of the car is slightly different than we expect. In many cases, we may not be able design a reward $r(s, a)$ that can satisfactorily achieve the goal, e.g., if we are building a robot for a complex task such as opening a fridge and fetching a bottle from inside it, then it is not easy to say how an action at some state (say when the door of the fridge is closed) relates to the eventual success of this task.

## Key Idea

Imitation Learning is a technique that allows us to solve complex problems without complete knowledge of the MDP (Markov decision process). The key idea is to learn from demonstrations, e.g., a human driving the car on icy asphalt, or a human opening the fridge and fetching the bottle, and simply "imitate" these actions. If the human can successfully execute these tasks, then they are deemed "the expert" and the robot can simply mimic the action taken by the expert without having to solve the MDP. The expert need not always be a human, it could also be a more sophisticated robot.

Imitation learning was one of the earliest success stories of both reinforcement learning and deep learning. The [ALVINN project at Carnegie Mellon University by Dean Pomerleau in 1988](@https://www.youtube.com/watch?v=2KMAAmkz9go}) used a two-layer neural network with 5 hidden neurons, about 1000 inputs from the pixels of a camera and 30 outputs. It successfully drove in different parts of the United States and Germany. Imitation learning has been used for impressive [acrobatic maneuvers on a remote controlled helicopter](@https://www.youtube.com/watch?v=qKZhtYviqSU).

## Behavior Cloning

We begin with a simple form of imitation learning. Let us imagine that we are given $n$ trajectories of the expert for our chosen task. This is our training dataset
$$D = \{ (s_t^i, a_t^i)_{t=0,1,\ldots,T-1} \}_{i=1,\ldots,n}$$

where for each trajectory $i$ of length $T$, for time instant $t$, we record the state $s_t^i$ of the expert and the action $a_t^i$ that they took. Behavior cloning learns a deterministic policy $\pi(s)$ for the robot such that $\pi(s_t^i) \approx a_t^i$ for all $i$ and $t$. We will use a deep neural network with weights $\varphi$ to learn this policy
$$\pi_\varphi: \mathcal{S} \mapsto \mathcal{A};$$

this network will take a state $s \in \mathcal{S}$ as the input and output an action $a \in \mathcal{A}$. To make things concrete, think of the gridworld example in :ref:`sec_mdp` where the the robot can take 4 actions $\mathcal{A} = \{ \text{go forward}, \text{turn right}, \text{turn left}, \text{stay at the same location} \}$. In this case, we can think of the policy $\pi_\varphi$ as a classifier with these 4 outputs and fit the network on our dataset to minimize the prediction error. Mathematically, this amounts to solving for
$$\hat{\varphi} = \mathrm{argmin}_\varphi \frac{1}{n T} \sum_{i=1}^n \sum_{t=0}^{T-1} l(a_t^i, \pi_\varphi(s_t^i))$$

where $l$ is the cross-entropy loss :numref:`subsec_softmax_and_derivatives`. If actions are continuous-valued $\mathcal{A} \subseteq \mathbb{R}^m$, then we obtain the policy using regression as
$$\hat{\varphi} = \mathrm{argmin}_\varphi \frac{1}{2n T} \sum_{i=1}^n \sum_{t=0}^{T-1} ||a_t^i - \pi_\varphi(s_t^i)||^2.$$

This optimization problem can be solved using any of the optimization algorithms for training a deep network, e.g., stochastic gradient descent :numref:`chap_optimization`. Given this policy, we can take the action $\pi_{\hat{\varphi}}(s)$ at any state $s \in \mathcal{S}$.

It is useful to note a few things. The policy obtained from behavior cloning can be used in the states for which we did not have any expert actions. We did not need to know anything about the MDP (except what the state space $\mathcal{S}$ and the action space $\mathcal{A}$ are) to use behavior cloning.

### When does behavior cloning work well?

Behavior cloning is no different from supervised learning where the ground-truth labels/targets are provided by the expert. Just like supervised learning can make mistakes on the test data even if it fits the training data well, for states that are not a part of the expert-provided training data, behavior cloning can take actions that may not be similar to the expert's actions. So behavior cloning works well when the expert-provided dataset covers a large part of the state space, i.e., when we know the expert's action for most states in $\mathcal{S}$. Similar considerations as standard supervised learning apply for behavior cloning, e.g., using cross-validation to ensure that we do not overfit to the training data.

## DAgger: Dataset Aggregation

There is a big issue with behavior cloning. Notice that if the action taking by our learned policy $\pi_{\hat{\varphi}}$ differs from that of the expert a state $s$, then the future state of the system obtained from $P(s' \mid s, \pi_{\hat{\varphi}}(s))$ can be quite different from the states that the expert visited in their dataset. In other words, the gap between the expert and the learned policy can diverge along the trajectory, and the robot can easily find itself in states that are very different from those of the expert.

DAgger, which stands for Dataset Aggregation, is an algorithm that aims to fix this issue. It works as follows.

Let $D^{(0)}$ be a dataset obtained from the expert. We learn a policy $\pi_{\varphi^{(0)}}$ with weights $\varphi^{(0)}$ using behavior cloning on this dataset.

1. At each iteration $k$ of DAgger, the robot collects data from $n$ trajectories, starting from say some initial state $s_0$. For each trajectory $i \in \{1,\ldots,n\}$, at each time instant $t$, the robot queries the expert with probability $p$ for an action and takes the action from its current policy $\pi_{\varphi^{(k-1)}}$ with probability $1-p$. In other words, at state $s_t$, the robot takes the action
$$a_t^i = \begin{cases}
    \pi_{\text{expert}}(s_t) & \text{with prob. } p\\
    \pi_{\varphi^{(k-1)}}(s_t)  & \text{with prob. } 1-p.
    \end{cases}$$

where we denote the expert as the policy $\pi_{\text{expert}}$.

2. The $k^{\text{th}}$ iteration therefore leads to the dataset
$$D_k = \{ (s_t^i, a_t^i)_{t=0,\ldots,T-1}\}_{i=1,\ldots,n}$$

which is appended to the original dataset to get
$$D^{(k)} = D^{(k-1)} \cup D_k.$$

3. Learn a new policy $\pi_{\varphi^{(k)}}$ using the dataset $D^{(k)}$ and proceed to the next iteration.

DAgger iteratively updates the behavior cloning policy by obtaining new data from the expert. As we discussed, the learned policy may veer off the expert's trajectory if the robot starts at states that are different from those of the expert, or if it takes a slightly different action than that of the expert. DAgger queries the expert with a probability $p$ to incrementally collect data for such states. This iteratively expands the dataset $D^{(k)}$ that is used to learn the policy.

![A schematic of the situation when the learned policy veers off the expert's trajectory. [We have to redraw this figure]](../img/dagger1.png)
:width:`250px`
:label:`fig_dagger1`

In the first few iterations, we may wish to be close to the expert's data and use a large value of $p$, as the fitted policy $\pi_{\varphi^{(k)}}$ becomes good, we can reduce the value of $p$ and rely less on the expert. We can stop DAgger with the new dataset $D^{(k)}$ is similar to the dataset of the previous iteration $D^{(k-1)}$.

DAgger is a conceptual framework. When we implement it on a real robot, or a real system, it is not practical to query the expert every few timesteps (e.g., the robot will have to halt while this query is made). It is much more reasonable to use the learned policy for the entire trajectory (i.e., set $p=0$) and have the expert label the actions at the end of the iteration. This is a more natural way of implementing DAgger.

## Summary
* Behavior cloning is an algorithm that uses data from an expert to learn a policy using standard supervised learning without the need to have complete knowledge of the MDP.
* DAgger is an algorithm for imitation learning that performs behavior cloning at each iteration while incrementally expanding the data collected from the expert.

## Exercises

1. 
