
# Value Iteration
:label:`sec_valueiter`

In this section we will discuss how to pick the best action for the robot at each state such that the *return* of the trajectory is maximized. We will describe an algorithm called Value Iteration and implement it for a simulated robot that travels over a frozen lake.

## Stochastic policy

A stochastic policy denoted as $\pi(a \mid s)$ (policy for short) is a conditional distribution over the actions $a \in \mathcal{A}$ given the state $s \in \mathcal{S}$, $\pi(a \mid s) \equiv P(a \mid s)$. As an example, if the robot has four actions $\mathcal{A}=$ {go left, go down, go right, go up}. The policy at a state $s \in \mathcal{S}$ for such a set of actions $\mathcal{A}$ is a categorical distribution where the probabilities of the four actions could be $[0.4, 0.2, 0.1, 0.3]$; at some other state $s' \in \mathcal{S}$ the probabilities $\pi(a \mid s')$ of the same four actions could be $[0.1, 0.1, 0.2, 0.6]$. Note that we should have $\sum_a \pi(a \mid s) = 1$ for any state $s$. A deterministic policy is a special case of a stochastic policy in that the distribution $\pi(a \mid s)$ only gives non-zero probability to one particular action, e.g., $[1, 0, 0, 0]$ for our example with four actions.

To keep the notation clear, we will often write $\pi(s)$ as the conditional distribution instead of $\pi(a \mid s)$.

## Value function

Imagine now that the robot starts at a state $s_0$ and at each time instant, it first samples an action from the policy $a_t \sim \pi(s_t)$ and takes this action to result in the next state $s_{t+1}$. The trajectory $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$, can be different depending upon which particular action $a_t$ is sampled at intermediate instants. We define the average *return* $R(\tau) = \sum_{t=0}^\infty \gamma^t r(s_t, a_t)$ of all such trajectories
$$V^\pi(s_0) = E_{a_t \sim \pi(s_t)} \Big[ R(\tau) \Big] = E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=0}^\infty \gamma^t r(s_t, a_t) \Big],$$

where $s_{t+1} \sim P(s_{t+1} \mid s_t, a_t)$ is the next state of the robot and $r(s_t, a_t)$ is the instantaneous reward obtained by taking action $a_t$ in state $s_t$ at time $t$. This is called the "value function" of the state $s_0$ for the policy $\pi$. In simple words, it is the expected $\gamma$-discounted *return* obtained by the robot if it takes actions from the policy $\pi$ at each time instant.

We next break down the trajectory into two stages (i) the first stage which corresponds to $s_0 \to s_1$ upon taking the action $a_0$, and (ii) a second stage which is the trajectory $\tau' = (s_1, a_1, r_1, \ldots)$ thereafter. The key idea behind all algorithms in reinforcement learning is to observe that the value of state $s_0$ can be written as the average reward obtained in the first stage and the value function averaged over all possible next states $s_1$. Mathematically, we write the two stages as

$$V^\pi(s_0) = r(s_0, a_0) + \gamma\ E_{a_0 \sim \pi(s_0)} \Big[ E_{s_1 \sim P(s_1 \mid s_0, a_0)} \Big[ V^\pi(s_1) \Big] \Big].$$:eqlabel:`eq_dynamic_programming`

Such a decomposition is an intuitive but very powerful idea and is known as the principle of dynamic programming. Notice that the second stage gets two expectations, one over the choices of the action $a_0$ taken in the first stage using the stochastic policy and another over the possible states $s_1$ obtained from the chosen action. We can write :eqref:`eq_dynamic_programming` using the transition probabilities in the Markov decision process (MDP) as

$$V^\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^\pi(s') \Big];\ \text{for all } s \in \mathcal{S}.$$:eqlabel:`eq_dynamic_programming_val`

An important thing to notice here is that the above identity holds for all states $s \in \mathcal{S}$ because we can think of a trajectory that begins at that state and break down the trajectory into two stages.

## Action-value function

In implementations, it is often useful to maintain a quantity called the "action value" function which is a closely related quantity to the value function. This is defined to be the average *return* of a trajectory that begins at $s_0$ but when the action of the first stage is fixed to be $a_0$

$$Q^\pi(s_0, a_0) = r(s_0, a_0) + E_{a_t \sim \pi(s_t)} \Big[ \sum_{t=1}^\infty \gamma^t r(s_t, a_t) \Big],$$

note that the summation inside the expectation is from $t=1,\ldots, \infty$ because the reward of the first stage is fixed in this case. We can also write the principle of dynamic programming in :eqref:`eq_dynamic_programming` using the action-value function as

$$Q^\pi(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q^\pi(s', a');\ \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$:eqlabel:`eq_dynamic_programming_q`

This version is the analog of :eqref:`eq_dynamic_programming_val` for the action value function.

## Optimal stochastic policy

Both the value function and the action-value function depend upon the policy that the robot chooses. We will next think of the "optimal policy" that achieves the maximal average *return*
$$\pi^* = \underset{\pi}{\text{argmax}} V^\pi(s_0).$$

Of all possible stochastic policies that the robot could have taken, the optimal policy $\pi^*$  achieves the largest average discounted *return* for trajectories starting from state $s_0$. Let us denote the value function and the action-value function of the optimal policy as $V^* \equiv V^{\pi^*}$ and $Q^* \equiv Q^{\pi^*}$.

For a deterministic policy where there is only one action that is possible at any given state, we can break down the trajectory of the optimal policy in two stages to observe that
$$\pi^*(s) = \underset{a \in \mathcal{A}}{\text{argmax}} \Big[ r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a)\ V^*(s') \Big],$$

afterall, the optimal action at state $s$ is the one that maximizes the sum of reward $r(s, a)$ from the first stage and the average *return* of the trajectories starting from the next sate $s'$, averaged over all possible next states $s'$ from the second stage.

## Value Iteration

The principle of dynamic programming in :eqref:`eq_dynamic_programming` or :eqref:`eq_dynamic_programming_q` can be turned into an algorithm to compute the optimal value function $V^*$ or the action-value function $Q^*$, respectively. Observe that :eqref:`eq_dynamic_programming_val` is also true for the optimal value function:
$$ V^*(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V^*(s') \Big];\ \text{for all } s \in \mathcal{S}.$$

The key idea behind value iteration is to think of this identity as a set of constraints that tie together $V^*(s)$ at different states $s \in \mathcal{S}$ and solve for the optimal value function that satisfies these constraints iteratively. Suppose we initialize the value function to some arbitrary values $V_0(s)$ for all states $s \in \mathcal{S}$. At the $k^{\text{th}}$ iteration of the algorithm, the Value Iteration algorithm updates the value function as
$$V_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \Big[ r(s,  a) + \gamma\  \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \Big];\ \text{for all } s \in \mathcal{S}.$$

It turns our that as $k \to \infty$ the value function estimated by the Value Iteration algorithm converges to the optimal value function irrespective of the initialization $V_0$,
$$V^*(s) = \lim_{k \to \infty} V_k(s);\ \text{for all states } s \in \mathcal{S}.$$

The same Value Iteration algorithm can be equivalently written using the action-value function as
$$Q_{k+1}(s, a) = r(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s')\ Q_k (s', a');\ \text{ for all } s \in \mathcal{S}, a \in \mathcal{A}.$$

In this case we initialize $Q_0(s, a)$ to some arbitrary values for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$. Again we have $Q^*(s, a) = \lim_{k \to \infty} Q_k(s, a)$ for all $s \in \mathcal{S}$ and $a \in \mathcal{A}$.


## Implementation of Value Iteration

We next show how to implement Value Iteration for a navigation problem called FrozenLake from [Open AI Gym](https://gym.openai.com). We first need to setup the enviroment as shown in the following code.
```{.python .input}
#@tab all

%matplotlib inline
%pip install gym==0.21.0
import numpy as np
import random
from d2l import torch as d2l

seed = 0       # random number generator seed
gamma = 0.95   # discount factor
num_iters = 10 # number of iterations
random.seed(seed) # set the random seed to ensure results can be reproduced
np.random.seed(seed)

# now set up the environment
env_info = d2l.make_env('FrozenLake-v1')
```

In the FrozenLake environment, a robot/agent moves on a $4 \times 4$ grid (these are the states) with actions that are "up" ($\uparrow$), "down" ($\rightarrow$), "left" ($\leftarrow$), and "right" ($\rightarrow$). The environment contains a number of holes (H) cells and forzen (F) cells as well as a goal (G), all of which are unknown to the robot. To keep the problem simple,
we assume the robot has reliable actions, i.e. $P(s' \mid s, a) = 1$ for all $s \in \mathcal{S}, a \in \mathcal{A}$. If the robot reaches the goal, the trial ends and the robot receives a reward of $1$ irrespective of the action. The reward at any other state is $0$ for all actions. The objective of the robot is to learn a policy that reaches the goal location (G) from a given start location (S) (this is $s_0$) to maximize the *return*.

The following function implements Value Iteration:

```{.python .input}
#@tab all

def value_iteration(einfo, gamma, num_iters):
    '''
        gamma: discount factor
        num_iters: number of iteration
        env_info: contains MDP and environment related information
    '''
    env_desc      = einfo['desc'] #2D array shows what each item means
    prob_idx      = einfo['trans_prob_idx']
    nextstate_idx = einfo['nextstate_idx']
    reward_idx    = einfo['reward_idx']
    num_states    = einfo['num_states']
    num_actions   = einfo['num_actions']
    mdp           = einfo['mdp']

    V  = np.zeros((num_iters + 1, num_states))
    Q  = np.zeros((num_iters + 1, num_states, num_actions))
    pi = np.zeros((num_iters + 1, num_states))

    for k in range(1, num_iters + 1):
        for s in range(num_states):
            for a in range(num_actions):
                # calculate \sum_{s'} p(s'\mid s,a) [r + \gamma v_k(s')]
                for pxrds in mdp[(s,a)]:
                    # mdp(s,a): [(p1,next1,r1,d1),(p2,next2,r2,d2),..]
                    pr = pxrds[prob_idx] # p(s'\mid s,a)
                    nextstate  = pxrds[nextstate_idx] # next state
                    reward     = pxrds[reward_idx] # reward
                    Q[k,s,a]  += pr * (reward + gamma * V[k - 1, nextstate])

            # record max value and max action
            V[k,s]  = np.max(Q[k,s,:])
            pi[k,s] = np.argmax(Q[k,s,:])

    d2l.show_value_function_progress(env_desc, V[:-1], pi[:-1])

value_iteration(env_info, gamma, num_iters)
```
The above pictures show the policy (the arrow indicates the action) and value function (the change in color shows how the value function changes over time from the initial value shown by dark color to the optimal value shown by light colors.). As we see, Value Iteration finds the optimal value function after 10 iterations and the goal state (G) can be reached starting from any state as long as it is not an H cell. Another interesting aspect of the implementation is that in addition to finding the optimal value function, we also automatically found the optimal policy $\pi^*$ corresponding to this value function.


## Summary
* The main idea behind the Value Iteration algorithm is to leverage dynamic programming to solve the larger, original problem by solving smaller sub-problems.
* The Value Iteration algorithm requires the Markov decision process (MDP) to be fully known.


## Exercises

1. Try increasing the grid size to $8 \times 8$. Compared with $4 \times 4$ grid, how many iterations does it take to find the optimal value function?
1. What is the computational complexity of the Value Iteration algorithm?
1. Run the Value Iteration algorithm again with $\gamma$ (i.e. "gamma" in the above code) when it equals to $0$, $0.5$, and $1$ and analyze its results. 
1. How does the value of $\gamma$ affect the number of iterations taken by Value Iteration to converge? What happens when $\gamma=1$?
