```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['pytorch', 'mxnet','tensorflow'])
```

# Code
:label:`sec_code_rl`


Add required RL libraries
```{.python .input}
#@tab pytorch

import gym #@save

```
Add FrozenLake enviroment
```{.python .input}
#@tab pytorch

def FrozenLake(): #@save
    '''
            Winter is here. You and your friends were tossing around a frisbee at the
            park when you made a wild throw that left the frisbee out in the middle of
            the lake. The water is mostly frozen, but there are a few holes where the
            ice has melted. If you step into one of those holes, you'll fall into the
            freezing water. At this time, there's an international frisbee shortage, so
            it's absolutely imperative that you navigate across the lake and retrieve
            the disc. However, the ice is slippery, so you won't always move in the
            direction you intend.
            The surface is described using a grid like the following
                SFFF
                FHFH
                FFFH
                HFFG
            S : starting point, safe
            F : frozen surface, safe
            H : hole, fall to your doom
            G : goal, where the frisbee is located
            The episode ends when you reach the goal or fall in a hole.
            You receive a reward of 1 if you reach the goal, and zero otherwise.
            You can only move in the following directions:
                LEFT which corresponds to action index 0.
                DOWN which corresponds to action index 1.
                RIGHT which corresponds to action index 2.
                UP which corresponds to action index 3.

    '''
    env = gym.make('FrozenLake-v1', is_slippery=False)
    einfo = {}
    einfo['desc'] = env.desc   # 2D array specifying what each grid item means
    einfo['num_states']     = env.nS # number of observations/states or obs/state dim
    einfo['num_actions']    = env.nA # number of actions or action dim
    # define indices for (transition probability, nextstate, reward, done) tuple
    einfo['trans_prob_idx'] = 0 #index in transition probability
    einfo['nextstate_idx']  = 1
    einfo['reward_idx']     = 2
    einfo['done_idx']       = 3

    # einfo['T'] is dict that contains (transition probability (pr), nextstate, reward, done) index by [s][a]
    # where s specifies state index and a specifies an action index.
    einfo['T']   = {
                     s : {a :
                            [pnrd for pnrd in pnrds] for (a, pnrds) in others.items() # pnrds: p, nextstate, reward, done
                         }
                        for (s, others) in env.P.items()
                    }
    return env, einfo

```

```{.python .input}
#@tab pytorch

def make_env(name =''): #@save
    '''
         Input parameters:
         name: specifies a gym environment.
         For Value iteration, only FrozenLake-v1 is supported.
    '''
    if name == 'FrozenLake-v1':
        return FrozenLake()

    else:
        raise ValueError("%s env is not supported in this Notebook")


```


```{.python .input}
#@tab pytorch

def show_value_function_progress(env, V_K, pi_K): #@save
    '''
     This function shows how value functions and policy changes through times.
     V_k:  [V(0), ..., V(K-1)]
     pi_K: [pi(0), ..., pi_K(K-1)]
     In both K V_k and pi_K, K idicates number of iterations.
    '''
    num_subplt_rows = len(V_K)//3 + 1
    idx = 1
    fig = plt.figure(figsize=(15,15))
    for (V, pi) in zip(V_K, pi_K):

        #plt.figure(figsize=(4,4))
        plt.subplot(num_subplt_rows, 4, idx)
        plt.imshow(V.reshape(4,4), cmap='gray', interpolation='nearest', clim=(0,1))
        ax = plt.gca()
        ax.set_facecolor('xkcd:salmon')
        ax.set_facecolor((1.0, 0.47, 0.02))

        ax.set_xticks(np.arange(5)-.5)
        ax.set_yticks(np.arange(5)-.5)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # we are showing 4x4 grid for now
        Y, X = np.mgrid[0:4, 0:4]
        a2uv = {0: (-1, 0), 1:(0, -1), 2:(1,0), 3:(-1, 0)}
        Pi = pi.reshape(4,4)

        for y in range(4):
            for x in range(4):
                a = Pi[y, x]
                u, v = a2uv[a]

                plt.text(x, y, str(env.desc[y,x].item().decode()),
                             color='g', size=15,  verticalalignment='center',
                             horizontalalignment='center', fontweight='bold')
                if env.desc[y,x].item().decode() != 'G' and env.desc[y,x].item().decode() != 'H':
                    plt.arrow(x, y,u*.3, -v*.3, color='r', head_width=0.2, head_length=0.1)

        plt.grid(color='w', lw=2, ls='-')
        plt.title("K = %s " % (idx));
        idx += 1

    plt.show()


```
