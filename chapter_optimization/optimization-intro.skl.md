# %%%1%%%

%%%2%%%

## %%%3%%%

%%%4%%%

%%%5%%%


## %%%6%%%

%%%7%%%

%%%8%%%

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mpl_toolkits import mplot3d
import numpy as np
```

### %%%9%%%

%%%10%%%

%%%11%%%

%%%12%%%

%%%13%%%

```{.python .input  n=2}
def f(x):
    return x * np.cos(np.pi * x)

gb.set_figsize((4.5, 2.5))
x = np.arange(-1.0, 2.0, 0.1)
fig,  = gb.plt.plot(x, f(x))
fig.axes.annotate('local minimum', xy=(-0.3, -0.25), xytext=(-0.77, -1.0),
                  arrowprops=dict(arrowstyle='->'))
fig.axes.annotate('global minimum', xy=(1.1, -0.95), xytext=(0.6, 0.8),
                  arrowprops=dict(arrowstyle='->'))
gb.plt.xlabel('x')
gb.plt.ylabel('f(x)');
```

%%%14%%%

### %%%15%%%

%%%16%%%

%%%17%%%

%%%18%%%

```{.python .input  n=3}
x = np.arange(-2.0, 2.0, 0.1)
fig, = gb.plt.plot(x, x**3)
fig.axes.annotate('saddle point', xy=(0, -0.2), xytext=(-0.52, -5.0),
                  arrowprops=dict(arrowstyle='->'))
gb.plt.xlabel('x')
gb.plt.ylabel('f(x)');
```

%%%19%%%

%%%20%%%

%%%21%%%

```{.python .input  n=4}
x, y = np.mgrid[-1: 1: 31j, -1: 1: 31j]
z = x**2 - y**2

ax = gb.plt.figure().add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z, **{'rstride': 2, 'cstride': 2})
ax.plot([0], [0], [0], 'rx')
ticks = [-1,  0, 1]
gb.plt.xticks(ticks)
gb.plt.yticks(ticks)
ax.set_zticks(ticks)
gb.plt.xlabel('x')
gb.plt.ylabel('y');
```

%%%22%%%

%%%23%%%

* %%%24%%%
* %%%25%%%
* %%%26%%%

%%%27%%%

%%%28%%%


## %%%29%%%

* %%%30%%%
* %%%31%%%


## %%%32%%%

* %%%33%%%


## %%%34%%%

![](../img/qr_optimization-intro.svg)


## %%%36%%%

%%%37%%% %%%38%%% %%%39%%% %%%40%%% %%%41%%%
