# Nadaraya-Watson Kernel Regression
:label:`sec_nadaraya-waston`

```{.python .input  n=2}
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn

npx.set_np()

def f(x):
    return 2 * d2l.sin(x) + x**0.8
```

```{.python .input}
n_train = 50
x_train = np.random.rand(n_train) * 5
y_train = f(x_train) + d2l.normal(0, 0.5, n_train)

x = np.arange(0, 5, 0.05)
y_truth = f(x)

def plot_kernel_reg(y_pred):
    d2l.plot(x, [y_truth, y_pred], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(x_train, y_train, 'o', alpha=0.5);
```

## Average Pooling

```{.python .input}
y_pred = np.repeat(y_train.mean(), len(x))
plot_kernel_reg(y_pred)
```

## Nonparametric Model

The Nadaraya--Watson estimator is:

$$
f(x) = \sum_i \frac{K(x - x_i)}{\sum_j K(x - x_j)} y_i
$$

where Gaussian kernel is:

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2})
$$


Thus,

$$\begin{aligned} f(x) &= \sum_i \alpha(x, x_i) y_i \\&= \sum_i \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_j \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_i \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i \end{aligned} $$

```{.python .input}
X_repeat = d2l.reshape(np.repeat(x, n_train), (-1, n_train))
attention_matrix = npx.softmax(-(X_repeat - x_train)**2 / 2)
y_pred = d2l.matmul(attention_matrix, y_train)
plot_kernel_reg(y_pred)
```

## Parametric Model

$$\begin{aligned}
f(x) &= \sum_i \alpha(x, x_i) y_i \\&= \sum_i \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_j \exp\left(-\frac{1}{2}((x - x_i)w)^2\right)} y_i \\&= \sum_i \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i
\end{aligned}$$

```{.python .input}
class NWKernelRegression(nn.Block):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = self.params.get('w', shape=(1,))
        
    def forward(self, q, K, V):
        Q = d2l.reshape(np.repeat(q, K.shape[1]), (-1, K.shape[1]))
        A = npx.softmax(-((Q - K) * self.w.data())**2 / 2)
        return npx.batch_dot(np.expand_dims(A, 1), V).reshape(-1)
```

```{.python .input}
q = x_train

X_tile = np.tile(x_train, (n_train, 1))
Y_tile = np.tile(y_train, (n_train, 1))

# Shape: ('n_train', num. of key-value pairs)
K = d2l.reshape(X_tile[(1 - np.eye(n_train)).astype('bool')], (n_train, -1))
# Shape: ('n_train', num. of key-value pairs, 1) 
V = d2l.reshape(Y_tile[(1 - np.eye(n_train)).astype('bool')],
                (n_train, -1, 1))
```

```{.python .input}
net = NWKernelRegression()
net.initialize()

loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 0.9})
animator = d2l.Animator(xlabel='epoch', ylabel='loss', xlim=[1, 5])

for epoch in range(5):
    with autograd.record():
        l = loss(net(x_train, K, V), y_train)
    l.backward()
    trainer.step(1)
    print(f'epoch {epoch + 1}, loss {float(l.sum()):.6f}')
    animator.add(epoch + 1, float(l.sum()))
```

```{.python .input}
K = np.tile(x_train, (len(x), 1))
V = np.expand_dims(np.tile(y_train, (len(x), 1)), -1)
y_pred = net(x, K, V)
plot_kernel_reg(y_pred)
```
