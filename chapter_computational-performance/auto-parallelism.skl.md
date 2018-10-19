# %%%1%%%

%%%2%%%

%%%3%%%

%%%4%%%

```{.python .input}
import gluonbook as gb
import mxnet as mx
from mxnet import nd
```

## %%%5%%%

%%%6%%%

```{.python .input}
def run(x):
    return [nd.dot(x, x) for _ in range(10)]
```

%%%7%%%

```{.python .input}
x_cpu = nd.random.uniform(shape=(2000, 2000))
x_gpu = nd.random.uniform(shape=(6000, 6000), ctx=mx.gpu(0))
```

%%%8%%%

```{.python .input}
run(x_cpu)  # %%%9%%%
run(x_gpu)
nd.waitall()  # %%%10%%%

with gb.Benchmark('Run on CPU.'):
    run(x_cpu)
    nd.waitall()

with gb.Benchmark('Then run on GPU.'):
    run(x_gpu)
    nd.waitall()
```

%%%11%%%

```{.python .input}
with gb.Benchmark('Run on both CPU and GPU in parallel.'):
    run(x_cpu)
    run(x_gpu)
    nd.waitall()
```

%%%12%%%


## %%%13%%%

%%%14%%%

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(mx.cpu()) for y in x]

with gb.Benchmark('Run on GPU.'):
    y = run(x_gpu)
    nd.waitall()

with gb.Benchmark('Then copy to CPU.'):
    copy_to_cpu(y)
    nd.waitall()
```

%%%15%%%

```{.python .input}
with gb.Benchmark('Run and copy in parallel.'):
    y = run(x_gpu)
    copy_to_cpu(y)
    nd.waitall()
```

%%%16%%%

## %%%17%%%

* %%%18%%%


## %%%19%%%

* %%%20%%%
* %%%21%%%
* %%%22%%%


## %%%23%%%

![](../img/qr_auto-parallelism.svg)
