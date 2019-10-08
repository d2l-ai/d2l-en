# Automatic Parallelism
:label:`sec_auto_para`

MXNet automatically constructs computational graphs at the back end. Using a
computational graph, the system is aware of all the computational dependencies,
and can selectively execute multiple non-interdependent tasks in parallel to
improve computing performance. For instance, the first example in
:numref:`sec_async` executes `a = np.ones((1, 2))` and `b = np.ones((1, 2))`
in turn. There is no dependency between these two steps, so the system can
choose to execute them in parallel.

Typically, a single operator will use all the computational resources on all CPUs or a single GPU. For example, the `dot` operator will use all threads on all CPUs (even if there are multiple CPU processors on a single machine) or a single GPU. If computational load of each operator is large enough and multiple operators are run in parallel on only on the CPU or a single GPU, then the operations of each operator can only receive a portion of computational resources of CPU or single GPU. Even if these computations can be parallelized, the ultimate increase in computing performance may not be significant. In this section, our discussion of automatic parallel computation mainly focuses on parallel computation using both CPUs and GPUs, as well as the parallelization of computation and communication.

First, import the required packages or modules for experiment in this section. Note that we need at least one GPU to run the experiment in this section.

```{.python .input}
import d2l
from mxnet import np, npx
npx.set_np()
```

## Parallel Computation using CPUs and GPUs

First, we will discuss parallel computation using CPUs and GPUs, for example, when computation in a program occurs both on the CPU and a GPU. First, define the `run` function so that it performs 10 matrix multiplications.

```{.python .input}
def run(x):
    return [x.dot(x) for _ in range(10)]
```

Next, create an `ndarray` on both the CPU and GPU.

```{.python .input}
x_cpu = np.random.uniform(size=(2000, 2000))
x_gpu = np.random.uniform(size=(6000, 6000), ctx=d2l.try_gpu())
```

Then, apply `run` function to the two `ndarray`s on both the CPU and GPU and print the time required.

```{.python .input}
run(x_cpu)  # Warm-up begins
run(x_gpu)
npx.waitall()  # Warm-up ends

timer = d2l.Timer()
run(x_cpu)
npx.waitall()
print('Run on %s: %.4f sec' % (x_cpu.context, timer.stop()))

timer.start()
run(x_gpu)
npx.waitall()
print('Run on %s: %.4f sec' % (x_gpu.context, timer.stop()))
```

We remove `npx.waitall()` between the two computing tasks `run(x_cpu)` and `run(x_gpu)` and hope the system can automatically parallel these two tasks.

```{.python .input}
timer.start()
run(x_cpu)
run(x_gpu)
npx.waitall()
print('Run on both %s and %s: %.4f sec' % (
    x_cpu.context, x_gpu.context, timer.stop()))
```

As we can see, when two computing tasks are executed together, the total execution time is less than the sum of their separate execution times. This means that MXNet can effectively automate parallel computation on CPUs and GPUs.


## Parallel Computation of Computing and Communication

In computations that use both the CPU and GPU, we often need to copy data between the CPU and GPU, resulting in data communication. In the example below, we compute on the GPU and then copy the results back to the CPU. We print the GPU computation time and the communication time from the GPU to CPU.

```{.python .input}
def copy_to_cpu(x):
    return [y.copyto(npx.cpu()) for y in x]

timer.start()
y = run(x_gpu)
npx.waitall()
print('Run on %s: %.4f sec' % (x_gpu.context, timer.stop()))

timer.start()
y_cpu = copy_to_cpu(y)
npx.waitall()
print('The copy to %s: %.4f sec' % (y_cpu[0].context, timer.stop()))
```

We remove the `waitall` function between computation and communication and print the total time need to complete both tasks.

```{.python .input}
timer.start()
y = run(x_gpu)
y_cpu = copy_to_cpu(y)
npx.waitall()
print('Run and copy in parallel: %.4f sec' % timer.stop())
```

As we can see, the total time required to perform computation and communication is less than the sum of their separate execution times. It should be noted that this computation and communication task is different from the parallel computation task that simultaneously used the CPU and GPU described earlier in this section. Here, there is a dependency between execution and communication: `y[i]` must be computed before it can be copied to the CPU. Fortunately, the system can copy `y[i-1]` when computing `y[i]` to reduce the total running time of computation and communication.

## Summary

* MXNet can improve computing performance through automatic parallel computation, such as parallel computation using the CPU and GPU and the parallelization of computation and communication.


## Exercises

* 10 operations were performed in the `run` function defined in this section. There are no dependencies between them. Design an experiment to see if MXNet will automatically execute them in parallel.
* Designing computation tasks that include more complex data dependencies, and run experiments to see if MXNet can obtain the correct results and improve computing performance.
* When the computational load of an operator is small enough, parallel computation on only the CPU or a single GPU may also improve the computing performance. Design an experiment to verify this.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2382)

![](../img/qr_auto-parallelism.svg)
