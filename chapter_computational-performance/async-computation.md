# Asynchronous Computing
:label:`chapter_async`

MXNet utilizes asynchronous programming to improve computing performance. Understanding how asynchronous programming works helps us to develop more efficient programs, by proactively reducing computational requirements and thereby minimizing the memory overhead required in the case of limited memory resources. First, we will import the package or module needed for this section’s experiment.

```{.python .input  n=1}
import d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import os
import subprocess
npx.set_np()
```

## Asynchronous Programming in MXNet

Broadly speaking, MXNet includes the front-end directly used by users for interaction, as well as the back-end used by the system to perform the computation. For example, users can write MXNet programs in various front-end languages, such as Python, R, Scala and C++. Regardless of the front-end programming language used, the execution of MXNet programs occurs primarily in the back-end of C++ implementations. In other words, front-end MXNet programs written by users are passed on to the back-end to be computed. The back-end possesses its own threads that continuously collect and execute queued tasks.

Through the interaction between front-end and back-end threads, MXNet is able to implement asynchronous programming. Asynchronous programming means that the front-end threads continue to execute subsequent instructions without having to wait for the back-end threads to return the results from the current instruction. For simplicity’s sake, assume that the Python front-end thread calls the following four instructions.

```{.python .input  n=2}
a = np.ones((1, 2))
b = np.ones((1, 2))
c = a * b + 2
c
```

In Asynchronous Computing, whenever the Python front-end thread executes one of the first three statements, it simply returns the task to the back-end queue. When the last statement’s results need to be printed, the Python front-end thread will wait for the C++ back-end thread to finish computing result of the variable `c`. One benefit of such as design is that the Python front-end thread in this example does not need to perform actual computations. Thus, there is little impact on the program’s overall performance, regardless of Python’s performance. MXNet will deliver consistently high performance, regardless of the front-end language’s performance, provided the C++ back-end can meet the efficiency requirements.

The following example uses timing to demonstrate the effect of asynchronous programming. As we can see, when `y = x.dot(x).sum()` is returned, it does not actually wait for the variable `y` to be calculated. Only when the `print` function needs to print the variable `y` must the function wait for it to be calculated.

```{.python .input  n=3}
timer = d2l.Timer()
x = np.random.uniform(size=(2000, 2000))
y = x.dot(x).sum()
print('Workloads are queued. Time %.4f sec' % timer.stop())

print('sum =', y)
print('Workloads are finished. Time %.4f sec' % timer.stop())
```

In truth, whether or not the current result is already calculated in-memory is irrelevant, unless we need to print or save the computation results. So long as the data are stored in `ndarray`s and the operators provided by MXNet are used, MXNet will utilize asynchronous programming by default to attain superior computing performance.


## Use of the Synchronization Function to Allow the Front-End to Wait for the Computation Results

In addition to the `print` function we just introduced, there are other ways to make the front-end thread wait for the completion of the back-end computations. The `wait_to_read` function can be used to make the front-end wait for the complete results of `ndarray` computation, and then execute following statement. Alternatively, we can use the `waitall` function to make the front-end wait for the completion of all previous computations. The latter is a common method used in performance testing.

Below, we use the `wait_to_read` function as an example. The time output includes the calculation time of `y`.

```{.python .input  n=4}
timer.start()
y = x.dot(x)
y.wait_to_read()
print('Done in %.4f sec' % timer.stop())
```

Below, we use `waitall` as an example. The time output includes the calculation time of `y` and `z` respectively.

```{.python .input  n=5}
timer.start()
y = x.dot(x)
z = x.dot(x)
npx.waitall()
print('Done in %.4f sec' % timer.stop())
```

Additionally, any operation that does not support asynchronous programming but converts the `ndarray` object from MXNet to an object in NumPy(a scientific computing package of Python) will cause the front-end to have to wait for computation results. For example, calling the `asnumpy` and `item` functions:

```{.python .input  n=6}
timer.start()
y = x.dot(x)
y.asnumpy()
print('Done in %.4f sec' % timer.stop())
```

```{.python .input  n=7}
timer.start()
y = x.dot(x)
np.abs(y).sum().item()
print('Done in %.4f sec' % timer.stop())
```

The `wait_to_read`, `waitall`, `asnumpy`, `item` and the`print` functions described above will cause the front-end to wait for the back-end computation results. Such functions are often referred to as synchronization functions.


## Using Asynchronous Programming to Improve Computing Performance

In the following example, we will use the “for” loop to continuously assign values to the variable `y`. Asynchronous programming is not used in tasks when the synchronization function `wait_to_read` is used in the “for” loop. However, when the synchronization function `waitall` is used outside of the “for” loop, asynchronous programming is used.

```{.python .input  n=8}
timer.start()
for _ in range(1000):
    y = x + 1
    y.wait_to_read()
print('Synchronous. Done in %.4f sec' % timer.stop())

timer.start()
for _ in range(1000):
    y = x + 1
npx.waitall()
print('Asynchronous. Done in %.4f sec' % timer.stop())
```

We have observed that certain aspects of computing performance can be improved by making use of asynchronous programming. To explain this, we will slightly simplify the interaction between the Python front-end thread and the C++ back-end thread. In each loop, the interaction between front and back-ends can be largely divided into three stages:

1. The front-end orders the back-end to insert the calculation task `y = x + 1` into the queue.
1. The back-end then receives the computation tasks from the queue and performs the actual computations.
1. The back-end then returns the computation results to the front-end.

Assume that the durations of these three stages are $t_1, t_2, t_3$, respectively. If we do not use asynchronous programming, the total time taken to perform 1000 computations is approximately $1000 (t_1+ t_2 + t_3)$. If asynchronous programming is used, the total time taken to perform 1000 computations can be reduced to $t_1 + 1000 t_2 + t_3$ (assuming $1000t_2 > 999t_1$), since the front-end does not have to wait for the back-end to return computation results for each loop.

## The Impact of Asynchronous Programming on Memory

In order to explain the impact of asynchronous programming on memory usage, recall what we learned in the previous chapters. Throughout the model training process implemented in the previous chapters, we usually evaluated things like the loss or accuracy of the model in each mini-batch. Detail-oriented readers may have discovered that such evaluations often make use of synchronization functions, such as `item` or `asnumpy`. If these synchronization functions are removed, the front-end will pass a large number of mini-batch computing tasks to the back-end in a very short time, which might cause a spike in memory usage. When the mini-batches makes use of synchronization functions, on each iteration, the front-end will only pass one mini-batch task to the back-end to be computed, which will typically reduce memory use.

Because the deep learning model is usually large and memory resources are usually limited, we recommend the use of synchronization functions for each mini-batch throughout model training, for example by using the `item` or `asnumpy` functions to evaluate model performance. Similarly, we also recommend utilizing synchronization functions for each mini-batch prediction (such as directly printing out the current batch’s prediction results), in order to reduce memory usage during model prediction.

Next, we will demonstrate asynchronous programming’s impact on memory. We will first define a data retrieval function `data_iter`, which upon being called, will start timing and regularly print out the time taken to retrieve data batches.

```{.python .input  n=9}
def data_iter():
    timer.start()
    num_batches, batch_size = 100, 1024
    for i in range(num_batches):
        X = np.random.normal(size=(batch_size, 512))
        y = np.ones((batch_size,))
        yield X, y
        if (i + 1) % 50 == 0:
            print('batch %d, time %.4f sec' % (i + 1, timer.stop()))
```

The multilayer perceptron, optimization algorithm, and loss function are defined below.

```{.python .input  n=10}
net = nn.Sequential()
net.add(nn.Dense(2048, activation='relu'),
        nn.Dense(512, activation='relu'),
        nn.Dense(1))
net.initialize()
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.005})
loss = gluon.loss.L2Loss()
```

A helper function to monitor memory use is defined here. It should be noted that this function can only be run on Linux or MacOS operating systems.

```{.python .input  n=11}
def get_mem():
    res = subprocess.check_output(['ps', 'u', '-p', str(os.getpid())])
    return int(str(res).split()[15]) / 1e3
```

Now we can begin testing. To initialize the `net` parameters we will try running the system once. See :numref:`chapter_deferred_init` for further discussions related to initialization.

```{.python .input  n=12}
for X, y in data_iter():
    break
loss(y, net(X)).wait_to_read()
```

For the `net` training model, the synchronization function `item` can naturally be used to record the loss of each mini-batch output by the `ndarray` object and to print out the model loss after each iteration. At this point, the generation interval of each mini-batch increases, but with a small memory overhead.

```{.python .input  n=13}
l_sum, mem = 0, get_mem()
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    # Use of the item synchronization function
    l_sum += l.sum().item()
    l.backward()
    trainer.step(X.shape[0])
npx.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

Even though each mini-batch’s generation interval is shorter, the memory usage may still be high during training if the synchronization function is removed. This is because, in default asynchronous programming, the front-end will pass on all mini-batch computations to the back-end in a short amount of time. As a result of this, a large amount of intermediate results cannot be released and may end up piled up in memory. In this experiment, we can see that all data (`X` and `y`) is generated in under a second. However, because of an insufficient training speed, this data can only be stored in the memory and cannot be cleared in time, resulting in extra memory usage.

```{.python .input  n=14}
mem = get_mem()
for X, y in data_iter():
    with autograd.record():
        l = loss(y, net(X))
    l.backward()
    trainer.step(X.shape[0])
npx.waitall()
print('increased memory: %f MB' % (get_mem() - mem))
```

## Summary

* MXNet includes the front-end used directly by users for interaction and the back-end used by the system to perform the computation.

* MXNet can improve computing performance through the use of asynchronous programming.

* We recommend using at least one synchronization function for each mini-batch training or prediction to avoid passing on too many computation tasks to the back-end in a short period of time.


## Exercises

* In the section "Use of Asynchronous Programming to Improve Computing Performance", we mentioned that using asynchronous computation can reduce the total amount of time needed to perform 1000 computations to $t_1 + 1000 t_2 + t_3$. Why do we have to assume $1000t_2 > 999t_1$ here?

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2381)

![](../img/qr_async-computation.svg)
