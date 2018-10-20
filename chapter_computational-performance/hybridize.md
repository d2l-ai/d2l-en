# A Hybrid of Imperative and Symbolic Programming

So far, this book has focused on imperative programming, which makes use of programming statements to change a program’s state. Consider the following example of simple imperative programming code.

```{.python .input}
def add(a, b):
    return a + b

def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g

fancy_func(1, 2, 3, 4)
```

As expected, Python will perform an addition when running the statement `e = add(a, b)`, and will store the result as the variable `e`, thereby changing the program’s state. The next two statements `f = add(c, d)` and `g = add(e, f)` will similarly perform additions and store the results as variables.

Although imperative programming is convenient, it may be inefficient. On the one hand, even if the `add` function is repeatedly called throughout the `fancy_func` function, Python will execute the three function calling statements individually, one after the other. On the other hand, we need to save the variable values of `e` and `f` until all the statements in `fancy_func` have been executed. This is because we do not know whether the variables `e` and `f` will be used by other parts of the program after the statements `e = add(a, b)` and `f = add(c, d)` have been executed.

Contrary to imperative programming, symbolic programming is usually performed after the computational process has been fully defined. Symbolic programming is used by multiple deep learning frameworks, including Theano and TensorFlow. The process of symbolic programming generally requires the following three steps:

1. Define the computation process.
2. Compile the computation process into an executable program.
3. Provide the required inputs and call on the compiled program for execution.

In the example below, we utilize symbolic programming to re-implement the imperative programming code provided at the beginning of this section.

```{.python .input}
def add_str():
    return '''
def add(a, b):
    return a + b
'''

def fancy_func_str():
    return '''
def fancy_func(a, b, c, d):
    e = add(a, b)
    f = add(c, d)
    g = add(e, f)
    return g
'''

def evoke_str():
    return add_str() + fancy_func_str() + '''
print(fancy_func(1, 2, 3, 4))
'''

prog = evoke_str()
print(prog)
y = compile(prog, '', 'exec')
exec(y)
```

The three functions defined above will only return the results of the computation process as a string. Finally, the complete computation process is compiled and run using the `compile` function. This leaves more room to optimize computation, since the system is able to view the entire program during its compilation. For example, during compilation, the program can be rewritten as `print((1 + 2) + (3 + 4))` or even directly rewritten as `print(10)`. Apart from reducing the amount of function calls, this process also saves memory.

A comparison of these two programming methods shows that

* imperative programming is easier. When imperative programming is used in Python, the majority of the code is straightforward and easy to write. At the same time, it is easier to debug imperative programming code. This is because it is easier to obtain and print all relevant intermediate variable values, or make use of Python’s built-in debugging tools.

* Symbolic programming is more efficient and easier to port. Symbolic programming makes it easier to better optimize the system during compilation, while also having the ability to port the program into a format independent of Python. This allows the program to be run in a non-Python environment, thus avoiding any potential performance issues related to the Python interpreter.


## Hybrid programming provides the best of both worlds.

Most deep learning frameworks choose either imperative or symbolic programming. For example, both Theano and TensorFlow (inspired by the latter) make use of symbolic programming, while Chainer and its predecessor PyTorch utilize imperative programming. When designing Gluon, developers considered whether it was possible to harness the benefits of both imperative and symbolic programming. The developers believed that users should be able to develop and debug using pure imperative programming, while having the ability to convert most programs into symbolic programming to be run when product-level computing performance and deployment are required This was achieved by Gluon through the introduction of hybrid programming.

In hybrid programming, we can build models using either the HybridBlock or the HybridSequential classes. By default, they are executed in the same way Block or Sequential classes are executed in imperative programming. When the `hybridize` function is called, Gluon will convert the program’s execution into the style used in symbolic programming. In fact, most models can make use of hybrid programming’s execution style.

Through the use of experiments, this section will demonstrate the benefits of hybrid programming.

## Constructing Models Using the HybridSequential Class

Previously, we learned how to use the Sequential class to concatenate multiple layers. Next, we will replace the Sequential class with the HybridSequential class in order to make use of hybrid programming.

```{.python .input}
from mxnet import nd, sym
from mxnet.gluon import nn
import time

def get_net():
    net = nn.HybridSequential()  # Here we use the class HybridSequential.
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(128, activation='relu'),
            nn.Dense(2))
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

By calling the `hybridize` function, we are able to compile and optimize the computation of the concatenation layer in the HybridSequential instance. The model’s computation result remains unchanged.

```{.python .input}
net.hybridize()
net(x)
```

It should be noted that only the layers inheriting the HybridBlock class will be optimized during computation. For example, the HybridSequential and `Dense` classes provided by Gluon are all subclasses of HybridBlock class, meaning they will both be optimized during computation. A layer will not be optimized if it inherits from the Block class rather than the HybridBlock class.


### Computing Performance

To demonstrate the performance improvement gained by the use of symbolic programming, we will compare the computation time before and after calling the `hybridize` function. Here we time 1000 `net` model computations. The model computations are based on imperative and symbolic programming, respectively, before and after `net` has called the `hybridize` function.

```{.python .input}
def benchmark(net, x):
    start = time.time()
    for i in range(1000):
        _ = net(x)
    nd.waitall()  # To facilitate timing, we wait for all computations to be completed.
    return time.time() - start

net = get_net()
print('before hybridizing: %.4f sec' % (benchmark(net, x)))
net.hybridize()
print('after hybridizing: %.4f sec' % (benchmark(net, x)))
```

As is observed in the above results, after a HybridSequential instance calls the `hybridize` function, computing performance is improved through the use of symbolic programming.


### Achieving Symbolic Programming

We can save the symbolic program and model parameters to the hard disk through the use of the `export` function after the `net` model has finished computing the output based on the input, such as in the case of `net(x)` in the `benchmark` function.

```{.python .input}
net.export('my_mlp')
```

The .json and .params files generated during this process are a symbolic program and a model parameter, respectively. They can be read by other front-end languages supported by Python or MXNet, such as C++, R, Scala, and Perl. This allows us to deploy trained models to other devices and easily use other front-end programming languages. At the same time, because symbolic programming was used during deployment, the computing performance is often superior to that based on imperative programming.

In MXNet, a symbolic program refers to a program that makes use of the Symbol type. We know that, when the NDArray input `x` is provided to `net`, `net(x)` will directly calculate the model output and return a result based on `x`. For models that have called the `hybridize` function, we can also provide a Symbol-type input variable, and `net(x)` will return Symbol type results.

```{.python .input}
x = sym.var('data')
net(x)
```

## Constructing Models Using the HybridBlock Class

Similar to the correlation between the Sequential Block classes, the HybridSequential class is a HybridBlock subclass. Contrary to the Block instance, which needs to use the `forward` function, for a HybridBlock instance we need to use the `hybrid_forward` function.

Earlier, we demonstrated that, after calling the `hybridize` function, the model is able to achieve superior computing performance and portability. In addition, model flexibility can be affected after calling the `hybridize` function. We will demonstrate this by constructing a model using the HybridBlock class.

```{.python .input}
class HybridNet(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(HybridNet, self).__init__(**kwargs)
        self.hidden = nn.Dense(10)
        self.output = nn.Dense(2)

    def hybrid_forward(self, F, x):
        print('F: ', F)
        print('x: ', x)
        x = F.relu(self.hidden(x))
        print('hidden: ', x)
        return self.output(x)
```

We need to add the additional input `F` to the `hybrid_forward` function when inheriting the HybridBlock class. We already know that MXNet uses both an NDArray class and a Symbol class, which are based on imperative programming and symbolic programming, respectively. Since these two classes perform very similar functions, MXNet will determine whether `F` will call NDArray or Symbol based on the input provided.

The following creates a HybridBlock instance. As we can see, by default, `F` uses NDArray. We also printed out the `x` input as well as the hidden layer’s output using the ReLU activation function.

```{.python .input}
net = HybridNet()
net.initialize()
x = nd.random.normal(shape=(1, 4))
net(x)
```

Repeating the forward computation will achieve the same results.

```{.python .input}
net(x)
```

Next, we will see what happens after we call the `hybridize` function.

```{.python .input}
net.hybridize()
net(x)
```

We can see that `F` turns into a Symbol. Moreover, even though the input data is still NDArray, the same input and intermediate output will all be converted to Symbol type in the `hybrid_forward` function.

Now, we repeat the forward computation.

```{.python .input}
net(x)
```

We can see that the three lines of print statements defined in the `hybrid_forward` function will not print anything. This is because a symbolic program has been produced since the last time `net(x)` was run by calling the `hybridize` function. Afterwards, when we run `net(x)` again, MXNet will no longer need to access Python code, but can directly perform symbolic programming at the C++ backend. This is another reason why model computing performance will be improve after the `hybridize` function is called. However, there is always the potential that any programs we write will suffer a loss in flexibility. If we want to use the three lines of print statements to debug the code in the above example, they will be skipped over and we would not be able to print when the symbolic program is executed. Additionally, in the case of a few functions not supported by Symbol (like `asnumpy`), and operations in-place like `a += b` and `a[:] = a + b` (must be rewritten as `a = a + b`). Therefore, we will not be able to use the `hybrid_forward` function or perform forward computation after the `hybridize` function has been called.


## Summary

* Both imperative and symbolic programming have their advantages as well as their disadvantages. Through hybrid programming, MXNet is able to combine the advantages of both.
* Models constructed by the HybridSequential and HybridBlock classes are able to convert imperative program into symbolic program by calling the `hybridize` function. We recommend using this method to improve computing performance.


## exercise

* Add `x.asnumpy()` to the first line of the `hybrid_forward` function of the HybridNet class in this section, run all the code in this section, and observe any error types and locations
* What happens if we add the Python statements `if` and `for` in the `hybrid_forward` function?
* Review the models that interest you in the previous chapters and use the HybridBlock class or HybridSequential class to implement them.


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/1665)

![](../img/qr_hybridize.svg)
