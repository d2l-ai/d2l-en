# Multilayer Perceptron

We have now introduced single-layer neural networks, including linear regression and Softmax regression. However, deep learning primarily focuses on multilayer models. In this section, using multilayer perceptron (MLP) as an example, we will introduce the concept of multilayer neural networks.

## Hidden Layer

Multilayer perceptrons import one or more hidden layers using the single-layer neural network. This hidden layer is located between the input and output layers. Figure 3.3 shows a neural network diagram of the multilayer perceptron.

![Multilayer perceptron with hidden layers. This example contains a hidden layer with 5 hidden units in it. ](../img/mlp.svg)

In the multilayer perceptron of Figure 3.3, the number of inputs and outputs is 4 and 3 respectively, and the hidden layer in the middle contains 5 hidden units. Since the input layer does not involve any calculations, there are a total of 2 layers of the multilayer perceptron in Figure 3.3. As shown in Figure 3.3, the neurons in the hidden layer are fully connected to the inputs within the input layer. The neurons in the output layer and the neurons in the hidden layer are also fully connected. Therefore, both the hidden layer and the output layer in the multilayer perceptron are fully connected layers.


More specifically, when given a mini-batch of samples $\boldsymbol{X}\in \mathbb{R}^{n \times d}$, the batch size is $n$, and the number of inputs is $d$. Assume that the multilayer perceptron has only one hidden layer, and the total number of hidden units is $h$. Record the output of the hidden layer (also known as ‘hidden layer variables’ or ‘hidden variables’) as $\boldsymbol{H}$, we have $\boldsymbol{H} \in \mathbb{R}^{n \times h}$. Because both the hidden layer and output layer are wholly connected layers, we can set the weight and bias parameters of the hidden layer to $\boldsymbol{W}_h \in \mathbb{R}^{d \times h}$ and $\boldsymbol{b}_h \in \mathbb{R}^{1 \times h}$, the weight and bias parameters of the output layer are $\boldsymbol{W}_o \in \mathbb{R}^{h \times q}$ and $\boldsymbol{b}_o \in \mathbb{R}^{1 \times q}$, respectively.

First, let’s take a look at the design of a multilayer perceptron with a single hidden layer. The output $\boldsymbol{O}\in \mathbb{R}^{n \times q}$ is calculated as

$$
\begin{aligned}
\boldsymbol{H} &= \boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h,\\
\boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
\end{aligned}      
$$

This is to say, the output of the hidden layer is used directly as the input of the output layer. If we combine the above two formulas, we can get

$
\boldsymbol{O}= (\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h)\boldsymbol{W}_o + \boldsymbol{b}_o = \boldsymbol{X} \boldsymbol{W}_h\boldsymbol{W}_o + \boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o.
$

It can be seen from the formula after combination, although the neural network introduces the hidden layer, it is still equivalent to a single-layer neural network: the weight parameter of output layer is $\boldsymbol{W}_h\boldsymbol{W}_o$, the bias parameter is $\boldsymbol{b}_h \boldsymbol{W}_o + \boldsymbol{b}_o$. It is not difficult to find, even if more hidden layers are added, as the above formula is only able to obtain a single-layer neural network when using the output layer alone.


## Activation Function

The root of the problem discussed above lies in the fully connected layer only performing affine transformation on the data, while the superposition of multiple affine transformations remain affine transformations. One way to solve this problem is to introduce nonlinear transformation. This can be done by transforming a hidden variable using a nonlinear function that operates by element, then continuing to have the nonlinear function act as the input to the next fully connected layer. This nonlinear function is referred to as an ‘activation function’. In the following section, we will introduce several commonly used activation functions.

### ReLU Function

The ReLU (rectified linear unit) function provides a very simple nonlinear transformation. Given the element $x$, the function is defined as

$\text{ReLU}(x) = \max(x, 0).$

It can be understood that the ReLU function retains only positive elements and dissipates negative elements. To observe this nonlinear transformation visually, we must first define a plotting function `xyplot`.

```{.python .input  n=6}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, nd

def xyplot(x_vals, y_vals, name):
    gb.set_figsize(figsize=(5, 2.5))
    gb.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    gb.plt.xlabel('x')
    gb.plt.ylabel(name + '(x)')
```

Then, we can plot the ReLU function using the `relu` function provided by NDArray. As you can see, the activation function is a two-stage linear function.

```{.python .input  n=7}
x = nd.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = x.relu()
xyplot(x, y, 'relu')
```

Obviously, when the input is negative, the derivative of ReLU function is 0; when the input is positive, the derivative of ReLU function is 1. Although the ReLU function is not derivable when the input is 0, we can set the derivative here to 0. The derivative of ReLU function is plotted below.

```{.python .input}
y.backward()
xyplot(x, x.grad, 'grad of relu')
```

### Sigmoid Function

The Sigmoid function can transform the value of an element between 0 and 1:

$\text{sigmoid}(x) = \frac{1}{1 + \exp(-x)}.$

The Sigmoid function is commonly used in early neural networks, but is currently being replaced by a simpler ReLU function. In the "Recurrent Neural Network" chapter, we will describe how to utilize the function’s ability to control the flow of information in the neural network thanks to its capacity to transform the value range between 0 and 1. The derivative of Sigmoid function is plotted below. When the input is close to 0, the Sigmoid function approaches linear transformation.

```{.python .input  n=8}
with autograd.record():
    y = x.sigmoid()
xyplot(x, y, 'sigmoid')
```

According to the chain rule, the derivative of Sigmoid function is as follows:

$\text{sigmoid}'(x) = \text{sigmoid}(x)\left(1-\text{sigmoid}(x)\right).$


The derivative of Sigmoid function is plotted below. When the input is 0, the derivative of the Sigmoid function reaches a maximum of 0.25; as the input deviates further from 0, the derivative of Sigmoid function approaches 0.

```{.python .input}
y.backward()
xyplot(x, x.grad, 'grad of sigmoid')
```

### Tanh Function

The Tanh (Hyperbolic Tangent) function transforms the value of an element between -1 and 1:

$\text{tanh}(x) = \frac{1 - \exp(-2x)}{1 + \exp(-2x)}.$

We can then plot the Tanh function. As the input nears 0, the Tanh function approaches linear transformation. Although the shape of the function is similar to that of the Sigmoid function, the Tanh function is symmetric at the origin of the coordinate system.

```{.python .input  n=9}
with autograd.record():
    y = x.tanh()
xyplot(x, y, 'tanh')
```

According to the chain rule, the derivative of Tanh function is:

$\text{tanh}'(x) = 1 - \text{tanh}^2(x).$

The derivative of Tanh function is plotted below. As the input nears 0, the derivative of the Tanh function approaches a maximum of 1; as the input deviates away from 0, the derivative of the Tanh function approaches 0.

```{.python .input}
y.backward()
xyplot(x, x.grad, 'grad of tanh')
```

## Multilayer Perceptron

The multilayer perceptron is the neural network composed of fully connected layers containing at least one hidden layer. Here, the output of each hidden layer is transformed by an activation function. The number of layers to the multilayer perceptron and the number of hidden units within each hidden layer are considered hyper-parameters. Taking a single hidden layer as an example and by following the key points previously defined in this section, the multilayer perceptron calculates the output as follows:

$$
\begin{aligned}
\boldsymbol{H} &= \phi(\boldsymbol{X} \boldsymbol{W}_h + \boldsymbol{b}_h),\\
\boldsymbol{O} &= \boldsymbol{H} \boldsymbol{W}_o + \boldsymbol{b}_o,
\end{aligned}
$$
 
$\phi$ represents the activation function. In the classification problem, we can perform a Softmax operation on the output $\boldsymbol{O}$, and use the cross-entropy loss function in Softmax regression.
In the regression problem, we set the number of outputs within the output layer to 1, and set the output directly $\boldsymbol{O}$ to the squared loss function we used in linear regression.  



## Summary

* The multilayer perceptron adds one or multiple fully connected hidden layers between the output and input layers and transforms the output of the hidden layer via an activation function.
* Commonly used activation functions include the ReLU function, the Sigmoid function, and the Tanh function.


## exercise

* Applying the chain rule, derive the mathematical expression of the derivative of Sigmoid function and Tanh function.
* Check out the information below to learn about other activation functions.


## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/6447)

![](../img/qr_mlp.svg)
