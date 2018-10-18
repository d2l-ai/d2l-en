# Forward Propagation, Back Propagation and Computational Graphs

In the previous sections we used a mini-batch stochastic gradient descent optimization algorithm to train the model. During the implementation of the algorithm, we only calculated the forward propagation of the model, which is to say, we calculated the model output for the input, then called the auto-generated `backward` function to then finally calculate the gradient through the `autograd` module. The automatic gradient calculation, when based on back-propagation, significantly simplifies the implementation of the deep learning model training algorithm. In this section we will use both mathematical and computational graphs to describe forward and back propagation. More specifically, we will explain forward and back propagation through a sample model with a single hidden layer perceptron with $L_2$ norm regularization.

## Forward Propagation

Forward propagation refers to the calculation and storage of intermediate variables (including outputs) for the neural network within the models in the order from input layer to output layer. For the sake of simplicity, let’s assume that the input example is $\boldsymbol{x}\in \mathbb{R}^d$ and the bias term is not considered, and that the intermediate variable is

$\boldsymbol{z}= \boldsymbol{W}^{(1)} \boldsymbol{x},$

where $\boldsymbol{W}^{(1)} \in \mathbb{R}^{h \times d}$ is the weight parameter of the hidden layer. After entering the intermediate variable $\boldsymbol{z}\in \mathbb{R}^h$ into the activation function $\phi$ operated by the basic elements, we will obtain a hidden layer variable with the vector length of $h$,

$\boldsymbol{h}= \phi (\boldsymbol{z}).$

The hidden variable $\boldsymbol{h}$ is also an intermediate variable. Assuming the parameters of the output layer only possess a weight of $\boldsymbol{W}^{(2)} \in \mathbb{R}^{q \times h}$, we can obtain an output layer variable with a vector length of $q$,

$\boldsymbol{o}= \boldsymbol{W}^{(2)} \boldsymbol{h}.$

Assuming the loss function is $\ell$ and the example label is $y$, we can then calculate the loss term for a single data example,

$L = \ell(\boldsymbol{o}, y).$

According to the definition of $L_2$ norm regularization, given the hyper-parameter $\lambda$, the regularization term is

$$s = \frac{\lambda}{2} \left(\|\boldsymbol{W}^{(1)}\|_F^2 + \|\boldsymbol{W}^{(2)}\|_F^2\right),$$

where the Frobenius norm of the matrix is equivalent to the calculation of the $L_2$ norm after flattening the matrix to a vector. Finally, the model's regularized loss on a given data example is

$J = L + s.$

We refer to $J$ as the objective function of a given data example and refer to it as the ‘objective function’ in the following discussion.


## Computational Graph of Forward Propagation

We usually plot computational graphs to help us visualize the dependencies of operators and variables within the calculation. Figure 3.6 plots the computational graph of the forward propagation of the sample model in this section. The lower left corner signifies the input and the upper right corner the output. Notice that the direction of the arrows in the figure are primarily rightward and upward.

![Computational Graph of Forward Propagation. These boxes represent variables, the circles represent operators, and the arrows represent any dependencies from input to output. ](../img/forward.svg)


## Back Propagation

Back propagation refers to the method of calculating the gradient of neural network parameters. In general, back propagation calculates and stores the intermediate variables of an objective function related to each layer of the neural network and the gradient of the parameters in the order of the output layer to the input layer according to the ‘chain rule’ in calculus. For functions $\mathsf{Y}=f(\mathsf{X})$ and $\mathsf{Z}=g(\mathsf{Y})$, in which the input and the output $\mathsf{X}, \mathsf{Y}, \mathsf{Z}$ are tensors of arbitrary shapes, through the chain rule, we then have

$\frac{\partial \mathsf{Z}}{\partial \mathsf{X}} = \text{prod}\left(\frac{\partial \mathsf{Z}}{\partial \mathsf{Y}}, \frac{\partial \mathsf{Y}}{\partial \mathsf{X}}\right),$

where the $\text{prod}$ operator will multiply the two inputs after the necessary operations (such as transposition and swapping input positions) based on their shape.

In reviewing the sample model in this section, it's parameters are $\boldsymbol{W}^{(1)}$ and $\boldsymbol{W}^{(2)}$, therefore the objective of back propagation is to calculate $\partial J/\partial \boldsymbol{W}^{(1)}$ and $\partial J/\partial \boldsymbol{W}^{(2)}$. We will then, in turn, apply the chain rule to calculate the gradient of each intermediate variable and parameter. Here, the order of calculations are opposite to that of the corresponding intermediate variables performed during forward propagation. First, calculate the gradients of the objective function $J=L+s$ with respect to the loss term $L$ and the regularization term $s$,

$\frac{\partial J}{\partial L} = 1, \quad \frac{\partial J}{\partial s} = 1.$, respectively

Second, calculate the gradient of the objective function with respect to the output layer variable according to the chain rule, $\partial J/\partial \boldsymbol{o}\in \mathbb{R}^q$:

$
\frac{\partial J}{\partial \boldsymbol{o}}
= \text{prod}\left(\frac{\partial J}{\partial L}, \frac{\partial L}{\partial \boldsymbol{o}}\right)
= \frac{\partial L}{\partial \boldsymbol{o}}.
$


Next, calculate the gradients of the regularization term with respect to two parameters:

$\frac{\partial s}{\partial \boldsymbol{W}^{(1)}} = \lambda \boldsymbol{W}^{(1)},\quad\frac{\partial s}{\partial \boldsymbol{W}^{(2)}} = \lambda \boldsymbol{W}^{(2)}.$


Now we are able calculate the gradient $\partial J/\partial \boldsymbol{W}^{(2)} \in \mathbb{R}^{q \times h}$ of the model parameters closest to the output layer. According to the chain rule, we then get

$
\frac{\partial J}{\partial \boldsymbol{W}^{(2)}}
= \text{prod}\left(\frac{\partial J}{\partial \boldsymbol{o}}, \frac{\partial \boldsymbol{o}}{\partial \boldsymbol{W}^{(2)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \boldsymbol{W}^{(2)}}\right)
= \frac{\partial J}{\partial \boldsymbol{o}} \boldsymbol{h}^\top + \lambda \boldsymbol{W}^{(2)}.
$


Continue back propagation along the output layer to the hidden layer. The gradient $\partial J/\partial \boldsymbol{h}\in \mathbb{R}^h$ of the hidden layer variable can be calculated as follows:

$
\frac{\partial J}{\partial \boldsymbol{h}}
= \text{prod}\left(\frac{\partial J}{\partial \boldsymbol{o}}, \frac{\partial \boldsymbol{o}}{\partial \boldsymbol{h}}\right)
= {\boldsymbol{W}^{(2)}}^\top \frac{\partial J}{\partial \boldsymbol{o}}.
$


Because the activation function $\phi$ is operated by basic elements, calculating the gradient $\partial J/\partial \boldsymbol{z}\in \mathbb{R}^h$ of the intermediate variable $\boldsymbol{z}$ requires the use of the multiplication operator via element $\odot$:

$
\frac{\partial J}{\partial \boldsymbol{z}}
= \text{prod}\left(\frac{\partial J}{\partial \boldsymbol{h}}, \frac{\partial \boldsymbol{h}}{\partial \boldsymbol{z}}\right)
= \frac{\partial J}{\partial \boldsymbol{h}} \odot \phi'\left(\boldsymbol{z}\right).
$

Finally, we can obtain the gradient $\partial J/\partial \boldsymbol{W}^{(1)} \in \mathbb{R}^{h \times d}$ of the model parameters closest to the input layer. According to the chain rule, we get

$
\frac{\partial J}{\partial \boldsymbol{W}^{(1)}}
= \text{prod}\left(\frac{\partial J}{\partial \boldsymbol{z}}, \frac{\partial \boldsymbol{z}}{\partial \boldsymbol{W}^{(1)}}\right) + \text{prod}\left(\frac{\partial J}{\partial s}, \frac{\partial s}{\partial \boldsymbol{W}^{(1)}}\right)
= \frac{\partial J}{\partial \boldsymbol{z}} \boldsymbol{x}^\top + \lambda \boldsymbol{W}^{(1)}.
$

## Training Deep Learning Model

When training the deep learning model, forward propagation and back propagation are interdependent of one another. Below, we will still use the sample models in this section to illustrate the dependencies between them.

一方面，正向传播的计算可能依赖于模型参数的当前值。而这些模型参数是在反向传播的梯度计算后通过优化算法迭代的。例如，计算正则化项$s = (\lambda/2) \left(\|\boldsymbol{W}^{(1)}\|_F^2 + \|\boldsymbol{W}^{(2)}\|_F^2\right)$依赖模型参数$\boldsymbol{W}^{(1)}$和$\boldsymbol{W}^{(2)}$的当前值。而这些当前值是优化算法最近一次根据反向传播算出梯度后迭代得到的。

However, the gradient calculation of back propagation may be dependent on the current value of each variable. The current values of these variables are also calculated by forward propagation. For example, the calculation of a parameter gradient $\partial J/\partial \boldsymbol{W}^{(2)} = (\partial J / \partial \boldsymbol{o}) \boldsymbol{h}^\top + \lambda \boldsymbol{W}^{(2)}$ needs to rely on the current value $\boldsymbol{h}$ of a hidden layer variable. This current value is obtained by calculation and storage via forward propagation from the input layer to the output layer.

Therefore, after the initialization of the model parameters has completed, we can then perform forward propagation and back propagation alternately, and update model parameters according to the gradient found through the back propagation calculation. Since we reused the intermediate variables calculated by forward propagation in back propagation as to avoid double counting, we also encounter the intermediate variable memory becoming unable to be released immediately after the forward propagation. This is also an important reason as to why training occupies more memory than prediction. Additionally, it should be noted that the number of intermediate variables is linearly related to the number of network layers. The size of each variable is also linearly related to the batch size and the number of inputs. They are primarily why the deeper neural network is more prone to exceeding the available memory when using larger batches for training.


## Summary

* Forward propagation sequentially calculates and stores intermediate variables within the neural network in order of input layer to output layer.
* Back propagation sequentially calculates and stores the gradients of intermediate variables and parameters within the neural network in order of output layer to input layer.
* When training deep learning models, forward propagation and back propagation are interdependent.


## exercise

* Add bias parameters to the hidden layer and output layer of the sample model in this section, then modify the computational graphs and mathematical expressions of forward propagation and back propagation.


## Scan the QR code to get to the [forum](https://discuss.gluon.ai/t/topic/3710)

![](../img/qr_backprop.svg)
