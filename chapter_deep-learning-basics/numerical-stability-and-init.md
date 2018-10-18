# Numerical Stability and Model Initialization

After gaining an understanding of forward propagation and back propagation, we can discuss the numerical stability problem of the deep learning model and the initialization method of the model parameters. Typical problems related to numerical stability for deep models are vanishing and explosion.


## Vanishing and Explosion

When the number of layers of the neural network is large, the numerical stability of the model tends to deteriorate. Suppose that the weight parameter of the $l$th layer $\boldsymbol{H}^{(l)}$ of a multilayer perceptron with a total of $L$ layers is $\boldsymbol{W}^{(l)}$ and the weight parameter of the output layer $\boldsymbol{H}^{(L)}$ is $\boldsymbol{W}^{(L)}$. In this discussion, we will not consider the bias parameters, and we will assume the activation function of all hidden layers is identity mapping $\phi(x) = x$. Given the input $\boldsymbol{X}$, the output of the $l$th layer of the multilayer perceptron is $\boldsymbol{H}^{(l)} = \boldsymbol{X} \boldsymbol{W}^{(1)} \boldsymbol{W}^{(2)} \ldots \boldsymbol{W}^{(l)}$. At this point, if the number of layer $l$ is large, the computation of $\boldsymbol{H}^{(l)}$ may vanish or explode. For example, suppose the weight parameters of the input and all layers are scalar, e.g. the weight parameters are 0.2 and 5. The layer 30 output of the multilayer perceptron is the product of input $\boldsymbol{X}$ with $0.2^{30} \approx 1 \times 10^{-21}$ (vanishing) and $5^{30} \approx 9 \times 10^{20}$ (explosion) respectively. Similarly, when the number of layers is large, the computation of the gradient is more prone to vanishing or explosion.

As we delve more deeply into this subject, we will further discuss numerical stability problems and solutions for deep learning in later chapters.


## Randomly Initialize Model Parameters

In neural networks, we often need to randomly initialize model parameters. Let us examine the reason we do this.

Review the multilayer perceptron described in Figure 3.3 in the [“Multilayer Perceptron”](mlp.md) section. For the sake of this explanation, assume that the output layer only holds one output unit, $o_1$ (deleting $o_2, o_3$ and the arrows pointing to them), and the hidden layer uses the same activation function. If the parameters of each hidden unit are initialized to equal values, each hidden unit will compute the same value based on the same input during forward propagation and pass it to the output layer. In back propagation, the parameter gradient values of each hidden unit are equal. Therefore, these parameters are still equal after iteration using the gradient-based optimization algorithm. The same is true for subsequent iterations. In this case, no matter how many hidden units there are, the hidden layer operates as if it had only one hidden unit. Therefore, as we did in the previous experiments, we usually randomly initialize the model parameters of the neural network, especially the weight parameters.


### Default Random Initialization in MXNet

There are many ways to randomly initialize model parameters. In the [“Gluon Implementation of Linear Regression”](linear-regression-gluon.md) section, we use `net.initialize(init.Normal(sigma=0.01))` to make the weight parameters of the model `net` use a random initialization method of normal distribution. If the initialization method is not specified, such as `net.initialize()`, then MXNet will use the default random initialization method: each element of the weight parameter is randomly sampled with an even distribution between -0.07 and 0.07, and the bias parameters are all reset to zero.


### Xavier Random Initialization

There is also a more commonly used random initialization method called the Xavier random initialization[1]. Suppose that the number of inputs of a fully connected layer is $a$ and the number of outputs is $b$. Xavier random initialization will cause each element of the weight parameter in the layer to be randomly sampled in an even distribution.

$U\left(-\sqrt{\frac{6}{a+b}}, \sqrt{\frac{6}{a+b}}\right).$

Its design mainly considers the fact that after the model parameters are initialized, the variance of each layer's output should not be affected by the number of inputs of the layer. The variance of each layer's gradient should also not be affected by the number of outputs of the layer.

## Summary

* Vanishing and explosion are typical problems related to numerical stability for deep models. When the number of layers of the neural network is large, the numerical stability of the model tends to deteriorate.
* We often need to randomly initialize the model parameters of neural networks.


## exercise

* Some people say that the random initialization of model parameters is done for the sake of "breaking the symmetry." How should "symmetry" be understood here?
* Can we initialize all weight parameters in linear regression or softmax regression to the same value?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/8052)

![](../img/qr_numerical-stability-and-init.svg)

## References

[1] Glorot, X., & Bengio, Y. (2010, March). Understanding the Difficulty of Training Deep Feedforward Neural Networks. In Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics (pp. 249-256).
