# Softmax Regression

The linear regression model introduced in the previous sections can be applied in situations where the output is a continuous value. In a other cases, the model output can be a discrete value like an image category. For this kind of discrete value prediction problem, we can use classification models such as softmax regression. Unlike linear regression, the output of softmax regression is changed from one unit to more than one, and the introduction of softmax operation makes the output more suitable for the prediction and training of discrete values.  In this section, we will use the softmax regression model to introduce classification models in neural networks.


## Classification Problems

Now, we will take a look at a simple image classification problem, where the input image has a height and width of 2 pixels and the color is grayscale. Thus, each pixel value can be represented by a scalar. We record the four pixels in the image as $x_1, x_2, x_3, x_4$. We assume that the actual labels of the images in the training data set are "dog", "cat", or "chicken" (assuming that the three animals can be represented by 4 pixels). These labels correspond to discrete values $y_1, y_2, y_3$, respectively.

We usually use discrete values to represent categories, such as $y_1=1, y_2=2, y_3=3$. Hence, the label of an image is one of the three values: 1, 2, and 3. Although we can still use regression models for modeling and set the predicted values to approach one of the three discrete values (1, 2, and 3), such conversion of continuous values to discrete values tends to affect the quality of the classification. Therefore, we generally use models that are more suitable for discrete value output to solve the classification problem.

## Softmax Regression Model

Like linear regression, softmax regression performs a linear blend between input features and weights. A major difference from linear regression is that the number of output values from the softmax regression is equal to the number of label categories. Because there are 4 features and 3 output animal categories, the weight contains 12 scalars ($w$ with subscripts) and the bias contains 3 scalars ($b$ with subscripts). We compute these three outputs, $o_1, o_2, and o_3$, for each input:

$$
\begin{aligned}
o_1 &= x_1 w_{11} + x_2 w_{21} + x_3 w_{31} + x_4 w_{41} + b_1,\\
o_2 &= x_1 w_{12} + x_2 w_{22} + x_3 w_{32} + x_4 w_{42} + b_2,\\
o_3 &= x_1 w_{13} + x_2 w_{23} + x_3 w_{33} + x_4 w_{43} + b_3.
\end{aligned}
$$


Figure 3.2 uses a neural network diagram to depict the calculation above.  Like linear regression, softmax regression is also a single-layer neural network.  Since the calculation of each output, $o_1, o_2, and o_3$, depends on all inputs, $x_1, x_2, x_3, and x_4$, the output layer of the softmax regression is also a fully connected layer.

![Softmax regression is a single-layer neural network.  ](../img/softmaxreg.svg)

### Softmax Operation

Since the classification problem requires discrete prediction output, we can use a simple approach to treat the output value $o_i$ as the confidence level of the prediction category $i$. We can choose the class with the largest output value as the predicted output, which is output $\ Operatorname*{argmax}_i o_i$. For example, if $o_1, o_2, and o_3$ are $0.1, 10, and 0.1$, respectively, then the prediction category is 2, which represents "cat".

However, there are two problems with using the output from the output layer directly. On the one hand, because the range of output values from the output layer is uncertain, it is difficult for us to visually judge the meaning of these values. For instance, the output value 10 from the previous example indicates a level of "very confident" that the image category is "cat". That is because its output value is 100 times that of the other two categories.  However, if $o_1=o_3=10^3$, then an output value of 10 means that the chance for the image category to be cat is very low.  On the other hand, since the actual label has discrete values, the error between these discrete values and the output values from an uncertain range is difficult to measure.

The softmax operator solves the two problems above. It transforms the output value into a probability distribution that has a positive value with a sum of 1:

$\hat{y}_1, \hat{y}_2, \hat{y}_3 = \text{softmax}(o_1, o_2, o_3),$

where

$
\hat{y}_1 = \frac{ \exp(o_1)}{\sum_{i=1}^3 \exp(o_i)},\quad
\hat{y}_2 = \frac{ \exp(o_2)}{\sum_{i=1}^3 \exp(o_i)},\quad
\hat{y}_3 = \frac{ \exp(o_3)}{\sum_{i=1}^3 \exp(o_i)}.
$

It is easy to see $\hat{y}_1 + \hat{y}_2 + \hat{y}_3 = 1$ with $0 \leq \hat{y}_1, \hat{y}_2, and \hat{y}_3 \leq 1$. Thus, $\hat{y}_1, \hat{y}_2, and \hat{y}_3$ is a legal probability distribution.  Now, if $\hat{y}_2=0.8$, regardless of the value of $\hat{y}_1$ and $\hat{y}_3$, we will know that the probability of the image category being cat is 80%. In addition, we noticed

$\operatorname*{argmax}_i o_i = \operatorname*{argmax}_i \hat y_i,$

So, the softmax operation does not change the prediction category output.

## Vector Calculation Expression for Single Example Classification

In order to improve the computational efficiency, we can use vector calculation to express the classification for a single example.  In the image classification problem above, it is assumed that the weight and bias parameters of the softmax regression are

$$
\boldsymbol{W} =
\begin{bmatrix}
    w_{11} & w_{12} & w_{13} \\
    w_{21} & w_{22} & w_{23} \\
    w_{31} & w_{32} & w_{33} \\
    w_{41} & w_{42} & w_{43}
\end{bmatrix},\quad
\boldsymbol{b} =
\begin{bmatrix}
    b_1 & b_2 & b_3
\end{bmatrix},
$$



Set the features of the example image $i$ with a height and width of 2 pixels to be

$\boldsymbol{x}^{(i)} = \begin{bmatrix}x_1^{(i)} & x_2^{(i)} & x_3^{(i)} & x_4^{(i)}\end{bmatrix},$

The output from the output layer is

$\boldsymbol{o}^{(i)} = \begin{bmatrix}o_1^{(i)} & o_2^{(i)} & o_3^{(i)}\end{bmatrix},$

The probability distribution for the prediction to be dog, cat or chicken is

$\boldsymbol{\hat{y}}^{(i)} = \begin{bmatrix}\hat{y}_1^{(i)} & \hat{y}_2^{(i)} & \hat{y}_3^{(i)}\end{bmatrix}.$


The vector calculation expression of softmax regression for the example $i$ classification is

$$
\begin{aligned}
\boldsymbol{o}^{(i)} &= \boldsymbol{x}^{(i)} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{y}}^{(i)} &= \text{softmax}(\boldsymbol{o}^{(i)}).
\end{aligned}
$$

## Vector Calculation Expression for Mini-batch Example Classification


To further improve computational efficiency, we usually carry out vector calculations for mini-batches of data. If we are given a mini-batch example with batch size $n$, the number of inputs (number of features) is $d$ and the number of outputs (number of categories) is $q$. Set the batch feature to be $\boldsymbol{X}\in \mathbb{R}^{n \times d}$. Assume the weight and bias parameters of softmax regression to be $\boldsymbol{W}\in \mathbb{R} ^{d \times q}, \boldsymbol{b} \in \mathbb{R}^{1 \times q}$. The vector calculation expression of softmax regression is

$$
\begin{aligned}
\boldsymbol{O} &= \boldsymbol{X} \boldsymbol{W} + \boldsymbol{b},\\
\boldsymbol{\hat{Y}} &= \text{softmax}(\boldsymbol{O}),
\end{aligned}
$$

In this case, the additional operation uses a broadcast mechanism $\boldsymbol{O}, \boldsymbol{\hat{Y}} \in \mathbb{R}^{n \times q}$, while row $i$ of the two matrices are the output $\boldsymbol{o}^{(i)} and probability distribution $\boldsymbol{\hat{y}}^{(i)}$ of example $i$, respectively.


## Cross-entropy Loss Function

As mentioned earlier, using softmax operations makes it easier to calculate errors with discrete labels. We already know that the softmax operation transforms the output into a legal class prediction distribution. As a matter of fact, actual labels can be expressed by classification distribution: for example $i$, we construct vector $\boldsymbol{y}^{(i)}\in \mathbb{R}^{q}$ to make its $y^{(i)}$th element (the discrete value of example $i$) equal to 1, and the rest equal to 0.  This way, our training goal can be set to make the predicted probability distribution $\boldsymbol{\hat y}^{(i)} stay as close as possible to the probability distribution $\boldsymbol{y}^{(i)}$ of the actual label.

我们可以像线性回归那样使用平方损失函数$\|\boldsymbol{\hat y}^{(i)}-\boldsymbol{y}^{(i)}\|^2/2$。然而，想要预测分类结果正确，我们其实并不需要预测概率完全等于标签概率。例如在图像分类的例子里，如果$y^{(i)}=3$，那么我们只需要$\hat{y}^{(i)}_3$比其他两个预测值$\hat{y}^{(i)}_1$和$\hat{y}^{(i)}_2$大就行了。即使$\hat{y}^{(i)}_3$值为0.6，不管其他两个预测值为多少，类别预测均正确。而平方损失则过于严格，例如$\hat y^{(i)}_1=\hat y^{(i)}_2=0.2$比$\hat y^{(i)}_1=0, \hat y^{(i)}_2=0.4$的损失要小很多，虽然两者都有同样正确的分类预测结果。

One way to address this issue is to use a measurement function that is more suitable for measuring the difference between two probability distributions. For cases like this, cross entropy is a commonly used measurement method:

$H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ) = -\sum_{j=1}^q y_j^{(i)} \log \hat y_j^{(i)},$

The subscripted $y_j^{(i)}$ is an element of the vector $\boldsymbol y^{(i)}$ that is either 0 or 1. You need to pay attention to its difference from the category discrete value of example $i$, which is $y^{(i)}$ without subscript. From the formula above, we know that only the $y^{(i)}$th element $y^{(i)}_{y^{(i)}}$, from the vector $\boldsymbol y^{(i)}$, is 1, and all the rest are 0. Hence, $H(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^ {(i)}) = -\log \hat y_{y^{(i)}}^{(i)}$. That is to say, cross entropy only concerns the prediction probability of the correct category. As long as its value is large enough, we can ensure that the classification result is correct. Of course, when an example has multiple labels, as when there is more than one object in the image, we cannot use this simplification method. Even in this case, the cross entropy is only concerned with the prediction probability of the class of objects appearing in the image.


Let us assume that the number of examples in the training data set is $n$, and the cross-entropy loss function is defined as
$\ell(\boldsymbol{\Theta}= \frac{1}{n}  \sum_{i=1}^n H\left(\boldsymbol y^{(i)}, \boldsymbol {\hat y}^{(i)}\right ),$

Here, $\boldsymbol{\Theta}$ represents the model parameters.  Similarly, if there is only one label per example, the cross-entropy loss can be abbreviated as $\ell(\boldsymbol{\Theta}= -(1/n) \sum_{i=1}^n \log \hat y_{y^{(i)}}^{(i)}$. From another perspective, we know that the minimization of $\ell(\boldsymbol{\Theta})$ is equal to the maximization of $\exp(-n\ell(\boldsymbol{\Theta}))=\prod_{i=1}^n \hat y_{y^{(i)}}^{(i)}$. In other words, minimizing the cross-entropy loss function is equivalent to maximizing the joint prediction probability of all label categories of the training data set.



## Model Prediction and Evaluation

After training the softmax regression model, given any example features, we can predict the probability of each output category. Normally, we use the category with the highest predicted probability as the output category. The prediction is correct if it is consistent with the actual category (label).  In the next part of the experiment, we will use accuracy to evaluate the model’s performance. This is equal to the ratio between the number of correct predictions and the total number of predictions.

## Summary

* Softmax regression applies to classification problems. It uses the probability distribution of the output category in the softmax operation.
* Softmax regression is a single-layer neural network, and the number of outputs is equal to the number of categories in the classification problem.
* Cross entropy is a good measure of the difference between two probability distributions.


## exercise

* Review the materials to understand the concept of maximum likelihood.  What similarities and differences does it have with the minimized cross-entropy loss function?


## Scan the QR Code to Access [Discussions](https://discuss.gluon.ai/t/topic/6403)

![](../img/qr_softmax-regression.svg)
