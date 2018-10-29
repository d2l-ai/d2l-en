# Model Selection, Underfitting and Overfitting

In the previous sections we evaluated the machine learning model’s performance using the training and testing data sets using experiments based on the Fashion-MNIST data set.  If you altered the model structure or the hyper-parameters during the experiment, you may have found that the model might not have been as accurate when using the testing data set compared to when the training data set was used.  Why is this?


## Training Error and Generalization Error

Before we can explain the aforementioned phenomena, we need to differentiate between training and a generalization error.  In layman's terms, training error refers to the error rate exhibited by the model during its use of the training data set and generalization error refers to any expected error rate exhibited by the model during the use of training data samples. The latter is often estimated based on the errors observed during the use of the testing data set.  In the previously discussed loss functions, for example, the squared loss function used for linear regression or the cross-entropy loss function used for softmax regression, can be used to calculate training and generalization error rates. 

To explain the two error concepts in further detail, we’ll be using the college entrance examination example.  In this example, the error rate when performing the college entrance examination tests (training) in previous years can be considered as the training error, while the estimated answer error rate when the actual college entrance examinations (test) are taken can be considered to be the generalization error.   In this example we have assumed that the training and test samples have been taken at random from an undefined, large test bank, in accordance with the examination outlines.  If a primary school student, who does not possess the same knowledge as that of a secondary school student, is asked to answer such questions then the answer error rates for both the training and the test could potentially be very similar.  However, if a third-grade middle school student, who repeatedly practices the exercises (training), were to attempt the training exercises and achieve an error rate of 0, it would not necessarily mean that the actual college examination (test) results will show a similar error rate. 

In machine learning we generally assume that each sample in both the training (exercises) and testing (test) data sets have been generated independently from the same probability distribution.  Based on this independent and identical distribution hypothesis, the training and generalization error expectations are identical when making use of any given machine learning model (with parameters).  For example, if we assigned random values (the primary school student) to the model parameters, the training and generalization error rates will be very similar.  However, in the previous chapters we learned that the selection of the parameters learned by the model while training on the training data set is based on the minimization of the training error rate (as in the case of the third-grade middle school examinee).  Therefore, the expectation of a training error is less than or equal to that of a generalization error. This means that, in general, the model parameters learned from the training data set will result in a model performance equal to or better than its performance when using the testing data set.  Since the generalization error rate cannot be estimated based on the training error rate, simply minimizing the training error rate will not necessarily mean a reduction in the generalization error rate. 

Machine learning models should therefore focus on the reduction of generalization errors.


## Model Selection

In machine learning we usually select our model based on an evaluation of the performance of several candidate models.  This process is called model selection. The candidate models can be similar models using different hyper-parameters.  Using the multilayer perceptron as an example, we can select the number of hidden layers as well as the number of hidden units, and activation functions in each hidden layer.  A significant effort in model selection is usually required in order to end up with an effective model.  In the following section we will be describing the validation data set often used in model selection. 


### Validation Data Set

Strictly speaking, the test set can only be used after all the hyper-parameters and model parameters have been selected.  The test data cannot be used in model selection process, such as in the tuning of hyper-parameters.  We should not rely solely on the training data during model selection, since the generalization error rate cannot be estimated from the training error rate.  Bearing this in mind, we can reserve a portion of data outside of the training and testing data sets to be used in model selection.  This reserved data is known as the validation data set, or validation set.  For example, a small, randomly selected portion from a given training set can be used as a validation set, with the remainder used as the true training set. 

However, in practical applications the test data is rarely discarded after one use since it’s not easily obtainable.  Therefore, in practice, there may be unclear boundaries between validation and testing data sets.  Unless explicitly stated, the test data sets used in the experiments provided in this book should be considered as validation sets, and the test accuracy in the experiment report are for validation accuracy. 


### $K$-Fold Cross-Validation

When there is not enough training data, it is considered excessive to reserve a large amount of validation data, since the validation data set does not play a part in model training.  A solution to this is the $K$-fold cross-validation method. In  $K$-fold cross-validation, the original training data set is split into $K$ non-coincident sub-data sets. Next, the model training and validation process is repeated $K$ times.  Every time the validation process is repeated, we validate the model using a sub-data set and use the $K-1$ sub-data set to train the model.  The sub-data set used to validate the model is continuously changed throughout this $K$ training and validation process.  Finally, the average $K$ training and validation error rates are calculated respectively. 



## Underfitting and Overfitting

Next, we will look into two common problems that occur during model training.  One type of problem occurs when the model unable to reduce training errors. This phenomenon is known as underfitting.  Another type of problem is when the number of model training errors is significantly less than that of the testing data set. This phenomenon is known as overfitting.  In practice, both underfitting and overfitting should be dealt with simultaneously whenever possible.  Although many factors could cause the above two fitting problems, for the time being we’ll be focusing primarily on two factors: model complexity and training data set size. 


### Model Complexity

A polynomial function fitting will be used as an example to explain model complexity.  Given a training data set consisting of the scalar data feature $x$ and the corresponding scalar label $y$, the polynomial function fitting’s goal is to find a $K$ order of a polynomial function 

$$\hat{y}= b + \sum_{k=1}^K x^k w_k$$

to estimate $y$. In the above formula, $w_k$ refers to the model’s weight parameter while $b$ is the bias parameter. Similar to linear regression, polynomial fitting also makes use of a squared loss function.  Particularly, first-order polynomial function fitting is also referred to as linear function fitting.

The higher order polynomial function is more complex than the lower order polynomial function, since the higher-order polynomial function model features more parameters and the model function’s selection range is wider.  Therefore, using the same training data set, higher order polynomial functions are able to achieve a lower training error rate easier than lower order polynomial functions.  Bearing in mind the given training data set, the typical relationship between model complexity and error is shown in Figure 3.4. In reference to the given training data set, if the model complex enough, it is easy for underfitting to occur. Similarly, if the model is too complex, it is easy for overfitting to happen.  Choosing an appropriately complex model for the data set is one way to avoid underfitting and overfitting. 


![Influence of Model Complexity on Underfitting and Overfitting](../img/capacity_vs_error.svg)


### Training Data Set Size

Another influencing factor with regards to underfitting and overfitting is the size of the training data set. Typically, if there are not enough samples in the training data set, especially if the number of samples is less than the number of model parameters (count by element), overfitting is more likely to occur.  Additionally, if the number of samples in the training data set increases, the generalization error rate will not increase.  Therefore when within the allowable range of computing resource, we typically expect larger training data set, especially for highly complex models. For example, in the case of a deep learning model with a large number of layers. 


## Polynomial Function Fitting Experiment

In order to understand the influence model complexity and the training data set size have on underfitting and overfitting, we’ll experiment by using a polynomial function fitting as an example. First, import the package or module needed for the experiment.

```{.python .input  n=1}
%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
```

### Generating Data Sets

A data set will be manually generated.  Given the sample feature $x$, we will use the following third-order polynomial function to generate the sample label in the both the training and testing data sets:  

$$y = 1.2x - 3.4x^2 + 5.6x^3 + 5 + \epsilon,$$

The noise term $\epsilon$ obeys a normal distribution with a mean of 0 and a standard deviation of 0.1.  The number of samples for both the training and the testing data sets  is set to 100.

```{.python .input  n=2}
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5
features = nd.random.normal(shape=(n_train + n_test, 1))
poly_features = nd.concat(features, nd.power(features, 2),
                          nd.power(features, 3))
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)
labels += nd.random.normal(scale=0.1, shape=labels.shape)
```

Take a look at the first 2 samples from the generated data set.

```{.python .input  n=3}
features[:2], poly_features[:2], labels[:2]
```

### Defining, Training and Testing Model

We first define the plotting function`semilogy`, where the $y$ axis makes use of the logarithmic scale.

```{.python .input  n=4}
# This function has been saved in the gluonbook package for future use.
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    gb.set_figsize(figsize)
    gb.plt.xlabel(x_label)
    gb.plt.ylabel(y_label)
    gb.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        gb.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        gb.plt.legend(legend)
```

Similar to linear regression, polynomial function fitting also makes use of a squared loss function.  Since we will be attempting to fit the generated data set using models of varying complexity, we insert the model definition into the `fit_and_plot` function. The training and testing steps involved in polynomial function fitting are similar to those previously described in softmax regression.

```{.python .input  n=5}
num_epochs, loss = 100, gloss.L2Loss()

def fit_and_plot(train_features, test_features, train_labels, test_labels):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    batch_size = min(10, train_labels.shape[0])
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'sgd',
                            {'learning_rate': 0.01})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])
    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())
```

### Third-order Polynomial Function Fitting (Normal)

We will begin by first using a third-order polynomial function with the same order as the data generation function.  Experiment results show that this model’s training error rate when using the testing data set is low.  The trained model parameters are also close to the true values: $w_1 = 1.2, w_2=-3.4, w_3=5.6, b = 5$.

```{.python .input  n=6}
fit_and_plot(poly_features[:n_train, :], poly_features[n_train:, :],
             labels[:n_train], labels[n_train:])
```

### Linear Function Fitting (Underfitting)

Let’s take another look at linear function fitting.  Naturally, after the decline in the early epoch, it’s difficult to further decrease this model’s training error rate.  After the last epoch iteration has been completed, the training error rate is still high.  When used in data sets generated by non-linear models (like the third-order polynomial function) linear models are susceptible to underfitting.

```{.python .input  n=7}
fit_and_plot(features[:n_train, :], features[n_train:, :], labels[:n_train],
             labels[n_train:])
```

### Insufficient Training (Overfitting)

In practice, if the model hasn’t been trained sufficiently, it is still easy to overfit even if a third-order polynomial function with the same order as the data generation model is used.  Let's train the model using just two samples. There apparently naturally aren’t enough training samples as they’re less than the number of model parameters.  This will result in a model that’s too complex to be easily influenced by noise in the training data.  Even if the training error rate is low, the testing error data rate will still be high in the iteration process.  This is a typical case of overfitting.

```{.python .input  n=8}
fit_and_plot(poly_features[0:2, :], poly_features[n_train:, :], labels[0:2],
             labels[n_train:])
```

Further along in later chapters, we will continue discussing overfitting problems and methods for dealing with them, such as weight decay and dropout. 


## Summary

* Since the generalization error rate cannot be estimated based on the training error rate, simply minimizing the training error rate will not necessarily mean a reduction in the generalization error rate. Machine learning models should therefore focus on the reduction of generalization errors.
* The validation data set can be used in model selection. 
* Underfitting means that the model is not able to reduce the training error rate while overfitting is a result of the model training error rate being much lower than the testing data set rate.  
* We should choose an appropriately complex model and avoid using insufficient training samples.


## exercise

* What problems may occur if a third-order polynomial model is used to fit the data generated by a linear model? Why? 
* In the third-order polynomial fitting problem mentioned in this section, is it possible to reduce the training error expectations of 100 samples to 0; and why? (Hint: Consider the existence of noise terms. )


## Scan the QR code to access [forum](https://discuss.gluon.ai/t/topic/983)

![](../img/qr_underfit-overfit.svg)
