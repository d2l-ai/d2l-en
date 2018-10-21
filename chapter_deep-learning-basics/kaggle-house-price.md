# Getting Started with Kaggle Competition: House Price Prediction

In the previous chapters, we learnt about the basics of deep learning. To summarize this, in this chapter we will be applying what we have learned.   Let’s get started with a Kaggle Competition: House Price Prediction. This section covers the data preprocessing, the design of models, and the selection of hyper-parameters.  We hope that by having the opportunity for hands-on involvement, you will be able to observe the experimental phenomena, analyze the results, and continuously adjust the methods used until you’re finally able to achieve satisfactory results. 

## Kaggle Competition

Kaggle（网站地址：https://www.kaggle.com ）是一个著名的供机器学习爱好者交流的平台。图3.7展示了Kaggle网站首页。为了便于提交结果，你需要注册Kaggle账号。

![Kaggle website home page ](../img/kaggle.png)

On the House Prices Prediction Competition page, we can learn more about the competition and also view other entrants’ scores. You can also download the relevant data set and submit your own predictions.  The competition’s web address is 

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques


Figure 3.8 shows the House Price Prediction Competition’s web page.

![House Price Prediction Competition web page.  You can access the competition data set by clicking  the "Data" tab. ](../img/house_pricing.png)

## Accessing and Reading Data Sets

The competition data is separated into two sets: training and testing.  Each data set includes the eigenvalues for each house, like street type, year of construction, roof type, basement condition, etc.  These eigenvalues include consecutive numbers, discrete labels, and even missing values (labelled "na"). The price of each house, namely the label, is only included in the training data set.  To download the data set, we can access the competition page and click the "Data" tab as shown in Figure 3.8.

The data will be read in and processed through the use of `pandas`. Make sure you have `pandas` installed before importing the packages required for this section, otherwise please refer to the code comments below.

```{.python .input  n=3}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
import gluonbook as gb
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd
```

The decompressed data includes two csv files and can be found in the `../data` directory. You will need to use `pandas` to read both files.

```{.python .input  n=14}
train_data = pd.read_csv('../data/kaggle_house_pred_train.csv')
test_data = pd.read_csv('../data/kaggle_house_pred_test.csv')
```

The training data set includes 1,460 examples, 80 features, and 1 label.

```{.python .input  n=11}
train_data.shape
```

The testing data set includes 1,459 examples and 80 features. In the testing data set, we need to predict the label for each example.

```{.python .input  n=5}
test_data.shape
```

Let’s take a look at the first 4 and last 2 features as well as the label (SalePrice) from the first 4 examples:

```{.python .input  n=28}
train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
```

We can see that in each example, the first feature is ID. This helps the model identify each training example, however it’s difficult to implement this in the testing example, so we don’t make use of it during training.  All 79 features of both the training and the testing data are linked by example.

```{.python .input  n=30}
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## Data Preprocessing

To standardize the features of continuous values we need to set the feature with a mean of $\mu$ and a standard deviation of $\sigma$ throughout the entire data set. Next, we can subtract each of the feature’s values by $\mu$ and divide them by $\sigma$ to obtain each normalized eigenvalues. Missing eigenvalues are replaced with the feature mean.

```{.python .input  n=6}
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(all_features.mean())
```

接下来将离散数值转成指示特征。举个例子，假设特征MSZoning里面有两个不同的离散值RL和RM，那么这一步转换将去掉MSZoning特征，并新加两个特征MSZoning\_RL和MSZoning\_RM，其值为0或1。如果一个样本原来在MSZoning里的值为RL，那么有MSZoning\_RL=0且MSZoning\_RM=1。

```{.python .input  n=7}
# Dummy_na=True refers to a missing value being a legal eigenvalue, and creates an indicative feature for it. 
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

You can see that this conversion increases the number of features from 79 to 331. 

Finally, through the  `values` attribute, we can obtain the NumPy format data and this can then be converted to NDArray for later training.

```{.python .input  n=9}
n_train = train_data.shape[0]
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))
```

## To train a model

To train the model, we use a basic linear regression model as well as a quadratic loss function.

```{.python .input  n=13}
loss = gloss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

Next, the root mean squared logarithmic error used by the competition is determined to evaluate the model.  The predicted values are preset to $\hat y_1, \ldots, \hat y_n$ and the corresponding real labels to $y_1,\ldots, y_n$, and are then defined as

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log(y_i)-\log(\hat y_i)\right)^2}.$$

```{.python .input  n=11}
def log_rmse(net, train_features, train_labels):
    # To further stabilize the value when the logarithm is taken, set the value less than 1 as 1. 
    clipped_preds = nd.clip(net(train_features), 1, float('inf'))
    rmse = nd.sqrt(2 * loss(clipped_preds.log(), train_labels.log()).mean())
    return rmse.asscalar()
```

Unlike in the previous sections, the following training functions use the Adam optimization algorithm.  Compared to the previously used mini-batch stochastic gradient descent, the Adam optimization algorithm is relatively less sensitive to learning rates.  This will be covered in further detail later on in the chapter “Optimization Algorithms”.

```{.python .input  n=14}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(
        train_features, train_labels), batch_size, shuffle=True)
    # The Adam optimization algorithm is used here.
    trainer = gluon.Trainer(net.collect_params(), 'adam', {
        'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls
```

## $K$-Fold Cross-Validation

The $K$-fold cross-validation is introduced in the section[“Model Selection, Under-fitting and Over-fitting"](underfit-overfit.md). It will be used to select the model design and to adjust the hyper-parameters. The following makes use of a function that returns the training and validation data needed for the `i`-fold cross-validation.

```{.python .input}
def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = nd.concat(X_train, X_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return X_train, y_train, X_valid, y_valid
```

The training and verification error averages are returned when we train $K$ times in the $K$-fold cross-validation.

```{.python .input  n=15}
def k_fold(k, X_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                  weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            gb.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse',
                        range(1, num_epochs + 1), valid_ls,
                        ['train', 'valid'])
        print('fold %d, train rmse: %f, valid rmse: %f' % (
            i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
```

## Model Selection

To calculate the amount of cross-validation errors, a set of un-tuned hyper parameters are used.  These hyper-parameters can be changed to minimize the testing error average.

```{.python .input  n=16}
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
verbose_epoch = num_epochs - 2
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                         weight_decay, batch_size)
print('%d-fold validation: avg train rmse: %f, avg valid rmse: %f'
      % (k, train_l, valid_l))
```

You will notice that sometimes the number of training errors for a set of hyper-parameters can be very low, while the number of errors for the $K$-fold cross validation may be higher.  This is most likely a consequence of over fitting. Therefore, when we reduce the amount of training errors, we need to observe whether the amount of errors in the $K$-fold cross-validation have also been reduced accordingly. 

##  Determine your predictions and submit them on Kaggle. 

The prediction function is as defined below. The completed training data set will be used to retrain the model before prediction and the results will be saved in the format required for submission.

```{.python .input  n=18}
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    gb.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

After the model has been designed and the hyper-parameters have been adjusted, the next step is to predict the prices of the sample houses provided in the testing data set.  If the amount of training errors are similar to those of the cross-validation, then the predictions are potentially ideal and can be submitted on Kaggle.

```{.python .input  n=19}
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

A file, “submission.csv”, will be generated after the above code has been executed.  CSV is one of the file formats outlined in the Kaggle competition requirements.  Next, we can submit our predictions on Kaggle and compare them to the actual house price (label) on the testing data set, checking for errors.  The steps are as follows: Log in to the Kaggle website and visit the House Price Prediction Competition page. Next, click the “Submit Predictions” or “Late Submission” button on the right.  Next, click the “Upload Submission File” button in the dashed box at the bottom of the page and select the prediction file you wish to upload.  Finally, click the “Make Submission” button at the bottom of the page to view your results.  As shown in Figure 3.9.

![Kaggle's house price predictions submission page ](../img/kaggle_submit2.png)


## Summary

* Real data usually needs to be preprocessed. 
* We can use $K$-fold cross validation to select the model and adjust the hyper-parameters.


## exercise

* Submit your predictions for this tutorial to Kaggle. What can your predictions score on Kaggle?
* Can you improve the score on Kaggle by comparing the $K$-fold cross-validation results and by modifying the model (for example, by adding hidden layers), and tuning parameters?
* What happens if we do not standardize the continuous numerical features like we have done in this section?
* Scan the QR code to access the forum and exchange ideas about the methods used and the results obtained with the community. Can you identify any better techniques?

## Scan the QR code to access the [forum](https://discuss.gluon.ai/t/topic/1039)

![](../img/qr_kaggle-house-price.svg)
