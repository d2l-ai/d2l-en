# Predicting House Prices on Kaggle
:label:`sec_kaggle_house`

Now that we have introduced some basic tools
for building and training deep networks
and regularizing them with techniques including
dimensionality reduction, weight decay, and dropout,
we are ready to put all this knowledge into practice
by participating in a Kaggle competition.
[Predicting house prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) is a great place to start.
The data is fairly generic and does not exhibit exotic structure
that might require specialized models (as audio or video might).
This dataset, collected by Bart de Cock in 2011 :cite:`De-Cock.2011`,
covers house prices in Ames, IA from the period of 2006-2010.
It is considerably larger than the famous [Boston housing dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.names) of Harrison and Rubinfeld (1978),
boasting both more examples and more features.


In this section, we will walk you through details of 
data preprocessing, model design, and hyperparameter selection.
We hope that through a hands-on approach,
you will gain some intuitions that will guide you
in your career as a data scientist.


## Downloading and Caching Datasets

Throughout the book, we will train and test models 
on various downloaded datasets. 
Here, we implement several utility functions 
to facilitate data downloading. 
First, we maintain a dictionary `DATA_HUB` 
that maps a string (the *name* of the dataset)
to a tuple containing both a URL to locate the dataset
and a SHA-1 key which we will use 
to verify the integrity of the file. 
All of our datasets are hosted on site
whose address is assigned to `DATA_URL` below.

```{.python .input  n=1}
import os
from mxnet import gluon
import zipfile
import tarfile

# Saved in the d2l package for later use
DATA_HUB = dict()

# Saved in the d2l package for later use
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
```

The following `download` function downloads the dataset,
caching it in a local directory (in `../data` by default)
and returns the name of the downloaded file.
If a file corrsponding to this dataset 
already exists in the cache directory 
and its SHA-1 matches the one stored in `DATA_HUB`, 
our code will use the cached file to avoid 
clogging up your internet with redundant downloads.

```{.python .input  n=2}
# Saved in the d2l package for later use
def download(name, cache_dir=os.path.join('..', 'data')):
    """Download a file inserted into DATA_HUB, return the local filename."""
    assert name in DATA_HUB, "%s doesn't exist" % name
    url, sha1 = DATA_HUB[name]
    d2l.mkdir_if_not_exist(cache_dir)
    return gluon.utils.download(url, cache_dir, sha1_hash=sha1)
```

We also implement two additional functions: 
one is to download and extract a zip/tar file
and the other to download all the files from `DATA_HUB`
(most of the datasets used in this book) into the cache directory.

```{.python .input  n=3}
# Saved in the d2l package for later use
def download_extract(name, folder=None):
    """Download and extract a zip/tar file."""
    fname = download(name)
    base_dir = os.path.dirname(fname) 
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, 'Only zip/tar files can be extracted'
    fp.extractall(base_dir)
    if folder:
        return os.path.join(base_dir, folder)
    else:
        return data_dir

# Saved in the d2l package for later use
def download_all():
    """Download all files in the DATA_HUB"""
    for name in DATA_HUB:
        download(name)
```

## Kaggle

[Kaggle](https://www.kaggle.com) is a popular platform
that hosts machine learning competitions.
Each competition centers on a dataset and many
are sponsored by stakeholders who offer prizes
to the winning solutions.
The platform helps users to share interact 
via forums and shared code, 
fostering both collaboration and competition.
While leaderboard chasing often spirals out of control,
with researchers focusing myopically on pre-processing steps
rather than asking fundamental questions,
there's also tremendous value in the objectivity of a platform
that facillitates direct quantitative comparisons
between competing approaches as well as code sharing
so that everyone can learn what did and did not work.
If you want to participate in a Kaggle competitions,
you will first need to register for an account
(see :numref:`fig_kaggle`).

![Kaggle website](../img/kaggle.png)
:width:`400px`
:label:`fig_kaggle`

On the House Prices Prediction page, as illustrated 
in :numref:`fig_house_pricing`,
you can find the dataset (under the "Data" tab),
submit predictions, see your ranking, etc.,
The URL is right here:

> https://www.kaggle.com/c/house-prices-advanced-regression-techniques

![House Price Prediction](../img/house_pricing.png)
:width:`400px`
:label:`fig_house_pricing`

## Accessing and Reading the Dataset

Note that the competition data is separated 
into training and test sets.
Each record includes the property value of the house
and attributes such as street type, year of construction,
roof type, basement condition, etc.
The features consist of various data types.
For example, the year of construction
is represented by an integer,
the roof type by discrete categorical assignments,
and other features by floating point numbers.
And here is where reality complicates things:
for some examples, some data is altogether missing
with the missing value marked simply as *na*.
The price of each house is included 
for the training set only
(it is a competition after all).
We will want to partition the training set
to create a validation set,
but we only get to evaluate our models on the official test set
after uploading predictions to Kaggle.
The "Data" tab on the competition tab 
has links to download the data.


To get started, we will read in and process the data
using `pandas`, an [efficient data analysis toolkit](http://pandas.pydata.org/pandas-docs/stable/),
so you will want to make sure that you have `pandas` installed
before proceeding further.
Fortunately, if you are reading in Jupyter,
we can install pandas without even leaving the notebook.

```{.python .input  n=4}
# If pandas is not installed, please uncomment the following line:
# !pip install pandas

%matplotlib inline
import d2l
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

For convenience, we can download and cache 
the Kaggle housing dataset 
using the script we defined above.

```{.python .input  n=5}
# Saved in the d2l package for later use        
DATA_HUB['kaggle_house_train'] = (
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

# Saved in the d2l package for later use  
DATA_HUB['kaggle_house_test'] = (
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
```

To load the two csv files containing training 
and test data respectively we use Pandas.

```{.python .input  n=6}
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
```

```{.json .output n=6}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Downloading ../data/kaggle_house_pred_train.csv from http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_train.csv...\nDownloading ../data/kaggle_house_pred_test.csv from http://d2l-data.s3-accelerate.amazonaws.com/kaggle_house_pred_test.csv...\n"
 }
]
```

The training dataset includes $1460$ examples, 
$80$ features, and $1$ label, while the test data 
contains $1459$ examples and $80$ features.

```{.python .input  n=7}
print(train_data.shape)
print(test_data.shape)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(1460, 81)\n(1459, 80)\n"
 }
]
```

Let’s take a look at the first $4$ and last $2$ features
as well as the label (SalePrice) from the first $4$ examples:

```{.python .input  n=8}
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
```

```{.json .output n=8}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "   Id  MSSubClass MSZoning  LotFrontage SaleType SaleCondition  SalePrice\n0   1          60       RL         65.0       WD        Normal     208500\n1   2          20       RL         80.0       WD        Normal     181500\n2   3          60       RL         68.0       WD        Normal     223500\n3   4          70       RL         60.0       WD       Abnorml     140000\n"
 }
]
```

We can see that in each example, the first feature is the ID.
This helps the model identify each training example.
While this is convenient, it does not carry
any information for prediction purposes.
Hence, we remove it from the dataset 
before feeding the data into the network.

```{.python .input  n=9}
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
```

## Data Preprocessing

As stated above, we have a wide variety of data types.
We'll need to process the data before we can start modeling.
Let's start with the numerical features.
First, we apply a heuristic, 
replacing all missing values 
by the corresponding variable's mean.
Then, to put all variables on a common scale,
we rescale them to zero mean and unit variance:

$$x \leftarrow \frac{x - \mu}{\sigma}.$$

To verify that this indeed transforms
our variable such that it has zero mean and unit variance,
note that $E[(x-\mu)/\sigma] = (\mu - \mu)/\sigma = 0$
and that $E[(x-\mu)^2] = \sigma^2$.
Intuitively, we *normalize* the data
for two reasons. 
First, it proves convenient for optimization.
Second, because we do not know *a priori*
which features will be relevant,
we do not want to penalize coefficients
assigned to one variable more than on any other.

```{.python .input  n=10}
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# After standardizing the data all means vanish, hence we can set missing
# values to 0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
```

Next we deal with discrete values.
This includes variables such as 'MSZoning'.
We replace them by a one-hot encoding
in the same way that we previously transformed
multiclass labels into vectors.
For instance, 'MSZoning' assumes the values 'RL' and 'RM'.
These map onto vectors $(1, 0)$ and $(0, 1)$ respectively.
Pandas does this automatically for us.

```{.python .input  n=11}
# Dummy_na=True refers to a missing value being a legal eigenvalue, and
# creates an indicative feature for it
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
```

```{.json .output n=11}
[
 {
  "data": {
   "text/plain": "(2919, 331)"
  },
  "execution_count": 11,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

You can see that this conversion increases 
the number of features from 79 to 331.
Finally, via the `values` attribute,
we can extract the NumPy format from the Pandas dataframe
and convert it into MXNet's native `ndarray` 
representation for training.

```{.python .input  n=12}
n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values, dtype=np.float32)
test_features = np.array(all_features[n_train:].values, dtype=np.float32)
train_labels = np.array(train_data.SalePrice.values,
                        dtype=np.float32).reshape(-1, 1)
```

## Training

To get started we train a linear model with squared loss.
Not surprisingly, our linear model will not lead
to a competition-winning submission
but it provides a sanity check to see whether
there is meaningful information in the data.
If we cannot do better than random guessing here,
then there might be a good chance
that we have a data processing bug.
And if things work, the linear model will serve as a baseline
giving us some intuition about how close the simple model
gets to the best reported models, giving us a sense
of how much gain we should expect from fancier models.

```{.python .input  n=13}
loss = gluon.loss.L2Loss()

def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net
```

With house prices, as with stock prices,
we care about relative quantities 
more than absolute quantities.
Thus we tend to care more about 
the relative error $\frac{y - \hat{y}}{y}$
than about the absolute error $y - \hat{y}$.
For instance, if our prediction is off by USD 100,000
when estimating the price of a house in Rural Ohio,
where the value of a typical house is 125,000 USD,
then we are probably doing a horrible job.
On the other hand, if we err by this amount 
in Los Altos Hills, California,
this might represent a stunningly accurate prediction
(there, the median house price exceeds 4 million USD).

One way to address this problem is to
measure the discrepancy in the logarithm of the price estimates.
In fact, this is also the official error metric
used by the competition to measure the quality of submissions.
After all, a small value $\delta$ of $\log y - \log \hat{y}$
translates into $e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$.
This leads to the following loss function:

$$L = \sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

```{.python .input  n=14}
def log_rmse(net, features, labels):
    # To further stabilize the value when the logarithm is taken, set the
    # value less than 1 as 1
    clipped_preds = np.clip(net(features), 1, float('inf'))
    return np.sqrt(2 * loss(np.log(clipped_preds), np.log(labels)).mean())
```

Unlike in previous sections, our training functions 
will rely on the Adam optimizer
(a slight variant on SGD that we will describe 
in greater detail later).
The main appeal of Adam vs vanilla SGD 
is that the Adam optimizer, 
despite doing no better (and sometimes worse)
given unlimited resources for hyperparameter optimization,
people tend to find that it is significantly less sensitive
to the initial learning rate.
This will be covered in further detail later on
when we discuss the details in :numref:`chap_optimization`.

```{.python .input  n=15}
def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # The Adam optimization algorithm is used here
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

## k-Fold Cross-Validation

If you are reading in a linear fashion,
you might recall that we introduced k-fold cross-validation
in the section where we discussed how to deal
with model section (:numref:`sec_model_selection`).
We will put this to good use to select the model design
and to adjust the hyperparameters.
We first need a function that returns
the $i^\mathrm{th}$ fold of the data
in a k-fold cross-validation procedure.
It proceeds by slicing out the $i^\mathrm{th}$ segment
as validation data and returning the rest as training data.
Note that this is not the most efficient way of handling data
and we would definitely do something much smarter
if our dataset was considerably larger.
But this added complexity might obfuscate our code unnecessarily
so we can safely omit here owing to the simplicity of our problem.

```{.python .input  n=16}
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
            X_train = np.concatenate((X_train, X_part), axis=0)
            y_train = np.concatenate((y_train, y_part), axis=0)
    return X_train, y_train, X_valid, y_valid
```

The training and verification error averages are returned
when we train $k$ times in the k-fold cross-validation.

```{.python .input  n=17}
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
            d2l.plot(list(range(1, num_epochs+1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse',
                     legend=['train', 'valid'], yscale='log')
        print('fold %d, train rmse: %f, valid rmse: %f' % (
            i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k
```

## Model Selection

In this example, we pick an untuned set of hyperparameters
and leave it up to the reader to improve the model.
Finding a good choice can take time,
depending on how many variables one optimizes over.
With a large enough dataset, 
and the normal sorts of hyperparameters reason,
k-fold cross-validation tends to be 
reasonably resilient against multiple testing.
However, if we try an unreasonably large number of options
we might just get lucky and find that our validation
performance is no longer representative of the true error.

```{.python .input  n=18}
k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print('%d-fold validation: avg train rmse: %f, avg valid rmse: %f'
      % (k, train_l, valid_l))
```

```{.json .output n=18}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "fold 0, train rmse: 0.169841, valid rmse: 0.157357\nfold 1, train rmse: 0.162251, valid rmse: 0.189077\nfold 2, train rmse: 0.163547, valid rmse: 0.167988\nfold 3, train rmse: 0.167754, valid rmse: 0.154793\nfold 4, train rmse: 0.162302, valid rmse: 0.182880\n5-fold validation: avg train rmse: 0.165139, avg valid rmse: 0.170419\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 248.644602 180.65625\" width=\"248.644602pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 248.644602 180.65625 \nL 248.644602 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 45.478125 143.1 \nL 240.778125 143.1 \nL 240.778125 7.2 \nL 45.478125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 52.562009 143.1 \nL 52.562009 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"mc158da89e2\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"52.562009\" xlink:href=\"#mc158da89e2\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(49.380759 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 88.429778 143.1 \nL 88.429778 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"88.429778\" xlink:href=\"#mc158da89e2\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 20 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(82.067278 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 124.297546 143.1 \nL 124.297546 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"124.297546\" xlink:href=\"#mc158da89e2\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 40 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(117.935046 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 160.165315 143.1 \nL 160.165315 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"160.165315\" xlink:href=\"#mc158da89e2\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 60 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(153.802815 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 196.033084 143.1 \nL 196.033084 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"196.033084\" xlink:href=\"#mc158da89e2\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 80 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(189.670584 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 231.900852 143.1 \nL 231.900852 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"231.900852\" xlink:href=\"#mc158da89e2\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 100 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n      </defs>\n      <g transform=\"translate(222.357102 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_7\">\n     <!-- epoch -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n     </defs>\n     <g transform=\"translate(127.9 171.376563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-104\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#p52291daf63)\" d=\"M 45.478125 64.904278 \nL 240.778125 64.904278 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"mbe80740292\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"45.478125\" xlink:href=\"#mbe80740292\" y=\"64.904278\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- $\\mathdefault{10^{0}}$ -->\n      <g transform=\"translate(20.878125 68.703497)scale(0.1 -0.1)\">\n       <use transform=\"translate(0 0.765625)\" xlink:href=\"#DejaVuSans-49\"/>\n       <use transform=\"translate(63.623047 0.765625)\" xlink:href=\"#DejaVuSans-48\"/>\n       <use transform=\"translate(128.203125 39.046875)scale(0.7)\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_15\">\n      <defs>\n       <path d=\"M 0 0 \nL -2 0 \n\" id=\"mc112932943\" style=\"stroke:#000000;stroke-width:0.6;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"127.582374\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"111.791904\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"100.588387\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"91.898264\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_19\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"84.797917\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_7\">\n     <g id=\"line2d_20\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"78.794659\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_8\">\n     <g id=\"line2d_21\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"73.594401\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_9\">\n     <g id=\"line2d_22\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"69.007447\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_10\">\n     <g id=\"line2d_23\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"37.910291\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_11\">\n     <g id=\"line2d_24\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"22.119822\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_12\">\n     <g id=\"line2d_25\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mc112932943\" y=\"10.916305\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_9\">\n     <!-- rmse -->\n     <defs>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-109\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n     </defs>\n     <g transform=\"translate(14.798437 87.669531)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"39.363281\" xlink:href=\"#DejaVuSans-109\"/>\n      <use x=\"136.775391\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"188.875\" xlink:href=\"#DejaVuSans-101\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_26\">\n    <path clip-path=\"url(#p52291daf63)\" d=\"M 54.355398 13.508307 \nL 56.148786 21.306906 \nL 57.942175 26.720993 \nL 59.735563 31.045763 \nL 61.528951 34.751498 \nL 63.32234 38.032606 \nL 65.115728 41.021857 \nL 66.909117 43.78522 \nL 68.702505 46.371877 \nL 70.495894 48.826672 \nL 72.289282 51.171208 \nL 74.08267 53.414034 \nL 75.876059 55.586004 \nL 77.669447 57.690184 \nL 79.462836 59.723521 \nL 81.256224 61.716428 \nL 83.049613 63.657679 \nL 84.843001 65.566032 \nL 86.636389 67.44487 \nL 88.429778 69.286238 \nL 90.223166 71.105468 \nL 92.016555 72.892721 \nL 93.809943 74.644452 \nL 95.603332 76.392386 \nL 97.39672 78.130402 \nL 99.190108 79.826825 \nL 100.983497 81.518725 \nL 102.776885 83.192041 \nL 104.570274 84.860097 \nL 106.363662 86.493228 \nL 108.157051 88.127028 \nL 109.950439 89.752122 \nL 111.743827 91.357637 \nL 113.537216 92.946578 \nL 115.330604 94.529503 \nL 117.123993 96.075871 \nL 118.917381 97.597839 \nL 120.71077 99.126764 \nL 122.504158 100.646894 \nL 124.297546 102.130135 \nL 126.090935 103.581377 \nL 127.884323 105.029267 \nL 129.677712 106.459118 \nL 131.4711 107.841199 \nL 133.264489 109.219034 \nL 135.057877 110.54915 \nL 136.851265 111.858765 \nL 138.644654 113.11267 \nL 140.438042 114.364618 \nL 142.231431 115.576808 \nL 144.024819 116.725613 \nL 145.818208 117.878591 \nL 147.611596 118.965169 \nL 149.404985 119.991592 \nL 151.198373 120.975446 \nL 152.991761 121.963362 \nL 154.78515 122.868139 \nL 156.578538 123.715846 \nL 158.371927 124.559806 \nL 160.165315 125.332631 \nL 161.958704 126.058325 \nL 163.752092 126.721811 \nL 165.54548 127.381336 \nL 167.338869 127.986509 \nL 169.132257 128.541437 \nL 170.925646 129.05066 \nL 172.719034 129.544931 \nL 174.512423 129.970233 \nL 176.305811 130.38402 \nL 178.099199 130.746141 \nL 179.892588 131.085449 \nL 181.685976 131.397875 \nL 183.479365 131.676922 \nL 185.272753 131.93486 \nL 187.066142 132.16397 \nL 188.85953 132.376109 \nL 190.652918 132.564767 \nL 192.446307 132.740462 \nL 194.239695 132.888722 \nL 196.033084 133.026766 \nL 197.826472 133.143242 \nL 199.619861 133.257677 \nL 201.413249 133.350542 \nL 203.206637 133.430099 \nL 205.000026 133.506941 \nL 206.793414 133.566719 \nL 208.586803 133.625023 \nL 210.380191 133.675855 \nL 212.17358 133.715926 \nL 213.966968 133.747057 \nL 215.760356 133.789918 \nL 217.553745 133.806569 \nL 219.347133 133.834907 \nL 221.140522 133.845903 \nL 222.93391 133.867664 \nL 224.727299 133.871511 \nL 226.520687 133.901524 \nL 228.314075 133.914898 \nL 230.107464 133.923679 \nL 231.900852 133.94802 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"line2d_27\">\n    <path clip-path=\"url(#p52291daf63)\" d=\"M 54.355398 13.377273 \nL 56.148786 21.133027 \nL 57.942175 26.519222 \nL 59.735563 30.824089 \nL 61.528951 34.509087 \nL 63.32234 37.770381 \nL 65.115728 40.739189 \nL 66.909117 43.489231 \nL 68.702505 46.057784 \nL 70.495894 48.501912 \nL 72.289282 50.829458 \nL 74.08267 53.059379 \nL 75.876059 55.216488 \nL 77.669447 57.305044 \nL 79.462836 59.327347 \nL 81.256224 61.305104 \nL 83.049613 63.23287 \nL 84.843001 65.128899 \nL 86.636389 66.994942 \nL 88.429778 68.823196 \nL 90.223166 70.634511 \nL 92.016555 72.409884 \nL 93.809943 74.154225 \nL 95.603332 75.893656 \nL 97.39672 77.622931 \nL 99.190108 79.314149 \nL 100.983497 80.997622 \nL 102.776885 82.669382 \nL 104.570274 84.333903 \nL 106.363662 85.966625 \nL 108.157051 87.599537 \nL 109.950439 89.228186 \nL 111.743827 90.839187 \nL 113.537216 92.435191 \nL 115.330604 94.028409 \nL 117.123993 95.589702 \nL 118.917381 97.128923 \nL 120.71077 98.677554 \nL 122.504158 100.220701 \nL 124.297546 101.732829 \nL 126.090935 103.215766 \nL 127.884323 104.702234 \nL 129.677712 106.17317 \nL 131.4711 107.601193 \nL 133.264489 109.032176 \nL 135.057877 110.419927 \nL 136.851265 111.793113 \nL 138.644654 113.114375 \nL 140.438042 114.443402 \nL 142.231431 115.73496 \nL 144.024819 116.969727 \nL 145.818208 118.216166 \nL 147.611596 119.395434 \nL 149.404985 120.525665 \nL 151.198373 121.610647 \nL 152.991761 122.708637 \nL 154.78515 123.725779 \nL 156.578538 124.686225 \nL 158.371927 125.650396 \nL 160.165315 126.540507 \nL 161.958704 127.384234 \nL 163.752092 128.161135 \nL 165.54548 128.941076 \nL 167.338869 129.65541 \nL 169.132257 130.323431 \nL 170.925646 130.939134 \nL 172.719034 131.541382 \nL 174.512423 132.062568 \nL 176.305811 132.572475 \nL 178.099199 133.017727 \nL 179.892588 133.442677 \nL 181.685976 133.834307 \nL 183.479365 134.188229 \nL 185.272753 134.514983 \nL 187.066142 134.802419 \nL 188.85953 135.073333 \nL 190.652918 135.311196 \nL 192.446307 135.543836 \nL 194.239695 135.727405 \nL 196.033084 135.904635 \nL 197.826472 136.050931 \nL 199.619861 136.192992 \nL 201.413249 136.30732 \nL 203.206637 136.400503 \nL 205.000026 136.49941 \nL 206.793414 136.571948 \nL 208.586803 136.644368 \nL 210.380191 136.696752 \nL 212.17358 136.750616 \nL 213.966968 136.78394 \nL 215.760356 136.830452 \nL 217.553745 136.846037 \nL 219.347133 136.870013 \nL 221.140522 136.876982 \nL 222.93391 136.891974 \nL 224.727299 136.893072 \nL 226.520687 136.909818 \nL 228.314075 136.922727 \nL 230.107464 136.913803 \nL 231.900852 136.921296 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 45.478125 143.1 \nL 45.478125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 240.778125 143.1 \nL 240.778125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 45.478125 143.1 \nL 240.778125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 45.478125 7.2 \nL 240.778125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"legend_1\">\n    <g id=\"patch_7\">\n     <path d=\"M 177.826562 44.55625 \nL 233.778125 44.55625 \nQ 235.778125 44.55625 235.778125 42.55625 \nL 235.778125 14.2 \nQ 235.778125 12.2 233.778125 12.2 \nL 177.826562 12.2 \nQ 175.826562 12.2 175.826562 14.2 \nL 175.826562 42.55625 \nQ 175.826562 44.55625 177.826562 44.55625 \nz\n\" style=\"fill:#ffffff;opacity:0.8;stroke:#cccccc;stroke-linejoin:miter;\"/>\n    </g>\n    <g id=\"line2d_28\">\n     <path d=\"M 179.826562 20.298437 \nL 199.826562 20.298437 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_29\"/>\n    <g id=\"text_10\">\n     <!-- train -->\n     <defs>\n      <path d=\"M 18.3125 70.21875 \nL 18.3125 54.6875 \nL 36.8125 54.6875 \nL 36.8125 47.703125 \nL 18.3125 47.703125 \nL 18.3125 18.015625 \nQ 18.3125 11.328125 20.140625 9.421875 \nQ 21.96875 7.515625 27.59375 7.515625 \nL 36.8125 7.515625 \nL 36.8125 0 \nL 27.59375 0 \nQ 17.1875 0 13.234375 3.875 \nQ 9.28125 7.765625 9.28125 18.015625 \nL 9.28125 47.703125 \nL 2.6875 47.703125 \nL 2.6875 54.6875 \nL 9.28125 54.6875 \nL 9.28125 70.21875 \nz\n\" id=\"DejaVuSans-116\"/>\n      <path d=\"M 34.28125 27.484375 \nQ 23.390625 27.484375 19.1875 25 \nQ 14.984375 22.515625 14.984375 16.5 \nQ 14.984375 11.71875 18.140625 8.90625 \nQ 21.296875 6.109375 26.703125 6.109375 \nQ 34.1875 6.109375 38.703125 11.40625 \nQ 43.21875 16.703125 43.21875 25.484375 \nL 43.21875 27.484375 \nz\nM 52.203125 31.203125 \nL 52.203125 0 \nL 43.21875 0 \nL 43.21875 8.296875 \nQ 40.140625 3.328125 35.546875 0.953125 \nQ 30.953125 -1.421875 24.3125 -1.421875 \nQ 15.921875 -1.421875 10.953125 3.296875 \nQ 6 8.015625 6 15.921875 \nQ 6 25.140625 12.171875 29.828125 \nQ 18.359375 34.515625 30.609375 34.515625 \nL 43.21875 34.515625 \nL 43.21875 35.40625 \nQ 43.21875 41.609375 39.140625 45 \nQ 35.0625 48.390625 27.6875 48.390625 \nQ 23 48.390625 18.546875 47.265625 \nQ 14.109375 46.140625 10.015625 43.890625 \nL 10.015625 52.203125 \nQ 14.9375 54.109375 19.578125 55.046875 \nQ 24.21875 56 28.609375 56 \nQ 40.484375 56 46.34375 49.84375 \nQ 52.203125 43.703125 52.203125 31.203125 \nz\n\" id=\"DejaVuSans-97\"/>\n      <path d=\"M 9.421875 54.6875 \nL 18.40625 54.6875 \nL 18.40625 0 \nL 9.421875 0 \nz\nM 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 64.59375 \nL 9.421875 64.59375 \nz\n\" id=\"DejaVuSans-105\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-110\"/>\n     </defs>\n     <g transform=\"translate(207.826562 23.798437)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-116\"/>\n      <use x=\"39.208984\" xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"80.322266\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"141.601562\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"169.384766\" xlink:href=\"#DejaVuSans-110\"/>\n     </g>\n    </g>\n    <g id=\"line2d_30\">\n     <path d=\"M 179.826562 34.976562 \nL 199.826562 34.976562 \n\" style=\"fill:none;stroke:#bf00bf;stroke-dasharray:5.55,2.4;stroke-dashoffset:0;stroke-width:1.5;\"/>\n    </g>\n    <g id=\"line2d_31\"/>\n    <g id=\"text_11\">\n     <!-- valid -->\n     <defs>\n      <path d=\"M 2.984375 54.6875 \nL 12.5 54.6875 \nL 29.59375 8.796875 \nL 46.6875 54.6875 \nL 56.203125 54.6875 \nL 35.6875 0 \nL 23.484375 0 \nz\n\" id=\"DejaVuSans-118\"/>\n      <path d=\"M 9.421875 75.984375 \nL 18.40625 75.984375 \nL 18.40625 0 \nL 9.421875 0 \nz\n\" id=\"DejaVuSans-108\"/>\n      <path d=\"M 45.40625 46.390625 \nL 45.40625 75.984375 \nL 54.390625 75.984375 \nL 54.390625 0 \nL 45.40625 0 \nL 45.40625 8.203125 \nQ 42.578125 3.328125 38.25 0.953125 \nQ 33.9375 -1.421875 27.875 -1.421875 \nQ 17.96875 -1.421875 11.734375 6.484375 \nQ 5.515625 14.40625 5.515625 27.296875 \nQ 5.515625 40.1875 11.734375 48.09375 \nQ 17.96875 56 27.875 56 \nQ 33.9375 56 38.25 53.625 \nQ 42.578125 51.265625 45.40625 46.390625 \nz\nM 14.796875 27.296875 \nQ 14.796875 17.390625 18.875 11.75 \nQ 22.953125 6.109375 30.078125 6.109375 \nQ 37.203125 6.109375 41.296875 11.75 \nQ 45.40625 17.390625 45.40625 27.296875 \nQ 45.40625 37.203125 41.296875 42.84375 \nQ 37.203125 48.484375 30.078125 48.484375 \nQ 22.953125 48.484375 18.875 42.84375 \nQ 14.796875 37.203125 14.796875 27.296875 \nz\n\" id=\"DejaVuSans-100\"/>\n     </defs>\n     <g transform=\"translate(207.826562 38.476562)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-118\"/>\n      <use x=\"59.179688\" xlink:href=\"#DejaVuSans-97\"/>\n      <use x=\"120.458984\" xlink:href=\"#DejaVuSans-108\"/>\n      <use x=\"148.242188\" xlink:href=\"#DejaVuSans-105\"/>\n      <use x=\"176.025391\" xlink:href=\"#DejaVuSans-100\"/>\n     </g>\n    </g>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p52291daf63\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"45.478125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

Notice that someimes the number of training errors
for a set of hyperparameters can be very low,
even as the number of errors on $k$-fold cross-validation 
is considerably higher. 
This indicates that we are overfitting.
Throughout training you will want to monitor both numbers.
No overfitting might indicate that our data can support a more powerful model. 
Massive overfitting might suggest that we can gain 
by incorporating regularization techniques.

##  Predict and Submit

Now that we know what a good choice of hyperparameters should be,
we might as well use all the data to train on it
(rather than just $1-1/k$ of the data
that is used in the cross-validation slices).
The model that we obtain in this way
can then be applied to the test set.
Saving the estimates in a CSV file
will simplify uploading the results to Kaggle.

```{.python .input  n=19}
def train_and_pred(train_features, test_feature, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='rmse', yscale='log')
    print('train rmse %f' % train_ls[-1])
    # Apply the network to the test set
    preds = net(test_features).asnumpy()
    # Reformat it for export to Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)
```

One nice sanity check is to see
whether the predictions on the test set
resemble those of the k-fold cross-validation process.
If they do, it is time to upload them to Kaggle.
The following code will generate a file called `submission.csv`
(CSV is one of the file formats accepted by Kaggle):

```{.python .input  n=20}
train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
```

```{.json .output n=20}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "train rmse 0.162543\n"
 },
 {
  "data": {
   "image/svg+xml": "<?xml version=\"1.0\" encoding=\"utf-8\" standalone=\"no\"?>\n<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n  \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n<!-- Created with matplotlib (https://matplotlib.org/) -->\n<svg height=\"180.65625pt\" version=\"1.1\" viewBox=\"0 0 248.644602 180.65625\" width=\"248.644602pt\" xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\">\n <defs>\n  <style type=\"text/css\">\n*{stroke-linecap:butt;stroke-linejoin:round;}\n  </style>\n </defs>\n <g id=\"figure_1\">\n  <g id=\"patch_1\">\n   <path d=\"M 0 180.65625 \nL 248.644602 180.65625 \nL 248.644602 0 \nL 0 0 \nz\n\" style=\"fill:none;\"/>\n  </g>\n  <g id=\"axes_1\">\n   <g id=\"patch_2\">\n    <path d=\"M 45.478125 143.1 \nL 240.778125 143.1 \nL 240.778125 7.2 \nL 45.478125 7.2 \nz\n\" style=\"fill:#ffffff;\"/>\n   </g>\n   <g id=\"matplotlib.axis_1\">\n    <g id=\"xtick_1\">\n     <g id=\"line2d_1\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 52.562009 143.1 \nL 52.562009 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_2\">\n      <defs>\n       <path d=\"M 0 0 \nL 0 3.5 \n\" id=\"m8bddee5c1b\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"52.562009\" xlink:href=\"#m8bddee5c1b\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_1\">\n      <!-- 0 -->\n      <defs>\n       <path d=\"M 31.78125 66.40625 \nQ 24.171875 66.40625 20.328125 58.90625 \nQ 16.5 51.421875 16.5 36.375 \nQ 16.5 21.390625 20.328125 13.890625 \nQ 24.171875 6.390625 31.78125 6.390625 \nQ 39.453125 6.390625 43.28125 13.890625 \nQ 47.125 21.390625 47.125 36.375 \nQ 47.125 51.421875 43.28125 58.90625 \nQ 39.453125 66.40625 31.78125 66.40625 \nz\nM 31.78125 74.21875 \nQ 44.046875 74.21875 50.515625 64.515625 \nQ 56.984375 54.828125 56.984375 36.375 \nQ 56.984375 17.96875 50.515625 8.265625 \nQ 44.046875 -1.421875 31.78125 -1.421875 \nQ 19.53125 -1.421875 13.0625 8.265625 \nQ 6.59375 17.96875 6.59375 36.375 \nQ 6.59375 54.828125 13.0625 64.515625 \nQ 19.53125 74.21875 31.78125 74.21875 \nz\n\" id=\"DejaVuSans-48\"/>\n      </defs>\n      <g transform=\"translate(49.380759 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_2\">\n     <g id=\"line2d_3\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 88.429778 143.1 \nL 88.429778 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_4\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"88.429778\" xlink:href=\"#m8bddee5c1b\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_2\">\n      <!-- 20 -->\n      <defs>\n       <path d=\"M 19.1875 8.296875 \nL 53.609375 8.296875 \nL 53.609375 0 \nL 7.328125 0 \nL 7.328125 8.296875 \nQ 12.9375 14.109375 22.625 23.890625 \nQ 32.328125 33.6875 34.8125 36.53125 \nQ 39.546875 41.84375 41.421875 45.53125 \nQ 43.3125 49.21875 43.3125 52.78125 \nQ 43.3125 58.59375 39.234375 62.25 \nQ 35.15625 65.921875 28.609375 65.921875 \nQ 23.96875 65.921875 18.8125 64.3125 \nQ 13.671875 62.703125 7.8125 59.421875 \nL 7.8125 69.390625 \nQ 13.765625 71.78125 18.9375 73 \nQ 24.125 74.21875 28.421875 74.21875 \nQ 39.75 74.21875 46.484375 68.546875 \nQ 53.21875 62.890625 53.21875 53.421875 \nQ 53.21875 48.921875 51.53125 44.890625 \nQ 49.859375 40.875 45.40625 35.40625 \nQ 44.1875 33.984375 37.640625 27.21875 \nQ 31.109375 20.453125 19.1875 8.296875 \nz\n\" id=\"DejaVuSans-50\"/>\n      </defs>\n      <g transform=\"translate(82.067278 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-50\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_3\">\n     <g id=\"line2d_5\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 124.297546 143.1 \nL 124.297546 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_6\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"124.297546\" xlink:href=\"#m8bddee5c1b\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_3\">\n      <!-- 40 -->\n      <defs>\n       <path d=\"M 37.796875 64.3125 \nL 12.890625 25.390625 \nL 37.796875 25.390625 \nz\nM 35.203125 72.90625 \nL 47.609375 72.90625 \nL 47.609375 25.390625 \nL 58.015625 25.390625 \nL 58.015625 17.1875 \nL 47.609375 17.1875 \nL 47.609375 0 \nL 37.796875 0 \nL 37.796875 17.1875 \nL 4.890625 17.1875 \nL 4.890625 26.703125 \nz\n\" id=\"DejaVuSans-52\"/>\n      </defs>\n      <g transform=\"translate(117.935046 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-52\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_4\">\n     <g id=\"line2d_7\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 160.165315 143.1 \nL 160.165315 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_8\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"160.165315\" xlink:href=\"#m8bddee5c1b\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_4\">\n      <!-- 60 -->\n      <defs>\n       <path d=\"M 33.015625 40.375 \nQ 26.375 40.375 22.484375 35.828125 \nQ 18.609375 31.296875 18.609375 23.390625 \nQ 18.609375 15.53125 22.484375 10.953125 \nQ 26.375 6.390625 33.015625 6.390625 \nQ 39.65625 6.390625 43.53125 10.953125 \nQ 47.40625 15.53125 47.40625 23.390625 \nQ 47.40625 31.296875 43.53125 35.828125 \nQ 39.65625 40.375 33.015625 40.375 \nz\nM 52.59375 71.296875 \nL 52.59375 62.3125 \nQ 48.875 64.0625 45.09375 64.984375 \nQ 41.3125 65.921875 37.59375 65.921875 \nQ 27.828125 65.921875 22.671875 59.328125 \nQ 17.53125 52.734375 16.796875 39.40625 \nQ 19.671875 43.65625 24.015625 45.921875 \nQ 28.375 48.1875 33.59375 48.1875 \nQ 44.578125 48.1875 50.953125 41.515625 \nQ 57.328125 34.859375 57.328125 23.390625 \nQ 57.328125 12.15625 50.6875 5.359375 \nQ 44.046875 -1.421875 33.015625 -1.421875 \nQ 20.359375 -1.421875 13.671875 8.265625 \nQ 6.984375 17.96875 6.984375 36.375 \nQ 6.984375 53.65625 15.1875 63.9375 \nQ 23.390625 74.21875 37.203125 74.21875 \nQ 40.921875 74.21875 44.703125 73.484375 \nQ 48.484375 72.75 52.59375 71.296875 \nz\n\" id=\"DejaVuSans-54\"/>\n      </defs>\n      <g transform=\"translate(153.802815 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-54\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_5\">\n     <g id=\"line2d_9\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 196.033084 143.1 \nL 196.033084 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_10\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"196.033084\" xlink:href=\"#m8bddee5c1b\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_5\">\n      <!-- 80 -->\n      <defs>\n       <path d=\"M 31.78125 34.625 \nQ 24.75 34.625 20.71875 30.859375 \nQ 16.703125 27.09375 16.703125 20.515625 \nQ 16.703125 13.921875 20.71875 10.15625 \nQ 24.75 6.390625 31.78125 6.390625 \nQ 38.8125 6.390625 42.859375 10.171875 \nQ 46.921875 13.96875 46.921875 20.515625 \nQ 46.921875 27.09375 42.890625 30.859375 \nQ 38.875 34.625 31.78125 34.625 \nz\nM 21.921875 38.8125 \nQ 15.578125 40.375 12.03125 44.71875 \nQ 8.5 49.078125 8.5 55.328125 \nQ 8.5 64.0625 14.71875 69.140625 \nQ 20.953125 74.21875 31.78125 74.21875 \nQ 42.671875 74.21875 48.875 69.140625 \nQ 55.078125 64.0625 55.078125 55.328125 \nQ 55.078125 49.078125 51.53125 44.71875 \nQ 48 40.375 41.703125 38.8125 \nQ 48.828125 37.15625 52.796875 32.3125 \nQ 56.78125 27.484375 56.78125 20.515625 \nQ 56.78125 9.90625 50.3125 4.234375 \nQ 43.84375 -1.421875 31.78125 -1.421875 \nQ 19.734375 -1.421875 13.25 4.234375 \nQ 6.78125 9.90625 6.78125 20.515625 \nQ 6.78125 27.484375 10.78125 32.3125 \nQ 14.796875 37.15625 21.921875 38.8125 \nz\nM 18.3125 54.390625 \nQ 18.3125 48.734375 21.84375 45.5625 \nQ 25.390625 42.390625 31.78125 42.390625 \nQ 38.140625 42.390625 41.71875 45.5625 \nQ 45.3125 48.734375 45.3125 54.390625 \nQ 45.3125 60.0625 41.71875 63.234375 \nQ 38.140625 66.40625 31.78125 66.40625 \nQ 25.390625 66.40625 21.84375 63.234375 \nQ 18.3125 60.0625 18.3125 54.390625 \nz\n\" id=\"DejaVuSans-56\"/>\n      </defs>\n      <g transform=\"translate(189.670584 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-56\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"xtick_6\">\n     <g id=\"line2d_11\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 231.900852 143.1 \nL 231.900852 7.2 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_12\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"231.900852\" xlink:href=\"#m8bddee5c1b\" y=\"143.1\"/>\n      </g>\n     </g>\n     <g id=\"text_6\">\n      <!-- 100 -->\n      <defs>\n       <path d=\"M 12.40625 8.296875 \nL 28.515625 8.296875 \nL 28.515625 63.921875 \nL 10.984375 60.40625 \nL 10.984375 69.390625 \nL 28.421875 72.90625 \nL 38.28125 72.90625 \nL 38.28125 8.296875 \nL 54.390625 8.296875 \nL 54.390625 0 \nL 12.40625 0 \nz\n\" id=\"DejaVuSans-49\"/>\n      </defs>\n      <g transform=\"translate(222.357102 157.698438)scale(0.1 -0.1)\">\n       <use xlink:href=\"#DejaVuSans-49\"/>\n       <use x=\"63.623047\" xlink:href=\"#DejaVuSans-48\"/>\n       <use x=\"127.246094\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_7\">\n     <!-- epoch -->\n     <defs>\n      <path d=\"M 56.203125 29.59375 \nL 56.203125 25.203125 \nL 14.890625 25.203125 \nQ 15.484375 15.921875 20.484375 11.0625 \nQ 25.484375 6.203125 34.421875 6.203125 \nQ 39.59375 6.203125 44.453125 7.46875 \nQ 49.3125 8.734375 54.109375 11.28125 \nL 54.109375 2.78125 \nQ 49.265625 0.734375 44.1875 -0.34375 \nQ 39.109375 -1.421875 33.890625 -1.421875 \nQ 20.796875 -1.421875 13.15625 6.1875 \nQ 5.515625 13.8125 5.515625 26.8125 \nQ 5.515625 40.234375 12.765625 48.109375 \nQ 20.015625 56 32.328125 56 \nQ 43.359375 56 49.78125 48.890625 \nQ 56.203125 41.796875 56.203125 29.59375 \nz\nM 47.21875 32.234375 \nQ 47.125 39.59375 43.09375 43.984375 \nQ 39.0625 48.390625 32.421875 48.390625 \nQ 24.90625 48.390625 20.390625 44.140625 \nQ 15.875 39.890625 15.1875 32.171875 \nz\n\" id=\"DejaVuSans-101\"/>\n      <path d=\"M 18.109375 8.203125 \nL 18.109375 -20.796875 \nL 9.078125 -20.796875 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.390625 \nQ 20.953125 51.265625 25.265625 53.625 \nQ 29.59375 56 35.59375 56 \nQ 45.5625 56 51.78125 48.09375 \nQ 58.015625 40.1875 58.015625 27.296875 \nQ 58.015625 14.40625 51.78125 6.484375 \nQ 45.5625 -1.421875 35.59375 -1.421875 \nQ 29.59375 -1.421875 25.265625 0.953125 \nQ 20.953125 3.328125 18.109375 8.203125 \nz\nM 48.6875 27.296875 \nQ 48.6875 37.203125 44.609375 42.84375 \nQ 40.53125 48.484375 33.40625 48.484375 \nQ 26.265625 48.484375 22.1875 42.84375 \nQ 18.109375 37.203125 18.109375 27.296875 \nQ 18.109375 17.390625 22.1875 11.75 \nQ 26.265625 6.109375 33.40625 6.109375 \nQ 40.53125 6.109375 44.609375 11.75 \nQ 48.6875 17.390625 48.6875 27.296875 \nz\n\" id=\"DejaVuSans-112\"/>\n      <path d=\"M 30.609375 48.390625 \nQ 23.390625 48.390625 19.1875 42.75 \nQ 14.984375 37.109375 14.984375 27.296875 \nQ 14.984375 17.484375 19.15625 11.84375 \nQ 23.34375 6.203125 30.609375 6.203125 \nQ 37.796875 6.203125 41.984375 11.859375 \nQ 46.1875 17.53125 46.1875 27.296875 \nQ 46.1875 37.015625 41.984375 42.703125 \nQ 37.796875 48.390625 30.609375 48.390625 \nz\nM 30.609375 56 \nQ 42.328125 56 49.015625 48.375 \nQ 55.71875 40.765625 55.71875 27.296875 \nQ 55.71875 13.875 49.015625 6.21875 \nQ 42.328125 -1.421875 30.609375 -1.421875 \nQ 18.84375 -1.421875 12.171875 6.21875 \nQ 5.515625 13.875 5.515625 27.296875 \nQ 5.515625 40.765625 12.171875 48.375 \nQ 18.84375 56 30.609375 56 \nz\n\" id=\"DejaVuSans-111\"/>\n      <path d=\"M 48.78125 52.59375 \nL 48.78125 44.1875 \nQ 44.96875 46.296875 41.140625 47.34375 \nQ 37.3125 48.390625 33.40625 48.390625 \nQ 24.65625 48.390625 19.8125 42.84375 \nQ 14.984375 37.3125 14.984375 27.296875 \nQ 14.984375 17.28125 19.8125 11.734375 \nQ 24.65625 6.203125 33.40625 6.203125 \nQ 37.3125 6.203125 41.140625 7.25 \nQ 44.96875 8.296875 48.78125 10.40625 \nL 48.78125 2.09375 \nQ 45.015625 0.34375 40.984375 -0.53125 \nQ 36.96875 -1.421875 32.421875 -1.421875 \nQ 20.0625 -1.421875 12.78125 6.34375 \nQ 5.515625 14.109375 5.515625 27.296875 \nQ 5.515625 40.671875 12.859375 48.328125 \nQ 20.21875 56 33.015625 56 \nQ 37.15625 56 41.109375 55.140625 \nQ 45.0625 54.296875 48.78125 52.59375 \nz\n\" id=\"DejaVuSans-99\"/>\n      <path d=\"M 54.890625 33.015625 \nL 54.890625 0 \nL 45.90625 0 \nL 45.90625 32.71875 \nQ 45.90625 40.484375 42.875 44.328125 \nQ 39.84375 48.1875 33.796875 48.1875 \nQ 26.515625 48.1875 22.3125 43.546875 \nQ 18.109375 38.921875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 75.984375 \nL 18.109375 75.984375 \nL 18.109375 46.1875 \nQ 21.34375 51.125 25.703125 53.5625 \nQ 30.078125 56 35.796875 56 \nQ 45.21875 56 50.046875 50.171875 \nQ 54.890625 44.34375 54.890625 33.015625 \nz\n\" id=\"DejaVuSans-104\"/>\n     </defs>\n     <g transform=\"translate(127.9 171.376563)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-101\"/>\n      <use x=\"61.523438\" xlink:href=\"#DejaVuSans-112\"/>\n      <use x=\"125\" xlink:href=\"#DejaVuSans-111\"/>\n      <use x=\"186.181641\" xlink:href=\"#DejaVuSans-99\"/>\n      <use x=\"241.162109\" xlink:href=\"#DejaVuSans-104\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"matplotlib.axis_2\">\n    <g id=\"ytick_1\">\n     <g id=\"line2d_13\">\n      <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 45.478125 64.161141 \nL 240.778125 64.161141 \n\" style=\"fill:none;stroke:#b0b0b0;stroke-linecap:square;stroke-width:0.8;\"/>\n     </g>\n     <g id=\"line2d_14\">\n      <defs>\n       <path d=\"M 0 0 \nL -3.5 0 \n\" id=\"m60a18640db\" style=\"stroke:#000000;stroke-width:0.8;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.8;\" x=\"45.478125\" xlink:href=\"#m60a18640db\" y=\"64.161141\"/>\n      </g>\n     </g>\n     <g id=\"text_8\">\n      <!-- $\\mathdefault{10^{0}}$ -->\n      <g transform=\"translate(20.878125 67.960359)scale(0.1 -0.1)\">\n       <use transform=\"translate(0 0.765625)\" xlink:href=\"#DejaVuSans-49\"/>\n       <use transform=\"translate(63.623047 0.765625)\" xlink:href=\"#DejaVuSans-48\"/>\n       <use transform=\"translate(128.203125 39.046875)scale(0.7)\" xlink:href=\"#DejaVuSans-48\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_2\">\n     <g id=\"line2d_15\">\n      <defs>\n       <path d=\"M 0 0 \nL -2 0 \n\" id=\"mb2cd030420\" style=\"stroke:#000000;stroke-width:0.6;\"/>\n      </defs>\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"128.617552\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_3\">\n     <g id=\"line2d_16\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"112.379071\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_4\">\n     <g id=\"line2d_17\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"100.857686\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_5\">\n     <g id=\"line2d_18\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"91.921006\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_6\">\n     <g id=\"line2d_19\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"84.619206\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_7\">\n     <g id=\"line2d_20\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"78.445623\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_8\">\n     <g id=\"line2d_21\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"73.097821\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_9\">\n     <g id=\"line2d_22\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"68.380726\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_10\">\n     <g id=\"line2d_23\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"36.401275\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_11\">\n     <g id=\"line2d_24\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"20.162795\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"ytick_12\">\n     <g id=\"line2d_25\">\n      <g>\n       <use style=\"stroke:#000000;stroke-width:0.6;\" x=\"45.478125\" xlink:href=\"#mb2cd030420\" y=\"8.64141\"/>\n      </g>\n     </g>\n    </g>\n    <g id=\"text_9\">\n     <!-- rmse -->\n     <defs>\n      <path d=\"M 41.109375 46.296875 \nQ 39.59375 47.171875 37.8125 47.578125 \nQ 36.03125 48 33.890625 48 \nQ 26.265625 48 22.1875 43.046875 \nQ 18.109375 38.09375 18.109375 28.8125 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 20.953125 51.171875 25.484375 53.578125 \nQ 30.03125 56 36.53125 56 \nQ 37.453125 56 38.578125 55.875 \nQ 39.703125 55.765625 41.0625 55.515625 \nz\n\" id=\"DejaVuSans-114\"/>\n      <path d=\"M 52 44.1875 \nQ 55.375 50.25 60.0625 53.125 \nQ 64.75 56 71.09375 56 \nQ 79.640625 56 84.28125 50.015625 \nQ 88.921875 44.046875 88.921875 33.015625 \nL 88.921875 0 \nL 79.890625 0 \nL 79.890625 32.71875 \nQ 79.890625 40.578125 77.09375 44.375 \nQ 74.3125 48.1875 68.609375 48.1875 \nQ 61.625 48.1875 57.5625 43.546875 \nQ 53.515625 38.921875 53.515625 30.90625 \nL 53.515625 0 \nL 44.484375 0 \nL 44.484375 32.71875 \nQ 44.484375 40.625 41.703125 44.40625 \nQ 38.921875 48.1875 33.109375 48.1875 \nQ 26.21875 48.1875 22.15625 43.53125 \nQ 18.109375 38.875 18.109375 30.90625 \nL 18.109375 0 \nL 9.078125 0 \nL 9.078125 54.6875 \nL 18.109375 54.6875 \nL 18.109375 46.1875 \nQ 21.1875 51.21875 25.484375 53.609375 \nQ 29.78125 56 35.6875 56 \nQ 41.65625 56 45.828125 52.96875 \nQ 50 49.953125 52 44.1875 \nz\n\" id=\"DejaVuSans-109\"/>\n      <path d=\"M 44.28125 53.078125 \nL 44.28125 44.578125 \nQ 40.484375 46.53125 36.375 47.5 \nQ 32.28125 48.484375 27.875 48.484375 \nQ 21.1875 48.484375 17.84375 46.4375 \nQ 14.5 44.390625 14.5 40.28125 \nQ 14.5 37.15625 16.890625 35.375 \nQ 19.28125 33.59375 26.515625 31.984375 \nL 29.59375 31.296875 \nQ 39.15625 29.25 43.1875 25.515625 \nQ 47.21875 21.78125 47.21875 15.09375 \nQ 47.21875 7.46875 41.1875 3.015625 \nQ 35.15625 -1.421875 24.609375 -1.421875 \nQ 20.21875 -1.421875 15.453125 -0.5625 \nQ 10.6875 0.296875 5.421875 2 \nL 5.421875 11.28125 \nQ 10.40625 8.6875 15.234375 7.390625 \nQ 20.0625 6.109375 24.8125 6.109375 \nQ 31.15625 6.109375 34.5625 8.28125 \nQ 37.984375 10.453125 37.984375 14.40625 \nQ 37.984375 18.0625 35.515625 20.015625 \nQ 33.0625 21.96875 24.703125 23.78125 \nL 21.578125 24.515625 \nQ 13.234375 26.265625 9.515625 29.90625 \nQ 5.8125 33.546875 5.8125 39.890625 \nQ 5.8125 47.609375 11.28125 51.796875 \nQ 16.75 56 26.8125 56 \nQ 31.78125 56 36.171875 55.265625 \nQ 40.578125 54.546875 44.28125 53.078125 \nz\n\" id=\"DejaVuSans-115\"/>\n     </defs>\n     <g transform=\"translate(14.798437 87.669531)rotate(-90)scale(0.1 -0.1)\">\n      <use xlink:href=\"#DejaVuSans-114\"/>\n      <use x=\"39.363281\" xlink:href=\"#DejaVuSans-109\"/>\n      <use x=\"136.775391\" xlink:href=\"#DejaVuSans-115\"/>\n      <use x=\"188.875\" xlink:href=\"#DejaVuSans-101\"/>\n     </g>\n    </g>\n   </g>\n   <g id=\"line2d_26\">\n    <path clip-path=\"url(#p37b1b10a6e)\" d=\"M 54.355398 13.377273 \nL 56.148786 21.97544 \nL 57.942175 27.976296 \nL 59.735563 32.798575 \nL 61.528951 36.952172 \nL 63.32234 40.657117 \nL 65.115728 44.047875 \nL 66.909117 47.181861 \nL 68.702505 50.148503 \nL 70.495894 52.964725 \nL 72.289282 55.67592 \nL 74.08267 58.278371 \nL 75.876059 60.794409 \nL 77.669447 63.261387 \nL 79.462836 65.64903 \nL 81.256224 67.99575 \nL 83.049613 70.307301 \nL 84.843001 72.566608 \nL 86.636389 74.807309 \nL 88.429778 76.992913 \nL 90.223166 79.158129 \nL 92.016555 81.301718 \nL 93.809943 83.439046 \nL 95.603332 85.533744 \nL 97.39672 87.605676 \nL 99.190108 89.652761 \nL 100.983497 91.679444 \nL 102.776885 93.715909 \nL 104.570274 95.677397 \nL 106.363662 97.644475 \nL 108.157051 99.558182 \nL 109.950439 101.471496 \nL 111.743827 103.373777 \nL 113.537216 105.224846 \nL 115.330604 107.045318 \nL 117.123993 108.808184 \nL 118.917381 110.52961 \nL 120.71077 112.220308 \nL 122.504158 113.86711 \nL 124.297546 115.450646 \nL 126.090935 116.989411 \nL 127.884323 118.454835 \nL 129.677712 119.880474 \nL 131.4711 121.246922 \nL 133.264489 122.514059 \nL 135.057877 123.732102 \nL 136.851265 124.904458 \nL 138.644654 125.968703 \nL 140.438042 126.976164 \nL 142.231431 127.897083 \nL 144.024819 128.780827 \nL 145.818208 129.560923 \nL 147.611596 130.282736 \nL 149.404985 130.977225 \nL 151.198373 131.576917 \nL 152.991761 132.124422 \nL 154.78515 132.623787 \nL 156.578538 133.066406 \nL 158.371927 133.462909 \nL 160.165315 133.820769 \nL 161.958704 134.13526 \nL 163.752092 134.410808 \nL 165.54548 134.659559 \nL 167.338869 134.88188 \nL 169.132257 135.072152 \nL 170.925646 135.232389 \nL 172.719034 135.383104 \nL 174.512423 135.506469 \nL 176.305811 135.612744 \nL 178.099199 135.709487 \nL 179.892588 135.784511 \nL 181.685976 135.853449 \nL 183.479365 135.91949 \nL 185.272753 135.951958 \nL 187.066142 136.003301 \nL 188.85953 136.043297 \nL 190.652918 136.075431 \nL 192.446307 136.108824 \nL 194.239695 136.131518 \nL 196.033084 136.158459 \nL 197.826472 136.162054 \nL 199.619861 136.18352 \nL 201.413249 136.223418 \nL 203.206637 136.230556 \nL 205.000026 136.275262 \nL 206.793414 136.299463 \nL 208.586803 136.309485 \nL 210.380191 136.349129 \nL 212.17358 136.378782 \nL 213.966968 136.401291 \nL 215.760356 136.43061 \nL 217.553745 136.502569 \nL 219.347133 136.531747 \nL 221.140522 136.585058 \nL 222.93391 136.64729 \nL 224.727299 136.672745 \nL 226.520687 136.755561 \nL 228.314075 136.801073 \nL 230.107464 136.852183 \nL 231.900852 136.922727 \n\" style=\"fill:none;stroke:#1f77b4;stroke-linecap:square;stroke-width:1.5;\"/>\n   </g>\n   <g id=\"patch_3\">\n    <path d=\"M 45.478125 143.1 \nL 45.478125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_4\">\n    <path d=\"M 240.778125 143.1 \nL 240.778125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_5\">\n    <path d=\"M 45.478125 143.1 \nL 240.778125 143.1 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n   <g id=\"patch_6\">\n    <path d=\"M 45.478125 7.2 \nL 240.778125 7.2 \n\" style=\"fill:none;stroke:#000000;stroke-linecap:square;stroke-linejoin:miter;stroke-width:0.8;\"/>\n   </g>\n  </g>\n </g>\n <defs>\n  <clipPath id=\"p37b1b10a6e\">\n   <rect height=\"135.9\" width=\"195.3\" x=\"45.478125\" y=\"7.2\"/>\n  </clipPath>\n </defs>\n</svg>\n",
   "text/plain": "<Figure size 252x180 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

Next, as demonstrated in :numref:`fig_kaggle_submit2`,
we can submit our predictions on Kaggle
and see how they compare to the actual house prices (labels) 
on the test set.
The steps are quite simple:

* Log in to the Kaggle website and visit the House Price Prediction Competition page.
* Click the “Submit Predictions” or “Late Submission” button (as of this writing, the button is located on the right).
* Click the “Upload Submission File” button in the dashed box at the bottom of the page and select the prediction file you wish to upload.
* Click the “Make Submission” button at the bottom of the page to view your results.

![Submitting data to Kaggle](../img/kaggle_submit2.png)
:width:`400px`
:label:`fig_kaggle_submit2`

## Summary

* Real data often contains a mix of different data types and needs to be preprocessed.
* Rescaling real-valued data to zero mean and unit variance is a good default. So is replacing missing values with their mean.
* Transforming categorical variables into indicator variables allows us to treat them like vectors.
* We can use k-fold cross validation to select the model and adjust the hyper-parameters.
* Logarithms are useful for relative loss.


## Exercises

1. Submit your predictions for this tutorial to Kaggle. How good are your predictions?
1. Can you improve your model by minimizing the log-price directly? What happens if you try to predict the log price rather than the price?
1. Is it always a good idea to replace missing values by their mean? Hint: can you construct a situation where the values are not missing at random?
1. Find a better representation to deal with missing values. Hint: what happens if you add an indicator variable?
1. Improve the score on Kaggle by tuning the hyperparameters through k-fold cross-validation.
1. Improve the score by improving the model (layers, regularization, dropout).
1. What happens if we do not standardize the continuous numerical features like we have done in this section?

## [Discussions](https://discuss.mxnet.io/t/2346)

![](../img/qr_kaggle-house-price.svg)
