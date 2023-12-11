```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Data Preprocessing
:label:`sec_pandas`

So far, we have been working with synthetic data
that arrived in ready-made tensors.
However, to apply deep learning in the wild
we must extract messy data 
stored in arbitrary formats,
and preprocess it to suit our needs.
Fortunately, the *pandas* [library](https://pandas.pydata.org/) 
can do much of the heavy lifting.
This section, while no substitute 
for a proper *pandas* [tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html),
will give you a crash course
on some of the most common routines.

## Reading the Dataset

Comma-separated values (CSV) files are ubiquitous 
for the storing of tabular (spreadsheet-like) data.
In them, each line corresponds to one record
and consists of several (comma-separated) fields, e.g.,
"Albert Einstein,March 14 1879,Ulm,Federal polytechnic school,field of gravitational physics".
To demonstrate how to load CSV files with `pandas`, 
we (**create a CSV file below**) `../data/house_tiny.csv`. 
This file represents a dataset of homes,
where each row corresponds to a distinct home
and the columns correspond to the number of rooms (`NumRooms`),
the roof type (`RoofType`), and the price (`Price`).

```{.python .input}
%%tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,RoofType,Price
NA,NA,127500
2,NA,106000
4,Slate,178100
NA,NA,140000''')
```

Now let's import `pandas` and load the dataset with `read_csv`.

```{.python .input}
%%tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Data Preparation

In supervised learning, we train models
to predict a designated *target* value,
given some set of *input* values. 
Our first step in processing the dataset
is to separate out columns corresponding
to input versus target values. 
We can select columns either by name or
via integer-location based indexing (`iloc`).

You might have noticed that `pandas` replaced
all CSV entries with value `NA`
with a special `NaN` (*not a number*) value. 
This can also happen whenever an entry is empty,
e.g., "3,,,270000".
These are called *missing values* 
and they are the "bed bugs" of data science,
a persistent menace that you will confront
throughout your career. 
Depending upon the context, 
missing values might be handled
either via *imputation* or *deletion*.
Imputation replaces missing values 
with estimates of their values
while deletion simply discards 
either those rows or those columns
that contain missing values. 

Here are some common imputation heuristics.
[**For categorical input fields, 
we can treat `NaN` as a category.**]
Since the `RoofType` column takes values `Slate` and `NaN`,
`pandas` can convert this column 
into two columns `RoofType_Slate` and `RoofType_nan`.
A row whose roof type is `Slate` will set values 
of `RoofType_Slate` and `RoofType_nan` to 1 and 0, respectively.
The converse holds for a row with a missing `RoofType` value.

```{.python .input}
%%tab all
inputs, targets = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

For missing numerical values, 
one common heuristic is to 
[**replace the `NaN` entries with 
the mean value of the corresponding column**].

```{.python .input}
%%tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Conversion to the Tensor Format

Now that [**all the entries in `inputs` and `targets` are numerical,
we can load them into a tensor**] (recall :numref:`sec_ndarray`).

```{.python .input}
%%tab mxnet
from mxnet import np

X, y = np.array(inputs.to_numpy(dtype=float)), np.array(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab pytorch
import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

X = tf.constant(inputs.to_numpy(dtype=float))
y = tf.constant(targets.to_numpy(dtype=float))
X, y
```

```{.python .input}
%%tab jax
from jax import numpy as jnp

X = jnp.array(inputs.to_numpy(dtype=float))
y = jnp.array(targets.to_numpy(dtype=float))
X, y
```

## Discussion

You now know how to partition data columns, 
impute missing variables, 
and load `pandas` data into tensors. 
In :numref:`sec_kaggle_house`, you will
pick up some more data processing skills. 
While this crash course kept things simple,
data processing can get hairy.
For example, rather than arriving in a single CSV file,
our dataset might be spread across multiple files
extracted from a relational database.
For instance, in an e-commerce application,
customer addresses might live in one table
and purchase data in another.
Moreover, practitioners face myriad data types
beyond categorical and numeric, for example,
text strings, images,
audio data, and point clouds. 
Oftentimes, advanced tools and efficient algorithms 
are required in order to prevent data processing from becoming
the biggest bottleneck in the machine learning pipeline. 
These problems will arise when we get to 
computer vision and natural language processing. 
Finally, we must pay attention to data quality.
Real-world datasets are often plagued 
by outliers, faulty measurements from sensors, and recording errors, 
which must be addressed before 
feeding the data into any model. 
Data visualization tools such as [seaborn](https://seaborn.pydata.org/), 
[Bokeh](https://docs.bokeh.org/), or [matplotlib](https://matplotlib.org/)
can help you to manually inspect the data 
and develop intuitions about 
the type of problems you may need to address.


## Exercises

1. Try loading datasets, e.g., Abalone from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets) and inspect their properties. What fraction of them has missing values? What fraction of the variables is numerical, categorical, or text?
1. Try indexing and selecting data columns by name rather than by column number. The pandas documentation on [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) has further details on how to do this.
1. How large a dataset do you think you could load this way? What might be the limitations? Hint: consider the time to read the data, representation, processing, and memory footprint. Try this out on your laptop. What happens if you try it out on a server? 
1. How would you deal with data that has a very large number of categories? What if the category labels are all unique? Should you include the latter?
1. What alternatives to pandas can you think of? How about [loading NumPy tensors from a file](https://numpy.org/doc/stable/reference/generated/numpy.load.html)? Check out [Pillow](https://python-pillow.org/), the Python Imaging Library. 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/28)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/29)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/195)
:end_tab:

:begin_tab:`jax`
[Discussions](https://discuss.d2l.ai/t/17967)
:end_tab:
