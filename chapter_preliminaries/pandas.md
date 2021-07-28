# Data Preprocessing
:label:`sec_pandas`

So far, we have been working with synthetic data
that arrived in ready-made tensors.
However, applying deep learning in the wild
typically requires processing messy
real-world data that may arrive
in a variety of formats.
Fortunately, the [*pandas* library](https://pandas.pydata.org/) 
can do much of the heavy lifting
for loading and preprocessing your data.
The following, while no substitute for a proper [*pandas* tutorial](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html), will show you 
how to read raw data in `pandas` and convert it into tensors. 


## Reading the Dataset

Let's begin by (**creating an artificial dataset that is stored in a
CSV (comma-separated values) file**)
`../data/house_tiny.csv`. Our dataset has four rows and three columns, where each row describes the number of rooms ("NumRooms"), the alley type ("Alley"), and the price ("Price") of a house. Data stored in other
formats may be processed similarly.

```{.python .input}
#@tab all
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('''NumRooms,Alley,Price
        NA,Pave,127500
        2,NA,106000
        4,NA,178100
        NA,NA,140000''')
```

To [**load the raw dataset from the created CSV file**],
we import `pandas` and use its `read_csv` function.

```{.python .input}
#@tab all
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Data Preparation

The first step is to separate inputs from outputs. We can accomplish this by selecting the relevant columns. We can do this by selecting column names or by using integer-location based indexing (`iloc`). Once this is done, we need to address the "NaN" entries, since they represent missing values. This can be handled, e.g., via *imputation* and *deletion*. Imputation replaces missing values with an estimate of their value, whereas deletion simply ignores them (or even the entire data column). Let's have a look at imputation. 

[**For categorical or discrete values in `inputs`, we consider "NaN" as a category.**]
Since the "Alley" column only takes two types of categorical values "Pave" and "NaN",
`pandas` can automatically convert this column into two columns "Alley_Pave" and "Alley_nan".
A row whose alley type is "Pave" will set values of "Alley_Pave" and "Alley_nan" to 1 and 0, respectively. The converse holds for a row with a missing alley type.

```{.python .input}
#@tab all
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

For missing numerical values 
we [**replace the "NaN" entries with the mean value of the same column**] to obtain a usable representation of the inputs.

```{.python .input}
#@tab all
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

## Conversion to the Tensor Format

Now that [**all the entries in `inputs` and `outputs` are numerical, they can be converted into tensors.**]
Once data is in this format, we can use the tools we introduced in :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

```{.python .input}
#@tab pytorch
import torch

X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
X, y
```

```{.python .input}
#@tab tensorflow
import tensorflow as tf

X, y = tf.constant(inputs.values), tf.constant(outputs.values)
X, y
```

## Summary

We barely scratched the surface of what is possible, namely how to partition data columns and how to deal with missing variables in their most basic form. Later on, we will pick up more data processing skills in :ref:`sec_kaggle_house`. The relevant part for now is that we can easily convert tensors from `pandas` into tensors managed by the framework of our choice. 

Going beyond basics, consider a situation where the data is not readily available in the form of a single CSV file but rather, it needs to be constituted from multiple individual tables.
For instance, users might have their address data stored in one table and their purchase data in another one. Much talent is required to generate representations that are effective for machine learning. Beyond table joins, we also need to deal with data types beyond categorical and numeric. For instance, some data might consist of strings, others of images, yet others of audio data, annotations, or point clouds. In all of these settings different tools and efficient algorithms are required to load the data in a way that data import itself does not become the bottleneck of the machine learning pipeline. We will encounter a number of such problems later in computer vision and natural language processing. 

A key aspect of good data analysis and processing is to address data quality. For instance, we might have outliers, faulty measurements from sensors, and recording errors. They need to be resolved before feeding the data into any model. Tools for visualization such as [seaborn](https://seaborn.pydata.org/), [Bokeh](https://docs.bokeh.org/), or [matplotlib](https://matplotlib.org/) can be very useful.


## Exercises

1. Try loading datasets, e.g., Abalone from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets.php) and inspect their properties. What fraction of them has missing values? What fraction of the variables is numerical, categorical, or text?
1. Try out indexing and selecting data columns by name rather than by column number. The pandas documentation on [indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html) has further details on how to do this.
1. How large a dataset do you think you could load this way? What might be the limitations? Hint: consider the time to read the data, representation, processing, and memory footprint. Try this out on your laptop. What changes if you try it out on a server? 
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
