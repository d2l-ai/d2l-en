# Data Preprocessing
:label:`sec_pandas`

So far we have introduced a variety of techniques for manipulating data that are already stored in `ndarray`s.
To apply deep learning to solving real-world problems,
we often begin with preprocessing raw data, rather than those nicely prepared data in the `ndarray` format.
Among popular data analytic tools in Python, the `pandas` package is commonly used.
Like many other extension packages in the vast ecosystem of Python,
`pandas` can work together with `ndarray`.
So, we will briefly walk through steps for preprocessing raw data with `pandas`
and converting them into the `ndarray` format.
We will cover more data preprocessing techniques in later chapters.

## Loading Data

As an example, we begin by creating an artificial dataset that is stored in a csv (comma-separated values) file. Data stored in other formats may be processed in similar ways.

```{.python .input}
# Write the dataset row by row into a csv file
data_file = '../data/house_tiny.csv'
with open(data_file, 'w') as f:
    f.write('NumRooms, Alley, Price\n')  # Column names
    f.write('NA, Pave, 127500\n')  # Each row is a data point
    f.write('2, NA, 106000\n')
    f.write('4, NA, 178100\n')
    f.write('NA, NA, 140000\n')
```

To load the raw dataset from the created csv file,
we import the `pandas` package and invoke the `read_csv` function.
This dataset has $4$ rows and $3$ columns, where each row describes the number of rooms ("NumRooms"), the alley type ("Alley"), and the price ("Price") of a house.

```{.python .input}
# If pandas is not installed, just uncomment the following line:
# !pip install pandas
import pandas as pd

data = pd.read_csv(data_file)
print(data)
```

## Handling Missing Data

Note that "NaN" entries are missing values.
To handle missing data, typical methods include *imputation* and *deletion*,
where imputation replaces missing values with substituted ones,
while deletion ignores missing values. Here we will consider imputation.

By integer-location based indexing (`iloc`), we split `data` into `inputs` and `outputs`,
where the former takes the first 2 columns while the later only keeps the last column.
For numerical values in `inputs` that are missing, we replace the "NaN" entries with the mean value of the same column.

```{.python .input}
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)
```

For categorical or discrete values in `inputs`, we consider "NaN" as a category.
Since the "Alley" column only takes 2 types of categorical values "Pave" and "NaN",
`pandas` can automatically convert this column to 2 columns "Alley_Pave" and "Alley_nan".
A row whose alley type is "Pave" will set values of "Alley_Pave" and "Alley_nan" to $1$ and $0$.
A row with a missing alley type will set their values to $0$ and $1$.

```{.python .input}
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
```

## Conversion to the  `ndarray` Format

Now that all the entries in `inputs` and `outputs` are numerical, they can be converted to the `ndarray` format.
Once data are in this format, they can be further manipulated with those `ndarray` functionalities that we have introduced in :numref:`sec_ndarray`.

```{.python .input}
from mxnet import np

X, y = np.array(inputs.values), np.array(outputs.values)
X, y
```

## Summary

* Like many other extension packages in the vast ecosystem of Python, `pandas` can work together with `ndarray`.
* Imputation and deletion can be used to handle missing data.


## Exercises

Create a raw dataset with more rows and columns.

1. Delete the column with the most missing values. 
2. Convert the preprocessed dataset to the `ndarray` format.


## [Discussions](https://discuss.mxnet.io/t/4973)

![](../img/qr_pandas.svg)
