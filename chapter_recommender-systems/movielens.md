#  The MovieLens Dataset

There are a number of data sets available for recommendation research. The [MovieLens](https://movielens.org/) data set is probably the most popular one. MovieLens is a non-commercial web-based movie recommender system, created in1997 and run by GroupLens, a research lab at the University of Minnesota, in order to gather movie rating data for research use.  MovieLens data has been critical for several research studies including personalized recommendation and social psychology.  


## Getting the data

The MovieLens data set is hosted by the [GroupLens](https://grouplens.org/datasets/movielens/) website. Several versions are available. We will use the MovieLens 100K Dataset.   This data set consists of 100,000 ratings (1-5) from 943 users on 1682 movies. It has been cleaned up so that each user has rated at least 20 movies. Some simple Simple demographic information such as age, gender, genres for the users and items are also available.  We can download the [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and extract the u.data file, which contains all the 100,000 ratings in csv format. There are many other files in the folder, a detailed description for each file can be found in [ReadMe](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) file of the dataset. At first, we import libraries that are needed.

```{.python .input  n=2}
import pandas as pd
import numpy as np
import zipfile

from mxnet import gluon
```

Then, we download the MovieLens 100k dataset and load the interactions as `DataFrame`.

```{.python .input  n=5}
# Save to the d2l package.
def read_data_ml100k(path = "ml-100k/u.data"):
    fname = gluon.utils.download(
        'http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    with zipfile.ZipFile(fname, 'r') as inzipfile:
        inzipfile.extract(path)
        return pd.read_csv(path, sep="\t", 
                           names=['user_id','item_id','rating','timestamp'],
                           engine='python')

df = read_data_ml100k()
num_users = df.user_id.unique().shape[0]
num_items = df.item_id.unique().shape[0]
print('number of users: %d, number of items: %d.'%(num_users, num_items))
print(df.head(5))
```

We can see that each line consists of four columns, including user id (1-943), item id (1-1682), rating (1-5) and timestamp. We can construct an interaction matrix of size $\text{num_users} * \text{num_items}$.  This data set only
records the existed ratings and most of the values in the interaction matrix are unknown as users have not rated the majority of movies. Clearly, the interaction matrix is extremely sparse (sparsity = 93.685%). The case in data sets from real-world applications can be even worse, and data sparsity has been a long- standing challenge in recommender systems.

## Splitting the data

Then, we randomly split the data set into training and test sets.  We use the 90% of the data as training samples and the rest 10% as test samples.

```{.python .input  n=6}
# Save to the d2l package.
def split_data(test_size = 0.1):
    msk = np.random.rand(len(df)) < (1 - test_size)
    train_data = df[msk]
    test_data = df[~msk]
    return train_data, test_data
```

## Loading the data
After dataset splitting, we will convert the training set and test set into lists and dictionaries for the sake of convienience. The following function reads the dataframe line by line and make the index of users/items start from zero, and returns lists of users, items, scores (e.g., ratings) and a dictiobnary that records the interactions.

```{.python .input  n=7}
# Save to the d2l package.
def load_dataset_ml100k(data, type_feedback="explicit"):
    users, items, scores = [], [], []
    interaction_dict = {}
    for line in data.itertuples():
        user_index = int(line[1] - 1)
        item_index = int(line[2] - 1)
        score = int(line[3]) if type_feedback == "explicit" else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        interaction_dict.setdefault(user_index, []).append(item_index)
    return users, items, scores, interaction_dict
```

Afterwards, we put the above steps together and save it for further use.

```{.python .input  n=8}
# Save to the d2l package.
def split_and_load_ml100k(test_size=0.1, batch_size=128):
    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data(test_size)
    return load_dataset_ml100k(train_data), load_dataset_ml100k(test_data)
```

The results can be used by gluon `Dataset` and `DataLoader`. For example, an `ArrayDataset` can be used to wrap the results and 'DataLoader' can be used for mini-batch sampling.

## Summary 
* MovieLens datasets are widely used for recommendation research. It is publicly available and free to use.
* We downloaded and preprocessed the MovieLens 100k dataset for further use in later sections. 

## Exercise
* What other similar recommendation datasets can you find?
* Read the provided README file to see
