
#  The MovieLens Dataset

There are a number of data sets available for recommendation research. The [MovieLens](https://movielens.org/) data set is probably the most popular one. MovieLens is a non-commercial web-based movie recommender system, created in1997 and run by GroupLens, a research lab at the University of Minnesota, in order to gather movie rating data for research use.  MovieLens data has been critical for several research studies including personalized recommendation and social psychology.  


## Getting the data

The MovieLens data set is hosted by the [GroupLens](https://grouplens.org/datasets/movielens/) website. Several versions are available. We will use the MovieLens 100K Dataset.   This data set consists of 100,000 ratings (1-5) from 943 users on 1682 movies. It has been cleaned up so that each user has rated at least 20 movies. Some simple Simple demographic information such as age, gender, genres for the users and items are also available.  We can download the [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and extract the u.data file, which contains all the 100,000 ratings in csv format. There are many other files in the folder, a detailed description for each file can be found in [ReadMe](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) file of the dataset. At first, we import libraries that are needed.


```
import pandas as pd
import zipfile
from mxnet import np, npx, gluon
```

Then, we download the MovieLens 100k dataset and load the interactions as `DataFrame`.
# Save to the d2l package.
def read_data_ml100k(path = "ml-100k/u.data", 
                     names=['user_id','item_id','rating','timestamp'],
                     sep="\t"):
    fname = gluon.utils.download(
        'http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    with zipfile.ZipFile(fname, 'r') as inzipfile:
        inzipfile.extract(path)
        df = pd.read_csv(path, sep=sep, names=names, engine='python')
        num_users = df.user_id.unique().shape[0]
        num_items = df.item_id.unique().shape[0]
        return df, num_users, num_items

df, num_users, num_items = read_data_ml100k()
#print('number of users: %d, number of items: %d.'%(num_users, num_items))
#print(df.head(5))
We can see that each line consists of four columns, including user id (1-943), item id (1-1682), rating (1-5) and timestamp. We can construct an interaction matrix of size $\text{num_users} * \text{num_items}$.  This data set only
records the existed ratings and most of the values in the interaction matrix are unknown as users have not rated the majority of movies. Clearly, the interaction matrix is extremely sparse (sparsity = 93.685%). The case in data sets from real-world applications can be even worse, and data sparsity has been a long- standing challenge in recommender systems.

## Splitting the data

Then, we randomly split the data set into training and test sets.  We use the 90% of the data as training samples and the rest 10% as test samples.


```
# Save to the d2l package.
def split_data(df, test_size = 0.1):
    msk = [True if x == 1 else False for x in 
           np.random.uniform(0, 1, (len(df))) < 1 - test_size]
    neg_msk = [not x for x in msk]
    train_data = df[msk]
    test_data = df[neg_msk]
    return train_data, test_data
```

## Loading the data
After dataset splitting, we will convert the training set and test set into lists and dictionaries for the sake of convienience. The following function reads the dataframe line by line and make the index of users/items start from zero, and returns lists of users, items, scores (e.g., ratings) and a dictiobnary that records the interactions.


```
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

Afterwards, we put the above steps together and save it for further use. The results are wrapped with gluon `Dataset` and `DataLoader`. Note that the `last_batch` of `DataLoader` for training data is set to the `rollover` mode and orders are shuffled.


```
# Save to the d2l package.
def split_and_load_ml100k(test_size=0.1, batch_size=128):
    df, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data(df, test_size)
    train_u, train_i, train_s, _ = load_dataset_ml100k(train_data)
    test_u, test_i, test_s, _ = load_dataset_ml100k(test_data) 
    train_arraydataset = gluon.data.ArrayDataset(np.array(train_u), 
                                                 np.array(train_i), 
                                                 np.array(train_s))
    test_arraydataset = gluon.data.ArrayDataset(np.array(test_u), 
                                                 np.array(test_i), 
                                                 np.array(test_s))

    train_data = gluon.data.DataLoader(train_arraydataset, shuffle=True, 
                                       last_batch="rollover", 
                                       batch_size=batch_size)
    test_data = gluon.data.DataLoader(test_arraydataset, shuffle=False, 
                                      batch_size=len(test_data))

    return num_users, num_items, train_data, test_data
```

The results can be used by gluon `Dataset` and `DataLoader`. For example, an `ArrayDataset` can be used to wrap the results and 'DataLoader' can be used for mini-batch sampling.

## Summary 
* MovieLens datasets are widely used for recommendation research. It is public available and free to use.
* We downloaded and preprocessed the MovieLens 100k dataset for further use in later sections. 

## Exercise
* What other similar recommendation datasets can you find?
* Read the README file of the MovieLens datasets.
