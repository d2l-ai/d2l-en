#  MovieLens Dataset

There are a number of datasets that are available for recommendation research. Among then, the [MovieLens](https://movielens.org/) dataset is probably the most popular one. MovieLens is a non-commercial web-based movie recommender system, created in 1997 and run by GroupLens, a research lab at the University of Minnesota, in order to gather movie rating data for research use.  MovieLens data has been critical for several research studies including personalized recommendation and social psychology.


## Getting the Data

The MovieLens dataset is hosted by the [GroupLens](https://grouplens.org/datasets/movielens/) website. Several versions are available. We will use the MovieLens 100K dataset.  This dataset is comprised of 100,000 ratings, ranging from 1 to 5 stars, from 943 users on 1682 movies. It has been cleaned up so that each user has rated at least 20 movies. Some simple demographic information such as age, gender, genres for the users and items are also available.  We can download the [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and extract the `u.data` file, which contains all the 100,000 ratings in the csv format. There are many other files in the folder, a detailed description for each file can be found in the [README](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) file of the dataset. 

To begin with, let us import the packages required to run this section’s experiments.

```{.python .input  n=1}
import sys
sys.path.insert(0, '..')
import d2l
from mxnet import gluon, np, npx
import pandas as pd
import zipfile
```

Then, we download the MovieLens 100k dataset and load the interactions as `DataFrame`.

```{.python .input  n=2}
# Save to the d2l package
def read_data_ml100k(path="../data/", member="ml-100k/u.data",
              names=['user_id','item_id','rating','timestamp'], sep="\t"):
    fname = gluon.utils.download(
        'http://files.grouplens.org/datasets/movielens/ml-100k.zip',
    path=path)
    with zipfile.ZipFile(fname, 'r') as inzipfile:
        inzipfile.extract(member, path)
        data = pd.read_csv(path + member, sep, names=names, engine='python')
        num_users = data.user_id.unique().shape[0]
        num_items = data.item_id.unique().shape[0]
        return data, num_users, num_items
```

## Statistics of the Dataset

Let us load up the data and inspect the first five records manually. It is an effective way to learn the data structure and verify that they have been loaded properly.

```{.python .input  n=3}
data, num_users, num_items = read_data_ml100k()
sparsity = 1 - len(data) / (num_users * num_items)
print('number of users: %d, number of items: %d.'%(num_users, num_items))
print('matrix sparsity: %f' % sparsity)
print(data.head(5))
```

```{.json .output n=3}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "number of users: 943, number of items: 1682.\nmatrix sparsity: 0.936953\n   user_id  item_id  rating  timestamp\n0      196      242       3  881250949\n1      186      302       3  891717742\n2       22      377       1  878887116\n3      244       51       2  880606923\n4      166      346       1  886397596\n"
 }
]
```

We can see that each line consists of four columns, including "user id" (1--943), "item id" (1--1682), "rating" (1--5) and "timestamp". We can construct an interaction matrix of size $n \times m$, where $n$ and $m$ are the number of users and the number of items respectively. This dataset only records the existing ratings and most of the values in the interaction matrix are unknown as users have not rated the majority of movies. Clearly, the interaction matrix is extremely sparse (sparsity = 93.695%). The case in datasets from large scale real-world applications can be even worse, and data sparsity has been a long-standing challenge in building recommender systems.

We then plot the distribution of the count of different ratings. As expected, it appears to be a normal distribution, with most ratings beings that a movie was good but not amazing.

```{.python .input  n=4}
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel("Rating")
d2l.plt.ylabel("Count")
d2l.plt.title("Distribution of Ratings in MovieLens 100K")
d2l.plt.show()
```

```{.json .output n=4}
[
 {
  "data": {
   "image/png": "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\n",
   "text/plain": "<Figure size 432x288 with 1 Axes>"
  },
  "metadata": {
   "needs_background": "light"
  },
  "output_type": "display_data"
 }
]
```

## Splitting the dataset

Then, we split the dataset into training and test sets. The following function provides two split modes including `random` and `time-aware`. In the `time-aware` mode, we leave out the latest rated item of each user for test and the rest for training. In the `random` mode, the function splits the dataset randomly and uses the 90% of the data as training samples and the rest 10% as test samples by default.

```{.python .input  n=5}
# Save to the d2l package
def split_data_ml100k(data, num_users, num_items, 
               split_mode="random", test_size = 0.1):
    """Split the dataset in random mode or time-aware mode."""
    if split_mode == "time-aware":
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u,[]).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users + 1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        mask = [True if x == 1 else False for x in np.random.uniform(
            0, 1, (len(data))) < 1 - test_size]
        neg_mask = [not x for x in mask]
        train_data, test_data = data[mask], data[neg_mask]
    return train_data, test_data
```

## Loading the data

After dataset splitting, we will convert the training set and test set into lists and dictionaries/matrix for the sake of convenience. The following function reads the dataframe line by line and make the index of users/items start from zero. The function then returns lists of users, items, ratings and a dictionary/matrix that records the interactions. We can specify the type of feedback to either `explicit` or `implicit`.

```{.python .input  n=6}
# Save to the d2l package
def load_data_ml100k(data, num_users, num_items, feedback="explicit"):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == "explicit" else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if feedback == "explicit" else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == "implicit":
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Afterwards, we put the above steps together and save it for further use. The results are wrapped with `Dataset` and `DataLoader`. Note that the `last_batch` of `DataLoader` for training data is set to the `rollover` mode and orders are shuffled.

```{.python .input  n=7}
# Save to the d2l package
def split_and_load_ml100k(split_mode="seq-aware", feedback="explicit", 
                          test_size=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data_ml100k(
        data, num_users, num_items, split_mode, test_size)
    train_u, train_i, train_r, _ = load_data_ml100k(
        train_data, num_users, num_items, feedback)
    test_u, test_i, test_r, _ = load_data_ml100k(
        test_data, num_users, num_items, feedback) 
    train_arraydataset = gluon.data.ArrayDataset(
        np.array(train_u), np.array(train_i), np.array(train_r))
    test_arraydataset = gluon.data.ArrayDataset(
        np.array(test_u), np.array(test_i), np.array(test_r))
    train_data = gluon.data.DataLoader(
        train_arraydataset, shuffle=True, last_batch="rollover",
        batch_size=batch_size)
    test_data = gluon.data.DataLoader(
        test_arraydataset, batch_size=batch_size)
    return num_users, num_items, train_data, test_data
```

## Summary

* MovieLens datasets are widely used for recommendation research. It is public available and free to use.
* We define functions to download and preprocess the MovieLens 100k dataset for further use in later sections. 


## Exercises

* What other similar recommendation datasets can you find?
* Go through the [https://movielens.org/](https://movielens.org/) site for more information about MovieLens.

```{.python .input}

```
