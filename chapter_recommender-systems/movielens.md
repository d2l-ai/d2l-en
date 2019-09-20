#  Exploring the MovieLens Dataset

There are a number of data sets available for recommendation research. The [MovieLens](https://movielens.org/) data set is probably the most popular one. MovieLens is a non-commercial web-based movie recommender system, created in1997 and run by GroupLens, a research lab at the University of Minnesota, in order to gather movie rating data for research use.  MovieLens data has been critical for several research studies including personalized recommendation and social psychology.  


## Getting the data

The MovieLens data set is hosted by the [GroupLens](https://grouplens.org/datasets/movielens/) website. Several versions are available. We will use the MovieLens 100K Dataset.   This data set is comprised of 100,000 ratings, ranging from 1 to 5 stars, from 943 users on 1682 movies. It has been cleaned up so that each user has rated at least 20 movies. Some simple demographic information such as age, gender, genres for the users and items are also available.  We can download the [ml-100k.zip](http://files.grouplens.org/datasets/movielens/ml-100k.zip) and extract the u.data file, which contains all the 100,000 ratings in csv format. There are many other files in the folder, a detailed description for each file can be found in the [ReadMe](http://files.grouplens.org/datasets/movielens/ml-100k-README.txt) file of the dataset. 

At first, we import libraries that are needed.

```{.python .input  n=2}
import d2l
import pandas as pd
import zipfile
from mxnet import gluon, np, npx 
```

Then, we download the MovieLens 100k dataset and load the interactions as `DataFrame`.

```{.python .input  n=3}
# Save to the d2l package.
def read_data(path = "ml-100k/u.data", 
                     names=['user_id','item_id','rating','timestamp'],
                     sep="\t"):
    fname = gluon.utils.download(
        'http://files.grouplens.org/datasets/movielens/ml-100k.zip')
    with zipfile.ZipFile(fname, 'r') as inzipfile:
        inzipfile.extract(path)
        data = pd.read_csv(path, sep=sep, names=names, engine='python')
        num_users = data.user_id.unique().shape[0]
        num_items = data.item_id.unique().shape[0]
        return data, num_users, num_items
```

## Statistics of the Dataset
Let's load up the data and inspect the first five records manually. It is an effective way to learn about the data structure and verify that they have been loaded properly.

```{.python .input  n=4}
data, num_users, num_items = read_data()
sparsity = 1 - len(data) / (num_users * num_items)
print('number of users: %d, number of items: %d.'%(num_users, num_items))
print('matrix sparsity: %f' % sparsity)
print(data.head(5))
```

We can see that each line consists of four columns, including user id (1-943), item id (1-1682), rating (1-5) and timestamp. We can construct an interaction matrix of size $\text{num_users} * \text{num_items}$.  This data set only records the existed ratings and most of the values in the interaction matrix are unknown as users have not rated the majority of movies. Clearly, the interaction matrix is extremely sparse (sparsity = 93.695%). The case in data sets from large scale real-world applications can be even worse, and data sparsity has been a long- standing challenge in building recommender systems.

We then plot the distribution of the count of different ratings. As expected, it appears to be a normal distribution, with most ratings beings that a movie was good but not amazing.

```{.python .input  n=5}
d2l.plt.hist(data['rating'], bins=5, ec='black')
d2l.plt.xlabel("Rating")
d2l.plt.ylabel("Count")
d2l.plt.title("Distribution of Ratings in MovieLens 100K")
d2l.plt.show()
```

## Splitting the data

Then, we split the dataset into training and test sets. The following function provides two split modes including `random` and `time-aware`. In the `time-aware` mode, we leave out the latest rated item of each user for test and the rest for training. In the `random` mode, it splits the data set randomly and uses the 90% of the data as training samples and the rest 10% as test samples by default.

```{.python .input  n=6}
# Save to the d2l package.
def split_data(data, num_users, num_items, 
               split_mode="random", test_size = 0.1):
    """Split the dataset in random mode or time-aware mode."""
    if split_mode == "time-aware":
        train_items, test_items, train_list = {}, {}, []
        for line in data.itertuples():
            u, i, rating, time = line[1], line[2], line[3], line[4]
            train_items.setdefault(u,[]).append((u, i, rating, time))
            if u not in test_items or test_items[u][-1] < time:
                test_items[u] = (i, rating, time)
        for u in range(1, num_users+1):
            train_list.extend(sorted(train_items[u], key=lambda k: k[3]))
        test_data = [(key, *value) for key, value in test_items.items()]
        train_data = [item for item in train_list if item not in test_data]
        train_data = pd.DataFrame(train_data)
        test_data = pd.DataFrame(test_data)
    else:
        msk = [True if x == 1 else False for x in 
        np.random.uniform(0, 1, (len(data))) < 1 - test_size]
        neg_msk = [not x for x in msk]
        train_data, test_data = data[msk], data[neg_msk]
    return train_data, test_data
```

## Loading the data
After dataset splitting, we will convert the training set and test set into lists and dictionaries/matrix for the sake of convienience. The following function reads the dataframe line by line and make the index of users/items start from zero, and returns lists of users, items, ratings and a dictiobnary/matrix that records the interactions. We can specify the type of feedback to either `explicit` or `implicit`.

```{.python .input  n=25}
# Save to the d2l package.
def load_dataset(data, num_users, num_items, feedback="explicit"):
    users, items, scores = [], [], []
    inter = np.zeros((num_items, num_users)) if feedback == "explicit" else {}
    for line in data.itertuples():
        user_index, item_index = int(line[1] - 1), int(line[2] - 1)
        score = int(line[3]) if type_feedback == "explicit" else 1
        users.append(user_index)
        items.append(item_index)
        scores.append(score)
        if feedback == "implicit":
            inter.setdefault(user_index, []).append(item_index)
        else:
            inter[item_index, user_index] = score
    return users, items, scores, inter
```

Afterwards, we put the above steps together and save it for further use. The results are wrapped with gluon `Dataset` and `DataLoader`. Note that the `last_batch` of `DataLoader` for training data is set to the `rollover` mode and orders are shuffled.

```{.python .input  n=26}
# Save to the d2l package.
def split_and_load_ml100k(split_mode="time-aware", feedback="explicit", 
                          test_size=0.1, batch_size=256):
    data, num_users, num_items = read_data_ml100k()
    train_data, test_data = split_data(data, split_mode, test_size)
    train_u, train_i, train_r, _ = load_dataset(train_data, feedback)
    test_u, test_i, test_r, _ = load_dataset(test_data, feedback) 
    train_arraydataset = gluon.data.ArrayDataset(np.array(train_u), 
                                                 np.array(train_i), 
                                                 np.array(train_r))
    test_arraydataset = gluon.data.ArrayDataset(np.array(test_u), 
                                                 np.array(test_i), 
                                                 np.array(test_r))
    train_data = gluon.data.DataLoader(train_arraydataset, shuffle=True, 
                                       last_batch="rollover", 
                                       batch_size=batch_size)
    test_data = gluon.data.DataLoader(test_arraydataset, 
                                      batch_size=batch_size)

    return num_users, num_items, train_data, test_data
```

## Summary 
* MovieLens datasets are widely used for recommendation research. It is public available and free to use.
* We downloaded and preprocessed the MovieLens 100k dataset for further use in later sections. 

## Exercise
* What other similar recommendation datasets can you find?
* Read the README file of the MovieLens datasets.
