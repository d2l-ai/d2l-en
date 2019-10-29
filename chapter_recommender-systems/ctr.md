# Feature-Rich Recommender Sytems

Interaction data is the most basic indication of users' preferences and interets. It plays a critical role in former introduced models. Yet, interaction data is usually extremely sparse and can be noisy at times. To address this issue, we can integrate side information such as features of items, profiles of users, and even in which context that the interaction occurred into the recommendation model. Utilizing these features are helpful in making recommendations in that these features can be an effective predictor of users interests epecially when interaction data is lacking. As such, it is essential for recommendation models also have the capability to deal with those features and give the model some content/context awareness. To demonstrate this type of recommendation models, we introduce another task on click-through rate (CTR) for online advertisement recommendations and present an anoynoumous advertising data. Targeted advertisement services have attracted widespread attention and are often framed as recommendation engines. Recommending advertisements that match users' personal taste and interest is important for click-through rate improvement.


Digital marketers use online advertising to display advertisements to customers. Click-through rate is a metric that measures the number of clicks advertisers receive on their ads per number of impressions and it is expressed as a percentage calculated with the formula: 

$$ \text{CTR} = \frac{\#\text{Clicks}} {\#\text{Impressions}} \times 100 \% .$$

Click-through rate is an important signal that indicates the effectiveness of prediction algorithms. Click-through rate prediction is a task of predicting the likelihood that something on a website will be clicked. Models on CTR prediction can not only be employed in targeted advertising systems but also in general item (e.g., movies, news, products) recommender systems, email campaigns, and even search engines. It is also closely related to user satisfaction, conversion rate, and can be helpful in setting campaign goals as it can help advertisers to set realistic expectations. 


## An Online Advertising Dataset

With the considerable advancements of internet and mobile technology, online advertising has become an important income resource and generates vast majority of revenue in the internet industry. It is important to display relevant advertisements or advertisements that pique users' interests so that casual visitors can be converted into paying customers. The dataset we introduced is an online advertising dataset. It consists of 34 fields, with the first column representing the target variable that indicates if an ad was clicked (1) or not (0). All the other columns are categorical features. The columns might represent the advertisement id, site or application id, device id, time, user profiles and so on. The real semantics of the features are undisclosed due to anonymization and privacy concern.

The following code downloads the dataset from our server and saves it into the local data folder.

```{.python .input  n=15}
from collections import defaultdict
from mxnet import gluon, np
from mxnet.gluon.data import Dataset
import os
import sys

data_path = "https://apache-mxnet.s3-accelerate.amazonaws.com/gluon/dataset/"
train_sha1 = "6dec3052e49ce0d1cec5ebc6f5ded1172be0befb"
test_sha1 ="c265e3c1fad0ed4caf8c1a373c580465a8096eb0"

def read_data_ctr(path="../data/", train="ctr/train.csv", 
                  test="ctr/test.csv"):
    ctr_path = path+"ctr"
    os.makedirs(ctr_path, exist_ok=True)
    gluon.utils.download(data_path + train, ctr_path, train_sha1)
    gluon.utils.download(data_path + test, ctr_path, test_sha1)

read_data_ctr()
```

There are a training set and a test set, consisting of 15000 and 3000 samples/lines, respectively.

## Dataset Wrapper
For the convience of data loading, we impement a `CTRDataset` which loads the advertising dataset from the CSV file and can be used by `DataLoader`.

```{.python .input  n=13}
# Saved in the d2l package for later use
class CTRDataset(Dataset):
    def __init__(self, data_path, feat_mapper=None, defaults=None, 
                 min_threshold=4, num_feat=34):
        self.NUM_FEATS, self.count, self.data = num_feat, 0, {}
        feat_cnts = defaultdict(lambda: defaultdict(int))
        self.feat_mapper, self.defaults = feat_mapper, defaults
        self.field_dims = np.zeros(self.NUM_FEATS, dtype=np.int64)
        with open(data_path) as f:
            for line in f:
                instance = {}
                values = line.rstrip('\n').split('\t')
                if len(values) != self.NUM_FEATS + 1:
                    continue
                label = np.float32([0,0])
                label[int(values[0])] = 1
                instance['y'] = [np.float32(values[0])]
                for i in range(1, self.NUM_FEATS + 1):
                    feat_cnts[i][values[i ]] += 1
                    instance.setdefault('x',[]).append(values[i ])
                self.data[self.count] = instance
                self.count = self.count + 1
        if self.feat_mapper is None and self.defaults is None:
            feat_mapper = {i: {feat for feat, c in cnt.items() if c >= 
                               min_threshold} for i, cnt in feat_cnts.items()}
            self.feat_mapper = {i: {feat: idx for idx, feat in enumerate(cnt)}
                                for i, cnt in feat_mapper.items()}
            self.defaults = {i: len(cnt) for i, cnt in feat_mapper.items()}
        for i, fm in self.feat_mapper.items():
            self.field_dims[i - 1] = len(fm) + 1
        self.offsets = np.array((0, *np.cumsum(self.field_dims).asnumpy()
                                 [:-1]))
    def __len__(self):
        return self.count
    def __getitem__(self, idx):
        feat = np.array([self.feat_mapper[i + 1].get(v, self.defaults[i + 1]) 
                         for i, v in enumerate(self.data[idx]['x'])])
        return feat + self.offsets, self.data[idx]['y']
```

The following example loads the training data and print out the first record.

```{.python .input  n=16}
train_data = CTRDataset(data_path="../data/ctr/train.csv")
train_data[0]
```

As can be seen, all the features are categorical. Each value represents the one-hot index of the corresponding entry. The label $0$ means that it is not clicked. This `CTRDataset` can also be used to load other datasets such as the Criteo display advertising challenge [Dataset](https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and the Avazu click-through rate prediction [Dataset](https://www.kaggle.com/c/avazu-ctr-prediction).  

## Summary 
* Click-through rate is an important metric that is used to measure the effectiveness of advertising systems and recommender systems.
* Click-through rate prediction is usually converted to a binary classification problem. The target is to predict whether an ad/item will be clicked or not based on given features.

## Exercise
* Can you load the Criteo and Avazu dataset with the provided `CTRDataset`. It is worth noting that the Criteo dataset consisting of real-valued features so you may have to revise the code a bit.
