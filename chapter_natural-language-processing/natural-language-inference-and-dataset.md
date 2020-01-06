# Natural language inference and the Dataset

In the section “Text Sentiment Classification: Using Recurrent Neural Networks”, we discussed the model of using recurrent neural networks for text classification. Text classification task aims to identify the category of any given text sequence. But in actual scenarios, sometimes we need two given sentences in order to classify the relationship between them, i.e. sentence pair classification. For example, on a shopping website, the online customer service system needs to decide whether users’ questions have the same meanings as the existing ones in the knowledge base, which is the task of identifying the relationship between two sentences. In this case, we can not solve the issue by using the model of categorizing the single text sequence.

Now, we are about to analyze the relationship between the two sentence classification. The most typical example of these tasks is natural language inference.

## Natural language inference

Natural language inference is also known as text entailment. Natural language inference is an important issue of natural language processing. On one hand, semantic relations exit among the seemingly isolated texts. On the other hand, semantic relations of texts enable machine to truly understand and utilize the semantic information.

To be more specific, the task of natural language inference refers to identifying the inference relationship between hypothesis and premise. There are three different types of inference relationships. 1. Entailment, i.e. People think the semanteme of hypothesis can be inferred from that of premise. 2. Contradiction, i.e. People believe that hypothesis is false according to the semanteme of premise. 3. Neutral, i.e. People can not decide the semanteme of hypothesis according to that of the premise.

Because this task inputs a sentence pair of premise and hypothesis, natural language inference is to classify the sentence pair.

Here are three examples:

The first example is entailment. “Showing affection” in the hypothesis can be inferred from “hugging one another” in the premise.
> Premise：Two blond women are hugging one another.
> Hypothesis：There are women showing affection.

The second example is contradiction. The premise mentions what a person inspects. It can be inferred that “sleeping” can not happen simultaneously in the hypothesis.
> Premise：A man inspects the uniform of a figure in some East Asian country.
> Hypothesis：The man is sleeping.

The third example is neutrality. There is no relationship between premise and hypothesis.
> Premise：A boy is jumping on skateboard in the middle of a red bridge.
> Hypothesis：The boy skates down the sidewalk.


## Stanford natural language inference(SNLI) dataset

Commonly used datasets in natural language inference include Stanford natural language inference(SNLI) and multiple natural language inference (MultiNLI) dataset. SNLI has more than 500,000 manually written English sentence pairs, which are divided into three types of relationships: entailment, contradiction and neutrality. MultiNLI is the upgraded version of SNLI. However, the difference between them is that MultiNLI dataset also involves relevant spoken and written texts. For this reason, it has more variations in comparison with Stanford natural language inference dataset.

In this section, we will use Stanford natural language inference dataset. To make better use of this dataset, we need to input the packages or modules needed for the experiment.

```{.python .input  n=1}
import collections
import d2l
import os
from mxnet import gluon, np, npx
import zipfile

npx.set_np()
```

We download the RAR of this dataset to ../data. The capacity of the RAR is around 100MB. After decompaction, the dataset is stored in ../data/snli_1.0

```{.python .input  n=2}
# Saved in the d2l package for later use
def download_snli(data_dir='../data/'):
    url = ('https://nlp.stanford.edu/projects/snli/snli_1.0.zip')
    sha1 = '9fcde07509c7e87ec61c640c1b2753d9041758e4'
    fname = gluon.utils.download(url, data_dir, sha1_hash=sha1)
    with zipfile.ZipFile(fname, 'r') as f:
        f.extractall(data_dir)
        
download_snli()
```

Having entered `../data/snli_1.0`, we can get access to different components of dataset .The dataset includes separated training set, verification set and test set. Each line of the dataset files includes 14 rows of original sentences and corresponding tags, etc. The dataset files also provide two forms of analytic trees of sentences. The analytic tree shows the text texture in the form of tree according to the grammatical relationship between words and phrases. In this section, we need to use the first three rows of dataset files. The first row is the tag. The second row is the analytical tree of premise. The third row is the analytical tree of hypothesis.

Next, we read the training dataset and testing dataset, and retain samples with valid tags. Brackets represent the level of analytic tree. We need to eliminate brackets, merely retain original texts and conduct word segmentation.

```{.python .input  n=3}
# Saved in the d2l package for later use
def read_file_snli(filename):
    label_set = set(["entailment", "contradiction", "neutral"])
    def tokenized(text): 
        # The brackets represent the level of the parse tree. 
        # We need to remove the brackets to keep only the original text.
        return text.replace("(", "").replace(")", "").strip().lower().split()
    with open(os.path.join('../data/snli_1.0/', filename), 'r') as f:
        examples = [row.split('\t') for row in f.readlines()[1:]]
    return [(tokenized(row[1]), tokenized(row[2]), row[0]) 
             for row in examples if row[0] in label_set]

train_data, test_data = [read_file_snli('snli_1.0_'+ split + '.txt') 
                         for split in ["train", "test"]]
```

We output the first five sentence pairs of premise and hypothesis, as well as corresponding tags of inference relationship.

```{.python .input  n=4}
train_data[:5] 
```

```{.json .output n=4}
[
 {
  "data": {
   "text/plain": "[(['a',\n   'person',\n   'on',\n   'a',\n   'horse',\n   'jumps',\n   'over',\n   'a',\n   'broken',\n   'down',\n   'airplane',\n   '.'],\n  ['a',\n   'person',\n   'is',\n   'training',\n   'his',\n   'horse',\n   'for',\n   'a',\n   'competition',\n   '.'],\n  'neutral'),\n (['a',\n   'person',\n   'on',\n   'a',\n   'horse',\n   'jumps',\n   'over',\n   'a',\n   'broken',\n   'down',\n   'airplane',\n   '.'],\n  ['a',\n   'person',\n   'is',\n   'at',\n   'a',\n   'diner',\n   ',',\n   'ordering',\n   'an',\n   'omelette',\n   '.'],\n  'contradiction'),\n (['a',\n   'person',\n   'on',\n   'a',\n   'horse',\n   'jumps',\n   'over',\n   'a',\n   'broken',\n   'down',\n   'airplane',\n   '.'],\n  ['a', 'person', 'is', 'outdoors', ',', 'on', 'a', 'horse', '.'],\n  'entailment'),\n (['children', 'smiling', 'and', 'waving', 'at', 'camera'],\n  ['they', 'are', 'smiling', 'at', 'their', 'parents'],\n  'neutral'),\n (['children', 'smiling', 'and', 'waving', 'at', 'camera'],\n  ['there', 'are', 'children', 'present'],\n  'entailment')]"
  },
  "execution_count": 4,
  "metadata": {},
  "output_type": "execute_result"
 }
]
```

According to rough statistics, we find approximately 550,000 training set samples. Three relationship tags account for around 180,000 respectively. We also find about 10,000 testing dataset samples. Three relationship tags account for around 3000 respectively. Each type of tag shows the basically equivalent amount.

```{.python .input  n=5}
print("Training pairs: %d" % len(train_data))
print("Test pairs: %d" % len(test_data))

print("Train labels: {'entailment': %d, 'contradiction': %d, 'neutral': %d}" %
      ([row[2] for row in train_data].count('entailment'), 
       [row[2] for row in train_data].count('contradiction'), 
       [row[2] for row in train_data].count('neutral')))
print("Test labels: {'entailment': %d, 'contradiction': %d, 'neutral': %d}" %
      ([row[2] for row in test_data].count('entailment'), 
       [row[2] for row in test_data].count('contradiction'), 
       [row[2] for row in test_data].count('neutral')))
```

```{.json .output n=5}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Training pairs: 549367\nTest pairs: 9824\nTrain labels: {'entailment': 183416, 'contradiction': 183187, 'neutral': 182764}\nTest labels: {'entailment': 3368, 'contradiction': 3237, 'neutral': 3219}\n"
 }
]
```

### Self-defining dataset
By inheriting `Dataset` by Gluon, we self-defined a natural language inference dataset `SNLIDataset`. By realizing `__getitem__` function, we can get access to the sentence pairs with idx index and relevant categories.

```{.python .input  n=11}
# Saved in the d2l package for later use
class SNLIDataset(gluon.data.Dataset):
    def __init__(self, dataset, vocab=None):
        self.max_len = 50  # We fix the length of each sentence to 50.
        self.data = read_file_snli('snli_1.0_'+ dataset + '.txt')
        if vocab is None:
            self.vocab = d2l.Vocab([row[0] for row in self.data] + \
                                   [row[1] for row in self.data],
                                   min_freq = 5) # Filter words less than 5 times

        else:
            self.vocab = vocab
        self.premise, self.hypothesis, self.labels =  \
                                self.preprocess(self.data, self.vocab)
        print('read ' + str(len(self.premise)) + ' examples')

    def preprocess(self, data, vocab):
        LABEL_TO_IDX = {'entailment': 0, 'contradiction': 1, 'neutral': 2}

        def pad(x):
            return x[:self.max_len] if len(x) > self.max_len \
                                    else x + [0] * (self.max_len - len(x))

        premise = np.array([pad(vocab[x[0]]) for x in data])
        hypothesis = np.array([pad(vocab[x[1]]) for x in data])
        labels = np.array([LABEL_TO_IDX[x[2]] for x in data])
        return premise, hypothesis, labels

    def __getitem__(self, idx):
        return (self.premise[idx], self.hypothesis[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premise)
```

### Read dataset

Training set and testing set examples are respectively established on the basis of self-defining `SNLIDataset`. We define 50 as the maximum text length. Then, we can check the number of samples retained in training set and testing set.

```{.python .input  n=7}
train_set = SNLIDataset("train")
test_set = SNLIDataset("test", train_set.vocab)
```

```{.json .output n=7}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "read 549367 examples\nread 9824 examples\n"
 }
]
```

Assume batch size is 128, respectively define the iterators of training set and testing set.

```{.python .input  n=8}
batch_size = 128
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(test_set, batch_size)
```

Output the size of the word list, showing 18677 valid words.

```{.python .input  n=9}
print('Vocab size:', len(train_set.vocab))
```

```{.json .output n=9}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "Vocab size: 16673\n"
 }
]
```

Print the form of the first small batch. What is different from text classification task is the data here consist of triples (Sentence 1, Sentence 2, Label)

```{.python .input  n=10}
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

```{.json .output n=10}
[
 {
  "name": "stdout",
  "output_type": "stream",
  "text": "(128, 50)\n(128, 50)\n(128,)\n"
 }
]
```

## Summary
- Natural language inference task aims to identify the inference relationship between premise and hypothesis.
- In natural language inference task, the sentences have three types of inference relationships: entailment, contradiction and neutral.
- An important dataset of natural language inference task is known as Stanford natural language inference (SNLI) dataset.
