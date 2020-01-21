# Natural Language Inference and the Dataset

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


SNLI :cite:`Bowman.Angeli.Potts.ea.2015`

Commonly used datasets in natural language inference include Stanford natural language inference(SNLI) and multiple natural language inference (MultiNLI) dataset. SNLI has more than 500,000 manually written English sentence pairs, which are divided into three types of relationships: entailment, contradiction and neutrality. MultiNLI is the upgraded version of SNLI. However, the difference between them is that MultiNLI dataset also involves relevant spoken and written texts. For this reason, it has more variations in comparison with Stanford natural language inference dataset.

In this section, we will use Stanford natural language inference dataset. To make better use of this dataset, we need to input the packages or modules needed for the experiment.

```{.python .input  n=28}
import collections
import d2l
import os
from mxnet import gluon, np, npx
import re
import zipfile

npx.set_np()
```

We download the RAR of this dataset to ../data. The capacity of the RAR is around 100MB. After decompaction, the dataset is stored in ../data/snli_1.0

```{.python .input  n=11}
# Saved in the d2l package for later use
d2l.DATA_HUB['SNLI'] = (
    'https://nlp.stanford.edu/projects/snli/snli_1.0.zip',
    '9fcde07509c7e87ec61c640c1b2753d9041758e4')

data_dir = d2l.download_extract('SNLI')
```

Having entered `../data/snli_1.0`, we can get access to different components of dataset .The dataset includes separated training set, verification set and test set. Each line of the dataset files includes 14 rows of original sentences and corresponding tags, etc. The dataset files also provide two forms of analytic trees of sentences. The analytic tree shows the text texture in the form of tree according to the grammatical relationship between words and phrases. In this section, we need to use the first three rows of dataset files. The first row is the tag. The second row is the analytical tree of premise. The third row is the analytical tree of hypothesis.

Next, we read the training dataset and testing dataset, and retain samples with valid tags. Brackets represent the level of analytic tree. We need to eliminate brackets, merely retain original texts and conduct word segmentation.

```{.python .input  n=66}
# Saved in the d2l package for later use
def read_snli(data_dir, is_train):
    """Read the SNLI dataset into premises, hypotheses, and labels."""
    def extract_text(s):
        # Remove information that will not be used by us
        s = re.sub('\(', '', s) 
        s = re.sub('\)', '', s)
        # Substitute two or more consecutive whitespace with space
        s = re.sub('\s{2,}', ' ', s)
        return s.strip()
    label_set = {'entailment': 0, 'contradiction': 1, 'neutral': 2}
    file_name = (data_dir + 'snli_1.0_'+ ('train' if is_train else 'test')
                 + '.txt')
    with open(file_name, 'r') as f:
        rows = [row.split('\t') for row in f.readlines()[1:]]
    premises = [extract_text(row[1]) for row in rows if row[0] in label_set]
    hypotheses = [extract_text(row[2]) for row in rows if row[0] in label_set]
    labels = [label_set[row[0]] for row in rows if row[0] in label_set]
    return premises, hypotheses, labels
```

We output the first five sentence pairs of premise and hypothesis, as well as corresponding tags of inference relationship.

```{.python .input  n=70}
train_data = read_snli(data_dir, is_train=True)
for x0, x1, y in zip(train_data[0][:3], train_data[1][:3], train_data[2][:3]):
    print('premise:', x0)
    print('hypothesis:', x1)
    print('label:', y)
```

According to rough statistics, we find approximately 550,000 training set samples. Three relationship tags account for around 180,000 respectively. We also find about 10,000 testing dataset samples. Three relationship tags account for around 3000 respectively. Each type of tag shows the basically equivalent amount.

```{.python .input}
test_data = read_snli(data_dir, is_train=False)
for data in [train_data, test_data]:
    print([[row for row in data[2]].count(i) for i in range(3)])
```

### Self-defining dataset
By inheriting `Dataset` by Gluon, we self-defined a natural language inference dataset `SNLIDataset`. By realizing `__getitem__` function, we can get access to the sentence pairs with idx index and relevant categories.

```{.python .input  n=115}
# Saved in the d2l package for later use
class SNLIDataset(gluon.data.Dataset):
    """A customized dataset to load the SNLI dataset."""
    def __init__(self, dataset, vocab=None):
        self.num_steps = 50  # We fix the length of each sentence to 50.
        p_tokens = d2l.tokenize(dataset[0])
        h_tokens = d2l.tokenize(dataset[1])
        if vocab is None:
            self.vocab = d2l.Vocab(p_tokens + h_tokens, min_freq=5,
                                   reserved_tokens=['<pad>'])
        else:
            self.vocab = vocab
        self.premises = self.pad(p_tokens)
        self.hypotheses = self.pad(h_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.premises)) + ' examples')

    def pad(self, lines):
        return np.array([d2l.trim_pad(self.vocab[line], self.num_steps, 
                                      self.vocab['<pad>']) for line in lines])

    def __getitem__(self, idx):
        return (self.premises[idx], self.hypotheses[idx]), self.labels[idx]

    def __len__(self):
        return len(self.premises)
```

### Read dataset

Training set and testing set examples are respectively established on the basis of self-defining `SNLIDataset`. We define 50 as the maximum text length. Then, we can check the number of samples retained in training set and testing set.

```{.python .input  n=114}
# Saved in the d2l package for later use
def load_data_snli(batch_size, num_steps=50):
    """Download the SNLI dataset and return its Dataloader instances."""
    data_dir = d2l.download_extract('SNLI')
    train_data = read_snli(data_dir, True)
    test_data = read_snli(data_dir, False)
    train_set = SNLIDataset(train_data)
    test_set = SNLIDataset(test_data, train_set.vocab)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
    test_iter = gluon.data.DataLoader(test_set, batch_size, shuffle=False)
    return train_iter, test_iter, train_set.vocab
```

Assume batch size is 128, respectively define the iterators of training set and testing set.

```{.python .input  n=111}
batch_size = 128
train_iter, test_iter, vocab = load_data_snli(batch_size)
```

Output the size of the word list, showing 18677 valid words.

```{.python .input  n=112}
print('Vocab size:', len(vocab))
```

Print the form of the first small batch. What is different from text classification task is the data here consist of triples (Sentence 1, Sentence 2, Label)

```{.python .input  n=113}
for X, Y in train_iter:
    print(X[0].shape)
    print(X[1].shape)
    print(Y.shape)
    break
```

## Summary
- Natural language inference task aims to identify the inference relationship between premise and hypothesis.
- In natural language inference task, the sentences have three types of inference relationships: entailment, contradiction and neutral.
- An important dataset of natural language inference task is known as Stanford natural language inference (SNLI) dataset.

## [Discussions](https://discuss.mxnet.io/t/5517)

![](../img/qr_natural-language-inference-and-dataset.svg)
