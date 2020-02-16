# BERT-NLI

![This section feeds pretrained BERT to an MLP-based architecture for natural language inference.](../img/nlp-map-nli-bert.svg)
:label:`fig_nlp-map-nli-bert`

```{.python .input  n=1}
import d2l
from mxnet import autograd, gluon, init, np, npx
from mxnet.gluon import nn

npx.set_np()
batch_size, ctx = 512, d2l.try_all_gpus()
bert_train_iter, vocab = d2l.load_data_wiki(batch_size, 'wikitext-2')

bert = d2l.BERTModel(len(vocab), embed_size=128, hidden_size=256, num_heads=2,
                     num_layers=2, dropout=0.2)
bert.initialize(init.Xavier(), ctx=ctx)
nsp_loss = gluon.loss.SoftmaxCELoss()
mlm_loss = gluon.loss.SoftmaxCELoss()

d2l.train_bert(bert_train_iter, bert, nsp_loss, mlm_loss, len(vocab), ctx, 20, 2000)
```

...

```{.python .input  n=65}
class SNLIBERTDataset(gluon.data.Dataset):
    def __init__(self, dataset, vocab = None):
        self.num_steps = 50  # We fix the length of each sentence to 50.
        p_tokens = d2l.tokenize(dataset[0], token='word')
        h_tokens = d2l.tokenize(dataset[1], token='word')
        self.vocab = vocab
        
        self.tokens, self.segment_ids, self.valid_lengths = self.preprocess(p_tokens, h_tokens)
        self.labels = np.array(dataset[2])
        print('read ' + str(len(self.tokens)) + ' examples')
    def preprocess(self, p_tokens, h_tokens):
        def pad(data):
            return d2l.trim_pad(data, self.num_steps, 0)
        
        tokens, segment_ids, valid_lengths = [], [], []
        
        for i in range(len(p_tokens)):
            token, segment_id = d2l.get_tokens_and_segment(p_tokens[i][:self.num_steps], 
                                                           h_tokens[i][:self.num_steps])
            tokens.append(self.vocab[pad(token)])
            segment_ids.append(np.array(pad(segment_id)))
            valid_lengths.append(np.array(len(token)))
            
        return tokens, segment_ids, valid_lengths
    

    def __getitem__(self, idx):
        return (self.tokens[idx], self.segment_ids[idx], self.valid_lengths[idx]), self.labels[idx]

    def __len__(self):
        return len(self.tokens)
```

...

```{.python .input  n=66}
data_dir = d2l.download_extract('SNLI')
train_data = d2l.read_snli(data_dir, True)
test_data = d2l.read_snli(data_dir, False)
train_set = SNLIBERTDataset(train_data, vocab)
test_set = SNLIBERTDataset(test_data, vocab)
```

...

```{.python .input  n=67}
batch_size = 512
train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
test_iter = gluon.data.DataLoader(test_set, batch_size)
```

...

```{.python .input  n=82}
class BERTClassifier(nn.Block):
    def __init__(self, bert, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.classifier = gluon.nn.Sequential()
        self.classifier.add(gluon.nn.Dense(256, flatten=False, activation='relu'))
        self.classifier.add(gluon.nn.Dense(num_classes))

    def forward(self, X):
        inputs, segment_types, seq_len = X
        seq_encoding, _, _ = self.bert(inputs, segment_types, seq_len)
        return self.classifier(seq_encoding[:, 0, :])
```

...

```{.python .input  n=83}
net = BERTClassifier(bert, 3)
net.classifier.initialize(ctx=ctx)
```

...

```{.python .input  n=87}
lr, num_epochs = 1e-4, 5
trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr})
loss = gluon.loss.SoftmaxCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, ctx, d2l.split_batch_multi_inputs)
```

...
