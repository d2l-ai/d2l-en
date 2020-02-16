...

```{.python .input  n=1}
import collections
import d2l
import mxnet as mx
from mxnet import autograd, gluon, init, np, npx
from mxnet.contrib import text
import os
import random
import time
import zipfile

npx.set_np()
```

```{.python .input  n=2}
# Saved in the d2l package for later use
d2l.DATA_HUB['wikitext-2'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip',
    '3c914d17d80b1459be871a5039ac23e752a53cbe')

# Saved in the d2l package for later use
d2l.DATA_HUB['wikitext-103'] = (
    'https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip',
    '0aec09a7537b58d4bb65362fee27650eeaba625a')
```

...

```{.python .input  n=3}
# Saved in the d2l package for later use
def read_wiki(data_dir):
    file_name = os.path.join(data_dir, 'wiki.train.tokens')
    with open(file_name, 'r') as f:
        raw = f.readlines()
    data = [line.strip().lower().split(' . ')
            for line in raw if len(line.split(' . '))>=2]
    random.shuffle(data)
    return data
```

...

```{.python .input  n=4}
# Saved in the d2l package for later use
def get_next_sentence(sentence, next_sentence, all_documents):
    if random.random() < 0.5:
        tokens_a = sentence
        tokens_b = next_sentence
        is_next = True
    else:
        random_sentence = random.choice(random.choice(all_documents))
        tokens_a = sentence
        tokens_b = random_sentence
        is_next = False
    return tokens_a, tokens_b, is_next
```

...

```{.python .input  n=5}
# Saved in the d2l package for later use
def get_tokens_and_segment(tokens_a, tokens_b):
    tokens = [] 
    segment_ids = [] 

    tokens.append('[CLS]')
    segment_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append('[SEP]')
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append('[SEP]')
    segment_ids.append(1)
    
    return tokens, segment_ids
```

...

```{.python .input  n=6}
# Saved in the d2l package for later use
def create_next_sentence(document, all_documents, vocab, max_length):
    instances = []
    for i in range(len(document)-1):
        tokens_a, tokens_b, is_next = get_next_sentence(document[i], document[i+1], all_documents)
        
        if len(tokens_a) + len(tokens_b) + 3 > max_length:
             continue
        tokens, segment_ids = get_tokens_and_segment(tokens_a, tokens_b)
        instances.append((tokens, segment_ids, is_next))
    return instances
```

...

```{.python .input  n=7}
# Saved in the d2l package for later use
def choice_mask_tokens(tokens, cand_indexes, num_to_predict, vocab):
    output_tokens = list(tokens)
    masked_lms = []
    random.shuffle(cand_indexes)
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        for index in index_set:
            masked_token = None
            if random.random() < 0.8:
                masked_token = '[MASK]'
            else:
                if random.random() < 0.5:
                    masked_token = tokens[index]
                else:
                    masked_token = random.randint(0, len(vocab) - 1)

            output_tokens[index] = masked_token
            masked_lms.append((index, tokens[index]))
    return output_tokens, masked_lms
```

...

```{.python .input  n=8}
# Saved in the d2l package for later use
def create_masked_lm(tokens, vocab):
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue
        cand_indexes.append([i])
        
    num_to_predict = max(1, int(round(len(tokens) * 0.15)))
    
    output_tokens, masked_lms = choice_mask_tokens(tokens, cand_indexes,
                                                   num_to_predict, vocab)
            
    masked_lms = sorted(masked_lms, key=lambda x: x[0])
    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p[0])
        masked_lm_labels.append(p[1])
        
    return vocab[output_tokens], masked_lm_positions, vocab[masked_lm_labels]
```

...

```{.python .input  n=9}
# Saved in the d2l package for later use
def convert_numpy(instances, max_length):
    input_ids, segment_ids, masked_lm_positions, masked_lm_ids = [], [], [], []
    masked_lm_weights, next_sentence_labels, valid_lengths = [], [], []
    for instance in instances:
        input_id = instance[0] + [0] * (max_length - len(instance[0]))
        segment_id = instance[3] + [0] * (max_length - len(instance[3]))
        masked_lm_position = instance[1] + [0] * (20 - len(instance[1]))
        masked_lm_id = instance[2] + [0] * (20 - len(instance[2]))
        masked_lm_weight = [1.0] * len(instance[2]) + [0.0] * (20 - len(instance[1]))
        next_sentence_label = instance[4]
        valid_length = len(instance[0])

        input_ids.append(np.array(input_id, dtype='int32'))
        segment_ids.append(np.array(segment_id, dtype='int32'))
        masked_lm_positions.append(np.array(masked_lm_position, dtype='int32'))
        masked_lm_ids.append(np.array(masked_lm_id, dtype='int32'))
        masked_lm_weights.append(np.array(masked_lm_weight, dtype='float32'))
        next_sentence_labels.append(np.array(next_sentence_label))
        valid_lengths.append(np.array(valid_length))
    return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths
```

...

```{.python .input  n=10}
# Saved in the d2l package for later use
def create_training_instances(train_data, vocab, max_length):
    instances = []
    for i, document in enumerate(train_data):
        instances.extend(create_next_sentence(document, train_data, vocab, max_length))
    instances = [(create_masked_lm(tokens, vocab) + (segment_ids, is_random_next))
                 for (tokens, segment_ids, is_random_next) in instances]
    input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths = convert_numpy(instances, max_length)
    return input_ids, masked_lm_ids, masked_lm_positions, masked_lm_weights,\
           next_sentence_labels, segment_ids, valid_lengths
```

...

```{.python .input  n=11}
# Saved in the d2l package for later use
class WikiDataset(gluon.data.Dataset):
    def __init__(self, dataset, max_length = 128):
        train_tokens = [d2l.tokenize(row, token='word') for row in dataset]
        
        text_list=[]
        [text_list.extend(row) for row in train_tokens]
        self.vocab = d2l.Vocab(text_list, min_freq=5, 
                               reserved_tokens=['[MASK]', '[CLS]', '[SEP]'])
        self.input_ids, self.masked_lm_ids, self.masked_lm_positions,\
        self.masked_lm_weights, self.next_sentence_labels, self.segment_ids,\
        self.valid_lengths = create_training_instances(train_tokens, self.vocab, max_length)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masked_lm_ids[idx], self.masked_lm_positions[idx], self.masked_lm_weights[idx],\
           self.next_sentence_labels[idx], self.segment_ids[idx], self.valid_lengths[idx]

    def __len__(self):
        return len(self.input_ids)
```

```{.python .input}
# Saved in the d2l package for later use
def load_data_wiki(batch_size, data_set = 'wikitext-2', num_steps=128):
    data_dir = d2l.download_extract(data_set, data_set)
    train_data = read_wiki(data_dir)
    train_set = WikiDataset(train_data, num_steps)
    train_iter = gluon.data.DataLoader(train_set, batch_size, shuffle=True)
    return train_iter, train_set.vocab
```

```{.python .input}
batch_size = 512
train_iter, vocab = load_data_wiki(batch_size, 'wikitext-2')
```

...

...

...

```{.python .input  n=14}
for _, data_batch in enumerate(train_iter):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = data_batch
    print(input_id.shape, masked_id.shape, masked_position.shape, masked_weight.shape,\
          next_sentence_label.shape, segment_id.shape, valid_length.shape)
    break
```

...

```{.python .input  n=15}
net = d2l.BERTModel(len(vocab), embed_size=128, hidden_size=256, 
                    num_heads=2, num_layers=2, dropout=0.2)
ctx = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=ctx)
nsp_loss = mx.gluon.loss.SoftmaxCELoss()
mlm_loss = mx.gluon.loss.SoftmaxCELoss()
```

...

```{.python .input  n=16}
# Saved in the d2l package for later use
def _get_batch_bert(batch, ctx):
    (input_id, masked_id, masked_position, masked_weight, \
     next_sentence_label, segment_id, valid_length) = batch
    
    return (gluon.utils.split_and_load(input_id, ctx, even_split=False),
            gluon.utils.split_and_load(masked_id, ctx, even_split=False),
            gluon.utils.split_and_load(masked_position, ctx, even_split=False),
            gluon.utils.split_and_load(masked_weight, ctx, even_split=False),
            gluon.utils.split_and_load(next_sentence_label, ctx, even_split=False),
            gluon.utils.split_and_load(segment_id, ctx, even_split=False),
            gluon.utils.split_and_load(valid_length.astype('float32'), ctx, even_split=False))
```

...

```{.python .input  n=17}
# Saved in the d2l package for later use
def batch_loss_bert(net, nsp_loss, mlm_loss, input_id, masked_id, masked_position,
                    masked_weight, next_sentence_label, segment_id, valid_length, vocab_size):
    ls = []
    ls_mlm = []
    ls_nsp = []
    for i_id, m_id, m_pos, m_w, nsl, s_i, v_l in zip(input_id, masked_id, masked_position, masked_weight,\
                                                      next_sentence_label, segment_id, valid_length):
        num_masks = m_w.sum() + 1e-8
        _, classified, decoded = net(i_id, s_i, v_l.reshape(-1),m_pos)
        l_mlm = mlm_loss(decoded.reshape((-1, vocab_size)),m_id.reshape(-1), m_w.reshape((-1, 1)))
        l_mlm = l_mlm.sum() / num_masks
        l_nsp = nsp_loss(classified, nsl)
        l_nsp = l_nsp.mean()
        l = l_mlm + l_nsp
        ls.append(l)
        ls_mlm.append(l_mlm)
        ls_nsp.append(l_nsp)
        npx.waitall()
    return ls, ls_mlm, ls_nsp
```

...

```{.python .input  n=18}
# Saved in the d2l package for later use
def train_bert(data_eval, net, nsp_loss, mlm_loss, vocab_size, ctx, log_interval, max_step):
    trainer = gluon.Trainer(net.collect_params(), 'adam')
    step_num = 0
    while step_num < max_step:
        eval_begin_time = time.time()
        begin_time = time.time()
        
        running_mlm_loss = running_nsp_loss = 0
        total_mlm_loss = total_nsp_loss = 0
        running_num_tks = 0
        for _, data_batch in enumerate(data_eval):
            (input_id, masked_id, masked_position, masked_weight, \
             next_sentence_label, segment_id, valid_length) = _get_batch_bert(data_batch, ctx)
            
            step_num += 1
            with autograd.record():
                ls, ls_mlm, ls_nsp = batch_loss_bert(net, nsp_loss, mlm_loss, input_id, masked_id, masked_position, masked_weight, next_sentence_label, segment_id, valid_length, vocab_size)
            for l in ls:
                l.backward()
            
            trainer.step(1)
            
            running_mlm_loss += sum([l for l in ls_mlm])
            running_nsp_loss += sum([l for l in ls_nsp])

            if (step_num + 1) % (log_interval) == 0:
                total_mlm_loss += running_mlm_loss
                total_nsp_loss += running_nsp_loss
                begin_time = time.time()
                running_mlm_loss = running_nsp_loss = 0

        eval_end_time = time.time()
        if running_mlm_loss != 0:
            total_mlm_loss += running_mlm_loss
            total_nsp_loss += running_nsp_loss
        total_mlm_loss /= step_num
        total_nsp_loss /= step_num
        print('Eval mlm_loss={:.3f}\tnsp_loss={:.3f}\t'
                     .format(float(total_mlm_loss),
                             float(total_nsp_loss)))
        print('Eval cost={:.1f}s'.format(eval_end_time - eval_begin_time))
```

...

```{.python .input  n=19}
train_bert(train_iter, net, nsp_loss, mlm_loss, len(vocab), ctx, 20, 1)
```
