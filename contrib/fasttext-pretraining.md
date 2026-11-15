# fastText išankstinis apmokymas
:label:`sec_word2vec_gluon`

Šiame skyriuje
apmokysime :numref:`sec_word2vec` apibrėžtą skip-gram modelį.

Pirmiausia importuokime
eksperimentui reikalingus paketus ir modulius ir įkelkime PTB duomenų rinkinį.

```{.python .input  n=1}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from functools import partial
from mxnet import autograd, gluon, init, np, npx, cpu
from mxnet.gluon import nn
import random

npx.set_np()
```

```{.python .input  n=2}
#@tab mxnet
def compute_subword(token):
    if token[0] != '<' and token[-1] != '>':
        token = '<' + token + '>'
        subwords = {token}
        for i in range(len(token)-3):
            for j in range(i + 3, len(token)+1):
                if j - i <= 6:
                    subwords.add(token[i:j])
        return subwords
    else:
        return [token]
```

```{.python .input  n=3}
#@tab mxnet
def get_subword_map(vocab):
    tokenid_to_subword, subword_to_idx = defaultdict(list), defaultdict(int)
    for token, tokenid in vocab.token_to_idx.items():
        subwords = compute_subword(token)
        for subword in subwords:
            if subword not in subword_to_idx:
                subword_to_idx[subword] = len(subword_to_idx)
            tokenid_to_subword[tokenid].append(subword_to_idx[subword])
    return tokenid_to_subword, subword_to_idx
```

```{.python .input  n=4}
#@tab mxnet
def token_transform(tokens, vocab, subword_map):
    if not isinstance(tokens, (list, tuple)):
        return d2l.truncate_pad(subword_map[tokens],
                                 64, vocab['<pad>'])
    return [token_transform(token, vocab, subword_map) for token in tokens]
```

```{.python .input  n=5}
#@tab mxnet
def batchify(data, vocab, subword_map):
    max_len = max(len(c) + len(n) for _, c, n in data)
    centers, contexts_negatives, masks, labels = [], [], [], []
    for center, context, negative in data:
        cur_len = len(context) + len(negative)
        centers += [token_transform([center], vocab, subword_map)]
        contexts_negatives += [token_transform(context + negative + \
                               [1] * (max_len - cur_len), vocab, subword_map)]
        masks += [[1] * cur_len + [0] * (max_len - cur_len)]
        labels += [[1] * len(context) + [0] * (max_len - len(context))]
    return (np.array(centers), np.array(contexts_negatives),
            np.array(masks), np.array(labels))
```

```{.python .input  n=6}
#@tab mxnet
def load_data_ptb(batch_size, max_window_size, num_noise_words):
    num_workers = d2l.get_dataloader_workers()
    sentences = d2l.read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=10, reserved_tokens=['<pad>'])
    subsampled = d2l.subsampling(sentences, vocab)
    corpus = [vocab[line] for line in subsampled]
    all_centers, all_contexts = d2l.get_centers_and_contexts(
        corpus, max_window_size)
    all_negatives = d2l.get_negatives(all_contexts, corpus, num_noise_words)
    dataset = gluon.data.ArrayDataset(
        all_centers, all_contexts, all_negatives)
    subword_map, subword_to_idx = get_subword_map(vocab)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      batchify_fn=partial(batchify, vocab=vocab, subword_map=subword_map),
                                      num_workers=num_workers)
    return data_iter, vocab, subword_to_idx
```

```{.python .input  n=7}
#@tab mxnet
batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab, subword_to_idx = load_data_ptb(batch_size, max_window_size, num_noise_words)
```

```{.python .input  n=8}
#@tab mxnet
names = ['centers', 'contexts_negatives', 'masks', 'labels']
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## Skip-gram modelis

Skip-gram modelį realizuosime naudodami įterpimo
sluoksnius ir mažų imčių (angl. *minibatch*) daugybą. Šie metodai taip pat dažnai naudojami
kitoms natūraliosios kalbos apdorojimo taikymo sritims realizuoti.

### Įterpimo sluoksnis
Sluoksnis, kuriame gautas žodis įterpiamas, vadinamas įterpimo sluoksniu;
jį galima gauti sukūrus `nn.Embedding` objektą Gluon. Įterpimo sluoksnio
svoris yra matrica, kurios eilučių skaičius lygus žodyno
dydžiui (`input_dim`), o stulpelių skaičius – kiekvieno žodžio
vektoriaus dimensijai (`output_dim`). Žodyno dydį nustatome $20$, o žodžio vektoriaus
dimensiją – $4$.

```{.python .input  n=9}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

### Skip-gram modelio tiesioginis skaičiavimas

Tiesioginio skaičiavimo metu skip-gram modelio įvestį sudaro centrinio tikslinio žodžio indeksas `center` ir
sujungti konteksto bei triukšmo žodžių indeksai `contexts_and_negatives`. Čia
kintamojo `center` forma yra (mažos imties dydis, 1), o
kintamojo `contexts_and_negatives` forma yra (mažos imties dydis, `max_len`). Šie
du kintamieji pirmiausia žodžių įterpimo sluoksniu paverčiami iš žodžių indeksų į žodžių vektorius, o tada
mažų imčių daugyba gaunama (mažos imties dydis, 1, `max_len`) formos išvestis. Kiekvienas išvesties elementas yra
centrinio tikslinio žodžio vektoriaus ir konteksto žodžio vektoriaus arba triukšmo
žodžio vektoriaus skaliarinė sandauga.

```{.python .input  n=10}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u, padding):
    v_embedding = embed_v(center)
    v_mask = (center!=padding).astype('float32')
    v = (v_embedding * np.expand_dims(v_mask, axis=-1)).sum(-1)/(np.expand_dims(v_mask.sum(-1), axis=-1)+1e-5)
    u_embedding = embed_u(contexts_and_negatives)
    u_mask = (contexts_and_negatives!=padding).astype('float32')
    u = (u_embedding * np.expand_dims(u_mask, axis=-1)).sum(-1)/(np.expand_dims(u_mask.sum(-1), axis=-1)+1e-5)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

Patikrinkime, kad išvesties forma turėtų būti (mažos imties dydis, 1, `max_len`).

```{.python .input  n=12}
#@tab mxnet
skip_gram(np.ones((2, 1, 64)), np.ones((2, 6, 64)), embed, embed, vocab['<pad>']).shape
```

## Mokymas

Prieš apmokydami žodžių įterpimo modelį, turime apibrėžti modelio
nuostolių funkciją.

### Dvejetainės kryžminės entropijos nuostolių funkcija

Pagal
nuostolių funkcijos apibrėžimą neigiamame atrinkime, galime tiesiogiai naudoti
Gluon dvejetainės kryžminės entropijos nuostolių funkciją `SigmoidBinaryCrossEntropyLoss`.

```{.python .input  n=13}
#@tab mxnet
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
```

Verta paminėti, kad kaukės kintamąjį galime naudoti, norėdami nurodyti, kuri dalis
prognozuojamų reikšmių ir žymių dalyvauja skaičiuojant nuostolių funkciją mažoje imtyje: kai kaukė lygi 1, atitinkamos pozicijos
prognozuojama reikšmė ir žymė dalyvauja skaičiuojant nuostolių funkciją;
kai kaukė lygi 0, atitinkamos pozicijos prognozuojama reikšmė ir žymė
nedalyvauja skaičiuojant nuostolių funkciją. Kaip minėjome
anksčiau, kaukės kintamuosius galima naudoti siekiant išvengti užpildymo (angl. *padding*) įtakos nuostolių
funkcijos skaičiavimams.

Turint du identiškus pavyzdžius, skirtingos kaukės lemia
skirtingas nuostolių reikšmes.

```{.python .input  n=14}
#@tab mxnet
pred = np.array([[.5]*4]*2)
label = np.array([[1, 0, 1, 0]]*2)
mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask)
```

Kadangi pavyzdžių ilgiai skiriasi, kiekvieno pavyzdžio nuostolius galime normalizuoti.

```{.python .input  n=15}
#@tab mxnet
loss(pred, label, mask) / mask.sum(axis=1) * mask.shape[1]
```

### Modelio parametrų inicializavimas

Atitinkamai sukonstruojame centrinių ir konteksto žodžių įterpimo sluoksnius ir hiperparametrą – žodžio vektoriaus
dimensiją `embed_size` – nustatome 100.

```{.python .input  n=16}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(subword_to_idx), output_dim=embed_size),
        nn.Embedding(input_dim=len(subword_to_idx), output_dim=embed_size))
```

### Mokymas

Mokymo funkcija apibrėžta toliau. Dėl užpildymo
nuostolių funkcijos skaičiavimas šiek tiek skiriasi nuo
ankstesnių mokymo funkcijų.

```{.python .input  n=17}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(ctx) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1], vocab['<pad>'])
                l = (loss(pred.reshape(label.shape), label, mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter),
                             (metric[0]/metric[1],))
            npx.waitall()
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

Dabar galime apmokyti skip-gram modelį naudodami neigiamą atrinkimą.

```{.python .input  n=20}
#@tab mxnet
lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs)
```

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed, vocab, subword_to_idx):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    all_v = []
    for token in vocab.idx_to_token:
        subword = compute_subword(token)
        w_v = W[[subword_to_idx[s] for s in subword]].sum(0)
        all_v.append(np.expand_dims(w_v, 0))
    all_v = np.concatenate(all_v, 0)
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(all_v, x) / np.sqrt(np.sum(all_v * all_v, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print('cosine sim=%.3f: %s' % (cos[i], (vocab.idx_to_token[i])))
```

```{.python .input}
#@tab mxnet
get_similar_tokens('chip', 3, net[0], vocab, subword_to_idx)
```
