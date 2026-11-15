# GloVe išankstinis apmokymas
:label:`sec_GloVe_gluon`

Šiame skyriuje apmokysime
:numref:`sec_glove` apibrėžtą
GloVe modelį.

Pirmiausia importuokime
eksperimentui
reikalingus paketus ir modulius.

```{.python .input  n=1}
#@tab mxnet
from collections import defaultdict
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx, cpu
from mxnet.gluon import nn
import random

npx.set_np()
```

## Duomenų rinkinio parengimas
GloVe modelį apmokysime PTB duomenų rinkinyje.

Pirmiausia
nuskaitome PTB duomenų rinkinį, sudarome žodžių žodyną ir kiekvieną leksemą susiejame su
indeksu, kad sukonstruotume tekstyną.

```{.python .input  n=2}
#@tab mxnet
sentences = d2l.read_ptb()
vocab = d2l.Vocab(sentences, min_freq=10)
corpus = [vocab[line] for line in sentences]
```

### Bendro pasitaikymo skaičių sudarymas
Tegu žodžių poros bendro pasitaikymo skaičiai
žymimi $X$, kurios elementai $x_{ij}$ lentelėje pateikia, kiek kartų žodis $j$
pasitaiko žodžio $i$ kontekste.

Toliau apibrėžiame šią funkciją,
kuri išskiria visus centrinius tikslinius žodžius ir jų konteksto žodžius. Ji naudoja
mažėjančią svorio funkciją, kad žodžių poros, esančios $d$ žodžių atstumu,
prie bendro skaičiaus prisidėtų $1/d$. Tai vienas iš būdų atsižvelgti į tai,
kad labai nutolusių žodžių porose tikimasi mažiau svarbios informacijos
apie žodžių tarpusavio ryšį.

```{.python .input  n=3}
#@tab mxnet
def get_coocurrence_counts(corpus, window_size):
    centers, contexts = [], []
    cooccurence_counts = defaultdict(float)
    for line in corpus:
        # Each sentence needs at least 2 words to form a
        # "central target word - context word" pair
        if len(line) < 2:
            continue
        centers += line
        for i in range(len(line)):  # Context window centered at i
            left_indices = list(range(max(0, i - window_size), i))
            right_indices = list(range(i + 1,
                                       min(len(line), i + 1 + window_size)))
            left_context = [line[idx] for idx in left_indices]
            right_context = [line[idx] for idx in right_indices]
            for distance, word in enumerate(left_context[::-1]):
                cooccurence_counts[line[i], word] += 1 / (distance + 1)
            for distance, word in enumerate(right_context):
                cooccurence_counts[line[i], word] += 1 / (distance + 1)
    cooccurence_counts = [(word[0], word[1], count)
                          for word, count in cooccurence_counts.items()]
    return cooccurence_counts
```

Sukuriame dirbtinį duomenų rinkinį su dviem sakiniais, kuriuose atitinkamai yra 5 ir 2 žodžiai.
Tarkime, kad didžiausias konteksto langas yra 4. Tada išspausdiname
visų centrinių tikslinių žodžių ir konteksto žodžių
bendro pasitaikymo skaičius.

```{.python .input  n=4}
#@tab mxnet
tiny_dataset = [list(range(5)), list(range(5, 7))]
print('dataset', tiny_dataset)
for center, context, coocurrence in get_coocurrence_counts(tiny_dataset, 4):
        print('center: %s, context: %s, coocurrence: %.2f' %
          (center, context, coocurrence))
```

Didžiausią konteksto lango dydį nustatome 5. Toliau išskiriami visi
duomenų rinkinio centriniai tiksliniai žodžiai ir jų konteksto žodžiai bei apskaičiuojami jų
bendro pasitaikymo skaičiai

```{.python .input  n=5}
#@tab mxnet
coocurrence_matrix = get_coocurrence_counts(corpus, 5)
'# center-context pairs: %d' % len(coocurrence_matrix)
```

### Viskas kartu

Galiausiai apibrėžiame funkciją load_data_ptb_glove,
kuri nuskaito PTB duomenų rinkinį ir grąžina duomenų įkroviklį.

```{.python .input  n=16}
#@tab mxnet
def load_data_ptb_glove(batch_size, window_size):
    num_workers = d2l.get_dataloader_workers()
    sentences = d2l.read_ptb()
    vocab = d2l.Vocab(sentences, min_freq=5)
    corpus = [vocab[line] for line in sentences]
    coocurrence_matrix = get_coocurrence_counts(corpus, window_size)
    dataset = gluon.data.ArrayDataset(coocurrence_matrix)
    data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True,
                                      num_workers=num_workers)
    return data_iter, vocab

batch_size, window_size = 1024, 10
data_iter, vocab = load_data_ptb_glove(batch_size, window_size)
```

Išspausdinkime pirmąją duomenų iteratoriaus mažą imtį.

```{.python .input  n=17}
#@tab mxnet
names = ['center', 'context', 'Cooccurence']
for batch in data_iter:
    for name, data in zip(names, batch):
        print(name, 'shape:', data.shape)
    break
```

## GloVe modelis

15.1 skyrelyje pristatėme, kad GloVe tikslas yra
minimizuoti nuostolių funkciją.

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}}
h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j -
\log\,x_{ij}\right)^2.$$

GloVe modelį realizuosime realizuodami kiekvieną
nuostolių funkcijos dalį.

### Svorio funkcija

GloVe į nuostolių funkciją įvedė svorio
funkciją $h(x_{ij})$.

$$h(x_{ij})=\begin{cases}
(\frac{x}{x_{max}})^\alpha & x_{ij}<x_{max}\\
1 & otherwise
\end{cases}$$


Realizuojame
svorio funkciją $h(x_{ij})$. Kadangi $x_{ij}<x_{max}$
ekvivalentu $(\frac{x}{x_{max}})^\alpha < 1$, galime pateikti šią
realizaciją.

```{.python .input  n=18}
#@tab mxnet
def compute_weight(x, x_max = 30, alpha = 0.75):
    w = (x / x_max) ** alpha
    return np.minimum(w, 1)
```

Toliau išspausdinami visų centrinių tikslinių žodžių ir konteksto žodžių bendro pasitaikymo skaičių svoriai,
kai $x_{max}$ nustatytas 2, o $\alpha$ – 0,75

```{.python .input  n=19}
#@tab mxnet
for center, context, coocurrence in get_coocurrence_counts(tiny_dataset, 4)[:5]:
    print('center: %s, context: %s, coocurrence: %.2f, weight: %.2f' %
          (center, context, coocurrence, compute_weight(coocurrence, x_max = 2, alpha = 0.75)))
```

### Poslinkio narys

GloVe kiekvienam žodžiui $w_i$ turi du skaliarinius modelio parametrus:
poslinkio narius $b_i$ (centriniams tiksliniams žodžiams) ir $c_i$ (konteksto žodžiams).
Poslinkio narį galima realizuoti įterpimo sluoksniu. Įterpimo sluoksnio svoris
yra matrica, kurios eilučių skaičius lygus žodyno dydžiui (input_dim), o
stulpelių skaičius – vienas.

Žodyno dydį nustatome 20.

```{.python .input}
#@tab mxnet
embed_bias = nn.Embedding(input_dim=20, output_dim=1)
embed_bias.initialize()
embed_bias.weight
```

Įterpimo sluoksnio įvestis yra žodžio indeksas. Kai įvedame
žodžio indeksą $i$, įterpimo sluoksnis grąžina $i$-ąją svorio
reikšmių eilutę kaip jo poslinkio narį.

```{.python .input}
#@tab mxnet
x = np.array([1, 2, 3])
embed_bias(x)
```

### GloVe modelio tiesioginis skaičiavimas

Tiesioginio skaičiavimo metu GloVe modelio įvestį sudaro centrinio
tikslinio žodžio indeksas `center` ir konteksto žodžio
indeksas
`context`. Čia
kintamojo `center` forma yra (mažos imties
dydis, 1),
o kintamojo
`context` forma yra (mažos imties dydis,
1). Šie
du kintamieji
pirmiausia žodžių įterpimo
sluoksniu paverčiami iš žodžių indeksų į žodžių
vektorius.

```{.python .input  n=20}
#@tab mxnet
def GloVe(center, context, coocurrence, embed_v, embed_u,
          bias_v, bias_u, x_max, alpha):
    # Shape of v: (batch_size, embed_size)
    v = embed_v(center)
    # Shape of u: (batch_size, embed_size)
    u = embed_u(context)
    # Shape of b: (batch_size, )
    b = bias_v(center).squeeze()
    # Shape of c: (batch_size, )
    c = bias_u(context).squeeze()
    # Shape of embed_products: (batch_size,)
    embed_products = npx.batch_dot(np.expand_dims(v, 1),
                                   np.expand_dims(u, 2)).squeeze()
    # Shape of distance_expr: (batch_size,)
    distance_expr = np.power(embed_products + b +
                     c - np.log(coocurrence), 2)
    # Shape of weight: (batch_size,)
    weight = compute_weight(coocurrence)
    return weight * distance_expr
```

Patikrinkime, kad išvesties forma turėtų būti (mažos imties dydis, ).

```{.python .input  n=21}
#@tab mxnet
embed_word = nn.Embedding(input_dim=20, output_dim=4)
embed_word.initialize()
GloVe(np.ones((2)), np.ones((2)), np.ones((2)), embed_word, embed_word,
      embed_bias, embed_bias, x_max = 2, alpha = 0.75).shape
```

## Mokymas

Prieš apmokydami žodžių įterpimo modelį, turime apibrėžti
modelio nuostolių funkciją.

### Modelio parametrų inicializavimas
Sukonstruojame
žodžių įterpimo sluoksnius ir
papildomus poslinkius,
o hiperparametrą – žodžio
vektoriaus
dimensiją `embed_size` –
nustatome
100.

```{.python .input  n=22}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=1),
        nn.Embedding(input_dim=len(vocab), output_dim=1))
```

### Mokymas

Mokymo funkcija apibrėžta toliau.

```{.python .input  n=23}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, x_max, alpha, ctx=d2l.try_gpu()):
    net.initialize(ctx=ctx, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'AdaGrad',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[0, num_epochs])
    for epoch in range(num_epochs):
        timer = d2l.Timer()
        metric = d2l.Accumulator(2)  # loss_sum, num_tokens
        for i, batch in enumerate(data_iter):
            center, context, coocurrence = [
                data.as_in_context(ctx) for data in batch]
            with autograd.record():
                l = GloVe(center, context, coocurrence.astype('float32'),
                          net[0], net[1], net[2], net[3], x_max, alpha)
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i+1) % 50 == 0:
                animator.add(epoch+(i+1)/len(data_iter),
                             (metric[0]/metric[1],))
    print('loss %.3f, %d tokens/sec on %s ' % (
        metric[0]/metric[1], metric[1]/timer.stop(), ctx))
```

Dabar galime apmokyti GloVe modelį.

```{.python .input  n=12}
#@tab mxnet
lr, num_epochs = 0.1, 5
x_max, alpha = 100, 0.75
train(net, data_iter, lr, num_epochs, x_max, alpha)
```

## GloVe modelio taikymas

GloVe modelis sukuria du žodžių vektorių rinkinius,
`embed_v` ir `embed_u`. `embed_v` ir `embed_u` yra
ekvivalentūs ir skiriasi tik
dėl atsitiktinės inicializacijos; abu vektorių rinkiniai turėtų
veikti vienodai. Paprastai kaip
žodžių vektorius naudojame sumą `embed_v`+`embed_u`.


Apmokę GloVe modelį, žodžių reikšmių panašumą vis tiek galime išreikšti
pagal dviejų žodžių vektorių kosinusinį panašumą.

```{.python .input  n=13}
#@tab mxnet
def get_similar_tokens(query_token, k, embed_v, embed_u):
    W = embed_v.weight.data() + embed_u.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print('cosine sim=%.3f: %s' % (cos[i], (vocab.idx_to_token[i])))

get_similar_tokens('chip', 3, net[0], net[1])
```

## Santrauka

* Galime iš anksto apmokyti GloVe modelį.


## Pratimai



## Diskusijos
