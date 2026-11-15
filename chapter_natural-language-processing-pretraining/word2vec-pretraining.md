# word2vec išankstinis apmokymas
:label:`sec_word2vec_pretraining`


Toliau įgyvendiname skip-gram
modelį, apibrėžtą
:numref:`sec_word2vec`.
Tada
iš anksto apmokysime word2vec naudodami neigiamų pavyzdžių atranką
PTB duomenų rinkinyje.
Pirmiausia
gaukime duomenų iteratorių
ir šio duomenų rinkinio žodyną
iškviesdami funkciją `d2l.load_data_ptb`,
kuri buvo aprašyta :numref:`sec_word2vec_data`

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
npx.set_np()

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn

batch_size, max_window_size, num_noise_words = 512, 5, 5
data_iter, vocab = d2l.load_data_ptb(batch_size, max_window_size,
                                     num_noise_words)
```

## Skip-gram modelis

Skip-gram modelį įgyvendiname
naudodami įterpinių sluoksnius ir rinkinio matricų daugybą.
Pirmiausia prisiminkime,
kaip veikia įterpinių sluoksniai.


### Įterpinių sluoksnis

Kaip aprašyta :numref:`sec_seq2seq`,
įterpinių sluoksnis
atvaizduoja teksto vieneto indeksą į jo požymių vektorių.
Šio sluoksnio svoris
yra matrica, kurios eilučių skaičius lygus
žodyno dydžiui (`input_dim`), o
stulpelių skaičius lygus
kiekvieno teksto vieneto vektoriaus dimensijai (`output_dim`).
Apmokius žodžių įterpinių modelį,
būtent šio svorio mums ir reikia.

```{.python .input}
#@tab mxnet
embed = nn.Embedding(input_dim=20, output_dim=4)
embed.initialize()
embed.weight
```

```{.python .input}
#@tab pytorch
embed = nn.Embedding(num_embeddings=20, embedding_dim=4)
print(f'Parameter embedding_weight ({embed.weight.shape}, '
      f'dtype={embed.weight.dtype})')
```

Įterpinių sluoksnio įvestis yra
teksto vieneto (žodžio) indeksas.
Bet kuriam teksto vieneto indeksui $i$
jo vektorinį vaizdavimą
galima gauti iš
įterpinių sluoksnio svorių matricos
$i$-osios eilutės.
Kadangi vektoriaus dimensija (`output_dim`)
buvo nustatyta 4,
įterpinių sluoksnis
grąžina vektorius, kurių forma (2, 3, 4),
teksto vienetų indeksų minirinkiniui, kurio forma
(2, 3).

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Tiesioginio sklidimo apibrėžimas

Tiesioginiame sklidime
skip-gram modelio įvestis
apima
centrinių žodžių indeksus `center`,
kurių forma (rinkinio dydis, 1),
ir
sukonkatenuotus konteksto ir triukšmo žodžių indeksus `contexts_and_negatives`,
kurių forma (rinkinio dydis, `max_len`),
kur `max_len`
apibrėžta
:numref:`subsec_word2vec-minibatch-loading`.
Šie du kintamieji pirmiausia paverčiami iš
teksto vienetų indeksų į vektorius per įterpinių sluoksnį,
tada jų rinkinio matricų daugyba
(aprašyta :numref:`subsec_batch_dot`)
grąžina
išvestį, kurios forma (rinkinio dydis, 1, `max_len`).
Kiekvienas išvesties elementas yra
centrinio žodžio vektoriaus ir konteksto ar triukšmo žodžio vektoriaus skaliarinė sandauga.

```{.python .input}
#@tab mxnet
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = npx.batch_dot(v, u.swapaxes(1, 2))
    return pred
```

```{.python .input}
#@tab pytorch
def skip_gram(center, contexts_and_negatives, embed_v, embed_u):
    v = embed_v(center)
    u = embed_u(contexts_and_negatives)
    pred = torch.bmm(v, u.permute(0, 2, 1))
    return pred
```

Atspausdinkime šios funkcijos `skip_gram` išvesties formą kelioms pavyzdinėms įvestims.

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Apmokymas

Prieš apmokydami skip-gram modelį su neigiamų pavyzdžių atranka,
pirmiausia apibrėžkime jo nuostolių funkciją.


### Dvejetainės kryžminės entropijos nuostolis

Pagal neigiamų pavyzdžių atrankos nuostolių funkcijos apibrėžimą
:numref:`subsec_negative-sampling`,
naudosime
dvejetainės kryžminės entropijos nuostolį.

```{.python .input}
#@tab mxnet
loss = gluon.loss.SigmoidBCELoss()
```

```{.python .input}
#@tab pytorch
class SigmoidBCELoss(nn.Module):
    # Binary cross-entropy loss with masking
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target, mask=None):
        out = nn.functional.binary_cross_entropy_with_logits(
            inputs, target, weight=mask, reduction="none")
        return out.mean(dim=1)

loss = SigmoidBCELoss()
```

Prisiminkite mūsų aprašymus
apie kaukės kintamąjį
ir žymenų kintamąjį
:numref:`subsec_word2vec-minibatch-loading`.
Toliau pateiktas kodas apskaičiuoja
dvejetainės kryžminės entropijos nuostolį
duotiems kintamiesiems.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Toliau parodoma,
kaip aukščiau pateikti rezultatai apskaičiuojami
(mažiau efektyviu būdu)
naudojant
sigmoidinę aktyvinimo funkciją
dvejetainės kryžminės entropijos nuostolyje.
Dvi išvestis galime laikyti
dviem normuotais nuostoliais,
kurie vidurkinami per nemaskuotas prognozes.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Modelio parametrų inicializavimas

Apibrėžiame du įterpinių sluoksnius
visiems žodyno žodžiams,
kai jie naudojami atitinkamai kaip centriniai žodžiai
ir konteksto žodžiai.
Žodžių vektoriaus dimensija
`embed_size` nustatoma 100.

```{.python .input}
#@tab mxnet
embed_size = 100
net = nn.Sequential()
net.add(nn.Embedding(input_dim=len(vocab), output_dim=embed_size),
        nn.Embedding(input_dim=len(vocab), output_dim=embed_size))
```

```{.python .input}
#@tab pytorch
embed_size = 100
net = nn.Sequential(nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size),
                    nn.Embedding(num_embeddings=len(vocab),
                                 embedding_dim=embed_size))
```

### Apmokymo ciklo apibrėžimas

Apmokymo ciklas apibrėžiamas toliau. Dėl papildymų nuostolių funkcijos skaičiavimas šiek tiek skiriasi nuo ankstesnių apmokymo funkcijų.

```{.python .input}
#@tab mxnet
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    net.initialize(ctx=device, force_reinit=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': lr})
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            center, context_negative, mask, label = [
                data.as_in_ctx(device) for data in batch]
            with autograd.record():
                pred = skip_gram(center, context_negative, net[0], net[1])
                l = (loss(pred.reshape(label.shape), label, mask) *
                     mask.shape[1] / mask.sum(axis=1))
            l.backward()
            trainer.step(batch_size)
            metric.add(l.sum(), l.size)
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

```{.python .input}
#@tab pytorch
def train(net, data_iter, lr, num_epochs, device=d2l.try_gpu()):
    def init_weights(module):
        if type(module) == nn.Embedding:
            nn.init.xavier_uniform_(module.weight)
    net.apply(init_weights)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[1, num_epochs])
    # Sum of normalized losses, no. of normalized losses
    metric = d2l.Accumulator(2)
    for epoch in range(num_epochs):
        timer, num_batches = d2l.Timer(), len(data_iter)
        for i, batch in enumerate(data_iter):
            optimizer.zero_grad()
            center, context_negative, mask, label = [
                data.to(device) for data in batch]

            pred = skip_gram(center, context_negative, net[0], net[1])
            l = (loss(pred.reshape(label.shape).float(), label.float(), mask)
                     / mask.sum(axis=1) * mask.shape[1])
            l.sum().backward()
            optimizer.step()
            metric.add(l.sum(), l.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[1],))
    print(f'loss {metric[0] / metric[1]:.3f}, '
          f'{metric[1] / timer.stop():.1f} tokens/sec on {str(device)}')
```

Dabar galime apmokyti skip-gram modelį naudodami neigiamų pavyzdžių atranką.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Žodžių įterpinių taikymas
:label:`subsec_apply-word-embed`


Apmokę word2vec modelį,
galime naudoti apmokyto modelio
žodžių vektorių kosinuso panašumą,
kad
rastume žodyno žodžius,
semantiškai panašiausius
į įvesties žodį.

```{.python .input}
#@tab mxnet
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data()
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = np.dot(W, x) / np.sqrt(np.sum(W * W, axis=1) * np.sum(x * x) + 1e-9)
    topk = npx.topk(cos, k=k+1, ret_typ='indices').asnumpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

```{.python .input}
#@tab pytorch
def get_similar_tokens(query_token, k, embed):
    W = embed.weight.data
    x = W[vocab[query_token]]
    # Compute the cosine similarity. Add 1e-9 for numerical stability
    cos = torch.mv(W, x) / torch.sqrt(torch.sum(W * W, dim=1) *
                                      torch.sum(x * x) + 1e-9)
    topk = torch.topk(cos, k=k+1)[1].cpu().numpy().astype('int32')
    for i in topk[1:]:  # Remove the input words
        print(f'cosine sim={float(cos[i]):.3f}: {vocab.to_tokens(i)}')

get_similar_tokens('chip', 3, net[0])
```

## Santrauka

* Skip-gram modelį su neigiamų pavyzdžių atranka galime apmokyti naudodami įterpinių sluoksnius ir dvejetainės kryžminės entropijos nuostolį.
* Žodžių įterpinių taikymai apima semantiškai panašių žodžių radimą duotam žodžiui pagal žodžių vektorių kosinuso panašumą.


## Pratimai

1. Naudodami apmokytą modelį, raskite semantiškai panašių žodžių kitiems įvesties žodžiams. Ar galite pagerinti rezultatus derindami hiperparametrus?
1. Kai apmokymo tekstynas yra milžiniškas, centriniams dabartinio minirinkinio žodžiams dažnai atrenkame konteksto ir triukšmo žodžius *atnaujindami modelio parametrus*. Kitaip tariant, tas pats centrinis žodis skirtingose apmokymo epochose gali turėti skirtingus konteksto ar triukšmo žodžius. Kokie šio metodo privalumai? Pabandykite įgyvendinti šį apmokymo metodą.

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1335)
:end_tab:
