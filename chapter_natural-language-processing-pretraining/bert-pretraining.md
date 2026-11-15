# BERT išankstinis apmokymas
:label:`sec_bert-pretraining`

Turėdami BERT modelį, įgyvendintą :numref:`sec_bert`,
ir išankstinio apmokymo pavyzdžius, sugeneruotus iš WikiText-2 duomenų rinkinio :numref:`sec_bert-dataset`, šiame skyrelyje iš anksto apmokysime BERT WikiText-2 duomenų rinkinyje.

```{.python .input}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, init, np, npx

npx.set_np()
```

```{.python .input}
#@tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
```

Pradėkime nuo WikiText-2 duomenų rinkinio įkėlimo kaip minirinkinių
išankstinio apmokymo pavyzdžių užmaskuotam kalbos modeliavimui ir kito sakinio prognozavimui.
Rinkinio dydis yra 512, o didžiausias BERT įvesties sekos ilgis yra 64.
Atkreipkite dėmesį, kad originaliame BERT modelyje didžiausias ilgis yra 512.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## BERT išankstinis apmokymas

Originalus BERT turi dvi skirtingų modelio dydžių versijas :cite:`Devlin.Chang.Lee.ea.2018`.
Bazinis modelis ($\textrm{BERT}_{\textrm{BASE}}$) naudoja 12 sluoksnių (transformerio kodavimo blokų)
su 768 paslėptaisiais vienetais (paslėptuoju dydžiu) ir 12 saviraiškės dėmesio sutelkimo transformacijos vienetų.
Didysis modelis ($\textrm{BERT}_{\textrm{LARGE}}$) naudoja 24 sluoksnius
su 1024 paslėptaisiais vienetais ir 16 saviraiškės dėmesio sutelkimo transformacijos vienetų.
Pažymėtina, kad pirmasis turi 110 milijonų parametrų, o antrasis – 340 milijonų parametrų.
Kad demonstravimas būtų paprastas,
apibrėžiame [**mažą BERT, naudojantį 2 sluoksnius, 128 paslėptuosius vienetus ir 2 saviraiškės dėmesio sutelkimo transformacijos vienetus**].

```{.python .input}
#@tab mxnet
net = d2l.BERTModel(len(vocab), num_hiddens=128, ffn_num_hiddens=256,
                    num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
net.initialize(init.Xavier(), ctx=devices)
loss = gluon.loss.SoftmaxCELoss()
```

```{.python .input}
#@tab pytorch
net = d2l.BERTModel(len(vocab), num_hiddens=128, 
                    ffn_num_hiddens=256, num_heads=2, num_blks=2, dropout=0.2)
devices = d2l.try_all_gpus()
loss = nn.CrossEntropyLoss()
```

Prieš apibrėždami apmokymo ciklą,
apibrėžiame pagalbinę funkciją `_get_batch_loss_bert`.
Turint apmokymo pavyzdžių fragmentą,
ši funkcija [**apskaičiuoja ir užmaskuoto kalbos modeliavimo, ir kito sakinio prognozavimo užduočių nuostolį**].
Atkreipkite dėmesį, kad galutinis BERT išankstinio apmokymo nuostolis
yra tiesiog užmaskuoto kalbos modeliavimo nuostolio
ir kito sakinio prognozavimo nuostolio suma.

```{.python .input}
#@tab mxnet
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X_shards,
                         segments_X_shards, valid_lens_x_shards,
                         pred_positions_X_shards, mlm_weights_X_shards,
                         mlm_Y_shards, nsp_y_shards):
    mlm_ls, nsp_ls, ls = [], [], []
    for (tokens_X_shard, segments_X_shard, valid_lens_x_shard,
         pred_positions_X_shard, mlm_weights_X_shard, mlm_Y_shard,
         nsp_y_shard) in zip(
        tokens_X_shards, segments_X_shards, valid_lens_x_shards,
        pred_positions_X_shards, mlm_weights_X_shards, mlm_Y_shards,
        nsp_y_shards):
        # Forward pass
        _, mlm_Y_hat, nsp_Y_hat = net(
            tokens_X_shard, segments_X_shard, valid_lens_x_shard.reshape(-1),
            pred_positions_X_shard)
        # Compute masked language model loss
        mlm_l = loss(
            mlm_Y_hat.reshape((-1, vocab_size)), mlm_Y_shard.reshape(-1),
            mlm_weights_X_shard.reshape((-1, 1)))
        mlm_l = mlm_l.sum() / (mlm_weights_X_shard.sum() + 1e-8)
        # Compute next sentence prediction loss
        nsp_l = loss(nsp_Y_hat, nsp_y_shard)
        nsp_l = nsp_l.mean()
        mlm_ls.append(mlm_l)
        nsp_ls.append(nsp_l)
        ls.append(mlm_l + nsp_l)
        npx.waitall()
    return mlm_ls, nsp_ls, ls
```

```{.python .input}
#@tab pytorch
#@save
def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         segments_X, valid_lens_x,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y, nsp_y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = net(tokens_X, segments_X,
                                  valid_lens_x.reshape(-1),
                                  pred_positions_X)
    # Compute masked language model loss
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
    mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)
    # Compute next sentence prediction loss
    nsp_l = loss(nsp_Y_hat, nsp_y)
    l = mlm_l + nsp_l
    return mlm_l, nsp_l, l
```

Iškviesdama dvi aukščiau minėtas pagalbines funkcijas,
toliau pateikta funkcija `train_bert`
apibrėžia procedūrą [**BERT (`net`) iš anksto apmokyti WikiText-2 (`train_iter`) duomenų rinkinyje**].
BERT apmokymas gali trukti labai ilgai.
Užuot nurodžius apmokymo epochų skaičių,
kaip funkcijoje `train_ch13` (žr. :numref:`sec_image_augmentation`),
toliau pateiktos funkcijos įvestis `num_steps`
nurodo apmokymo iteracijos žingsnių skaičių.

```{.python .input}
#@tab mxnet
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': 0.01})
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for batch in train_iter:
            (tokens_X_shards, segments_X_shards, valid_lens_x_shards,
             pred_positions_X_shards, mlm_weights_X_shards,
             mlm_Y_shards, nsp_y_shards) = [gluon.utils.split_and_load(
                elem, devices, even_split=False) for elem in batch]
            timer.start()
            with autograd.record():
                mlm_ls, nsp_ls, ls = _get_batch_loss_bert(
                    net, loss, vocab_size, tokens_X_shards, segments_X_shards,
                    valid_lens_x_shards, pred_positions_X_shards,
                    mlm_weights_X_shards, mlm_Y_shards, nsp_y_shards)
            for l in ls:
                l.backward()
            trainer.step(1)
            mlm_l_mean = sum([float(l) for l in mlm_ls]) / len(mlm_ls)
            nsp_l_mean = sum([float(l) for l in nsp_ls]) / len(nsp_ls)
            metric.add(mlm_l_mean, nsp_l_mean, batch[0].shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

```{.python .input}
#@tab pytorch
def train_bert(train_iter, net, loss, vocab_size, devices, num_steps):
    net(*next(iter(train_iter))[:4])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    trainer = torch.optim.Adam(net.parameters(), lr=0.01)
    step, timer = 0, d2l.Timer()
    animator = d2l.Animator(xlabel='step', ylabel='loss',
                            xlim=[1, num_steps], legend=['mlm', 'nsp'])
    # Sum of masked language modeling losses, sum of next sentence prediction
    # losses, no. of sentence pairs, count
    metric = d2l.Accumulator(4)
    num_steps_reached = False
    while step < num_steps and not num_steps_reached:
        for tokens_X, segments_X, valid_lens_x, pred_positions_X,\
            mlm_weights_X, mlm_Y, nsp_y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            valid_lens_x = valid_lens_x.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_y = mlm_Y.to(devices[0]), nsp_y.to(devices[0])
            trainer.zero_grad()
            timer.start()
            mlm_l, nsp_l, l = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y)
            l.backward()
            trainer.step()
            metric.add(mlm_l, nsp_l, tokens_X.shape[0], 1)
            timer.stop()
            animator.add(step + 1,
                         (metric[0] / metric[3], metric[1] / metric[3]))
            step += 1
            if step == num_steps:
                num_steps_reached = True
                break

    print(f'MLM loss {metric[0] / metric[3]:.3f}, '
          f'NSP loss {metric[1] / metric[3]:.3f}')
    print(f'{metric[2] / timer.sum():.1f} sentence pairs/sec on '
          f'{str(devices)}')
```

BERT išankstinio apmokymo metu galime nubraižyti ir užmaskuoto kalbos modeliavimo nuostolį, ir kito sakinio prognozavimo nuostolį.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## [**Teksto vaizdavimas naudojant BERT**]

Iš anksto apmokę BERT,
galime jį naudoti vienam tekstui, tekstų poroms ar bet kuriam jų teksto vienetui vaizduoti.
Toliau pateikta funkcija grąžina BERT (`net`) vaizdavimus visiems teksto vienetams
`tokens_a` ir `tokens_b`.

```{.python .input}
#@tab mxnet
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = np.expand_dims(np.array(vocab[tokens], ctx=devices[0]),
                               axis=0)
    segments = np.expand_dims(np.array(segments, ctx=devices[0]), axis=0)
    valid_len = np.expand_dims(np.array(len(tokens), ctx=devices[0]), axis=0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

```{.python .input}
#@tab pytorch
def get_bert_encoding(net, tokens_a, tokens_b=None):
    tokens, segments = d2l.get_tokens_and_segments(tokens_a, tokens_b)
    token_ids = torch.tensor(vocab[tokens], device=devices[0]).unsqueeze(0)
    segments = torch.tensor(segments, device=devices[0]).unsqueeze(0)
    valid_len = torch.tensor(len(tokens), device=devices[0]).unsqueeze(0)
    encoded_X, _, _ = net(token_ids, segments, valid_len)
    return encoded_X
```

[**Panagrinėkime sakinį „a crane is flying“.**]
Prisiminkite BERT įvesties vaizdavimą, aptartą :numref:`subsec_bert_input_rep`.
Įterpus specialius teksto vienetus „&lt;cls&gt;“ (naudojamą klasifikavimui)
ir „&lt;sep&gt;“ (naudojamą atskyrimui),
BERT įvesties sekos ilgis yra šeši.
Kadangi nulis yra teksto vieneto „&lt;cls&gt;“ indeksas,
`encoded_text[:, 0, :]` yra viso įvesties sakinio BERT vaizdavimas.
Kad įvertintume daugiareikšmį teksto vienetą „crane“,
taip pat išspausdiname pirmuosius tris šio teksto vieneto BERT vaizdavimo elementus.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

[**Dabar panagrinėkime sakinių porą
„a crane driver came“ ir „he just left“.**]
Panašiai `encoded_pair[:, 0, :]` yra viso sakinių poros užkoduotas rezultatas iš iš anksto apmokyto BERT.
Atkreipkite dėmesį, kad pirmieji trys daugiareikšmio teksto vieneto „crane“ elementai skiriasi nuo tų, kai kontekstas yra kitoks.
Tai patvirtina, kad BERT vaizdavimai priklauso nuo konteksto.

```{.python .input}
#@tab all
tokens_a, tokens_b = ['a', 'crane', 'driver', 'came'], ['he', 'just', 'left']
encoded_pair = get_bert_encoding(net, tokens_a, tokens_b)
# Tokens: '<cls>', 'a', 'crane', 'driver', 'came', '<sep>', 'he', 'just',
# 'left', '<sep>'
encoded_pair_cls = encoded_pair[:, 0, :]
encoded_pair_crane = encoded_pair[:, 2, :]
encoded_pair.shape, encoded_pair_cls.shape, encoded_pair_crane[0][:3]
```

:numref:`chap_nlp_app` patikslinsime iš anksto apmokytą BERT modelį
tolesniems natūraliosios kalbos apdorojimo taikymams.

## Santrauka

* Originalus BERT turi dvi versijas: bazinis modelis turi 110 milijonų parametrų, o didysis modelis – 340 milijonų parametrų.
* Iš anksto apmokę BERT, galime jį naudoti vienam tekstui, tekstų poroms ar bet kuriam jų teksto vienetui vaizduoti.
* Eksperimente tas pats teksto vienetas turi skirtingą BERT vaizdavimą, kai jų kontekstai skiriasi. Tai patvirtina, kad BERT vaizdavimai priklauso nuo konteksto.


## Pratimai

1. Eksperimente matome, kad užmaskuoto kalbos modeliavimo nuostolis yra gerokai didesnis už kito sakinio prognozavimo nuostolį. Kodėl?
2. Nustatykite didžiausią BERT įvesties sekos ilgį 512 (tokį patį kaip originaliame BERT modelyje). Naudokite originalaus BERT modelio konfigūracijas, pavyzdžiui, $\textrm{BERT}_{\textrm{LARGE}}$. Ar paleisdami šį skyrelį susiduriate su kokia nors klaida? Kodėl?

:begin_tab:`mxnet`
[Diskusijos](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Diskusijos](https://discuss.d2l.ai/t/1497)
:end_tab:
