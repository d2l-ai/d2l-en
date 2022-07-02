# Pré-entraînement de BERT
:label:`sec_bert-pretraining` 

 Avec le modèle BERT implémenté dans :numref:`sec_bert` 
 et les exemples de pré-entraînement générés à partir du jeu de données WikiText-2 dans :numref:`sec_bert-dataset` , nous allons pré-entraîner BERT sur le jeu de données WikiText-2 dans cette section.

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

Pour commencer, nous chargeons le jeu de données WikiText-2 sous forme de minibatchs
d'exemples de pré-entraînement pour la modélisation du langage masqué et la prédiction de la phrase suivante.
La taille du lot est de 512 et la longueur maximale d'une séquence d'entrée BERT est de 64.
Notez que dans le modèle BERT original, la longueur maximale est de 512.

```{.python .input}
#@tab all
batch_size, max_len = 512, 64
train_iter, vocab = d2l.load_data_wiki(batch_size, max_len)
```

## Préformation de BERT

Le BERT original a deux versions de modèles de tailles différentes :cite:`Devlin.Chang.Lee.ea.2018` .
Le modèle de base ($\text{BERT}_{\text{BASE}}$) utilise 12 couches (blocs de transformateurs-codeurs)
avec 768 unités cachées (taille cachée) et 12 têtes d'auto-attention.
Le grand modèle ($\text{BERT}_{\text{LARGE}}$) utilise 24 couches
avec 1024 unités cachées et 16 têtes d'auto-attention.
Notamment, le premier modèle a 110 millions de paramètres tandis que le second en a 340 millions.
Pour faciliter la démonstration,
nous définissons [**un petit BERT, utilisant 2 couches, 128 unités cachées, et 2 têtes d'auto-attention**].

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

Avant de définir la boucle d'apprentissage,
nous définissons une fonction d'aide `_get_batch_loss_bert`.
Étant donné le tesson d'exemples d'entraînement,
cette fonction [**calcule la perte pour les tâches de modélisation du langage masqué et de prédiction de la phrase suivante**].
Notez que la perte finale de la préformation de BERT
est juste la somme de la perte de modélisation du langage masqué
et de la perte de prédiction de la phrase suivante.

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

En invoquant les deux fonctions d'aide susmentionnées,
la fonction suivante `train_bert`
 définit la procédure pour [**prétraîner BERT (`net`) sur le jeu de données WikiText-2 (`train_iter`)**].
L'entraînement de BERT peut être très long.
Au lieu de spécifier le nombre d'époques pour l'entraînement
comme dans la fonction `train_ch13` (voir :numref:`sec_image_augmentation` ),
l'entrée `num_steps` de la fonction suivante
spécifie le nombre d'étapes d'itération pour l'entraînement.

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

Nous pouvons tracer à la fois la perte de modélisation du langage masqué et la perte de prédiction de la phrase suivante
pendant le pré-entraînement de BERT.

```{.python .input}
#@tab all
train_bert(train_iter, net, loss, len(vocab), devices, 50)
```

## [**Representing Text with BERT**]

Après le pré-entraînement de BERT,
nous pouvons l'utiliser pour représenter un texte unique, des paires de textes ou n'importe quel token.
La fonction suivante renvoie les représentations de BERT (`net`) pour tous les tokens
dans `tokens_a` et `tokens_b`.

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

[**Considérons la phrase "une grue est en train de voler".**]
Rappelons la représentation d'entrée de BERT telle que discutée dans :numref:`subsec_bert_input_rep` .
Après avoir inséré les tokens spéciaux "&lt;cls&gt;" (utilisés pour la classification)
et "&lt;sep&gt;" (utilisés pour la séparation),
la séquence d'entrée de BERT a une longueur de six.
Puisque zéro est l'index du token "&lt;cls&gt;",
`encoded_text[:, 0, :]` est la représentation BERT de la phrase d'entrée entière.
Pour évaluer le token polysémique "grue",
nous imprimons également les trois premiers éléments de la représentation BERT du token.

```{.python .input}
#@tab all
tokens_a = ['a', 'crane', 'is', 'flying']
encoded_text = get_bert_encoding(net, tokens_a)
# Tokens: '<cls>', 'a', 'crane', 'is', 'flying', '<sep>'
encoded_text_cls = encoded_text[:, 0, :]
encoded_text_crane = encoded_text[:, 2, :]
encoded_text.shape, encoded_text_cls.shape, encoded_text_crane[0][:3]
```

[**Considérons maintenant une paire de phrases
"un grutier est venu" et "il vient de partir".**]
De même, `encoded_pair[:, 0, :]` est le résultat encodé de la paire de phrases entière à partir du BERT pré-entraîné.
Notez que les trois premiers éléments du jeton polysémique "grue" sont différents de ceux qui sont codés lorsque le contexte est différent.
Cela confirme que les représentations de BERT sont sensibles au contexte.

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

Dans :numref:`chap_nlp_app` , nous affinerons un modèle BERT pré-entraîné
pour des applications de traitement du langage naturel en aval.


## Résumé

* Le BERT original a deux versions, où le modèle de base a 110 millions de paramètres et le grand modèle a 340 millions de paramètres.
* Après le pré-entraînement de BERT, nous pouvons l'utiliser pour représenter un texte unique, des paires de textes ou n'importe quel token dans ces textes.
* Dans l'expérience, le même token a une représentation BERT différente lorsque leurs contextes sont différents. Cela confirme que les représentations BERT sont sensibles au contexte.


## Exercices

1. Dans l'expérience, nous pouvons voir que la perte de modélisation du langage masqué est significativement plus élevée que la perte de prédiction de la phrase suivante. Pourquoi ?
2. Fixez la longueur maximale d'une séquence d'entrée BERT à 512 (comme le modèle BERT original). Utilisez les configurations du modèle BERT original telles que $\text{BERT}_{\text{LARGE}}$. Rencontrez-vous des erreurs en exécutant cette section ? Pourquoi ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/390)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1497)
:end_tab:
