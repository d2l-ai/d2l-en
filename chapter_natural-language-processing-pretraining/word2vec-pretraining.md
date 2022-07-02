# Pretraining word2vec
:label:`sec_word2vec_pretraining` 

 
 Nous poursuivons en implémentant le modèle de saut de programme
défini dans
:numref:`sec_word2vec` .
Ensuite,
nous allons pré-entraîner word2vec en utilisant l'échantillonnage négatif
sur le jeu de données PTB.
Tout d'abord,
nous obtenons l'itérateur de données
et le vocabulaire pour cet ensemble de données
en appelant la fonction `d2l.load_data_ptb`
 , qui a été décrite en :numref:`sec_word2vec_data`

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

## Le modèle de saut de gramme

Nous implémentons le modèle de saut de gramme
en utilisant des couches d'intégration et des multiplications matricielles par lots.
Tout d'abord, examinons
le fonctionnement des couches d'intégration.


### Couche d'incorporation

Comme décrit dans :numref:`sec_seq2seq` ,
une couche d'incorporation
fait correspondre l'index d'un token à son vecteur de caractéristiques.
Le poids de cette couche
est une matrice dont le nombre de lignes est égal à
, la taille du dictionnaire (`input_dim`) et
, le nombre de colonnes est égal à
, la dimension du vecteur pour chaque token (`output_dim`).
Après l'entraînement d'un modèle d'intégration de mots,
est le poids dont nous avons besoin.

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

L'entrée d'une couche d'incorporation est l'index
d'un token (mot).
Pour tout indice de token $i$,
sa représentation vectorielle
peut être obtenue à partir de
la ligne $i^\mathrm{th}$ de la matrice de poids
dans la couche d'intégration.
Comme la dimension du vecteur (`output_dim`)
a été fixée à 4,
la couche d'intégration
renvoie des vecteurs de forme (2, 3, 4)
pour un minibatch d'indices de tokens de forme
(2, 3).

```{.python .input}
#@tab all
x = d2l.tensor([[1, 2, 3], [4, 5, 6]])
embed(x)
```

### Définition de la propagation vers l'avant

Dans la propagation vers l'avant,
l'entrée du modèle de saut de programme
comprend
les indices du mot central `center`
 de forme (taille du lot, 1)
et
les indices concaténés du contexte et du mot de bruit `contexts_and_negatives`
 de forme (taille du lot, `max_len`),
où `max_len`
 est défini
dans :numref:`subsec_word2vec-minibatch-loading` .
Ces deux variables sont d'abord transformées en vecteurs à partir des indices de jetons
via la couche d'intégration,
puis leur multiplication matricielle par lot
(décrite dans :numref:`subsec_batch_dot` )
renvoie
une sortie de forme (taille du lot, 1, `max_len`).
Chaque élément de la sortie est le produit scalaire de
un vecteur de mot central et un vecteur de mot de contexte ou de bruit.

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

Imprimons la forme de la sortie de cette fonction `skip_gram` pour quelques exemples d'entrées.

```{.python .input}
#@tab mxnet
skip_gram(np.ones((2, 1)), np.ones((2, 4)), embed, embed).shape
```

```{.python .input}
#@tab pytorch
skip_gram(torch.ones((2, 1), dtype=torch.long),
          torch.ones((2, 4), dtype=torch.long), embed, embed).shape
```

## Formation

Avant de former le modèle de saut de programme avec échantillonnage négatif,
définissons d'abord sa fonction de perte.


### Perte d'entropie croisée binaire

Selon la définition de la fonction de perte
pour l'échantillonnage négatif dans :numref:`subsec_negative-sampling` , 
nous utiliserons 
la perte d'entropie croisée binaire.

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

Rappelez-vous nos descriptions
de la variable masque
et de la variable étiquette dans
:numref:`subsec_word2vec-minibatch-loading` .
Le tableau suivant
calcule la perte d'entropie croisée binaire 

 pour les variables données.

```{.python .input}
#@tab all
pred = d2l.tensor([[1.1, -2.2, 3.3, -4.4]] * 2)
label = d2l.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
mask = d2l.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
loss(pred, label, mask) * mask.shape[1] / mask.sum(axis=1)
```

Le tableau ci-dessous montre
comment les résultats ci-dessus sont calculés
(de manière moins efficace)
en utilisant la fonction d'activation sigmoïde

 dans la perte d'entropie croisée binaire.
Nous pouvons considérer 
les deux sorties comme
deux pertes normalisées
dont la moyenne est calculée sur les prédictions non masquées.

```{.python .input}
#@tab all
def sigmd(x):
    return -math.log(1 / (1 + math.exp(-x)))

print(f'{(sigmd(1.1) + sigmd(2.2) + sigmd(-3.3) + sigmd(4.4)) / 4:.4f}')
print(f'{(sigmd(-1.1) + sigmd(-2.2)) / 2:.4f}')
```

### Initialisation des paramètres du modèle

Nous définissons deux couches d'intégration
pour tous les mots du vocabulaire
lorsqu'ils sont utilisés comme mots centraux
et mots de contexte, respectivement.
La dimension du vecteur de mots
`embed_size` est fixée à 100.

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

### Définition de la boucle d'apprentissage

La boucle d'apprentissage est définie ci-dessous. En raison de l'existence du remplissage, le calcul de la fonction de perte est légèrement différent de celui des fonctions d'apprentissage précédentes.

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

Nous pouvons maintenant former un modèle de saut de programme en utilisant l'échantillonnage négatif.

```{.python .input}
#@tab all
lr, num_epochs = 0.002, 5
train(net, data_iter, lr, num_epochs)
```

## Application de l'intégration de mots
:label:`subsec_apply-word-embed` 

 
 Après avoir entraîné le modèle word2vec,
nous pouvons utiliser la similarité en cosinus
des vecteurs de mots du modèle entraîné
pour 
trouver les mots du dictionnaire
qui sont les plus similaires sémantiquement
à un mot d'entrée.

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

## Résumé

* Nous pouvons entraîner un modèle de saut de programme avec échantillonnage négatif en utilisant des couches d'intégration et la perte d'entropie croisée binaire.
* Les applications de l'intégration des mots incluent la recherche de mots sémantiquement similaires pour un mot donné, basée sur la similarité en cosinus des vecteurs de mots.


## Exercices

1. En utilisant le modèle formé, trouvez des mots sémantiquement similaires pour d'autres mots d'entrée. Pouvez-vous améliorer les résultats en ajustant les hyperparamètres ?
1. Lorsqu'un corpus d'entraînement est énorme, nous échantillonnons souvent des mots de contexte et des mots de bruit pour les mots centraux dans le minibatch actuel * lors de la mise à jour des paramètres du modèle *. En d'autres termes, le même mot central peut avoir différents mots de contexte ou mots de bruit dans différentes époques d'apprentissage. Quels sont les avantages de cette méthode ? Essayez de mettre en œuvre cette méthode de formation.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/384)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1335)
:end_tab:
