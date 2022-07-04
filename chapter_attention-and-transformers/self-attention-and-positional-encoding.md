```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Auto-attention et encodage positionnel
:label:`sec_self-attention-and-positional-encoding` 

En apprentissage profond,
nous utilisons souvent des CNN ou des RNN pour encoder une séquence.
Maintenant, avec les mécanismes d'attention,
imaginez que nous introduisons une séquence d'éléments verbaux
dans un groupement d'attention
de sorte que
le même ensemble d'éléments verbaux
agisse comme des requêtes, des clés et des valeurs.
Plus précisément,
chaque requête s'occupe de toutes les paires clé-valeur
et génère une sortie d'attention.
Puisque les requêtes, les clés et les valeurs
proviennent du même endroit,
réalise
une *auto-attention* :cite:`Lin.Feng.Santos.ea.2017,Vaswani.Shazeer.Parmar.ea.2017`, qui est également appelée *intra-attention* :cite:`Cheng.Dong.Lapata.2016,Parikh.Tackstrom.Das.ea.2016,Paulus.Xiong.Socher.2017`.
Dans cette section,
nous discuterons du codage de séquence utilisant l'auto-attention,
y compris l'utilisation d'informations supplémentaires pour l'ordre de la séquence.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, np, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

## [**Self-Attention**]

Étant donné une séquence de jetons d'entrée
$\mathbf{x}_1, \ldots, \mathbf{x}_n$ où tout $\mathbf{x}_i \in \mathbb{R}^d$ ($1 \leq i \leq n$),
son auto-attention produit
une séquence de même longueur
$\mathbf{y}_1, \ldots, \mathbf{y}_n$ ,
où

$$\mathbf{y}_i = f(\mathbf{x}_i, (\mathbf{x}_1, \mathbf{x}_1), \ldots, (\mathbf{x}_n, \mathbf{x}_n)) \in \mathbb{R}^d$$ 

selon la définition de la mise en commun de l'attention $f$ dans
:eqref:`eq_attn-pooling-def`.
En utilisant l'attention multi-têtes,
l'extrait de code suivant
calcule l'auto-attention d'un tenseur
avec la forme (taille du lot, nombre d'étapes temporelles ou longueur de la séquence en tokens, $d$).
Le tenseur de sortie a la même forme.

```{.python .input}
%%tab mxnet
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
attention.initialize()
```

```{.python .input}
%%tab pytorch
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab tensorflow
num_hiddens, num_heads = 100, 5
attention = d2l.MultiHeadAttention(num_hiddens, num_hiddens, num_hiddens,
                                   num_hiddens, num_heads, 0.5)
```

```{.python .input}
%%tab mxnet, pytorch
batch_size, num_queries, valid_lens = 2, 4, d2l.tensor([3, 2])
X = d2l.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens),
                (batch_size, num_queries, num_hiddens))
```

```{.python .input}
%%tab tensorflow
batch_size, num_queries, valid_lens = 2, 4, tf.constant([3, 2])
X = tf.ones((batch_size, num_queries, num_hiddens))
d2l.check_shape(attention(X, X, X, valid_lens, training=False),
                (batch_size, num_queries, num_hiddens))
```


## Comparaison des CNN, des RNN et de l'auto-attention
:label:`subsec_cnn-rnn-self-attention` 

Comparons les architectures permettant de mettre en correspondance
une séquence de $n$ tokens
avec une autre séquence de longueur égale,
où chaque token d'entrée ou de sortie est représenté par
un vecteur $d$-dimensionnel.
Plus précisément,
nous considérerons les CNN, les RNN et l'auto-attention.
Nous comparerons leur complexité de calcul,
les opérations séquentielles ,

et les longueurs maximales des chemins.
Notez que les opérations séquentielles empêchent le calcul parallèle,
alors qu'un chemin plus court entre
toute combinaison de positions de séquence
facilite l'apprentissage des dépendances à longue portée au sein de la séquence :cite:`Hochreiter.Bengio.Frasconi.ea.2001`.


![Comparing CNN (padding tokens are omitted) les architectures RNN, RNN et auto-attention](../img/cnn-rnn-self-attention.svg)
:label:`fig_cnn-rnn-self-attention` 

Considérons une couche convolutive dont la taille du noyau est $k$.
Nous fournirons plus de détails sur le traitement des séquences
à l'aide des CNN dans les chapitres suivants.
Pour l'instant,
nous avons seulement besoin de savoir que
puisque la longueur de la séquence est $n$,
le nombre de canaux d'entrée et de sortie est $d$,
la complexité de calcul de la couche convolutionnelle est $\mathcal{O}(knd^2)$.
Comme le montre :numref:`fig_cnn-rnn-self-attention`,
les CNN sont hiérarchiques donc 
il y a $\mathcal{O}(1)$ opérations séquentielles
et la longueur maximale du chemin est $\mathcal{O}(n/k)$.
Par exemple, $\mathbf{x}_1$ et $\mathbf{x}_5$
sont dans le champ réceptif d'un CNN à deux couches
avec un noyau de taille 3 dans :numref:`fig_cnn-rnn-self-attention`.

Lors de la mise à jour de l'état caché des RNN,
la multiplication de la matrice de poids $d \times d$
et de l'état caché $d$-dimensionnel a 
une complexité de calcul de $\mathcal{O}(d^2)$.
La longueur de la séquence étant $n$,
la complexité de calcul de la couche récurrente
est $\mathcal{O}(nd^2)$.
Selon :numref:`fig_cnn-rnn-self-attention`,
il existe $\mathcal{O}(n)$ des opérations séquentielles
qui ne peuvent pas être parallélisées
et la longueur maximale du chemin est également $\mathcal{O}(n)$.

Dans l'auto-attention,
les requêtes, les clés et les valeurs 
sont toutes des matrices $n \times d$.
Considérons l'attention par produit scalaire dans
:eqref:`eq_softmax_QK_V`,
où une matrice $n \times d$ est multipliée par
une matrice $d \times n$,
puis la matrice de sortie $n \times n$ est multipliée
par une matrice $n \times d$.
Par conséquent,
l'auto-attention
a une complexité computationnelle de $\mathcal{O}(n^2d)$.
Comme nous pouvons le voir sur :numref:`fig_cnn-rnn-self-attention`,
chaque jeton est directement connecté
à tout autre jeton via l'auto-attention.
Par conséquent, le calcul de
peut être parallèle à $\mathcal{O}(1)$ opérations séquentielles
et la longueur maximale du chemin est également $\mathcal{O}(1)$.

Dans l'ensemble,
les CNN et l'auto-attention bénéficient tous deux d'un calcul parallèle
et l'auto-attention présente la longueur de chemin maximale la plus courte.
Cependant, la complexité de calcul quadratique par rapport à la longueur de la séquence
rend l'auto-attention prohibitivement lente pour les très longues séquences.





## [**Positional Encoding**]
:label:`subsec_positional-encoding` 

 
Contrairement aux RNN qui traitent de manière récurrente
les tokens d'une séquence un par un,
l'auto-attention abandonne
les opérations séquentielles en faveur de 
le calcul parallèle.
Pour utiliser les informations sur l'ordre des séquences,
nous pouvons injecter
des informations positionnelles absolues ou relatives

en ajoutant un *codage positionnel*
aux représentations d'entrée.
Les codages positionnels peuvent être 
soit appris, soit fixés.
Dans ce qui suit, 
nous décrivons un codage positionnel fixe
basé sur les fonctions sinus et cosinus :cite:`Vaswani.Shazeer.Parmar.ea.2017`.

Supposons que
la représentation d'entrée $\mathbf{X} \in \mathbb{R}^{n \times d}$ contienne les enchâssements $d$-dimensionnels pour $n$ les tokens d'une séquence.
Le codage positionnel produit
$\mathbf{X} + \mathbf{P}$ 
en utilisant une matrice d'enchâssement positionnel $\mathbf{P} \in \mathbb{R}^{n \times d}$ de même forme,
dont l'élément sur la ligne $i^\mathrm{th}$ 
et la colonne $(2j)^\mathrm{th}$
ou $(2j + 1)^\mathrm{th}$ est

$$\begin{aligned} p_{i, 2j} &= \sin\left(\frac{i}{10000^{2j/d}}\right),\\p_{i, 2j+1} &= \cos\left(\frac{i}{10000^{2j/d}}\right).\end{aligned}$$ 
:eqlabel:`eq_positional-encoding-def` 

Au premier abord,
cette conception de la fonction trigonométrique
semble bizarre.
Avant d'expliquer cette conception,
commençons par l'implémenter dans la classe `PositionalEncoding` suivante.

```{.python .input}
%%tab mxnet
#@save
class PositionalEncoding(nn.Block):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len).reshape(-1, 1) / np.power(
            10000, np.arange(0, num_hiddens, 2) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].as_in_ctx(X.ctx)
        return self.dropout(X)
```

```{.python .input}
%%tab pytorch
#@save
class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Create a long enough P
        self.P = d2l.zeros((1, max_len, num_hiddens))
        X = d2l.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(
            0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)
```

```{.python .input}
%%tab tensorflow
#@save
class PositionalEncoding(tf.keras.layers.Layer):
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)
        
    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)
```

Dans la matrice d'encastrement positionnel $\mathbf{P}$,
[**les lignes correspondent à des positions dans une séquence et les colonnes représentent différentes dimensions d'encodage positionnel**].
Dans l'exemple ci-dessous,
nous pouvons voir que
les colonnes $6^{\mathrm{th}}$ et $7^{\mathrm{th}}$
de la matrice d'encastrement positionnel 
ont une fréquence plus élevée que 
les colonnes $8^{\mathrm{th}}$ et $9^{\mathrm{th}}$
 .
Le décalage entre les colonnes ,
$6^{\mathrm{th}}$ et $7^{\mathrm{th}}$ (idem pour les colonnes $8^{\mathrm{th}}$ et $9^{\mathrm{th}}$)
est dû à l'alternance des fonctions sinus et cosinus.

```{.python .input}
%%tab mxnet
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
pos_encoding.initialize()
X = pos_encoding(np.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab pytorch
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(d2l.zeros((1, num_steps, encoding_dim)))
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(d2l.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in d2l.arange(6, 10)])
```

```{.python .input}
%%tab tensorflow
encoding_dim, num_steps = 32, 60
pos_encoding = PositionalEncoding(encoding_dim, 0)
X = pos_encoding(tf.zeros((1, num_steps, encoding_dim)), training=False)
P = pos_encoding.P[:, :X.shape[1], :]
d2l.plot(np.arange(num_steps), P[0, :, 6:10].T, xlabel='Row (position)',
         figsize=(6, 2.5), legend=["Col %d" % d for d in np.arange(6, 10)])
```

### Information de position absolue

Pour voir comment la fréquence monotone décroissante
le long de la dimension de codage est liée à l'information de position absolue,
imprimons [**les représentations binaires**] de $0, 1, \ldots, 7$.
Comme nous pouvons le voir,
le bit le plus bas, le deuxième bit le plus bas et le troisième bit le plus bas alternent sur chaque nombre, tous les deux nombres et tous les quatre nombres, respectivement.

```{.python .input}
%%tab all
for i in range(8):
    print(f'{i} in binary is {i:>03b}')
```

Dans les représentations binaires,
un bit supérieur a une fréquence plus faible qu'un bit inférieur.
De même,
comme le montre la carte thermique ci-dessous,
[**le codage positionnel diminue les fréquences le long de la dimension de codage**]
en utilisant des fonctions trigonométriques.
Comme les sorties sont des nombres flottants,
de telles représentations continues
sont plus efficaces en termes d'espace
que les représentations binaires.

```{.python .input}
%%tab mxnet
P = np.expand_dims(np.expand_dims(P[0, :, :], 0), 0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab pytorch
P = P[0, :, :].unsqueeze(0).unsqueeze(0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

```{.python .input}
%%tab tensorflow
P = tf.expand_dims(tf.expand_dims(P[0, :, :], axis=0), axis=0)
d2l.show_heatmaps(P, xlabel='Column (encoding dimension)',
                  ylabel='Row (position)', figsize=(3.5, 4), cmap='Blues')
```

### Informations sur la position relative

Outre la capture d'informations sur la position absolue,
le codage positionnel ci-dessus
permet également à
un modèle d'apprendre facilement à assister à des positions relatives.
En effet,
pour tout décalage de position fixe $\delta$,
le codage positionnel à la position $i + \delta$
peut être représenté par une projection linéaire
de celui à la position $i$.


Cette projection peut être expliquée
mathématiquement.
En désignant
$\omega_j = 1/10000^{2j/d}$ ,
toute paire de $(p_{i, 2j}, p_{i, 2j+1})$ 
dans :eqref:`eq_positional-encoding-def` 
peut 
être projetée linéairement vers $(p_{i+\delta, 2j}, p_{i+\delta, 2j+1})$
pour tout décalage fixe $\delta$:

$$\begin{aligned}
&\begin{bmatrix} \cos(\delta \omega_j) & \sin(\delta \omega_j) \\  -\sin(\delta \omega_j) & \cos(\delta \omega_j) \\ \end{bmatrix}
\begin{bmatrix} p_{i, 2j} \\  p_{i, 2j+1} \\ \end{bmatrix}\\
=&\begin{bmatrix} \cos(\delta \omega_j) \sin(i \omega_j) + \sin(\delta \omega_j) \cos(i \omega_j) \\  -\sin(\delta \omega_j) \sin(i \omega_j) + \cos(\delta \omega_j) \cos(i \omega_j) \\ \end{bmatrix}\\
=&\begin{bmatrix} \sin\left((i+\delta) \omega_j\right) \\  \cos\left((i+\delta) \omega_j\right) \\ \end{bmatrix}\\
=& 
\begin{bmatrix} p_{i+\delta, 2j} \\  p_{i+\delta, 2j+1} \\ \end{bmatrix},
\end{aligned}$$

où la matrice de projection $2\times 2$ ne dépend d'aucun indice de position $i$.

## Résumé

* Dans l'auto-attention, les requêtes, les clés et les valeurs proviennent toutes du même endroit.
* Les CNN et l'auto-attention apprécient le calcul parallèle et l'auto-attention a la longueur de chemin maximale la plus courte. Cependant, la complexité quadratique du calcul par rapport à la longueur de la séquence rend l'auto-attention prohibitivement lente pour les très longues séquences.
* Pour utiliser les informations sur l'ordre des séquences, nous pouvons injecter des informations de position absolue ou relative en ajoutant un codage positionnel aux représentations d'entrée.


## Exercices

1. Supposons que nous concevions une architecture profonde pour représenter une séquence en empilant des couches d'auto-attention avec un encodage positionnel. Quels pourraient être les problèmes ?
1. Pouvez-vous concevoir une méthode d'encodage positionnel apprenable ?
1. Pouvons-nous attribuer différents encastrements appris en fonction des différents décalages entre les requêtes et les clés qui sont comparées dans l'auto-attention ? Indice : vous pouvez vous référer aux encastrements de position relative :cite:`shaw2018self,huang2018music`.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1651)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1652)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3870)
:end_tab:
