```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Transformateur
:label:`sec_transformer` 

 
 Nous avons comparé les CNN, les RNN et l'auto-attention dans
:numref:`subsec_cnn-rnn-self-attention` .
Notamment, l'auto-attention

 bénéficie à la fois du calcul parallèle et
de la longueur de chemin maximale la plus courte.
C'est pourquoi, naturellement,
il est intéressant de concevoir des architectures profondes
en utilisant l'auto-attention.
Contrairement aux modèles d'auto-attention antérieurs
qui reposent toujours sur des RNN pour les représentations d'entrée :cite:`Cheng.Dong.Lapata.2016,Lin.Feng.Santos.ea.2017,Paulus.Xiong.Socher.2017` ,
le modèle de transformation
est uniquement basé sur les mécanismes d'attention
sans aucune couche convolutive ou récurrente :cite:`Vaswani.Shazeer.Parmar.ea.2017` .

Bien qu'ils aient été proposés à l'origine
pour l'apprentissage de séquence à séquence sur des données textuelles, les transformateurs
ont été omniprésents dans un large éventail d'applications modernes d'apprentissage profond
,
notamment dans les domaines du langage, de la vision, du langage et de l'apprentissage par renforcement.

## Modèle

En tant qu'instance de l'architecture de l'encodeur-décodeur
,
l'architecture globale de
le transformateur
est présentée dans :numref:`fig_transformer` .
Comme on peut le voir,
le transformateur est composé d'un encodeur et d'un décodeur.
A la différence de
, l'attention de Bahdanau
pour l'apprentissage de séquence à séquence
, dans :numref:`fig_s2s_attention_details` ,
, les embeddings de séquence d'entrée (source) et de sortie (cible)

 sont ajoutés avec un encodage positionnel
avant d'être introduits dans
l'encodeur et le décodeur
qui empilent des modules basés sur l'auto-attention.

![The transformer architecture.](../img/transformer.svg)
:width:`500px`
:label:`fig_transformer`


Nous donnons maintenant un aperçu de l'architecture du transformateur
dans :numref:`fig_transformer` .
À un niveau élevé,
le codeur transformateur est une pile de plusieurs couches identiques,
où chaque couche
a deux sous-couches (l'une ou l'autre est désignée par $\mathrm{sublayer}$).
La première
est un réseau de mise en commun d'auto-attention à têtes multiples
et la seconde est un réseau à action directe par position.
Plus précisément,
dans l'auto-attention du codeur, les requêtes, les clés et les valeurs de
proviennent toutes des sorties de
de la couche précédente du codeur.
Inspiré par la conception du ResNet dans :numref:`sec_resnet` ,
une connexion résiduelle est employée
autour des deux sous-couches.
Dans le transformateur,
pour toute entrée $\mathbf{x} \in \mathbb{R}^d$ à toute position de la séquence,
nous exigeons que $\mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ pour que
la connexion résiduelle $\mathbf{x} + \mathrm{sublayer}(\mathbf{x}) \in \mathbb{R}^d$ soit réalisable.
Cette addition de la connexion résiduelle est immédiatement
suivie d'une normalisation de la couche :cite:`Ba.Kiros.Hinton.2016` .
En conséquence, le codeur transformateur produit une représentation vectorielle $d$-dimensionnelle pour chaque position de la séquence d'entrée.

Le décodeur de transformateur est également
une pile de plusieurs couches identiques avec des connexions résiduelles et des normalisations de couche.
Outre les deux sous-couches décrites dans
le codeur, le décodeur insère
une troisième sous-couche, appelée
l'attention du codeur-décodeur,
entre ces deux-là.
Dans l'attention de l'encodeur-décodeur,
les requêtes proviennent des sorties
de la couche précédente du décodeur,
et les clés et valeurs proviennent
des sorties de l'encodeur transformateur.
Dans l'auto-attention du décodeur,
les requêtes, les clés et les valeurs proviennent toutes des sorties
de la couche décodeur précédente.
Cependant,
chaque position du décodeur est
autorisée à ne s'occuper que de toutes les positions du décodeur
jusqu'à cette position.
Cette attention *masquée*
préserve la propriété auto-régressive,
garantissant que la prédiction ne dépend que des jetons de sortie qui ont été générés.


Nous avons déjà décrit et implémenté
l'attention multi-têtes basée sur des produits scalaires de points
dans :numref:`sec_multihead-attention` 
 et le codage positionnel dans :numref:`subsec_positional-encoding` .
Dans la suite,
nous implémenterons le reste du modèle de transformation.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
import math
from mxnet import autograd, init, np, npx
from mxnet.gluon import nn
import pandas as pd
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import math
import pandas as pd
import torch
from torch import nn
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import numpy as np
import pandas as pd
import tensorflow as tf
```

## [**Réseaux feed-forward par position**]
:label:`subsec_positionwise-ffn` 

 Le réseau feed-forward par position
transforme
la représentation à toutes les positions de la séquence
en utilisant le même MLP.
C'est pourquoi nous l'appelons *positionwise*.
Dans l'implémentation ci-dessous,
l'entrée `X` avec la forme
(taille du lot, nombre de pas de temps ou longueur de la séquence en tokens, nombre d'unités cachées ou dimension de la caractéristique)
sera transformée par un MLP à deux couches en
un tenseur de sortie de forme
(taille du lot, nombre de pas de temps, `ffn_num_outputs`).

```{.python .input}
%%tab mxnet
#@save
class PositionWiseFFN(nn.Block):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.Dense(ffn_num_hiddens, flatten=False,
                               activation='relu')
        self.dense2 = nn.Dense(ffn_num_outputs, flatten=False)

    def forward(self, X):
        return self.dense2(self.dense1(X))
```

```{.python .input}
%%tab pytorch
#@save
class PositionWiseFFN(nn.Module):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = nn.LazyLinear(ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

```{.python .input}
%%tab tensorflow
#@save
class PositionWiseFFN(tf.keras.layers.Layer):
    """Positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))
```

L'exemple suivant
montre que [**la dimension la plus intérieure
d'un tenseur change**] en
le nombre de sorties dans
le réseau feed-forward positionnel.
Puisque le même MLP transforme
à toutes les positions,
lorsque les entrées à toutes ces positions sont les mêmes,
leurs sorties sont également identiques.

```{.python .input}
%%tab mxnet
ffn = PositionWiseFFN(4, 8)
ffn.initialize()
ffn(np.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab pytorch
ffn = PositionWiseFFN(4, 8)
ffn.eval()
ffn(d2l.ones((2, 3, 4)))[0]
```

```{.python .input}
%%tab tensorflow
ffn = PositionWiseFFN(4, 8)
ffn(tf.ones((2, 3, 4)))[0]
```

## Connexion résiduelle et normalisation des couches

Concentrons-nous maintenant sur
la composante " add &amp; norm " de :numref:`fig_transformer` .
Comme nous l'avons décrit au début
de cette section,
il s'agit d'une connexion résiduelle immédiatement
suivie d'une normalisation de couche.
Ces deux éléments sont essentiels à l'efficacité des architectures profondes.

Dans :numref:`sec_batch_norm` ,
nous avons expliqué comment la normalisation par lot
recentre et redimensionne les exemples dans
un minibatch.
La normalisation par couche est identique à la normalisation par lot
, sauf que la première
normalise à travers la dimension des caractéristiques.

Malgré ses nombreuses applications
dans le domaine de la vision par ordinateur, la normalisation par lot

 est généralement moins efficace, d'un point de vue empirique, que la normalisation par couche
dans les tâches de traitement du langage naturel
, dont les entrées sont souvent des séquences de longueur variable
.

L'extrait de code suivant
[**compare la normalisation à travers différentes dimensions
par la normalisation par couche et la normalisation par lot**].

```{.python .input}
%%tab mxnet
ln = nn.LayerNorm()
ln.initialize()
bn = nn.BatchNorm()
bn.initialize()
X = d2l.tensor([[1, 2], [2, 3]])
# Compute mean and variance from X in the training mode
with autograd.record():
    print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab pytorch
ln = nn.LayerNorm(2)
bn = nn.LazyBatchNorm1d()
X = d2l.tensor([[1, 2], [2, 3]], dtype=torch.float32)
# Compute mean and variance from X in the training mode
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

```{.python .input}
%%tab tensorflow
ln = tf.keras.layers.LayerNormalization()
bn = tf.keras.layers.BatchNormalization()
X = tf.constant([[1, 2], [2, 3]], dtype=tf.float32)
print('layer norm:', ln(X), '\nbatch norm:', bn(X))
```

Nous pouvons maintenant mettre en œuvre la classe `AddNorm`
 [**en utilisant un branchement résiduel suivi d'une normalisation par couche**].
Le dropout est également appliqué pour la régularisation.

```{.python .input}
%%tab mxnet
#@save
class AddNorm(nn.Block):
    """Residual connection followed by layer normalization."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm()

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab pytorch
#@save
class AddNorm(nn.Module):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(norm_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)
```

```{.python .input}
%%tab tensorflow
#@save
class AddNorm(tf.keras.layers.Layer):
    """Residual connection followed by layer normalization."""
    def __init__(self, norm_shape, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization(norm_shape)

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)
```

La connexion résiduelle exige que
les deux entrées aient la même forme
de sorte que [**le tenseur de sortie ait également la même forme après l'opération d'addition**].

```{.python .input}
%%tab mxnet
add_norm = AddNorm(0.5)
add_norm.initialize()
d2l.check_shape(add_norm(d2l.ones((2, 3, 4)), d2l.ones((2, 3, 4))), (2, 3, 4))
```

```{.python .input}
%%tab pytorch
add_norm = AddNorm(4, 0.5)
d2l.check_shape(add_norm(d2l.ones((2, 3, 4)), d2l.ones((2, 3, 4))), (2, 3, 4))
```

```{.python .input}
%%tab tensorflow
# Normalized_shape is: [i for i in range(len(input.shape))][1:]
add_norm = AddNorm([1, 2], 0.5)
d2l.check_shape(add_norm(tf.ones((2, 3, 4)), tf.ones((2, 3, 4)),
                         training=False), (2, 3, 4))
```

## Encodeur
:label:`subsec_transformer-encoder` 

 Avec tous les composants essentiels pour assembler
l'encodeur transformateur,
commençons par
implémenter [**une couche unique dans l'encodeur**].
La classe suivante `TransformerEncoderBlock`
 contient deux sous-couches : des réseaux d'auto-attention à têtes multiples et des réseaux feed-forward par position,
où une connexion résiduelle suivie d'une normalisation de couche est employée
autour des deux sous-couches.

```{.python .input}
%%tab mxnet
#@save
class TransformerEncoderBlock(nn.Block):
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab pytorch
#@save
class TransformerEncoderBlock(nn.Module):
    """Transformer encoder block."""
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout,
                 use_bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                dropout, use_bias)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(num_hiddens, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))
```

```{.python .input}
%%tab tensorflow
#@save
class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Transformer encoder block."""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs),
                          **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)
```

Comme on peut le constater,
[**toute couche dans le codeur transformateur
ne change pas la forme de son entrée.**]

```{.python .input}
%%tab mxnet
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.initialize()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab pytorch
X = d2l.ones((2, 100, 24))
valid_lens = d2l.tensor([3, 2])
encoder_blk = TransformerEncoderBlock(24, 48, 8, 0.5)
encoder_blk.eval()
d2l.check_shape(encoder_blk(X, valid_lens), X.shape)
```

```{.python .input}
%%tab tensorflow
X = tf.ones((2, 100, 24))
valid_lens = tf.constant([3, 2])
norm_shape = [i for i in range(len(X.shape))][1:]
encoder_blk = TransformerEncoderBlock(24, 24, 24, 24, norm_shape, 48, 8, 0.5)
d2l.check_shape(encoder_blk(X, valid_lens, training=False), X.shape)
```

Dans l'implémentation suivante de [**l'encodeur transformateur**],
nous empilons `num_blks` instances des classes `TransformerEncoderBlock` ci-dessus.
Puisque nous utilisons l'encodage positionnel fixe
dont les valeurs sont toujours comprises entre -1 et 1,
nous multiplions les valeurs des encastrements d'entrée apprenables
par la racine carrée de la dimension d'encastrement
pour les remettre à l'échelle avant d'additionner l'encastrement d'entrée et l'encodage positionnel.

```{.python .input}
%%tab mxnet
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for _ in range(num_blks):
            self.blks.add(TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))
        self.initialize()

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab pytorch
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens,
                 num_heads, num_blks, dropout, use_bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, use_bias))

    def forward(self, X, valid_lens):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

```{.python .input}
%%tab tensorflow
#@save
class TransformerEncoder(d2l.Encoder):
    """Transformer encoder."""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout, bias=False):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerEncoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, bias) for _ in range(
            num_blks)]

    def call(self, X, valid_lens, **kwargs):
        # Since positional encoding values are between -1 and 1, the embedding
        # values are multiplied by the square root of the embedding dimension
        # to rescale before they are summed up
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens, **kwargs)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X
```

Nous spécifions ci-dessous les hyperparamètres pour [**créer un codeur transformateur à deux couches**].
La forme de la sortie du codeur transformateur
est (taille du lot, nombre de pas de temps, `num_hiddens`).

```{.python .input}
%%tab mxnet
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(np.ones((2, 100)), valid_lens), (2, 100, 24))
```

```{.python .input}
%%tab pytorch
encoder = TransformerEncoder(200, 24, 48, 8, 2, 0.5)
d2l.check_shape(encoder(d2l.ones((2, 100), dtype=torch.long), valid_lens),
                (2, 100, 24))
```

```{.python .input}
%%tab tensorflow
encoder = TransformerEncoder(200, 24, 24, 24, 24, [1, 2], 48, 8, 2, 0.5)
d2l.check_shape(encoder(tf.ones((2, 100)), valid_lens, training=False),
                (2, 100, 24))
```

## Décodeur

Comme indiqué sur :numref:`fig_transformer` ,
[**le décodeur transformateur
est composé de plusieurs couches identiques**].
Chaque couche est mise en œuvre dans la classe suivante
`TransformerDecoderBlock` ,
qui contient trois sous-couches :
attention du décodeur,
attention du codeur-décodeur,
et réseaux à action directe par position.
Ces sous-couches emploient
une connexion résiduelle autour d'elles
suivie d'une normalisation des couches.


Comme nous l'avons décrit précédemment dans cette section,
dans l'auto-attention du décodeur multi-tête masqué
(la première sous-couche),
les requêtes, les clés et les valeurs
proviennent toutes des sorties de la couche décodeur précédente.
Lors de la formation des modèles de séquence à séquence,
les tokens à toutes les positions (pas de temps)
de la séquence de sortie
sont connus.
Cependant,
pendant la prédiction
la séquence de sortie est générée token par token ;
ainsi,
à tout pas de temps du décodeur
seuls les tokens générés
peuvent être utilisés dans l'auto-attention du décodeur.
Pour préserver l'autorégression dans le décodeur,
son auto-attention masquée
spécifie `dec_valid_lens` de sorte que
toute requête
n'attend que
toutes les positions dans le décodeur
jusqu'à la position de la requête.

```{.python .input}
%%tab mxnet
class TransformerDecoderBlock(nn.Block):
    # The i-th block in the transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = np.concatenate((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values

        if autograd.is_training():
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = np.tile(np.arange(1, num_steps + 1, ctx=X.ctx),
                                     (batch_size, 1))
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab pytorch
class TransformerDecoderBlock(nn.Module):
    # The i-th block in the transformer decoder
    def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm1 = AddNorm(num_hiddens, dropout)
        self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
                                                 dropout)
        self.addnorm2 = AddNorm(num_hiddens, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(num_hiddens, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values
        if self.training:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = torch.arange(
                1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
        Y = self.addnorm1(X, X2)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state
```

```{.python .input}
%%tab tensorflow
class TransformerDecoderBlock(tf.keras.layers.Layer):
    # The i-th block in the transformer decoder
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_hiddens, num_heads, dropout, i):
        super().__init__()
        self.i = i
        self.attention1 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def call(self, X, state, **kwargs):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # During training, all the tokens of any output sequence are processed
        # at the same time, so state[2][self.i] is None as initialized. When
        # decoding any output sequence token by token during prediction,
        # state[2][self.i] contains representations of the decoded output at
        # the i-th block up to the current time step
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = tf.concat((state[2][self.i], X), axis=1)
        state[2][self.i] = key_values
        if kwargs["training"]:
            batch_size, num_steps, _ = X.shape
            # Shape of dec_valid_lens: (batch_size, num_steps), where every
            # row is [1, 2, ..., num_steps]
            dec_valid_lens = tf.repeat(
                tf.reshape(tf.range(1, num_steps + 1),
                           shape=(-1, num_steps)), repeats=batch_size, axis=0)
        else:
            dec_valid_lens = None
        # Self-attention
        X2 = self.attention1(X, key_values, key_values, dec_valid_lens,
                             **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,
                             **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs), state
```

Pour faciliter les opérations de produit scalaire
dans l'attention codeur-décodeur
et les opérations d'addition dans les connexions résiduelles,
[**la dimension caractéristique (`num_hiddens`) du décodeur est
la même que celle du codeur.**]

```{.python .input}
%%tab mxnet
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
decoder_blk.initialize()
X = np.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab pytorch
decoder_blk = TransformerDecoderBlock(24, 48, 8, 0.5, 0)
X = d2l.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state)[0], X.shape)
```

```{.python .input}
%%tab tensorflow
decoder_blk = TransformerDecoderBlock(24, 24, 24, 24, [1, 2], 48, 8, 0.5, 0)
X = tf.ones((2, 100, 24))
state = [encoder_blk(X, valid_lens), valid_lens, [None]]
d2l.check_shape(decoder_blk(X, state, training=False)[0], X.shape)
```

Maintenant, nous [**construisons le décodeur transformateur complet**]
composé de `num_blks` instances de `TransformerDecoderBlock`.
Au final,
une couche entièrement connectée calcule la prédiction
pour tous les `vocab_size` jetons de sortie possibles.
Les deux poids d'attention du décodeur
et les poids d'attention de l'encodeur-décodeur
sont stockés pour une visualisation ultérieure.

```{.python .input}
%%tab mxnet
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add(TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.Dense(vocab_size, flatten=False)
        self.initialize()

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab pytorch
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            self.blks.add_module("block"+str(i), TransformerDecoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
        self.dense = nn.LazyLinear(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # Decoder self-attention weights
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # Encoder-decoder attention weights
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

```{.python .input}
%%tab tensorflow
class TransformerDecoder(d2l.AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_hiddens, num_heads,
                 num_blks, dropout):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_blks = num_blks
        self.embedding = tf.keras.layers.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
        self.blks = [TransformerDecoderBlock(
            key_size, query_size, value_size, num_hiddens, norm_shape,
            ffn_num_hiddens, num_heads, dropout, i)
                     for i in range(num_blks)]
        self.dense = tf.keras.layers.Dense(vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens):
        return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

    def call(self, X, state, **kwargs):
        X = self.pos_encoding(self.embedding(X) * tf.math.sqrt(
            tf.cast(self.num_hiddens, dtype=tf.float32)), **kwargs)
        # 2 attention layers in decoder
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state, **kwargs)
            # Decoder self-attention weights
            self._attention_weights[0][i] = (
                blk.attention1.attention.attention_weights)
            # Encoder-decoder attention weights
            self._attention_weights[1][i] = (
                blk.attention2.attention.attention_weights)
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights
```

## [**Training**]

Instancions un modèle encodeur-décodeur
en suivant l'architecture du transformateur.
Ici, nous spécifions que
l'encodeur transformateur et le décodeur transformateur
ont tous deux 2 couches utilisant une attention à 4 têtes.
Comme pour :numref:`sec_seq2seq_training` ,
nous formons le modèle de transformateur
pour l'apprentissage de séquence à séquence sur le jeu de données de traduction automatique anglais-français.

```{.python .input}
%%tab all
data = d2l.MTFraEng(batch_size=128)
num_hiddens, num_blks, dropout = 256, 2, 0.2
ffn_num_hiddens, num_heads = 64, 4
if tab.selected('tensorflow'):
    key_size, query_size, value_size = 256, 256, 256
    norm_shape = [2]
if tab.selected('mxnet'):
    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
if tab.selected('pytorch'):
    encoder = TransformerEncoder(
        len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
    decoder = TransformerDecoder(
        len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
        num_blks, dropout)
if tab.selected('mxnet', 'pytorch'):
    model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                        lr=0.001)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1, num_gpus=1)
if tab.selected('tensorflow'):
    with d2l.try_gpu():
        encoder = TransformerEncoder(
            len(data.src_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        decoder = TransformerDecoder(
            len(data.tgt_vocab), key_size, query_size, value_size, num_hiddens,
            norm_shape, ffn_num_hiddens, num_heads, num_blks, dropout)
        model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
                            lr=0.001)
    trainer = d2l.Trainer(max_epochs=50, gradient_clip_val=1)
trainer.fit(model, data)
```

Après l'entraînement,
nous utilisons le modèle de transformation
pour [**traduire quelques phrases anglaises**] en français et calculer leurs scores BLEU.

```{.python .input}
%%tab all
engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
preds, _ = model.predict_step(
    data.build(engs, fras), d2l.try_gpu(), data.num_steps)
for en, fr, p in zip(engs, fras, preds):
    translation = []
    for token in data.tgt_vocab.to_tokens(p):
        if token == '<eos>':
            break
        translation.append(token)
    print(f'{en} => {translation}, bleu,'
          f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
```

Visualisons [**les poids d'attention du transformateur**] lors de la traduction de la dernière phrase anglaise en français.
La forme des poids d'auto-attention du codeur
est (nombre de couches du codeur, nombre de têtes d'attention, `num_steps` ou nombre de requêtes, `num_steps` ou nombre de paires clé-valeur).

```{.python .input}
%%tab all
_, dec_attention_weights = model.predict_step(
    data.build([engs[-1]], [fras[-1]]), d2l.try_gpu(), data.num_steps, True)
enc_attention_weights = d2l.reshape(
    d2l.concat(model.encoder.attention_weights, 0),
    (num_blks, num_heads, -1, data.num_steps))
d2l.check_shape(enc_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

Dans l'auto-attention du codeur,
, les requêtes et les clés proviennent de la même séquence d'entrée.
Comme les jetons de remplissage ne sont pas porteurs de sens,
avec une longueur valide spécifiée de la séquence d'entrée,
aucune requête ne s'intéresse aux positions des jetons de remplissage.
Dans ce qui suit,
deux couches de poids d'attention multi-têtes
sont présentées ligne par ligne.
Chaque tête s'occupe indépendamment de
sur la base d'un sous-espace de représentation distinct des requêtes, des clés et des valeurs.

```{.python .input}
%%tab mxnet, tensorflow
d2l.show_heatmaps(
    enc_attention_weights, xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

```{.python .input}
%%tab pytorch
d2l.show_heatmaps(
    enc_attention_weights.cpu(), xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

[**Pour visualiser à la fois les poids d'auto-attention du décodeur et les poids d'attention du codeur-décodeur,
nous avons besoin de plus de manipulations de données.**]
Par exemple,
nous remplissons les poids d'attention masqués avec zéro.
Notez que
les poids d'auto-attention du décodeur
et les poids d'attention de l'encodeur-décodeur
ont tous deux les mêmes requêtes :
le jeton de début de séquence suivi de
les jetons de sortie et éventuellement
les jetons de fin de séquence.

```{.python .input}
%%tab mxnet
dec_attention_weights_2d = [d2l.tensor(head[0]).tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, (
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.transpose(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab pytorch
dec_attention_weights_2d = [head[0].tolist()
                            for step in dec_attention_weights
                            for attn in step for blk in attn for head in blk]
dec_attention_weights_filled = d2l.tensor(
    pd.DataFrame(dec_attention_weights_2d).fillna(0.0).values)
dec_attention_weights = d2l.reshape(dec_attention_weights_filled, (
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = \
    dec_attention_weights.permute(1, 2, 3, 0, 4)
```

```{.python .input}
%%tab tensorflow
dec_attention_weights_2d = [head[0] for step in dec_attention_weights
                            for attn in step
                            for blk in attn for head in blk]
dec_attention_weights_filled = tf.convert_to_tensor(
    np.asarray(pd.DataFrame(dec_attention_weights_2d).fillna(
        0.0).values).astype(np.float32))
dec_attention_weights = tf.reshape(dec_attention_weights_filled, shape=(
    -1, 2, num_blks, num_heads, data.num_steps))
dec_self_attention_weights, dec_inter_attention_weights = tf.transpose(
    dec_attention_weights, perm=(1, 2, 3, 0, 4))
```

```{.python .input}
%%tab all
d2l.check_shape(dec_self_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
d2l.check_shape(dec_inter_attention_weights,
                (num_blks, num_heads, data.num_steps, data.num_steps))
```

En raison de la propriété auto-régressive de l'auto-attention du décodeur,
aucune requête n'attend les paires clé-valeur après la position de la requête.

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_self_attention_weights[:, :, :, :],
    xlabel='Key positions', ylabel='Query positions',
    titles=['Head %d' % i for i in range(1, 5)], figsize=(7, 3.5))
```

Comme dans le cas de l'auto-attention du codeur,
via la longueur valide spécifiée de la séquence d'entrée,
[**no query from the output sequence
attend to those padding tokens from the input sequence.**]

```{.python .input}
%%tab all
d2l.show_heatmaps(
    dec_inter_attention_weights, xlabel='Key positions',
    ylabel='Query positions', titles=['Head %d' % i for i in range(1, 5)],
    figsize=(7, 3.5))
```

Bien que l'architecture de transformateur
ait été proposée à l'origine pour l'apprentissage de séquence à séquence,
comme nous le découvrirons plus tard dans le livre,
soit le codeur transformateur
soit le décodeur transformateur
est souvent utilisé individuellement
pour différentes tâches d'apprentissage profond.


## Résumé

* Le transformateur est une instance de l'architecture encodeur-décodeur, bien que l'encodeur ou le décodeur puisse être utilisé individuellement dans la pratique.
* Dans le transformateur, l'auto-attention multi-tête est utilisée pour représenter la séquence d'entrée et la séquence de sortie, bien que le décodeur doive préserver la propriété auto-régressive via une version masquée.
* Les connexions résiduelles et la normalisation des couches dans le transformateur sont importantes pour l'apprentissage d'un modèle très profond.
* Le réseau feed-forward positionnel dans le modèle de transformation transforme la représentation à toutes les positions de la séquence en utilisant le même MLP.

## Exercices

1. Entraînez un transformateur plus profond dans les expériences. Comment cela affecte-t-il la vitesse d'apprentissage et les performances de traduction ?
1. Est-ce une bonne idée de remplacer l'attention du produit scalaire par une attention additive dans le transformateur ? Pourquoi ?
1. Pour la modélisation du langage, devrions-nous utiliser le transformateur encodeur, décodeur, ou les deux ? Comment concevoir cette méthode ?
1. Quels peuvent être les défis des transformateurs si les séquences d'entrée sont très longues ? Pourquoi ?
1. Comment améliorer l'efficacité des transformateurs en termes de calcul et de mémoire ? Conseil : vous pouvez vous référer au document d'étude de Tay et al. :cite:`Tay.Dehghani.Bahri.ea.2020` .


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/348)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1066)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3871)
:end_tab:
