```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

# Architecture codeur-décodeur
:label:`sec_encoder-decoder` 

Comme nous l'avons vu dans 
:numref:`sec_machine_translation`,
la traduction automatique
est un domaine problématique majeur pour les modèles de transduction de séquences,
dont l'entrée et la sortie sont
toutes deux des séquences de longueur variable.
Pour traiter ce type d'entrées et de sorties,
nous pouvons concevoir une architecture comportant deux composants principaux.
Le premier composant est un *codeur* :
il prend une séquence de longueur variable en entrée et la transforme en un état de forme fixe.
Le deuxième composant est un *décodeur* :
il transforme l'état codé de forme fixe
en une séquence de longueur variable.
C'est ce qu'on appelle une architecture d'*encodeur-décodeur*,
qui est représentée sur :numref:`fig_encoder_decoder`.

![The encoder-decoder architecture.](../img/encoder-decoder.svg)
:label:`fig_encoder_decoder`

Prenons l'exemple de la traduction automatique de l'anglais au français
.
Étant donné une séquence d'entrée en anglais :
"They", "are", "watching", ".",
cette architecture d'encodeur-décodeur
encode d'abord l'entrée de longueur variable dans un état,
puis décode l'état 
pour générer la séquence traduite mot à mot
en sortie :
"Ils", "regardent", ".".
Puisque l'architecture de l'encodeur-décodeur
constitue la base
de différents modèles de transduction de séquence
dans les sections suivantes,
cette section convertira cette architecture
en une interface qui sera implémentée ultérieurement.

## (**Encodeur**)

Dans l'interface de l'encodeur,
nous spécifions simplement que
l'encodeur prend en entrée des séquences de longueur variable `X`.
L'implémentation sera fournie 
par tout modèle qui hérite de cette classe de base `Encoder`.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet.gluon import nn

#@save
class Encoder(nn.Block):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
from torch import nn

#@save
class Encoder(nn.Module):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def forward(self, X, *args):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

#@save
class Encoder(tf.keras.layers.Layer):
    """The base encoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def call(self, X, *args):
        raise NotImplementedError
```

## [**Decoder**]

Dans l'interface de décodeur suivante,
nous ajoutons une fonction supplémentaire `init_state`
pour convertir la sortie de l'encodeur (`enc_outputs`)
en état encodé.
Notez que cette étape
peut nécessiter des entrées supplémentaires telles que 
la longueur valide de l'entrée,
qui a été expliquée
dans :numref:`sec_machine_translation`.
Pour générer une séquence de longueur variable jeton par jeton,
à chaque fois, le décodeur
peut mettre en correspondance une entrée (par exemple, le jeton généré au pas de temps précédent)
et l'état codé
en un jeton de sortie au pas de temps actuel.

```{.python .input}
%%tab mxnet
#@save
class Decoder(nn.Block):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab pytorch
#@save
class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError
```

```{.python .input}
%%tab tensorflow
#@save
class Decoder(tf.keras.layers.Layer):
    """The base decoder interface for the encoder-decoder architecture."""
    def __init__(self):
        super().__init__()

    # Later there can be additional arguments (e.g., length excluding padding)
    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def call(self, X, state):
        raise NotImplementedError
```

## [**Assemblage de l'encodeur et du décodeur**]

Au final,
l'architecture de l'encodeur-décodeur
contient à la fois un encodeur et un décodeur,
avec éventuellement des arguments supplémentaires.
Dans la propagation vers l'avant,
la sortie de l'encodeur
est utilisée pour produire l'état codé,
et cet état
sera ensuite utilisé par le décodeur comme l'une de ses entrées.

```{.python .input}
%%tab mxnet, pytorch
#@save
class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state)[0]
```

```{.python .input}
%%tab tensorflow
#@save
class EncoderDecoder(d2l.Classifier):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args, training=True)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        # Return decoder output only
        return self.decoder(dec_X, dec_state, training=True)[0]
```

Le terme " état " dans l'architecture codeur-décodeur
vous a probablement inspiré la mise en œuvre de cette architecture
à l'aide de réseaux neuronaux avec états.
Dans la section suivante,
nous verrons comment appliquer les RNN pour concevoir 
des modèles de transduction de séquence basés sur 
cette architecture codeur-décodeur.


## Résumé

* L'architecture encodeur-décodeur peut gérer des entrées et des sorties qui sont toutes deux des séquences de longueur variable, et convient donc aux problèmes de transduction de séquences tels que la traduction automatique.
* L'encodeur prend une séquence de longueur variable en entrée et la transforme en un état de forme fixe.
* Le décodeur transforme l'état codé de forme fixe en une séquence de longueur variable.


## Exercices

1. Supposons que nous utilisions des réseaux neuronaux pour mettre en œuvre l'architecture de l'encodeur-décodeur. L'encodeur et le décodeur doivent-ils être du même type de réseau neuronal ? 
1. Outre la traduction automatique, pouvez-vous imaginer une autre application pour laquelle l'architecture codeur-décodeur peut être utilisée ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/341)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1061)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3864)
:end_tab:
