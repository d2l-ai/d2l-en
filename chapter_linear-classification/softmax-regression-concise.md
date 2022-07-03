```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Mise en œuvre concise de la régression Softmax
:label:`sec_softmax_concise` 

 

Tout comme les cadres d'apprentissage profond de haut niveau
ont facilité la mise en œuvre de la régression linéaire
(voir :numref:`sec_linear_concise` ),
ils sont tout aussi pratiques ici.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, init, npx
from mxnet.gluon import nn
npx.set_np()
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
from torch.nn import functional as F
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
```

### Définition du modèle

Comme dans :numref:`sec_linear_concise`, 
nous construisons notre couche entièrement connectée 
en utilisant la couche intégrée. 
La méthode intégrée `__call__` invoque ensuite `forward` 
chaque fois que nous devons appliquer le réseau à une entrée.

:begin_tab:`mxnet`
Même si l'entrée `X` est un tenseur d'ordre 4, 
la couche intégrée `Dense` 
convertira automatiquement `X` en un tenseur d'ordre 2 
en gardant la dimensionnalité le long du premier axe inchangée.
:end_tab:

:begin_tab:`pytorch`
Nous utilisons une couche `Flatten` pour convertir le tenseur d'ordre 4 `X` en tenseur d'ordre 2 
en gardant la dimensionnalité le long du premier axe inchangée.

:end_tab:

:begin_tab:`tensorflow`
Nous utilisons une couche `Flatten` pour convertir le tenseur d'ordre 4 `X` 
en gardant la dimension le long du premier axe inchangée.
:end_tab:

```{.python .input}
%%tab all
class SoftmaxRegression(d2l.Classifier):
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Dense(num_outputs)
            self.net.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(nn.Flatten(),
                                     nn.LazyLinear(num_outputs))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential()
            self.net.add(tf.keras.layers.Flatten())
            self.net.add(tf.keras.layers.Dense(num_outputs))

    def forward(self, X):
        return self.net(X)
```

## Softmax Revisited
:label:`subsec_softmax-implementation-revisited` 

Dans :numref:`sec_softmax_scratch` nous avons calculé la sortie de notre modèle
et appliqué la perte d'entropie croisée. Bien que cette méthode soit parfaitement
raisonnable d'un point de vue mathématique, elle est risquée d'un point de vue informatique, en raison d'un débordement et d'un sous-débordement numérique dans l'exponentiation de
.

Rappelons que la fonction softmax calcule les probabilités via
$\hat y_j = \frac{\exp(o_j)}{\sum_k \exp(o_k)}$ .
Si certains des $o_k$ sont très grands, c'est-à-dire très positifs,
alors $\exp(o_k)$ pourrait être plus grand que le plus grand nombre
que nous pouvons avoir pour certains types de données. C'est ce qu'on appelle un *débordement*. De même,
si tous les arguments sont très négatifs, nous obtiendrons *underflow*.
Par exemple, les nombres à virgule flottante en simple précision, approximativement,
couvrent la plage de $10^{-38}$ à $10^{38}$. Par conséquent, si le plus grand terme de $\mathbf{o}$
se trouve en dehors de l'intervalle $[-90, 90]$, le résultat ne sera pas stable.
Une solution à ce problème consiste à soustraire $\bar{o} \stackrel{\mathrm{def}}{=} \max_k o_k$ de
toutes les entrées :

$$
\hat y_j = \frac{\exp o_j}{\sum_k \exp o_k} =
\frac{\exp(o_j - \bar{o}) \exp \bar{o}}{\sum_k \exp (o_k - \bar{o}) \exp \bar{o}} =
\frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})}.
$$

Par construction, nous savons que $o_j - \bar{o} \leq 0$ pour tous les $j$. Ainsi, pour un problème de classification $q$-classe,
le dénominateur est contenu dans l'intervalle $[1, q]$. De plus, le numérateur de
ne dépasse jamais $1$, ce qui empêche tout dépassement numérique. Le dépassement de capacité numérique
ne se produit que lorsque $\exp(o_j - \bar{o})$ est évalué numériquement comme $0$. Néanmoins, quelques étapes plus loin
nous pourrions nous trouver en difficulté lorsque nous voulons calculer $\log \hat{y}_j$ comme $\log 0$.
En particulier, dans le cadre de la rétropropagation,
nous pourrions nous retrouver face à un écran plein
des redoutables résultats `NaN` (Not a Number).

Heureusement, nous sommes sauvés par le fait que
même si nous calculons des fonctions exponentielles,
nous avons finalement l'intention de prendre leur log
(lors du calcul de la perte d'entropie croisée).
En combinant softmax et entropie croisée,
nous pouvons échapper complètement aux problèmes de stabilité numérique. Nous avons :

$$
\log \hat{y}_j =
\log \frac{\exp(o_j - \bar{o})}{\sum_k \exp (o_k - \bar{o})} =
o_j - \bar{o} - \log \sum_k \exp (o_k - \bar{o}).
$$

Cela permet d'éviter à la fois le débordement et le sous-écoulement.
Nous voudrons garder la fonction softmax conventionnelle à portée de main
au cas où nous voudrions évaluer les probabilités de sortie de notre modèle.
Mais au lieu de passer les probabilités de la softmax dans notre nouvelle fonction de perte,
nous avons juste
[**passer les logits et calculer la softmax et son log
en une seule fois dans la fonction de perte d'entropie croisée,**]
qui fait des choses intelligentes comme le ["LogSumExp trick"](https://en.wikipedia.org/wiki/LogSumExp).

```{.python .input  n=3}
%%tab all
@d2l.add_to_class(d2l.Classifier)  #@save
def loss(self, Y_hat, Y, averaged=True):
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    Y = d2l.reshape(Y, (-1,))
    if tab.selected('mxnet'):
        fn = gluon.loss.SoftmaxCrossEntropyLoss()
        l = fn(Y_hat, Y)
        return l.mean() if averaged else l
    if tab.selected('pytorch'):
        return F.cross_entropy(
            Y_hat, Y, reduction='mean' if averaged else 'none')
    if tab.selected('tensorflow'):
        fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(Y, Y_hat)
```

## Entrainement

Ensuite, nous formons notre modèle. Comme précédemment, nous utilisons les images Fashion-MNIST, aplaties en vecteurs de caractéristiques à 784 dimensions.

```{.python .input}
%%tab all
data = d2l.FashionMNIST(batch_size=256)
model = SoftmaxRegression(num_outputs=10, lr=0.1)
trainer = d2l.Trainer(max_epochs=10)
trainer.fit(model, data)
```

Comme précédemment, cet algorithme converge vers une solution
qui atteint une précision décente,
bien que cette fois-ci avec moins de lignes de code que précédemment.


## Résumé

Les API de haut niveau sont très pratiques pour cacher à leur utilisateur des aspects potentiellement dangereux, comme la stabilité numérique. De plus, elles permettent aux utilisateurs de concevoir des modèles de manière concise avec très peu de lignes de code. C'est à la fois une bénédiction et une malédiction. L'avantage évident est que cela rend les choses très accessibles, même pour les ingénieurs qui n'ont jamais suivi un seul cours de statistiques dans leur vie (c'est d'ailleurs l'un des publics cibles de ce livre). Mais cacher les arêtes vives a aussi un prix : cela dissuade d'ajouter des composants nouveaux et différents par soi-même, car il n'y a pas de mémoire musculaire pour le faire. De plus, il est plus difficile de *réparer* les choses lorsque le rembourrage protecteur de
ne couvre pas entièrement tous les cas de figure. Encore une fois, cela est dû à un manque de familiarité.

C'est pourquoi nous vous conseillons vivement de passer en revue les deux versions, la plus simple et la plus élégante, de la plupart des implémentations qui suivent. Bien que nous mettions l'accent sur la facilité de compréhension, les implémentations sont néanmoins généralement assez performantes (les convolutions sont la grande exception ici). Notre intention est de vous permettre de vous appuyer sur celles-ci lorsque vous inventez quelque chose de nouveau qu'aucun framework ne peut vous donner.


## Exercices

1. L'apprentissage profond utilise de nombreux formats de nombres différents, y compris FP64 double précision (utilisé extrêmement rarement),
FP32 simple précision, BFLOAT16 (bon pour les représentations compressées), FP16 (très instable), TF32 (un nouveau format de NVIDIA), et INT8. Calculez le plus petit et le plus grand argument de la fonction exponentielle pour lesquels le résultat n'entraîne pas de dépassement de capacité ou de sous-dépassement numérique.
1. INT8 est un format très limité avec des nombres non nuls allant de $1$ à $255$. Comment pourriez-vous étendre sa plage dynamique sans utiliser plus de bits ? La multiplication et l'addition standard fonctionnent-elles toujours ?
1. Augmentez le nombre d'époques pour la formation. Pourquoi la précision de la validation peut-elle diminuer après un certain temps ? Comment pouvons-nous y remédier ?
1. Que se passe-t-il lorsque vous augmentez le taux d'apprentissage ? Comparez les courbes de perte pour plusieurs taux d'apprentissage. Lequel fonctionne le mieux ? Quand ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/52)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/53)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/260)
:end_tab:
