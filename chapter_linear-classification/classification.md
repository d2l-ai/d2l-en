```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Le modèle de classification de base
:label:`sec_classification` 

 Vous avez peut-être remarqué que les implémentations à partir de zéro et l'implémentation concise utilisant la fonctionnalité du framework étaient assez similaires dans le cas de la régression. Il en va de même pour la classification. Étant donné qu'un grand nombre de modèles de ce livre traitent de la classification, il est intéressant d'ajouter une fonctionnalité pour prendre en charge ce paramètre spécifique. Cette section fournit une classe de base pour les modèles de classification afin de simplifier le code futur.

```{.python .input  n=2}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, gluon
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
from IPython import display
```

## La classe `Classifier`

 Nous définissons la classe `Classifier` ci-dessous. Dans la classe `validation_step`, nous reportons à la fois la valeur de la perte et la précision de la classification sur un lot de validation. Nous tirons une mise à jour pour chaque lot de `num_val_batches`. Cela présente l'avantage de générer la perte et la précision moyennes sur l'ensemble des données de validation. Ces chiffres moyens ne sont pas exactement corrects si le dernier lot contient moins d'exemples, mais nous ignorons cette différence mineure pour garder le code simple.

```{.python .input  n=5}
%%tab all
class Classifier(d2l.Module):  #@save
    def validation_step(self, batch):
        Y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(Y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(Y_hat, batch[-1]), train=False)
```

Par défaut, nous utilisons un optimiseur de descente de gradient stochastique, fonctionnant sur des minilots, comme nous l'avons fait dans le contexte de la régression linéaire.

```{.python .input  n=6}
%%tab mxnet
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    params = self.parameters()
    if isinstance(params, list):
        return d2l.SGD(params, self.lr)
    return gluon.Trainer(params, 'sgd', {'learning_rate': self.lr})
```

```{.python .input  n=7}
%%tab pytorch
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return torch.optim.SGD(self.parameters(), lr=self.lr)
```

```{.python .input  n=8}
%%tab tensorflow
@d2l.add_to_class(d2l.Module)  #@save
def configure_optimizers(self):
    return tf.keras.optimizers.SGD(self.lr)
```

## Précision

Étant donné la distribution de probabilité prédite `y_hat`,
nous choisissons généralement la classe avec la probabilité prédite la plus élevée
chaque fois que nous devons produire une prédiction difficile.
En effet, de nombreuses applications exigent que nous fassions un choix.
Par exemple, Gmail doit classer un courriel dans les catégories suivantes : " primaire ", " social ", " mises à jour ", " forums " ou " spam ".
Il peut estimer les probabilités en interne,
, mais au bout du compte, il doit choisir une des classes.

Lorsque les prédictions sont cohérentes avec la classe d'étiquettes `y`, elles sont correctes.
La précision de la classification est la fraction de toutes les prédictions qui sont correctes.
Bien qu'il puisse être difficile d'optimiser la précision directement (elle n'est pas différentiable),
c'est souvent la mesure de performance qui nous intéresse le plus. Elle est souvent *la*
quantité pertinente dans les benchmarks. C'est pourquoi nous l'indiquons presque toujours lorsque nous formons des classificateurs.

La précision est calculée comme suit.
Premièrement, si `y_hat` est une matrice,
nous supposons que la deuxième dimension stocke les scores de prédiction pour chaque classe.
Nous utilisons `argmax` pour obtenir la classe prédite par l'index de la plus grande entrée de chaque ligne.
Ensuite, nous [**comparons la classe prédite avec la vérité de base `y` par éléments.**]
L'opérateur d'égalité `==` étant sensible aux types de données,
nous convertissons le type de données de `y_hat` pour qu'il corresponde à celui de `y`.
Le résultat est un tenseur contenant des entrées de 0 (faux) et 1 (vrai).
En prenant la somme, on obtient le nombre de prédictions correctes.

```{.python .input  n=9}
%%tab all
@d2l.add_to_class(Classifier)  #@save
def accuracy(self, Y_hat, Y, averaged=True):
    """Compute the number of correct predictions."""
    Y_hat = d2l.reshape(Y_hat, (-1, Y_hat.shape[-1]))
    preds = d2l.astype(d2l.argmax(Y_hat, axis=1), Y.dtype)
    compare = d2l.astype(preds == d2l.reshape(Y, -1), d2l.float32)
    return d2l.reduce_mean(compare) if averaged else compare
```

```{.python .input  n=10}
%%tab mxnet

@d2l.add_to_class(d2l.Module)  #@save
def get_scratch_params(self):
    params = []
    for attr in dir(self):
        a = getattr(self, attr)
        if isinstance(a, np.ndarray):
            params.append(a)
        if isinstance(a, d2l.Module):
            params.extend(a.get_scratch_params())
    return params

@d2l.add_to_class(d2l.Module)  #@save
def parameters(self):
    params = self.collect_params()
    return params if isinstance(params, gluon.parameter.ParameterDict) and len(
        params.keys()) else self.get_scratch_params()
```

## Résumé

La classification est un problème suffisamment courant pour justifier ses propres fonctions pratiques. L'importance centrale de la classification est la *précision* du classificateur. Notez que, bien que la précision soit souvent au centre de nos préoccupations, nous formons des classificateurs pour optimiser une variété d'autres objectifs pour des raisons statistiques et informatiques. Cependant, quelle que soit la fonction de perte minimisée pendant la formation, il est utile de disposer d'une méthode pratique pour évaluer empiriquement la précision de notre classificateur. 


## Exercices

1. Désignez par $L_v$ la perte de validation, et laissez $L_v^q$ être son estimation rapide et sale calculée par la moyenne de la fonction de perte dans cette section. Enfin, désignez par $l_v^b$ la perte sur le dernier minibatch. Exprimez $L_v$ en termes de $L_v^q$, $l_v^b$, et de la taille de l'échantillon et du minilot.
1. Montrez que l'estimation rapide et sale $L_v^q$ est sans biais. C'est-à-dire, montrez que $E[L_v] = E[L_v^q]$. Pourquoi voudriez-vous toujours utiliser $L_v$ à la place ?
1. Étant donné une perte de classification multiclasse, désignant par $l(y,y')$ la pénalité d'estimation de $y'$ lorsque nous voyons $y$ et étant donné une probabilité $p(y|x)$, formulez la règle pour une sélection optimale de $y'$. Conseil : exprimez la perte attendue, en utilisant $l$ et $p(y|x)$.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/6808)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/6809)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/6810)
:end_tab:
