```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Underfitting and Overfitting 
:label:`sec_polynomial`

Dans cette section, nous testons certains des concepts que nous avons vus précédemment. Pour rester simple, nous utiliserons la régression polynomiale comme exemple.

```{.python .input  n=3}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import math
npx.set_np()
```

```{.python .input  n=4}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn
import math
```

```{.python .input  n=5}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf
import math
```

### Génération du Dataset

Tout d'abord, nous avons besoin de données. Étant donné $x$, nous allons [**utiliser le polynôme cubique suivant pour générer les étiquettes**] sur les données de formation et de test :

(**$$y = 5 + 1.2x - 3.4\frac{x^2}{2!} + 5.6 \frac{x^3}{3!} + \epsilon \text{ where }
\epsilon \sim \mathcal{N}(0, 0.1^2).$$**)

Le terme de bruit $\epsilon$ obéit à une distribution normale
avec une moyenne de 0 et un écart-type de 0,1.
Pour l'optimisation, nous voulons généralement éviter
de très grandes valeurs de gradients ou de pertes.
C'est pourquoi les *caractéristiques*
sont rééchelonnées de $x^i$ à $\frac{x^i}{i!}$.
Cela nous permet d'éviter les très grandes valeurs pour les grands exposants $i$.
Nous allons synthétiser 100 échantillons chacun pour l'ensemble d'entraînement et l'ensemble de test.

```{.python .input  n=6}
%%tab all
class Data(d2l.DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()        
        p, n = max(3, self.num_inputs), num_train + num_val
        w = d2l.tensor([1.2, -3.4, 5.6] + [0]*(p-3))
        if tab.selected('mxnet') or tab.selected('pytorch'):
            x = d2l.randn(n, 1)
            noise = d2l.randn(n, 1) * 0.1
        if tab.selected('tensorflow'):
            x = d2l.normal((n, 1))
            noise = d2l.normal((n, 1)) * 0.1
        X = d2l.concat([x ** (i+1) / math.gamma(i+2) for i in range(p)], 1)
        self.y = d2l.matmul(X, d2l.reshape(w, (-1, 1))) + noise
        self.X = X[:,:num_inputs]
        
    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
```

Encore une fois, les monômes stockés dans `poly_features`
sont remis à l'échelle par la fonction gamma,
où $\Gamma(n)=(n-1)!$.
[**Regardez les 2 premiers échantillons**] de l'ensemble de données généré.
La valeur 1 est techniquement une caractéristique,
à savoir la caractéristique constante correspondant au biais.

### [**Régression Polynomial du Troisième Ordre (Normal)**]

Nous commencerons par utiliser une fonction polynomiale du troisième ordre, qui est du même ordre que celui de la fonction de génération de données.
Les résultats montrent que les pertes d'apprentissage et de test de ce modèle peuvent toutes deux être efficacement réduites.
Les paramètres du modèle appris sont également proches
des valeurs réelles $w = [1.2, -3.4, 5.6], b=5$.

```{.python .input  n=7}
%%tab all
def train(p):
    if tab.selected('mxnet') or tab.selected('tensorflow'):
        model = d2l.LinearRegression(lr=0.01)
    if tab.selected('pytorch'):
        model = d2l.LinearRegression(p, lr=0.01)
    model.board.ylim = [1, 1e2]
    data = Data(200, 200, p, 20)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    print(model.get_w_b())
    
train(p=3)
```

### [**Régression linéaire (Underfitting)**]

Regardons à nouveau l'ajustement de la fonction linéaire.
Après la diminution dans les premières époques,
il devient difficile de diminuer davantage
la perte d'apprentissage de ce modèle.
Après que la dernière itération d'époque ait été complétée,
la perte d'apprentissage est toujours élevée.
Lorsqu'il est utilisé pour ajuster des modèles non linéaires
(comme la fonction polynomiale du troisième ordre ici)
les modèles linéaires sont susceptibles d'être sous-adaptés.

```{.python .input  n=8}
%%tab all
train(p=1)
```

### [**Régression Polynomial d'Ordre Supérieur (Overfitting)**]

Essayons maintenant d'entraîner le modèle
en utilisant un polynôme de degré trop élevé.
Ici, il n'y a pas assez de données pour apprendre que
les coefficients de degré supérieur devraient avoir des valeurs proches de zéro.
En conséquence, notre modèle trop complexe
est si sensible qu'il est influencé
par le bruit dans les données d'apprentissage.
Bien que la perte d'apprentissage puisse être efficacement réduite,
la perte de test est toujours beaucoup plus élevée.
Cela montre que
le modèle complexe s'adapte trop aux données.

```{.python .input  n=9}
%%tab all
train(p=10)
```

Dans les sections suivantes, nous continuerons
à discuter des problèmes d'overfitting
et des méthodes pour les traiter,
tels que la décroissance et l'abandon des poids.


## Résumé

* Comme l'erreur de généralisation ne peut pas être estimée à partir de l'erreur d'apprentissage, le fait de minimiser l'erreur d'apprentissage ne signifie pas nécessairement une réduction de l'erreur de généralisation. Les modèles d'apprentissage automatique doivent faire attention à ne pas être surajustés afin de minimiser l'erreur de généralisation.
* Un ensemble de validation peut être utilisé pour la sélection du modèle, à condition qu'il ne soit pas utilisé trop librement.
* Le sous-adaptation signifie qu'un modèle n'est pas en mesure de réduire l'erreur d'apprentissage. Lorsque l'erreur de formation est beaucoup plus faible que l'erreur de validation, il y a sur-ajustement.
* Nous devons choisir un modèle suffisamment complexe et éviter d'utiliser des échantillons d'entraînement insuffisants.


## Exercises

1. Pouvez-vous résoudre exactement le problème de la régression polynomiale ? Indice : utilisez l'algèbre linéaire.
1. Considérez la sélection de modèle pour les polynômes :
    1. Tracez la perte d'apprentissage en fonction de la complexité du modèle (degré du polynôme). Qu'observez-vous ? De quel degré de polynôme avez-vous besoin pour réduire la perte d'apprentissage à 0 ?
    1. Tracez la perte de test dans ce cas.
    1. Créez le même graphique en fonction de la quantité de données.
1. Que se passe-t-il si vous laissez tomber la normalisation ($1/i!$) des caractéristiques polynomiales $x^i$ ? Pouvez-vous corriger cela d'une autre manière ?
1. Pouvez-vous un jour vous attendre à voir une erreur de généralisation nulle ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/96)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/97)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/234)
:end_tab:
