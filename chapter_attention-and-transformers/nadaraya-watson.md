# Mise en commun de l'attention : Régression du noyau de Nadaraya-Watson
:label:`sec_nadaraya-watson` 

 Vous connaissez maintenant les principales composantes des mécanismes d'attention dans le cadre de :numref:`fig_qkv` .
Pour récapituler,
les interactions entre
les requêtes (indices volitifs) et les clés (indices non volitifs)
aboutissent à un *regroupement de l'attention*.
La mise en commun de l'attention agrège sélectivement les valeurs (entrées sensorielles) pour produire la sortie.
Dans cette section,
nous décrirons la mise en commun de l'attention plus en détail
pour vous donner une vue de haut niveau de
comment les mécanismes d'attention fonctionnent en pratique.
Plus précisément,
le modèle de régression à noyau de Nadaraya-Watson
proposé en 1964
est un exemple simple mais complet
pour démontrer l'apprentissage automatique avec les mécanismes d'attention.

```{.python .input}
%load_ext d2lbook.tab
tab.interact_select('mxnet', 'pytorch', 'tensorflow')
```

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
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

## [**Générer l'ensemble de données**]

Pour garder les choses simples,
considérons le problème de régression suivant :
étant donné un ensemble de données de paires d'entrées-sorties $\{(x_1, y_1), \ldots, (x_n, y_n)\}$,
comment apprendre $f$ à prédire la sortie $\hat{y} = f(x)$ pour toute nouvelle entrée $x$?

Nous générons ici un ensemble de données artificielles selon la fonction non linéaire suivante avec le terme de bruit $\epsilon$:

$$y_i = 2\sin(x_i) + x_i^{0.8} + \epsilon,$$ 

 où $\epsilon$ obéit à une distribution normale avec une moyenne nulle et un écart type de 0,5.
Nous générons 50 exemples d'apprentissage et 50 exemples de validation
.
Pour mieux visualiser le modèle d'attention par la suite, les entrées de formation sont triées.

```{.python .input}
%%tab all
class NonlinearData(d2l.DataModule):
    def __init__(self, n, batch_size):        
        self.save_hyperparameters()
        f = lambda x: 2 * d2l.sin(x) + x**0.8
        if tab.selected('pytorch'):
            self.x_train, _ = torch.sort(d2l.rand(n) * 5)
            self.y_train = f(self.x_train) + d2l.randn(n)
        if tab.selected('mxnet'):
            self.x_train = np.sort(d2l.rand(n) * 5)
            self.y_train = f(self.x_train) + d2l.randn(n)            
        if tab.selected('tensorflow'):
            self.x_train = tf.sort(d2l.rand((n,1)) * 5, 0)
            self.y_train = f(self.x_train) + d2l.normal((n,1))        
        self.x_val = d2l.arange(0, 5, 5.0/n)
        self.y_val = f(self.x_val)
        
    def get_dataloader(self, train):
        arrays = (self.x_train, self.y_train) if train else (self.x_val, self.y_val)
        return self.get_tensorloader(arrays, train)    
    
n = 50    
data = NonlinearData(n, batch_size=10)
```

La fonction suivante représente tous les exemples d'apprentissage (représentés par des cercles),
la fonction de génération de données de base `f` sans le terme de bruit (étiqueté par "Truth"), et la fonction de prédiction apprise (étiqueté par "Pred").

```{.python .input}
%%tab all
def plot_kernel_reg(y_hat):
    d2l.plot(data.x_val, [data.y_val, d2l.numpy(y_hat)], 'x', 'y', legend=['Truth', 'Pred'],
             xlim=[0, 5], ylim=[-1, 5])
    d2l.plt.plot(data.x_train, data.y_train, 'o', alpha=0.5);
```

## Mise en commun de la moyenne

Nous commençons par l'estimateur le plus " bête " du monde pour ce problème de régression :
en utilisant la mise en commun de la moyenne pour faire la moyenne de toutes les sorties de formation :

$$f(x) = \frac{1}{n}\sum_{i=1}^n y_i,$$ 
 :eqlabel:`eq_avg-pooling` 

 qui est représenté ci-dessous. Comme nous pouvons le voir, cet estimateur n'est pas si intelligent.

```{.python .input}
%%tab all
y_hat = d2l.repeat(d2l.reduce_mean(data.y_train), n)
plot_kernel_reg(y_hat)
```

## [**Mise en commun non paramétrique de l'attention**]

De toute évidence, la mise en commun de la moyenne de
omet les entrées $x_i$.
Une meilleure idée a été proposée
par Nadaraya :cite:`Nadaraya.1964` 
 et Watson :cite:`Watson.1964` 
 pour pondérer les sorties $y_i$ en fonction de l'emplacement de leurs entrées :

$$f(x) = \sum_{i=1}^n \frac{K(x - x_i)}{\sum_{j=1}^n K(x - x_j)} y_i,$$ 
 :eqlabel:`eq_nadaraya-watson` 

 où $K$ est un *noyau*.
L'estimateur dans :eqref:`eq_nadaraya-watson` 
 est appelé *régression à noyau de Nadaraya-Watson*.
Nous ne nous plongerons pas ici dans les détails des noyaux.
Rappelons le cadre des mécanismes d'attention dans :numref:`fig_qkv` .
Du point de vue de l'attention,
nous pouvons réécrire :eqref:`eq_nadaraya-watson` 
 sous une forme plus généralisée de *mise en commun de l'attention* :

$$f(x) = \sum_{i=1}^n \alpha(x, x_i) y_i,$$ 
 :eqlabel:`eq_attn-pooling` 

 
 où $x$ est la requête et $(x_i, y_i)$ la paire clé-valeur.
En comparant :eqref:`eq_attn-pooling` et :eqref:`eq_avg-pooling` ,
la mise en commun de l'attention ici
est une moyenne pondérée des valeurs $y_i$.
Le *poids d'attention* $\alpha(x, x_i)$
 dans :eqref:`eq_attn-pooling` 
 est attribué à la valeur correspondante $y_i$
 en fonction de l'interaction
entre la requête $x$ et la clé $x_i$
 modélisée par $\alpha$.
Pour toute requête, ses poids d'attention sur toutes les paires clé-valeur sont une distribution de probabilité valide : ils sont non négatifs et leur somme est égale à un.

Pour se faire une idée de la mise en commun de l'attention,
considère simplement un noyau *gaussien* défini comme suit

$$
K(u) = \frac{1}{\sqrt{2\pi}} \exp(-\frac{u^2}{2}).
$$


En introduisant le noyau gaussien dans
:eqref:`eq_attn-pooling` et
:eqref:`eq_nadaraya-watson` , on obtient

$$\begin{aligned} f(x) &=\sum_{i=1}^n \alpha(x, x_i) y_i\\ &= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}(x - x_i)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}(x - x_j)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}(x - x_i)^2\right) y_i. \end{aligned}$$ 
 :eqlabel:`eq_nadaraya-watson-gaussian` 

 Dans :eqref:`eq_nadaraya-watson-gaussian` ,
une clé $x_i$ qui est plus proche de la requête donnée $x$ recevra
*plus d'attention* via un *poids d'attention plus grand* attribué à la valeur correspondante de la clé $y_i$.

Notamment, la régression à noyau de Nadaraya-Watson est un modèle non paramétrique ;
donc :eqref:`eq_nadaraya-watson-gaussian` 
 est un exemple de *mise en commun non paramétrique de l'attention*.
Dans ce qui suit, nous traçons la prédiction basée sur ce modèle d'attention non paramétrique
.
La ligne prédite est lisse et plus proche de la vérité de base que celle produite par le regroupement moyen.

```{.python .input}
%%tab all
def diff(queries, keys):
    return d2l.reshape(queries, (-1, 1)) - d2l.reshape(keys, (1, -1))

def attention_pool(query_key_diffs, values):    
    if tab.selected('mxnet'):
        attention_weights = npx.softmax(- query_key_diffs**2 / 2, axis=1)
    if tab.selected('pytorch'):
        attention_weights = F.softmax(- query_key_diffs**2 / 2, dim=1)
    if tab.selected('tensorflow'):
        attention_weights = tf.nn.softmax(- query_key_diffs**2/2, axis=1)
    return d2l.matmul(attention_weights, values), attention_weights

y_hat, attention_weights = attention_pool(
    diff(data.x_val, data.x_train), data.y_train)
plot_kernel_reg(y_hat)
```

Examinons maintenant les [**poids d'attention**].
Ici, les entrées de validation sont des requêtes tandis que les entrées de formation sont des clés.
Puisque les deux entrées sont triées,
nous pouvons voir que plus la paire requête-clé est proche,
plus le poids d'attention est élevé dans le regroupement d'attention.

```{.python .input}
%%tab all
d2l.show_heatmaps([[attention_weights]],
                  xlabel='Sorted training inputs',
                  ylabel='Sorted validation inputs')
```

## **Parametric Attention Pooling**

La régression non paramétrique à noyau de Nadaraya-Watson
bénéficie de l'avantage de la *cohérence* :
étant donné un nombre suffisant de données, ce modèle converge vers la solution optimale.
Néanmoins,
nous pouvons facilement intégrer des paramètres apprenables dans la mise en commun de l'attention.

A titre d'exemple, légèrement différent de :eqref:`eq_nadaraya-watson-gaussian` ,
dans ce qui suit
la distance entre la requête $x$ et la clé $x_i$
 est multipliée par un paramètre apprenable $w$:


 $$\begin{aligned}f(x) &= \sum_{i=1}^n \alpha(x, x_i) y_i \\&= \sum_{i=1}^n \frac{\exp\left(-\frac{1}{2}((x - x_i)w)^2\right)}{\sum_{j=1}^n \exp\left(-\frac{1}{2}((x - x_j)w)^2\right)} y_i \\&= \sum_{i=1}^n \mathrm{softmax}\left(-\frac{1}{2}((x - x_i)w)^2\right) y_i.\end{aligned}$$ 
 :eqlabel:`eq_nadaraya-watson-gaussian-para` 

 Dans le reste de la section,
nous entraînerons ce modèle en apprenant le paramètre de
la mise en commun de l'attention dans :eqref:`eq_nadaraya-watson-gaussian-para` .


### Multiplication matricielle par lots
:label:`subsec_batch_dot` 

 Pour calculer plus efficacement l'attention
pour les minibatchs,
nous pouvons exploiter les utilitaires de multiplication matricielle par lots
fournis par les cadres d'apprentissage profond.


Supposons que le premier minibatch contienne $n$ matrices $\mathbf{X}_1, \ldots, \mathbf{X}_n$ de forme $a\times b$, et que le second minibatch contienne $n$ matrices $\mathbf{Y}_1, \ldots, \mathbf{Y}_n$ de forme $b\times c$. Leur multiplication matricielle par lot
produit des matrices
$n$ $\mathbf{X}_1\mathbf{Y}_1, \ldots, \mathbf{X}_n\mathbf{Y}_n$ de forme $a\times c$. Par conséquent, [**étant donné deux tenseurs de forme ($n$, $a$, $b$) et ($n$, $b$, $c$), la forme du résultat de leur multiplication matricielle par lot est ($n$, $a$, $c$).**]

```{.python .input}
%%tab all
X = d2l.ones((2, 1, 4))
Y = d2l.ones((2, 4, 6))
d2l.check_shape(d2l.batch_matmul(X, Y), (2, 1, 6))
```

Dans le contexte des mécanismes d'attention, nous pouvons [**utiliser la multiplication matricielle par lots pour calculer les moyennes pondérées des valeurs d'un lot.**]

```{.python .input}
%%tab mxnet
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20), (2, 10))
npx.batch_dot(np.expand_dims(weights, 1), np.expand_dims(values, -1)).shape
```

```{.python .input}
%%tab pytorch
weights = d2l.ones((2, 10)) * 0.1
values = d2l.reshape(d2l.arange(20.0), (2, 10))
torch.bmm(weights.unsqueeze(1), values.unsqueeze(-1))
```

```{.python .input}
%%tab tensorflow
weights = tf.ones((2, 10)) * 0.1
values = tf.reshape(tf.range(20.0), shape = (2, 10))
tf.matmul(tf.expand_dims(weights, axis=1), tf.expand_dims(values, axis=-1)).numpy()
```

### Définition du modèle

En utilisant la multiplication matricielle par minilots,
nous définissons ci-dessous la version paramétrique
de la régression à noyau de Nadaraya-Watson
basée sur le [**regroupement paramétrique de l'attention**] dans
:eqref:`eq_nadaraya-watson-gaussian-para` .

```{.python .input}
%%tab all
class NWKernelRegression(d2l.Module):
    def __init__(self, keys, values, lr):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.ones(1)
            self.w.attach_grad()            
        if tab.selected('pytorch'):
            self.w = d2l.ones(1, requires_grad=True)
        if tab.selected('tensorflow'):
            self.w = tf.Variable(d2l.ones(1), trainable=True)
                        
    def forward(self, queries):
        y_hat, self.attention_weights = attention_pool(
            diff(queries, self.keys) * self.w, self.values)
        return y_hat
    
    def loss(self, y_hat, y):
        l = (d2l.reshape(y_hat, -1) - d2l.reshape(y, -1)) ** 2 / 2
        return d2l.reduce_mean(l)

    def configure_optimizers(self):
        if tab.selected('mxnet') or tab.selected('pytorch'):
            return d2l.SGD([self.w], self.lr)
        if tab.selected('tensorflow'):
            return d2l.SGD(self.lr)
```

### Formation

Dans ce qui suit, nous [**transformons l'ensemble de données de formation
en clés et valeurs**] pour former le modèle d'attention.
Dans la mise en commun paramétrique de l'attention,
pour plus de simplicité
, toute entrée d'entraînement prend simplement des paires clé-valeur de tous les exemples d'entraînement pour prédire sa sortie.

```{.python .input}
%%tab all
model = NWKernelRegression(data.x_train, data.y_train, lr=1)
model.board.display = False
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
```

En essayant d'ajuster l'ensemble de données d'entraînement avec du bruit, la ligne prédite est moins lisse que son homologue non paramétrique tracée précédemment.

```{.python .input}
%%tab all
plot_kernel_reg(model.forward(data.x_val))
```

Par rapport à la mise en commun non paramétrique de l'attention,
[**la région avec de grands poids d'attention devient plus nette**]
dans le paramètre paramétrique.

```{.python .input}
%%tab all
d2l.show_heatmaps([[model.attention_weights]],
                  xlabel='Sorted training inputs',
                  ylabel='Sorted validation inputs')
```

## Résumé

* La régression à noyau de Nadaraya-Watson est un exemple d'apprentissage automatique avec des mécanismes d'attention.
* La mise en commun de l'attention de la régression à noyau de Nadaraya-Watson est une moyenne pondérée des sorties de formation. Du point de vue de l'attention, le poids de l'attention est attribué à une valeur en fonction d'une requête et de la clé qui est associée à la valeur.
* La mise en commun de l'attention peut être non paramétrique ou paramétrique.


## Exercices

1. Augmentez le nombre d'exemples d'entraînement. Pouvez-vous mieux apprendre la régression non paramétrique à noyau de Nadaraya-Watson ?
1. Quelle est la valeur de notre apprentissage de $w$ dans l'expérience paramétrique de mise en commun de l'attention ? Pourquoi la région pondérée est-elle plus nette lors de la visualisation des poids d'attention ?
1. Comment pouvons-nous ajouter des hyperparamètres à la régression non paramétrique de Nadaraya-Watson par noyau pour mieux prédire ?
1. Concevez un autre regroupement d'attention paramétrique pour la régression à noyau de cette section. Entraînez ce nouveau modèle et visualisez ses poids d'attention.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/1598)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1599)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/3866)
:end_tab:
