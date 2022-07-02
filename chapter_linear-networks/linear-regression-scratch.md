```{.python .input  n=1}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Implémentation de la régression linéaire à partir de zéro
:label:`sec_linear_scratch` 

 Nous sommes maintenant prêts à travailler sur 
une implémentation entièrement fonctionnelle 
de la régression linéaire. 
Dans cette section, 
(**nous implémenterons l'ensemble de la méthode à partir de zéro,
y compris (i) le modèle ; (ii) la fonction de perte ;
(iii) un optimiseur stochastique de descente de gradient en minibatch ;
et (iv) la fonction d'entraînement 
qui assemble tous ces éléments.**)
Enfin, nous allons exécuter notre générateur de données synthétiques
à partir de :numref:`sec_synthetic-regression-data` 
 et appliquer notre modèle
sur le jeu de données résultant. 
Bien que les cadres modernes d'apprentissage profond 
puissent automatiser la quasi-totalité de ce travail,
mettre en œuvre des choses à partir de zéro est la seule façon
de s'assurer que vous savez vraiment ce que vous faites.
En outre, lorsqu'il s'agit de personnaliser des modèles,
en définissant nos propres couches ou fonctions de perte,
comprendre comment les choses fonctionnent sous le capot s'avérera pratique.
Dans cette section, nous nous appuierons uniquement sur 
les tenseurs et la différenciation automatique.
Plus tard, nous présenterons une mise en œuvre plus concise,
tirant parti des cloches et des sifflets des cadres d'apprentissage profond 
tout en conservant la structure de ce qui suit.

```{.python .input  n=2}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input  n=3}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch
```

```{.python .input  n=4}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf
```

### Définition du modèle

[**Avant de pouvoir commencer à optimiser les paramètres de notre modèle**] par minibatch SGD,
(**nous devons d'abord avoir quelques paramètres.**)
Dans ce qui suit, nous initialisons les poids en tirant
des nombres aléatoires d'une distribution normale avec une moyenne de 0
et un écart-type de 0,01. 
Le nombre magique 0,01 fonctionne souvent bien dans la pratique, 
mais vous pouvez spécifier une valeur différente 
à travers l'argument `sigma`.
De plus, nous fixons le biais à 0.
Notez que pour une conception orientée objet
nous ajoutons le code à la méthode `__init__` d'une sous-classe de `d2l.Module` (introduite dans :numref:`oo-design-models` ).

```{.python .input  n=5}
%%tab all
class LinearRegressionScratch(d2l.Module):  #@save
    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1))
            self.b = d2l.zeros(1)
            self.w.attach_grad()
            self.b.attach_grad()
        if tab.selected('pytorch'):
            self.w = d2l.normal(0, sigma, (num_inputs, 1), requires_grad=True)
            self.b = d2l.zeros(1, requires_grad=True)
        if tab.selected('tensorflow'):
            w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
            b = tf.zeros(1)
            self.w = tf.Variable(w, trainable=True)
            self.b = tf.Variable(b, trainable=True)
```

Ensuite, nous devons [**définir notre modèle,
en reliant son entrée et ses paramètres à sa sortie.**]
Pour notre modèle linéaire, nous prenons simplement le produit matrice-vecteur
des caractéristiques d'entrée $\mathbf{X}$ 
 et des poids du modèle $\mathbf{w}$,
et nous ajoutons le décalage $b$ à chaque exemple.
$\mathbf{Xw}$ est un vecteur et $b$ un scalaire.
En raison du mécanisme de diffusion 
(voir :numref:`subsec_broadcasting` ),
lorsque nous ajoutons un vecteur et un scalaire,
le scalaire est ajouté à chaque composante du vecteur.
La fonction `forward` résultante 
est enregistrée en tant que méthode dans la classe `LinearRegressionScratch`
 via `add_to_class` (introduit dans :numref:`oo-design-utilities` ).

```{.python .input  n=6}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def forward(self, X):
    """The linear regression model."""
    return d2l.matmul(X, self.w) + self.b
```

## Définition de la fonction de perte

Puisque [**la mise à jour de notre modèle nécessite de prendre
le gradient de notre fonction de perte,**]
nous devons (**définir la fonction de perte en premier.**)
Ici, nous utilisons la fonction de perte au carré
dans :eqref:`eq_mse` .
Dans l'implémentation, nous devons transformer la valeur réelle `y`
 en la forme de la valeur prédite `y_hat`.
Le résultat renvoyé par la fonction suivante
aura également la même forme que `y_hat`. 
Nous retournons également la valeur de perte moyenne
parmi tous les exemples du minibatch.

```{.python .input  n=7}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def loss(self, y_hat, y):
    l = (y_hat - d2l.reshape(y, y_hat.shape)) ** 2 / 2
    return d2l.reduce_mean(l)
```

## Définition de l'algorithme d'optimisation

Comme nous l'avons vu dans :numref:`sec_linear_regression` ,
la régression linéaire a une solution à forme fermée.
Cependant, notre objectif ici est d'illustrer 
comment former des réseaux neuronaux plus généraux,
et cela nécessite que nous vous apprenions 
comment utiliser la SGD par minilots.
Nous profitons donc de l'occasion
pour vous présenter votre premier exemple fonctionnel de SGD.
À chaque étape, en utilisant un minilot 
tiré au hasard de notre ensemble de données,
nous estimons le gradient de la perte
par rapport aux paramètres.
Ensuite, nous mettons à jour les paramètres
dans la direction qui peut réduire la perte.

Le code suivant applique la mise à jour, 
étant donné un ensemble de paramètres, un taux d'apprentissage `lr`.
Puisque notre perte est calculée comme une moyenne sur le minilot, 
, nous n'avons pas besoin d'ajuster le taux d'apprentissage en fonction de la taille du lot. 
Dans les chapitres suivants, nous étudierons 
comment les taux d'apprentissage doivent être ajustés
pour les très grands minibatchs, comme cela se produit 
dans l'apprentissage distribué à grande échelle.
Pour l'instant, nous pouvons ignorer cette dépendance.

 


:begin_tab:`mxnet`
Nous définissons notre classe `SGD`, 
une sous-classe de `d2l.HyperParameters` (introduite dans :numref:`oo-design-utilities` ),
pour avoir une API similaire
à celle de l'optimiseur SGD intégré.
Nous mettons à jour les paramètres de la méthode `step`.
Elle accepte un argument `batch_size` qui peut être ignoré.
:end_tab:

:begin_tab:`pytorch`
Nous définissons notre classe `SGD`,
une sous-classe de `d2l.HyperParameters` (introduite dans :numref:`oo-design-utilities` ),
pour avoir une API similaire à 
comme l'optimiseur SGD intégré.
Nous mettons à jour les paramètres dans la méthode `step`.
La méthode `zero_grad` fixe tous les gradients à 0,
ce qui doit être exécuté avant une étape de rétropropagation. 
:end_tab:

:begin_tab:`tensorflow`
Nous définissons notre classe `SGD`,
une sous-classe de `d2l.HyperParameters` (introduite dans :numref:`oo-design-utilities` ),
pour avoir une API similaire
à celle de l'optimiseur SGD intégré.
Nous mettons à jour les paramètres dans la méthode `apply_gradients`.
Elle accepte une liste de paires de paramètres et de gradients 
:end_tab:

```{.python .input  n=8}
%%tab mxnet, pytorch
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, params, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()

    if tab.selected('mxnet'):
        def step(self, _):
            for param in self.params:
                param -= self.lr * param.grad
    
    if tab.selected('pytorch'):
        def step(self):
            for param in self.params:
                param -= self.lr * param.grad

        def zero_grad(self):
            for param in self.params:
                if param.grad is not None:
                    param.grad.zero_()
```

```{.python .input  n=9}
%%tab tensorflow
class SGD(d2l.HyperParameters):  #@save
    def __init__(self, lr):
        """Minibatch stochastic gradient descent."""
        self.save_hyperparameters()
    
    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)        
```

Nous définissons ensuite la méthode `configure_optimizers`, qui renvoie une instance de la classe `SGD`.

```{.python .input  n=10}
%%tab all
@d2l.add_to_class(LinearRegressionScratch)  #@save
def configure_optimizers(self):
    if tab.selected('mxnet') or tab.selected('pytorch'):
        return SGD([self.w, self.b], self.lr)
    if tab.selected('tensorflow'):
        return SGD(self.lr)
```

## Formation

Maintenant que tous les éléments sont en place
(paramètres, fonction de perte, modèle et optimiseur),
nous sommes prêts à [**mettre en œuvre la boucle de formation principale.**]
Il est essentiel que vous compreniez bien ce code
car vous utiliserez des boucles de formation similaires
pour tous les autres modèles d'apprentissage profond
abordés dans ce livre.
À chaque *époque*, nous itérons à travers 
l'ensemble des données d'apprentissage, 
en passant une fois par chaque exemple
(en supposant que le nombre d'exemples 
est divisible par la taille du lot). 
À chaque itération, nous sélectionnons un mini-batch d'exemples d'apprentissage,
et calculons sa perte par la méthode du modèle `training_step`. 
Ensuite, nous calculons les gradients par rapport à chaque paramètre. 
Enfin, nous appelons l'algorithme d'optimisation
pour mettre à jour les paramètres du modèle. 
En résumé, nous exécuterons la boucle suivante :

* Initialiser les paramètres $(\mathbf{w}, b)$
 * Répéter jusqu'à ce que cela soit fait
 * Calculer le gradient $\mathbf{g} \leftarrow \partial_{(\mathbf{w},b)} \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} l(\mathbf{x}^{(i)}, y^{(i)}, \mathbf{w}, b)$
 * Mettre à jour les paramètres $(\mathbf{w}, b) \leftarrow (\mathbf{w}, b) - \eta \mathbf{g}$
 
 Rappelons que l'ensemble de données de régression synthétique 
que nous avons généré dans :numref:`` sec_synthetic-regression-data`` 
ne fournit pas d'ensemble de données de validation. 
Dans la plupart des cas, cependant, 
nous utiliserons un ensemble de données de validation 
pour mesurer la qualité de notre modèle. 
Ici, nous passons le dataloader de validation 
une fois dans chaque époque pour mesurer la performance du modèle.
Conformément à notre conception orientée objet,
les fonctions `prepare_batch` et `fit_epoch`
 sont enregistrées en tant que méthodes de la classe `d2l.Trainer`
 (présentée dans :numref:`oo-design-training` ).

```{.python .input  n=11}
%%tab all    
@d2l.add_to_class(d2l.Trainer)  #@save
def prepare_batch(self, batch):
    return batch
```

```{.python .input  n=12}
%%tab pytorch
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.train()        
    for batch in self.train_dataloader:        
        loss = self.model.training_step(self.prepare_batch(batch))
        self.optim.zero_grad()
        with torch.no_grad():
            loss.backward()
            if self.gradient_clip_val > 0:
                self.clip_gradients(self.gradient_clip_val, self.model)
            self.optim.step()
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.eval()
    for batch in self.val_dataloader:
        with torch.no_grad():            
            self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=13}
%%tab mxnet
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    for batch in self.train_dataloader:
        with autograd.record():
            loss = self.model.training_step(self.prepare_batch(batch))
        loss.backward()
        if self.gradient_clip_val > 0:
            self.clip_gradients(self.gradient_clip_val, self.model)
        self.optim.step(1)
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

```{.python .input  n=14}
%%tab tensorflow
@d2l.add_to_class(d2l.Trainer)  #@save
def fit_epoch(self):
    self.model.training = True
    for batch in self.train_dataloader:            
        with tf.GradientTape() as tape:
            loss = self.model.training_step(self.prepare_batch(batch))
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.gradient_clip_val > 0:
            grads = self.clip_gradients(self.gradient_clip_val, grads)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        self.train_batch_idx += 1
    if self.val_dataloader is None:
        return
    self.model.training = False
    for batch in self.val_dataloader:        
        self.model.validation_step(self.prepare_batch(batch))
        self.val_batch_idx += 1
```

Nous sommes presque prêts à entraîner le modèle,
, mais nous avons d'abord besoin de données pour l'entraînement.
Ici, nous utilisons la classe `SyntheticRegressionData` 
 et nous lui passons quelques paramètres de base.
Ensuite, nous entraînons notre modèle avec 
le taux d'apprentissage `lr=0.03` 
 et l'ensemble `max_epochs=3`. 
Notez qu'en général, le nombre d'époques 
et le taux d'apprentissage sont des hyperparamètres.
En général, la définition des hyperparamètres est délicate
et nous voudrons généralement utiliser une division en trois parties,
un ensemble pour l'entraînement, 
un second pour la sélection des hyperparamètres,
et le troisième réservé à l'évaluation finale.
Nous éludons ces détails pour l'instant mais nous les réviserons plus tard
.

```{.python .input  n=15}
%%tab all
model = LinearRegressionScratch(2, lr=0.03)
data = d2l.SyntheticRegressionData(w=d2l.tensor([2, -3.4]), b=4.2)
trainer = d2l.Trainer(max_epochs=3)
trainer.fit(model, data)
```

Comme nous avons synthétisé nous-mêmes l'ensemble de données,
nous savons précisément quels sont les vrais paramètres.
Ainsi, nous pouvons [**évaluer notre succès dans l'entraînement
en comparant les vrais paramètres
avec ceux que nous avons appris**] à travers notre boucle d'entraînement.
En effet, ils s'avèrent être très proches les uns des autres.

```{.python .input  n=16}
%%tab all
print(f'error in estimating w: {data.w - d2l.reshape(model.w, data.w.shape)}')
print(f'error in estimating b: {data.b - model.b}')
```

Nous ne devons pas considérer comme acquise la capacité de récupérer exactement les paramètres réels de 
.
En général, pour les modèles profonds, il n'existe pas de solutions uniques
pour les paramètres,
et même pour les modèles linéaires,
la récupération exacte des paramètres
n'est possible que si aucune caractéristique 
ne dépend linéairement des autres.
Cependant, dans le domaine de l'apprentissage automatique, 
nous sommes souvent moins préoccupés
par la récupération des véritables paramètres sous-jacents,
et plus préoccupés par les paramètres 
qui conduisent à une prédiction très précise :cite:`Vapnik.1992` .
Heureusement, même pour les problèmes d'optimisation difficiles,
la descente de gradient stochastique peut souvent trouver des solutions remarquablement bonnes,
en partie grâce au fait que, pour les réseaux profonds,
il existe de nombreuses configurations des paramètres
qui conduisent à une prédiction très précise.


## Résumé

Dans cette section, nous avons fait un pas important 
vers la conception de systèmes d'apprentissage profond 
en implémentant un modèle de réseau neuronal 
entièrement fonctionnel et une boucle de formation.
Dans ce processus, nous avons construit un chargeur de données, 
un modèle, une fonction de perte, une procédure d'optimisation,
et un outil de visualisation et de surveillance. 
Pour ce faire, nous avons composé un objet Python 
qui contient tous les composants pertinents pour la formation d'un modèle. 
Bien qu'il ne s'agisse pas encore d'une implémentation de niveau professionnel
, elle est parfaitement fonctionnelle et un code comme celui-ci 
pourrait déjà vous aider à résoudre rapidement de petits problèmes.
Dans les sections suivantes, nous verrons comment faire
à la fois *plus concis* (en évitant le code passe-partout)
et *plus efficace* (en utilisant nos GPU à leur plein potentiel).



## Exercices

1. Que se passerait-il si nous initialisions les poids à zéro. L'algorithme fonctionnerait-il toujours ? Que se passerait-il si nous
 initialisions les paramètres avec la variance $1,000$ plutôt que $0.01$?
1. Supposons que vous essayez de trouver [Georg Simon Ohm](https://en.wikipedia.org/wiki/Georg_Ohm)
 un modèle pour les résistances qui relie la tension et le courant. Pouvez-vous utiliser la différenciation automatique
 pour apprendre les paramètres de votre modèle ?
1. Pouvez-vous utiliser [Planck's Law](https://en.wikipedia.org/wiki/Planck%27s_law) pour déterminer la température d'un objet
 en utilisant la densité spectrale d'énergie ? Pour référence, la densité spectrale $B$ du rayonnement émanant d'un corps noir est
 $B(\lambda, T) = \frac{2 hc^2}{\lambda^5} \cdot \left(\exp \frac{h c}{\lambda k T} - 1\right)^{-1}$ . Ici,
 $\lambda$ est la longueur d'onde, $T$ est la température, $c$ est la vitesse de la lumière, $h$ est le quantum de Planck et $k$ est la constante de Boltzmann
. Vous mesurez l'énergie pour différentes longueurs d'onde $\lambda$ et vous devez maintenant adapter la courbe de densité spectrale
 à la loi de Planck.
1. Quels sont les problèmes que vous pourriez rencontrer si vous vouliez calculer les dérivées secondes de la perte ? Comment
 pourriez-vous les résoudre ?
1. Pourquoi la méthode `reshape` est-elle nécessaire dans la fonction `loss`?
1. Faites des expériences en utilisant différents taux d'apprentissage pour déterminer à quelle vitesse la valeur de la fonction de perte diminue. Pouvez-vous réduire l'erreur
 en augmentant le nombre d'époques d'apprentissage ?
1. Si le nombre d'exemples ne peut pas être divisé par la taille du lot, qu'arrive-t-il à `data_iter` à la fin d'une époque ?
1. Essayez de mettre en œuvre une fonction de perte différente, telle que la perte de valeur absolue `(y_hat - d2l.reshape(y, y_hat.shape)).abs().sum()`.
   1. Vérifiez ce qui se passe pour les données régulières.
   1. Vérifiez s'il y a une différence de comportement si vous perturbez activement certaines entrées de $\mathbf{y}$,
 comme $y_5 = 10,000$.
 1. Pouvez-vous imaginer une solution bon marché pour combiner les meilleurs aspects de la perte au carré et de la perte en valeur absolue ?
      Conseil : comment éviter les valeurs de gradient vraiment importantes ?
1. Pourquoi devons-nous remanier l'ensemble de données ? Pouvez-vous concevoir un cas où un ensemble de données malicieux briserait l'algorithme d'optimisation
 autrement ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/42)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/43)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/201)
:end_tab:
