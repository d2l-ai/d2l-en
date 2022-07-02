```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Différenciation automatique
:label:`sec_autograd` 

 Rappelez-vous de :numref:`sec_calculus` 
 que le calcul des dérivées est l'étape cruciale
de tous les algorithmes d'optimisation
que nous utiliserons pour former des réseaux profonds.
Bien que les calculs soient simples,
les effectuer à la main peut être fastidieux et source d'erreurs, 
et ce problème ne fait que croître
à mesure que nos modèles deviennent plus complexes.

Heureusement, tous les cadres modernes d'apprentissage profond
nous soulagent de ce travail
en proposant la *différenciation automatique*
(souvent abrégée en *autograd*). 
Lorsque nous faisons passer des données par chaque fonction successive,
le framework construit un *graphe computationnel* 
qui suit la façon dont chaque valeur dépend des autres.
Pour calculer les dérivées, 
les paquets de différenciation automatique 
travaillent ensuite à rebours à travers ce graphe
en appliquant la règle de la chaîne. 
L'algorithme de calcul permettant d'appliquer la règle de la chaîne
de cette manière est appelé *rétropropagation*.

Alors que les bibliothèques autogrades sont devenues 
des préoccupations brûlantes au cours de la dernière décennie,
elles ont une longue histoire. 
En fait, les premières références à l'autograd
remontent à plus d'un demi-siècle :cite:`Wengert.1964` .
Les idées fondamentales de la rétropropagation moderne
remontent à une thèse de doctorat de 1980 :cite:`Speelpenning.1980` 
 et ont été développées à la fin des années 1980 :cite:`Griewank.1989` .
Si la rétropropagation est devenue la méthode par défaut 
pour le calcul des gradients, ce n'est pas la seule option. 
Par exemple, le langage de programmation Julia utilise la propagation directe 
 :cite:`Revels.Lubin.Papamarkou.2016` . 
Avant d'explorer les méthodes, 
, maîtrisons d'abord le paquetage autograd.


## Une fonction simple

Supposons que nous soyons intéressés par
pour (**différencier la fonction $y = 2\mathbf{x}^{\top}\mathbf{x}$ par rapport au vecteur colonne $\mathbf{x}$.**)
Pour commencer, nous attribuons une valeur initiale à `x`.

```{.python .input  n=1}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
import torch

x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf

x = tf.range(4, dtype=tf.float32)
x
```

[**Avant de calculer le gradient
de $y$ par rapport à $\mathbf{x}$,
nous avons besoin d'un endroit pour le stocker.**]
En général, nous évitons d'allouer une nouvelle mémoire
chaque fois que nous prenons une dérivée 
parce que l'apprentissage profond nécessite 
de calculer successivement les dérivées
par rapport aux mêmes paramètres
des milliers ou des millions de fois,
et nous risquons de manquer de mémoire.
Notez que le gradient d'une fonction à valeur scalaire
par rapport à un vecteur $\mathbf{x}$
 est à valeur vectorielle et a 
la même forme que $\mathbf{x}$.

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
x.requires_grad_(True)  # Better create `x = torch.arange(4.0, requires_grad=True)`
x.grad                  # The default value is None
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

(**Nous calculons maintenant notre fonction de `x` et assignons le résultat à `y`.**)

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

:begin_tab:`mxnet`
[**Nous pouvons maintenant prendre le gradient de `y` par rapport à `x`**] en appelant 
sa méthode `backward`.
Ensuite, nous pouvons accéder au gradient 
via l'attribut `grad` de `x`.
:end_tab:

:begin_tab:`pytorch`
[**Nous pouvons maintenant prendre le gradient de `y`
 par rapport à `x`**] en appelant 
sa méthode `backward`.
Ensuite, nous pouvons accéder au gradient 
via l'attribut `grad` de `x`.
:end_tab:

:begin_tab:`tensorflow`
[**Nous pouvons maintenant calculer le gradient de `y`
 par rapport à `x`**] en appelant 
la fonction `gradient`.
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

(**Nous savons déjà que le gradient de la fonction $y = 2\mathbf{x}^{\top}\mathbf{x}$ par rapport à $\mathbf{x}$ devrait être $4\mathbf{x}$.**)
Nous pouvons maintenant vérifier que le calcul automatique du gradient
et le résultat attendu sont identiques.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

:begin_tab:`mxnet`
[**Maintenant, calculons 
une autre fonction de `x`
 et prenons son gradient.**] 
Notez que MXNet réinitialise le tampon de gradient 
chaque fois que nous enregistrons un nouveau gradient. 
:end_tab:

:begin_tab:`pytorch`
[**Maintenant, calculons 
une autre fonction de `x`
 et prenons son gradient.**]
Notez que PyTorch ne réinitialise pas automatiquement 
le tampon de gradient 
lorsque nous enregistrons un nouveau gradient. 
Au lieu de cela, le nouveau gradient 
est ajouté au gradient déjà enregistré.
Ce comportement s'avère pratique
lorsque nous voulons optimiser la somme 
de plusieurs fonctions objectives.
Pour réinitialiser le tampon de gradient,
nous pouvons appeler `x.grad.zero()` comme suit:
:end_tab:

:begin_tab :`tensorflow`
 [**Calculons maintenant 
une autre fonction de `x`
 et prenons son gradient.**]
Notez que TensorFlow réinitialise la mémoire tampon des gradients 
chaque fois que nous enregistrons un nouveau gradient 
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

## Retour en arrière pour les variables non scalaires

Lorsque `y` est un vecteur, 
l'interprétation la plus naturelle 
de la dérivée de `y`
 par rapport à un vecteur `x` 
 est une matrice appelée la *Jacobienne*
qui contient les dérivées partielles
de chaque composante de `y` 
 par rapport à chaque composante de `x`.
De la même manière, pour les matrices d'ordre supérieur `y` et `x`,
le résultat de la différenciation pourrait être un tenseur d'ordre encore plus élevé.

Si les jacobiens sont utilisés dans certaines techniques avancées d'apprentissage automatique (
),
le plus souvent, nous voulons additionner 
les gradients de chaque composante de `y`
 par rapport au vecteur complet `x`,
pour obtenir un vecteur de la même forme que `x`.
Par exemple, nous avons souvent un vecteur 
représentant la valeur de notre fonction de perte
calculée séparément pour chacun parmi
un *lot* d'exemples d'apprentissage.
Ici, nous voulons simplement (**additionner les gradients calculés individuellement pour chaque exemple**).

:begin_tab:`mxnet`
MXNet gère ce problème en réduisant tous les tenseurs en scalaires 
en faisant la somme avant de calculer un gradient. 
En d'autres termes, plutôt que de renvoyer le jacobien 
$\partial_{\mathbf{x}} \mathbf{y}$ ,
il renvoie le gradient de la somme
$\partial_{\mathbf{x}} \sum_i y_i$ . 
:end_tab:

:begin_tab:`pytorch`
Comme les cadres d'apprentissage profond varient 
dans la façon dont ils interprètent les gradients des tenseurs non scalaires
,
PyTorch prend certaines mesures pour éviter toute confusion.
L'invocation de `backward` sur un tenseur non scalaire entraîne une erreur 
, sauf si nous indiquons à PyTorch comment réduire l'objet à un scalaire. 
Plus formellement, nous devons fournir un vecteur $\mathbf{v}$ 
 tel que `backward` calculera 
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ 
 plutôt que $\partial_{\mathbf{x}} \mathbf{y}$. 
La partie suivante peut prêter à confusion,
mais pour des raisons qui deviendront claires plus tard, 
cet argument (représentant $\mathbf{v}$) est nommé `gradient`. 
Pour une description plus détaillée, voir l'article de Yang Zhang 
[Medium post](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29) . 
:end_tab:

:begin_tab:`tensorflow`
Par défaut, TensorFlow renvoie le gradient de la somme.
En d'autres termes, plutôt que de renvoyer 
le Jacobien $\partial_{\mathbf{x}} \mathbf{y}$,
il renvoie le gradient de la somme
$\partial_{\mathbf{x}} \sum_i y_i$ 
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as `y = tf.reduce_sum(x * x)`
```

## Détacher des calculs

Parfois, nous souhaitons [**déplacer certains calculs
en dehors du graphe de calcul enregistré.**]
Par exemple, disons que nous utilisons l'entrée 
pour créer certains termes intermédiaires auxiliaires 
pour lesquels nous ne voulons pas calculer de gradient. 
Dans ce cas, nous devons *détacher* 
le graphe d'influence de calcul respectif 
du résultat final. 
L'exemple suivant permet d'y voir plus clair : 
supposons que nous avons `z = x * y` et `y = x * x` 
 mais que nous voulons nous concentrer sur l'influence *directe* de `x` sur `z` 
 plutôt que sur l'influence véhiculée par `y`. 
Dans ce cas, nous pouvons créer une nouvelle variable `u`
 qui prend la même valeur que `y` 
 mais dont la *provenance* (comment elle a été créée)
a été effacée.
Ainsi, `u` n'a pas d'ancêtres dans le graphe
et les gradients ne passent pas de `u` à `x`.
Par exemple, si l'on prend le gradient de `z = x * u`
 , on obtient le résultat `x`,
(et non `3 * x * x` comme on aurait pu s'y attendre de 
puisque `z = x * x * x`).

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set `persistent=True` to preserve the compute graph. 
# This lets us run `t.gradient` more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

Notez que pendant que cette procédure
détache les ancêtres
de `y` du graphe menant à `z`, 
le graphe de calcul menant à `y` 
 persiste et nous pouvons donc calculer
le gradient de `y` par rapport à `x`.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

## Gradients et flux de contrôle Python

Jusqu'à présent, nous avons examiné les cas où le chemin de l'entrée à la sortie 
était bien défini via une fonction telle que `z = x * x * x`.
La programmation nous offre beaucoup plus de liberté dans la façon dont nous calculons les résultats. 
Par exemple, nous pouvons les faire dépendre de variables auxiliaires 
ou conditionner des choix sur des résultats intermédiaires. 
Un avantage de l'utilisation de la différenciation automatique
est que [**même si**] la construction du graphe de calcul de 
(**une fonction a nécessité de passer par un labyrinthe de flux de contrôle Python**)
(par ex, conditionnelles, boucles et appels de fonctions arbitraires),
(**nous pouvons toujours calculer le gradient de la variable résultante.**)
Pour illustrer cela, considérons l'extrait de code suivant où 
le nombre d'itérations de la boucle `while`
 et l'évaluation de l'instruction `if`
 dépendent tous deux de la valeur de l'entrée `a`.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

Ci-dessous, nous appelons cette fonction en passant une valeur aléatoire en entrée.
Comme l'entrée est une variable aléatoire, 
nous ne savons pas quelle forme 
prendra le graphe de calcul.
Cependant, chaque fois que nous exécutons `f(a)` 
 sur une entrée spécifique, nous réalisons 
un graphe de calcul spécifique
et pouvons ensuite exécuter `backward`.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

Même si notre fonction `f` est un peu 
inventée à des fins de démonstration,
sa dépendance à l'égard de l'entrée est assez simple : 
c'est une fonction *linéaire* de `a` 
 avec une échelle définie par morceaux. 
En tant que telle, `f(a) / a` est un vecteur d'entrées constantes 
et, de plus, `f(a) / a` doit correspondre à 
le gradient de `f(a)` par rapport à `a`.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

Le flux de contrôle dynamique est très courant dans l'apprentissage profond. 
Par exemple, lors du traitement du texte, le graphe de calcul
dépend de la longueur de l'entrée. 
Dans ces cas, la différenciation automatique 
devient vitale pour la modélisation statistique 
puisqu'il est impossible de calculer le gradient a priori. 


## Discussion

Vous avez maintenant eu un aperçu de la puissance de la différenciation automatique. 
Le développement de bibliothèques permettant de calculer les dérivées
de manière automatique et efficace 
a permis d'augmenter considérablement la productivité
des praticiens de l'apprentissage profond,
les libérant ainsi pour qu'ils puissent se concentrer sur des préoccupations plus importantes.
De plus, autograd nous permet de concevoir des modèles massifs
pour lesquels les calculs de gradients par stylo et papier 
prendraient un temps prohibitif.
Il est intéressant de noter que si nous utilisons autograd pour *optimiser* les modèles
(au sens statistique)
, la *optimisation* des bibliothèques autograd elles-mêmes
(au sens computationnel)
est un sujet riche
d'un intérêt vital pour les concepteurs de cadres.
Ici, les outils des compilateurs et de la manipulation des graphes 
sont exploités pour calculer les résultats 
de la manière la plus rapide et la plus efficace en termes de mémoire. 

Pour l'instant, essayez de vous souvenir des principes de base suivants : (i) attacher des gradients aux variables pour lesquelles nous voulons des dérivées ; (ii) enregistrer le calcul de la valeur cible ; (iii) exécuter la fonction de rétropropagation ; et (iv) accéder au gradient résultant.


## Exercices

1. Pourquoi la dérivée seconde est-elle beaucoup plus coûteuse à calculer que la dérivée première ?
1. Après avoir exécuté la fonction de rétropropagation, exécutez-la à nouveau immédiatement et voyez ce qui se passe. Pourquoi ?
1. Dans l'exemple de flux de contrôle où nous calculons la dérivée de `d` par rapport à `a`, que se passerait-il si nous changions la variable `a` en un vecteur aléatoire ou une matrice ? À ce stade, le résultat du calcul `f(a)` n'est plus un scalaire. Que se passe-t-il avec le résultat ? Comment l'analyser ?
1. Soit $f(x) = \sin(x)$. Tracez le graphique de $f$ et de sa dérivée $f'$. N'exploitez pas le fait que $f'(x) = \cos(x)$ mais utilisez plutôt la différenciation automatique pour obtenir le résultat. 
1. Soit $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Rédigez un graphe de dépendance retraçant les résultats de $x$ à $f(x)$. 
1. Utilisez la règle de la chaîne pour calculer la dérivée $\frac{df}{dx}$ de la fonction susmentionnée, en plaçant chaque terme sur le graphe de dépendance que vous avez construit précédemment. 
1. Étant donné le graphique et les résultats de la dérivée intermédiaire, vous avez plusieurs options pour calculer le gradient. Évaluez le résultat une fois à partir de $x$ vers $f$ et une fois à partir de $f$ en remontant vers $x$. Le chemin de $x$ à $f$ est communément appelé *différenciation avant*, tandis que le chemin de $f$ à $x$ est appelé différenciation arrière. 
1. Quand voudriez-vous utiliser la différenciation vers l'avant et quand la différenciation vers l'arrière ? Conseil : tenez compte de la quantité de données intermédiaires nécessaires, de la possibilité de paralléliser les étapes et de la taille des matrices et des vecteurs concernés 

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/34)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/35)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/200)
:end_tab:
