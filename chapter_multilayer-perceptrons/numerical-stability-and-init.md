```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Stabilité numérique et initialisation
:label:`sec_numerical_stability` 

 
Jusqu'à présent, chaque modèle que nous avons mis en œuvre
exigeait que nous initialisions ses paramètres
selon une certaine distribution pré-spécifiée.
Jusqu'à présent, nous avons considéré que le schéma d'initialisation allait de soi,
passant sous silence les détails de la façon dont ces choix sont faits.
Vous avez peut-être même eu l'impression que ces choix
ne sont pas particulièrement importants.
Au contraire, le choix du schéma d'initialisation
joue un rôle significatif dans l'apprentissage des réseaux neuronaux,
et il peut être crucial pour maintenir la stabilité numérique.
De plus, ces choix peuvent être liés de manière intéressante
au choix de la fonction d'activation non linéaire.
La fonction que nous choisissons et la manière dont nous initialisons les paramètres
peuvent déterminer la vitesse de convergence de notre algorithme d'optimisation.
De mauvais choix peuvent entraîner l'explosion ou la disparition de gradients lors de l'apprentissage.
 
Dans cette section, nous approfondissons ces sujets
et discutons de quelques heuristiques utiles
qui vous seront utiles
tout au long de votre carrière en apprentissage profond.


## Gradients évanescents et explosifs

Considérons un réseau profond avec $L$ couches,
entrée $\mathbf{x}$ et sortie $\mathbf{o}$.
Avec chaque couche $l$ définie par une transformation $f_l$
paramétrée par des poids $\mathbf{W}^{(l)}$,
dont la sortie de la couche cachée est $\mathbf{h}^{(l)}$ (let $\mathbf{h}^{(0)} = \mathbf{x}$),
notre réseau peut être exprimé comme suit :

$$\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \text{ and thus } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x}).$$

Si la sortie de la couche cachée et l'entrée sont des vecteurs,
nous pouvons écrire le gradient de $\mathbf{o}$ par rapport à
tout ensemble de paramètres $\mathbf{W}^{(l)}$ comme suit :

$$\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.$$ 

En d'autres termes, ce gradient est
le produit de $L-l$ matrices
$\mathbf{M}^{(L)} \cdot \ldots \cdot \mathbf{M}^{(l+1)}$ 
et du vecteur gradient $\mathbf{v}^{(l)}$.
Nous sommes donc susceptibles de rencontrer les mêmes problèmes de débordement numérique
qui apparaissent souvent
lorsque l'on multiplie ensemble un trop grand nombre de probabilités.
Lorsqu'on traite des probabilités, une astuce courante consiste à
passer dans l'espace logarithmique, c'est-à-dire à déplacer la pression
de la mantisse vers l'exposant
de la représentation numérique.
Malheureusement, notre problème ci-dessus est plus grave :
initialement, les matrices $\mathbf{M}^{(l)}$ peuvent avoir une grande variété de valeurs propres.
Elles peuvent être petites ou grandes, et
leur produit peut être *très grand* ou *très petit*.

Les risques posés par les gradients instables
vont au-delà de la représentation numérique.
Les gradients de magnitude imprévisible
menacent également la stabilité de nos algorithmes d'optimisation.
Nous pouvons être confrontés à des mises à jour de paramètres qui sont soit
(i) excessivement grandes, détruisant notre modèle
(le problème du *gradient explosif*) ;
ou (ii) excessivement petites
(le problème du *gradient évanescent*),
rendant l'apprentissage impossible car les paramètres
bougent à peine à chaque mise à jour.


### (**Vanishing Gradients**)

Un coupable fréquent à l'origine du problème du gradient de fuite
est le choix de la fonction d'activation $\sigma$
qui est ajoutée à la suite des opérations linéaires de chaque couche.
Historiquement, la fonction sigmoïde
$1/(1 + \exp(-x))$ (introduite dans :numref:`sec_mlp` )
était populaire car elle ressemble à une fonction de seuillage.
Comme les premiers réseaux neuronaux artificiels étaient inspirés
par les réseaux neuronaux biologiques,
l'idée de neurones qui se déclenchent soit *complètement* soit *pas du tout*
(comme les neurones biologiques) semblait séduisante.
Examinons de plus près la sigmoïde
pour voir pourquoi elle peut provoquer des gradients évanescents.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

x = np.arange(-8.0, 8.0, 0.1)
x.attach_grad()
with autograd.record():
    y = npx.sigmoid(x)
y.backward()

d2l.plot(x, [y, x.grad], legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

x = torch.arange(-8.0, 8.0, 0.1, requires_grad=True)
y = torch.sigmoid(x)
y.backward(torch.ones_like(x))

d2l.plot(x.detach().numpy(), [y.detach().numpy(), x.grad.numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

x = tf.Variable(tf.range(-8.0, 8.0, 0.1))
with tf.GradientTape() as t:
    y = tf.nn.sigmoid(x)
d2l.plot(x.numpy(), [y.numpy(), t.gradient(y, x).numpy()],
         legend=['sigmoid', 'gradient'], figsize=(4.5, 2.5))
```

Comme vous pouvez le voir, (**le gradient de la sigmoïde disparaît
à la fois lorsque ses entrées sont grandes et lorsqu'elles sont petites**).
De plus, lors de la rétropropagation à travers de nombreuses couches,
à moins que nous nous trouvions dans la zone Goldilocks, où
les entrées de nombreuses sigmoïdes sont proches de zéro,
les gradients du produit global peuvent disparaître.
Lorsque notre réseau comporte de nombreuses couches,
si nous ne faisons pas attention, le gradient
sera probablement coupé à une certaine couche.
En effet, ce problème était autrefois un fléau pour l'entrainement des réseaux profonds.
Par conséquent, les ReLU, qui sont plus stables
(mais moins plausibles sur le plan neuronal),
sont devenus le choix par défaut des praticiens.


### [**Exploding Gradients**]

Le problème opposé, lorsque les gradients explosent,
peut être tout aussi vexant.
Pour illustrer cela un peu mieux,
nous tirons 100 matrices aléatoires gaussiennes
et les multiplions avec une certaine matrice initiale.
Pour l'échelle que nous avons choisie
(le choix de la variance $\sigma^2=1$),
le produit matriciel explose.
Lorsque cela se produit en raison de l'initialisation
d'un réseau profond, nous n'avons aucune chance de faire converger
un optimiseur à descente de gradient.

```{.python .input}
%%tab mxnet
M = np.random.normal(size=(4, 4))
print('a single matrix', M)
for i in range(100):
    M = np.dot(M, np.random.normal(size=(4, 4)))

print('after multiplying 100 matrices', M)
```

```{.python .input}
%%tab pytorch
M = torch.normal(0, 1, size=(4, 4))
print('a single matrix \n',M)
for i in range(100):
    M = M @ torch.normal(0, 1, size=(4, 4))

print('after multiplying 100 matrices\n', M)
```

```{.python .input}
%%tab tensorflow
M = tf.random.normal((4, 4))
print('a single matrix \n', M)
for i in range(100):
    M = tf.matmul(M, tf.random.normal((4, 4)))

print('after multiplying 100 matrices\n', M.numpy())
```

### Briser la symétrie

Un autre problème dans la conception des réseaux neuronaux
est la symétrie inhérente à leur paramétrage.
Supposons que nous ayons un MLP simple
avec une couche cachée et deux unités.
Dans ce cas, nous pourrions permuter les poids $\mathbf{W}^{(1)}$
de la première couche et permuter de la même manière
les poids de la couche de sortie
pour obtenir la même fonction.
Il n'y a rien de spécial qui différencie
la première unité cachée de la deuxième unité cachée.
En d'autres termes, nous avons une symétrie de permutation
entre les unités cachées de chaque couche.

C'est plus qu'une simple nuisance théorique.
Considérons le MLP à une couche cachée
mentionné plus haut avec deux unités cachées.
À titre d'illustration,
supposez que la couche de sortie transforme les deux unités cachées en une seule unité de sortie.
Imaginez ce qui se passerait si nous initialisions
tous les paramètres de la couche cachée
comme $\mathbf{W}^{(1)} = c$ pour une certaine constante $c$.
Dans ce cas, pendant la propagation vers l'avant,
chaque unité cachée prend les mêmes entrées et paramètres,
produisant la même activation,
qui est transmise à l'unité de sortie.
Pendant la rétropropagation,
la différenciation de l'unité de sortie par rapport aux paramètres $\mathbf{W}^{(1)}$ donne un gradient dont les éléments prennent tous la même valeur.
Ainsi, après une itération basée sur le gradient (par exemple, la descente de gradient stochastique en minibatch),
tous les éléments de $\mathbf{W}^{(1)}$ prennent toujours la même valeur.
De telles itérations
ne *briseraient jamais la symétrie* par elles-mêmes
et nous pourrions ne jamais être en mesure de réaliser
le pouvoir expressif du réseau.
La couche cachée se comporterait
comme si elle ne comportait qu'une seule unité.
Notez que si la descente de gradient stochastique en minibatch ne rompt pas cette symétrie, la régularisation par abandon 
 (qui sera introduite plus tard) le ferait !


## Initialisation des paramètres

Une façon d'aborder - ou du moins d'atténuer - les problèmes soulevés ci-dessus par
est de procéder à une initialisation minutieuse.
Comme nous le verrons plus tard,
une attention supplémentaire pendant l'optimisation
et une régularisation appropriée peuvent encore améliorer la stabilité.


### Initialisation par défaut

Dans les sections précédentes, par exemple dans :numref:`sec_linear_concise`,
nous avons utilisé une distribution normale
pour initialiser les valeurs de nos poids.
Si nous ne spécifions pas la méthode d'initialisation, le framework utilisera
une méthode d'initialisation aléatoire par défaut, qui fonctionne souvent bien en pratique
pour des problèmes de taille modérée.






### Xavier Initialisation
:label:`subsec_xavier` 

Examinons la distribution d'échelle de
une sortie $o_{i}$ pour une couche entièrement connectée
*sans non-linéarités*.
Avec $n_\mathrm{in}$ entrées $x_j$
et leurs poids associés $w_{ij}$ pour cette couche,
une sortie est donnée par

$$o_{i} = \sum_{j=1}^{n_\mathrm{in}} w_{ij} x_j.$$ 

Les poids $w_{ij}$ sont tous tirés
indépendamment de la même distribution.
De plus, supposons que cette distribution
a une moyenne nulle et une variance $\sigma^2$.
Notez que cela ne signifie pas que la distribution doit être gaussienne,
mais simplement que la moyenne et la variance doivent exister.
Pour l'instant, supposons que les entrées de la couche $x_j$
ont également une moyenne et une variance de zéro $\gamma^2$
et qu'elles sont indépendantes de $w_{ij}$ et indépendantes les unes des autres.
Dans ce cas, nous pouvons calculer la moyenne et la variance de $o_i$ comme suit :

$$
\begin{aligned}
    E[o_i] & = \sum_{j=1}^{n_\mathrm{in}} E[w_{ij} x_j] \\&= \sum_{j=1}^{n_\mathrm{in}} E[w_{ij}] E[x_j] \\&= 0, \\
    \mathrm{Var}[o_i] & = E[o_i^2] - (E[o_i])^2 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij} x^2_j] - 0 \\
        & = \sum_{j=1}^{n_\mathrm{in}} E[w^2_{ij}] E[x^2_j] \\
        & = n_\mathrm{in} \sigma^2 \gamma^2.
\end{aligned}
$$

Une façon de garder la variance fixée
est de fixer $n_\mathrm{in} \sigma^2 = 1$.
Considérons maintenant la rétropropagation.
Nous sommes confrontés à un problème similaire,
mais les gradients sont propagés à partir des couches les plus proches de la sortie.
En utilisant le même raisonnement que pour la propagation en avant,
nous voyons que la variance des gradients peut faire exploser
à moins que $n_\mathrm{out} \sigma^2 = 1$,
où $n_\mathrm{out}$ est le nombre de sorties de cette couche.
Cela nous laisse face à un dilemme :
nous ne pouvons pas satisfaire les deux conditions simultanément.
Au lieu de cela, nous essayons simplement de les satisfaire :

$$
\begin{aligned}
\frac{1}{2} (n_\mathrm{in} + n_\mathrm{out}) \sigma^2 = 1 \text{ or equivalently }
\sigma = \sqrt{\frac{2}{n_\mathrm{in} + n_\mathrm{out}}}.
\end{aligned}
$$

C'est le raisonnement qui sous-tend l'initialisation désormais standard
et pratiquement bénéfique *Xavier*,
du nom du premier auteur de ses créateurs :cite:`Glorot.Bengio.2010`.
Typiquement, l'initialisation Xavier
échantillonne les poids à partir d'une distribution gaussienne
avec une moyenne et une variance nulles
$\sigma^2 = \frac{2}{n_\mathrm{in} + n_\mathrm{out}}$ .
Nous pouvons également adapter l'intuition de Xavier pour
choisir la variance lors de l'échantillonnage des poids
à partir d'une distribution uniforme.
Notez que la distribution uniforme $U(-a, a)$ a une variance $\frac{a^2}{3}$.
En ajoutant $\frac{a^2}{3}$ à notre condition sur $\sigma^2$
 , on obtient la suggestion d'initialiser selon

$$U\left(-\sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}, \sqrt{\frac{6}{n_\mathrm{in} + n_\mathrm{out}}}\right).$$ 

Bien que l'hypothèse de non-existence de non-linéarités
dans le raisonnement mathématique ci-dessus
puisse être facilement violée dans les réseaux neuronaux,
la méthode d'initialisation de Xavier
s'avère bien fonctionner en pratique.


### Au-delà de

Le raisonnement ci-dessus ne fait qu'effleurer la surface
des approches modernes de l'initialisation des paramètres.
Un cadre d'apprentissage profond met souvent en œuvre plus d'une douzaine d'heuristiques différentes.
De plus, l'initialisation des paramètres continue d'être
un domaine chaud de recherche fondamentale en apprentissage profond.
Parmi celles-ci, on trouve des heuristiques spécialisées pour les paramètres liés (partagés),
la super-résolution, les modèles de séquence
et d'autres situations.
Par exemple,
Xiao et al. ont démontré la possibilité de former
des réseaux neuronaux à 10000 couches sans astuces architecturales
en utilisant une méthode d'initialisation soigneusement conçue :cite:`Xiao.Bahri.Sohl-Dickstein.ea.2018`.

Si le sujet vous intéresse, nous vous suggérons
de vous plonger dans les offres de ce module,
de lire les articles qui ont proposé et analysé chaque heuristique,
puis d'explorer les dernières publications sur le sujet.
Peut-être tomberez-vous sur ou même inventerez-vous
une idée intelligente et contribuerez à une mise en œuvre dans les cadres d'apprentissage profond.


## Résumé

* La disparition et l'explosion des gradients sont des problèmes courants dans les réseaux profonds. Une grande attention dans l'initialisation des paramètres est nécessaire pour s'assurer que les gradients et les paramètres restent bien contrôlés.
* Des heuristiques d'initialisation sont nécessaires pour s'assurer que les gradients initiaux ne sont ni trop grands ni trop petits.
* Les fonctions d'activation ReLU atténuent le problème des gradients évanescents. Cela peut accélérer la convergence.
* L'initialisation aléatoire est essentielle pour garantir que la symétrie est brisée avant l'optimisation.
* L'initialisation Xavier suggère que, pour chaque couche, la variance de toute sortie n'est pas affectée par le nombre d'entrées, et la variance de tout gradient n'est pas affectée par le nombre de sorties.

## Exercices

1. Pouvez-vous concevoir d'autres cas où un réseau neuronal pourrait présenter une symétrie nécessitant une rupture, outre la symétrie de permutation dans les couches d'un MLP ?
1. Peut-on initialiser tous les paramètres de poids dans la régression linéaire ou dans la régression softmax à la même valeur ?
1. Recherchez les limites analytiques des valeurs propres du produit de deux matrices. Qu'est-ce que cela vous apprend sur la nécessité de s'assurer que les gradients sont bien conditionnés ?
1. Si nous savons que certains termes divergent, pouvons-nous y remédier après coup ? Consultez l'article sur l'échelonnement adaptatif du taux par couche pour vous inspirer de :cite:`You.Gitman.Ginsburg.2017`.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/103)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/104)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/235)
:end_tab:
