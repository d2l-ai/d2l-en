# Maximum de vraisemblance
:label:`sec_maximum_likelihood` 

L'un des modes de pensée les plus couramment rencontrés en apprentissage automatique est le point de vue du maximum de vraisemblance.  Il s'agit du concept selon lequel, lorsque l'on travaille avec un modèle probabiliste dont les paramètres sont inconnus, les paramètres qui confèrent aux données la plus forte probabilité sont les plus probables.

## Le principe du maximum de vraisemblance

Il s'agit d'une interprétation bayésienne à laquelle il peut être utile de réfléchir.  Supposons que nous ayons un modèle avec des paramètres $\boldsymbol{\theta}$ et une collection d'exemples de données $X$.  Pour être plus concret, nous pouvons imaginer que $\boldsymbol{\theta}$ est une valeur unique représentant la probabilité qu'une pièce de monnaie tombe sur face lorsqu'elle est tirée à pile ou face, et que $X$ est une séquence de tirages à pile ou face indépendants.  Nous étudierons cet exemple en profondeur plus tard.

Si nous voulons trouver la valeur la plus probable pour les paramètres de notre modèle, cela signifie que nous voulons trouver

$$\mathop{\mathrm{argmax}} P(\boldsymbol{\theta}\mid X).$$ 
:eqlabel:`eq_max_like` 

Par la règle de Bayes, c'est la même chose que

$$
\mathop{\mathrm{argmax}} \frac{P(X \mid \boldsymbol{\theta})P(\boldsymbol{\theta})}{P(X)}.
$$

L'expression $P(X)$, une probabilité agnostique de générer les données, ne dépend pas du tout de $\boldsymbol{\theta}$ et peut donc être abandonnée sans modifier le meilleur choix de $\boldsymbol{\theta}$. De même, nous pouvons maintenant affirmer que nous n'avons aucune hypothèse préalable sur le jeu de paramètres qui est meilleur que les autres, et nous pouvons donc déclarer que $P(\boldsymbol{\theta})$ ne dépend pas non plus de thêta !  Cela a du sens, par exemple, dans notre exemple de lancer de pièce de monnaie où la probabilité qu'elle tombe sur face peut être n'importe quelle valeur dans $[0,1]$ sans qu'il y ait de croyance préalable qu'elle soit juste ou non (souvent appelée une *antériorité non informative*).  Nous voyons donc que l'application de la règle de Bayes montre que notre meilleur choix de $\boldsymbol{\theta}$ est l'estimation du maximum de vraisemblance pour $\boldsymbol{\theta}$:

$$
\hat{\boldsymbol{\theta}} = \mathop{\mathrm{argmax}} _ {\boldsymbol{\theta}} P(X \mid \boldsymbol{\theta}).
$$

Dans la terminologie courante, la probabilité des données compte tenu des paramètres ($P(X \mid \boldsymbol{\theta})$) est appelée la *vraisemblance*.

### Un exemple concret

Voyons comment cela fonctionne dans un exemple concret.  Supposons que nous ayons un seul paramètre $\theta$ représentant la probabilité qu'un lancer de pièce soit face.  La probabilité d'obtenir un pile est alors $1-\theta$, et donc si nos données observées $X$ sont une séquence avec $n_H$ pile et $n_T$ pile, nous pouvons utiliser le fait que les probabilités indépendantes se multiplient pour voir que 

$$
P(X \mid \theta) = \theta^{n_H}(1-\theta)^{n_T}.
$$

Si nous tirons à pile ou face $13$ et que nous obtenons la séquence "HHHTHTTHHHHHT", qui comporte $n_H = 9$ et $n_T = 4$, nous constatons que cela correspond à

$$
P(X \mid \theta) = \theta^9(1-\theta)^4.
$$

Ce qui est bien dans cet exemple, c'est que nous connaissons la réponse dès le départ.  En effet, si nous disions verbalement : " J'ai tiré à pile ou face 13 pièces, et 9 sont tombées sur face, quelle est notre meilleure estimation de la probabilité que la pièce tombe sur face ? ", tout le monde devinerait correctement $9/13$.  La méthode du maximum de vraisemblance nous permettra d'obtenir ce chiffre à partir des principes de base, d'une manière qui pourra être généralisée à des situations beaucoup plus complexes.

Pour notre exemple, le tracé de $P(X \mid \theta)$ est le suivant :

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import autograd, np, npx
npx.set_np()

theta = np.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

theta = torch.arange(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

theta = tf.range(0, 1, 0.001)
p = theta**9 * (1 - theta)**4.

d2l.plot(theta, p, 'theta', 'likelihood')
```

Sa valeur maximale se situe quelque part près de notre valeur attendue $9/13 \approx 0.7\ldots$.  Pour voir si c'est exactement là, nous pouvons nous tourner vers le calcul.  Remarquez qu'au maximum, le gradient de la fonction est plat.  Ainsi, nous pouvons trouver l'estimation du maximum de vraisemblance :eqref:`eq_max_like` en trouvant les valeurs de $\theta$ où la dérivée est nulle, et en trouvant celle qui donne la probabilité la plus élevée.  Nous calculons :

$$
\begin{aligned}
0 & = \frac{d}{d\theta} P(X \mid \theta) \\
& = \frac{d}{d\theta} \theta^9(1-\theta)^4 \\
& = 9\theta^8(1-\theta)^4 - 4\theta^9(1-\theta)^3 \\
& = \theta^8(1-\theta)^3(9-13\theta).
\end{aligned}
$$

Il y a trois solutions : $0$, $1$ et $9/13$.  Les deux premières sont clairement des minima, et non des maxima, car elles attribuent la probabilité $0$ à notre séquence.  La valeur finale n'attribue *pas* une probabilité nulle à notre séquence, et doit donc être l'estimation du maximum de vraisemblance $\hat \theta = 9/13$.

## Optimisation numérique et log-vraisemblance négative

L'exemple précédent est intéressant, mais que se passe-t-il si nous avons des milliards de paramètres et d'exemples de données ?

Tout d'abord, remarquez que si nous faisons l'hypothèse que tous les exemples de données sont indépendants, nous ne pouvons plus pratiquement considérer la vraisemblance elle-même car elle est un produit de nombreuses probabilités.  En effet, chaque probabilité est dans $[0,1]$, disons typiquement d'une valeur d'environ $1/2$, et le produit de $(1/2)^{1000000000}$ est bien en dessous de la précision de la machine.  Nous ne pouvons pas travailler avec cela directement. 

Cependant, rappelez-vous que le logarithme transforme les produits en sommes, auquel cas 

$$
\log((1/2)^{1000000000}) = 1000000000\cdot\log(1/2) \approx -301029995.6\ldots
$$

Ce nombre s'inscrit parfaitement dans un flotteur à simple précision $32$-bit.  Nous devrions donc considérer la *log-vraisemblance*, soit

$$
\log(P(X \mid \boldsymbol{\theta})).
$$

Comme la fonction $x \mapsto \log(x)$ est croissante, maximiser la vraisemblance est la même chose que maximiser la log-vraisemblance.  En effet, dans :numref:`sec_naive_bayes`, nous verrons ce raisonnement appliqué en travaillant avec l'exemple spécifique du classifieur de Bayes naïf.

Nous travaillons souvent avec des fonctions de perte, où nous souhaitons minimiser la perte.  Nous pouvons transformer le maximum de vraisemblance en minimisation d'une perte en prenant $-\log(P(X \mid \boldsymbol{\theta}))$, qui est la *log-vraisemblance négative*.

Pour illustrer cela, reprenons le problème du tirage à pile ou face que nous avons vu précédemment, et supposons que nous ne connaissons pas la solution sous forme fermée.  Nous pouvons calculer que

$$
-\log(P(X \mid \boldsymbol{\theta})) = -\log(\theta^{n_H}(1-\theta)^{n_T}) = -(n_H\log(\theta) + n_T\log(1-\theta)).
$$

Cette solution peut être écrite en code, et optimisée librement, même pour des milliards de tirages à pile ou face.

```{.python .input}
#@tab mxnet
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = np.array(0.5)
theta.attach_grad()

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with autograd.record():
        loss = -(n_H * np.log(theta) + n_T * np.log(1 - theta))
    loss.backward()
    theta -= lr * theta.grad

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab pytorch
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = torch.tensor(0.5, requires_grad=True)

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    loss = -(n_H * torch.log(theta) + n_T * torch.log(1 - theta))
    loss.backward()
    with torch.no_grad():
        theta -= lr * theta.grad
    theta.grad.zero_()

# Check output
theta, n_H / (n_H + n_T)
```

```{.python .input}
#@tab tensorflow
# Set up our data
n_H = 8675309
n_T = 25624

# Initialize our paramteres
theta = tf.Variable(tf.constant(0.5))

# Perform gradient descent
lr = 0.00000000001
for iter in range(10):
    with tf.GradientTape() as t:
        loss = -(n_H * tf.math.log(theta) + n_T * tf.math.log(1 - theta))
    theta.assign_sub(lr * t.gradient(loss, theta))

# Check output
theta, n_H / (n_H + n_T)
```

La commodité numérique n'est pas la seule raison pour laquelle les gens aiment utiliser les log-vraisemblances négatives. Il existe plusieurs autres raisons pour lesquelles elle est préférable.



La deuxième raison pour laquelle nous considérons la log-vraisemblance est l'application simplifiée des règles de calcul. Comme nous l'avons vu plus haut, en raison des hypothèses d'indépendance, la plupart des probabilités que nous rencontrons en apprentissage automatique sont des produits de probabilités individuelles.

$$
P(X\mid\boldsymbol{\theta}) = p(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta})\cdots p(x_n\mid\boldsymbol{\theta}).
$$

Cela signifie que si nous appliquons directement la règle du produit pour calculer une dérivée, nous obtenons

$$
\begin{aligned}
\frac{\partial}{\partial \boldsymbol{\theta}} P(X\mid\boldsymbol{\theta}) & = \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right)\cdot P(x_2\mid\boldsymbol{\theta})\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_2\mid\boldsymbol{\theta})\right)\cdots P(x_n\mid\boldsymbol{\theta}) \\
& \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad \vdots \\
& \quad + P(x_1\mid\boldsymbol{\theta})\cdot P(x_2\mid\boldsymbol{\theta}) \cdots \left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
\end{aligned}
$$

Cela nécessite $n(n-1)$ multiplications, ainsi que $(n-1)$ additions, ce qui est donc proportionnel à un temps quadratique des entrées !  Une astuce suffisante pour regrouper les termes ramènera ce temps à un temps linéaire, mais cela demande un peu de réflexion.  Pour la log-vraisemblance négative, nous avons plutôt

$$
-\log\left(P(X\mid\boldsymbol{\theta})\right) = -\log(P(x_1\mid\boldsymbol{\theta})) - \log(P(x_2\mid\boldsymbol{\theta})) \cdots - \log(P(x_n\mid\boldsymbol{\theta})),
$$

ce qui donne

$$
- \frac{\partial}{\partial \boldsymbol{\theta}} \log\left(P(X\mid\boldsymbol{\theta})\right) = \frac{1}{P(x_1\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_1\mid\boldsymbol{\theta})\right) + \cdots + \frac{1}{P(x_n\mid\boldsymbol{\theta})}\left(\frac{\partial}{\partial \boldsymbol{\theta}}P(x_n\mid\boldsymbol{\theta})\right).
$$

Cela ne nécessite que $n$ diviseurs et $n-1$ sommes, et donc un temps linéaire dans les entrées.

La troisième et dernière raison de considérer la log-vraisemblance négative est la relation avec la théorie de l'information, que nous aborderons en détail dans :numref:`sec_information_theory`.  Il s'agit d'une théorie mathématique rigoureuse qui donne un moyen de mesurer le degré d'information ou d'aléa d'une variable aléatoire.  L'objet d'étude clé dans ce domaine est l'entropie qui est 

$$
H(p) = -\sum_{i} p_i \log_2(p_i),
$$

qui mesure le caractère aléatoire d'une source. Remarquez que ce n'est rien d'autre que la probabilité moyenne $-\log$. Ainsi, si nous prenons notre log-vraisemblance négative et la divisons par le nombre d'exemples de données, nous obtenons un relatif de l'entropie connu sous le nom d'entropie croisée.  Cette interprétation théorique serait à elle seule suffisamment convaincante pour motiver la déclaration de la log-vraisemblance négative moyenne sur l'ensemble de données comme moyen de mesurer la performance du modèle.

## Maximum de vraisemblance pour les variables continues

Tout ce que nous avons fait jusqu'à présent suppose que nous travaillons avec des variables aléatoires discrètes, mais que faire si nous voulons travailler avec des variables continues ?

En résumé, rien ne change, si ce n'est que nous remplaçons toutes les instances de la probabilité par la densité de probabilité.  Si l'on se souvient que les densités s'écrivent en minuscules à l'adresse $p$, cela signifie que, par exemple, nous disons désormais

$$
-\log\left(p(X\mid\boldsymbol{\theta})\right) = -\log(p(x_1\mid\boldsymbol{\theta})) - \log(p(x_2\mid\boldsymbol{\theta})) \cdots - \log(p(x_n\mid\boldsymbol{\theta})) = -\sum_i \log(p(x_i \mid \theta)).
$$

La question qui se pose est la suivante : "Pourquoi est-ce correct ?"  Après tout, la raison pour laquelle nous avons introduit les densités était que les probabilités d'obtenir des résultats spécifiques étaient elles-mêmes nulles, et donc la probabilité de générer nos données pour tout ensemble de paramètres n'est-elle pas nulle ?

En effet, c'est le cas, et comprendre pourquoi nous pouvons passer aux densités est un exercice qui consiste à retracer ce qui arrive aux epsilons.

Commençons par redéfinir notre objectif.  Supposons que pour les variables aléatoires continues, nous ne voulons plus calculer la probabilité d'obtenir exactement la bonne valeur, mais plutôt la correspondance dans un certain intervalle $\epsilon$.  Pour simplifier, nous supposons que nos données sont des observations répétées $x_1, \ldots, x_N$ de variables aléatoires distribuées de manière identique $X_1, \ldots, X_N$.  Comme nous l'avons vu précédemment, cela peut s'écrire comme suit

$$
\begin{aligned}
&P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta}) \\
\approx &\epsilon^Np(x_1\mid\boldsymbol{\theta})\cdot p(x_2\mid\boldsymbol{\theta}) \cdots p(x_n\mid\boldsymbol{\theta}).
\end{aligned}
$$

Ainsi, si nous prenons le logarithme négatif de cette valeur, nous obtenons

$$
\begin{aligned}
&-\log(P(X_1 \in [x_1, x_1+\epsilon], X_2 \in [x_2, x_2+\epsilon], \ldots, X_N \in [x_N, x_N+\epsilon]\mid\boldsymbol{\theta})) \\
\approx & -N\log(\epsilon) - \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
\end{aligned}
$$

Si nous examinons cette expression, le seul endroit où se trouve $\epsilon$ est dans la constante additive $-N\log(\epsilon)$.  Celle-ci ne dépend pas du tout des paramètres $\boldsymbol{\theta}$, donc le choix optimal de $\boldsymbol{\theta}$ ne dépend pas de notre choix de $\epsilon$!  Si nous demandons quatre chiffres ou quatre cents, le choix optimal de $\boldsymbol{\theta}$ reste le même, donc nous pouvons librement laisser tomber l'epsilon pour voir que ce que nous voulons optimiser est

$$
- \sum_{i} \log(p(x_i\mid\boldsymbol{\theta})).
$$

Ainsi, nous voyons que le point de vue du maximum de vraisemblance peut fonctionner avec des variables aléatoires continues aussi facilement qu'avec des variables discrètes en remplaçant les probabilités par des densités de probabilité.

## Résumé
* Le principe du maximum de vraisemblance nous dit que le modèle le mieux adapté pour un ensemble de données donné est celui qui génère les données avec la probabilité la plus élevée.
* On travaille souvent avec la log-vraisemblance négative pour diverses raisons : stabilité numérique, conversion des produits en sommes (et simplification des calculs de gradient qui en résulte), et liens théoriques avec la théorie de l'information.
* Bien que la motivation soit plus simple dans le cadre discret, elle peut être librement généralisée au cadre continu en maximisant la densité de probabilité attribuée aux points de données.

## Exercices
1. Supposons que vous sachiez qu'une variable aléatoire non négative a une densité $\alpha e^{-\alpha x}$ pour une certaine valeur $\alpha>0$.  Vous obtenez une seule observation de la variable aléatoire, à savoir le nombre $3$.  Quelle est l'estimation du maximum de vraisemblance pour $\alpha$?
2. Supposons que vous disposiez d'un ensemble de données d'échantillons $\{x_i\}_{i=1}^N$ tirés d'une gaussienne de moyenne inconnue, mais de variance $1$.  Quelle est l'estimation du maximum de vraisemblance pour la moyenne ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/416)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1096)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1097)
:end_tab:
