# Descente de gradient
:label:`sec_gd` 

 Dans cette section, nous allons présenter les concepts de base qui sous-tendent la *descente de gradient*.
Bien qu'elle soit rarement utilisée directement en apprentissage profond, la compréhension de la descente de gradient est essentielle pour comprendre les algorithmes de descente de gradient stochastique. 
Par exemple, le problème d'optimisation peut diverger en raison d'un taux d'apprentissage trop élevé. Ce phénomène peut déjà être observé dans la descente de gradient. De même, le préconditionnement est une technique courante dans la descente de gradient et se retrouve dans des algorithmes plus avancés.
Commençons par un cas particulier simple.


## Descente de gradient unidimensionnelle

La descente de gradient en une dimension est un excellent exemple pour expliquer pourquoi l'algorithme de descente de gradient peut réduire la valeur de la fonction objectif. Considérons une fonction réelle continuellement différentiable $f: \mathbb{R} \rightarrow \mathbb{R}$. En utilisant une expansion de Taylor, nous obtenons

$$f(x + \epsilon) = f(x) + \epsilon f'(x) + \mathcal{O}(\epsilon^2).$$ 
 :eqlabel:`gd-taylor` 

 C'est-à-dire que dans l'approximation du premier ordre, $f(x+\epsilon)$ est donné par la valeur de la fonction $f(x)$ et la première dérivée $f'(x)$ à $x$. Il n'est pas déraisonnable de supposer que pour de petites valeurs de $\epsilon$, un déplacement dans la direction du gradient négatif diminuera $f$. Pour garder les choses simples, nous choisissons une taille de pas fixe $\eta > 0$ et choisissons $\epsilon = -\eta f'(x)$. En intégrant ce résultat dans l'expansion de Taylor ci-dessus, nous obtenons

$$f(x - \eta f'(x)) = f(x) - \eta f'^2(x) + \mathcal{O}(\eta^2 f'^2(x)).$$ 
 :eqlabel:`gd-taylor-2` 

 Si la dérivée $f'(x) \neq 0$ ne disparaît pas, nous progressons depuis $\eta f'^2(x)>0$. De plus, nous pouvons toujours choisir $\eta$ suffisamment petit pour que les termes d'ordre supérieur ne soient plus pertinents. Nous arrivons donc à

$$f(x - \eta f'(x)) \lessapprox f(x).$$ 

 Cela signifie que, si nous utilisons

$$x \leftarrow x - \eta f'(x)$$ 

 pour itérer $x$, la valeur de la fonction $f(x)$ pourrait diminuer. Par conséquent, dans la descente par gradient, nous choisissons d'abord une valeur initiale $x$ et une constante $\eta > 0$, puis nous les utilisons pour itérer continuellement $x$ jusqu'à ce que la condition d'arrêt soit atteinte, par exemple, lorsque l'amplitude du gradient $|f'(x)|$ est suffisamment faible ou que le nombre d'itérations a atteint une certaine valeur.

Par souci de simplicité, nous choisissons la fonction objectif $f(x)=x^2$ pour illustrer la mise en œuvre de la descente par gradient. Bien que nous sachions que $x=0$ est la solution pour minimiser $f(x)$, nous utilisons toujours cette fonction simple pour observer comment $x$ évolue.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import numpy as np
import torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import numpy as np
import tensorflow as tf
```

```{.python .input}
#@tab all
def f(x):  # Objective function
    return x ** 2

def f_grad(x):  # Gradient (derivative) of the objective function
    return 2 * x
```

Ensuite, nous utilisons $x=10$ comme valeur initiale et supposons $\eta=0.2$. En utilisant la descente de gradient pour itérer $x$ pendant 10 fois, nous pouvons voir que, finalement, la valeur de $x$ se rapproche de la solution optimale.

```{.python .input}
#@tab all
def gd(eta, f_grad):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x)
        results.append(float(x))
    print(f'epoch 10, x: {x:f}')
    return results

results = gd(0.2, f_grad)
```

La progression de l'optimisation sur $x$ peut être tracée comme suit.

```{.python .input}
#@tab all
def show_trace(results, f):
    n = max(abs(min(results)), abs(max(results)))
    f_line = d2l.arange(-n, n, 0.01)
    d2l.set_figsize()
    d2l.plot([f_line, results], [[f(x) for x in f_line], [
        f(x) for x in results]], 'x', 'f(x)', fmts=['-', '-o'])

show_trace(results, f)
```

### Taux d'apprentissage
:label:`subsec_gd-learningrate` 

 Le taux d'apprentissage $\eta$ peut être fixé par le concepteur de l'algorithme. Si nous utilisons un taux d'apprentissage trop faible, $x$ se mettra à jour très lentement, ce qui nécessitera davantage d'itérations pour obtenir une meilleure solution. Pour montrer ce qui se passe dans un tel cas, considérons la progression du même problème d'optimisation pour $\eta = 0.05$. Comme nous pouvons le constater, même après 10 étapes, nous sommes encore très loin de la solution optimale.

```{.python .input}
#@tab all
show_trace(gd(0.05, f_grad), f)
```

Inversement, si nous utilisons un taux d'apprentissage excessivement élevé, $\left|\eta f'(x)\right|$ peut être trop grand pour la formule d'expansion de Taylor du premier ordre. En d'autres termes, le terme $\mathcal{O}(\eta^2 f'^2(x))$ dans :eqref:`gd-taylor-2` pourrait devenir significatif. Dans ce cas, nous ne pouvons pas garantir que l'itération de $x$ sera capable de réduire la valeur de $f(x)$. Par exemple, lorsque nous fixons le taux d'apprentissage à $\eta=1.1$, $x$ dépasse la solution optimale $x=0$ et diverge progressivement.

```{.python .input}
#@tab all
show_trace(gd(1.1, f_grad), f)
```

### Minima locaux

Pour illustrer ce qui se passe pour les fonctions non convexes, considérons le cas de $f(x) = x \cdot \cos(cx)$ pour une certaine constante $c$. Cette fonction possède une infinité de minima locaux. En fonction de notre choix du taux d'apprentissage et de la qualité du problème, nous pouvons obtenir une solution parmi de nombreuses autres. L'exemple ci-dessous illustre comment un taux d'apprentissage élevé (irréaliste) conduira à un mauvais minimum local.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

show_trace(gd(2, f_grad), f)
```

## Descente de gradient multivariée

Maintenant que nous avons une meilleure intuition du cas univarié, considérons la situation où $\mathbf{x} = [x_1, x_2, \ldots, x_d]^\top$. C'est-à-dire que la fonction objectif $f: \mathbb{R}^d \to \mathbb{R}$ transforme des vecteurs en scalaires. Par conséquent, son gradient est également multivarié. Il s'agit d'un vecteur composé de $d$ dérivées partielles :

$$\nabla f(\mathbf{x}) = \bigg[\frac{\partial f(\mathbf{x})}{\partial x_1}, \frac{\partial f(\mathbf{x})}{\partial x_2}, \ldots, \frac{\partial f(\mathbf{x})}{\partial x_d}\bigg]^\top.$$ 

 Chaque élément de dérivée partielle $\partial f(\mathbf{x})/\partial x_i$ dans le gradient indique le taux de variation de $f$ à $\mathbf{x}$ par rapport à l'entrée $x_i$. Comme précédemment dans le cas univarié, nous pouvons utiliser l'approximation de Taylor correspondante pour les fonctions multivariées pour avoir une idée de ce que nous devons faire. En particulier, nous avons que

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \mathbf{\boldsymbol{\epsilon}}^\top \nabla f(\mathbf{x}) + \mathcal{O}(\|\boldsymbol{\epsilon}\|^2).$$ 
 :eqlabel:`gd-multi-taylor` 

 En d'autres termes, jusqu'aux termes du second ordre dans $\boldsymbol{\epsilon}$ la direction de la descente la plus raide est donnée par le gradient négatif $-\nabla f(\mathbf{x})$. En choisissant un taux d'apprentissage approprié $\eta > 0$, on obtient l'algorithme prototype de descente par gradient :

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \nabla f(\mathbf{x}).$$ 

 Pour voir comment l'algorithme se comporte en pratique, construisons une fonction objectif $f(\mathbf{x})=x_1^2+2x_2^2$ avec un vecteur bidimensionnel $\mathbf{x} = [x_1, x_2]^\top$ en entrée et un scalaire en sortie. Le gradient est donné par $\nabla f(\mathbf{x}) = [2x_1, 4x_2]^\top$. Nous allons observer la trajectoire de $\mathbf{x}$ par descente de gradient à partir de la position initiale $[-5, -2]$. 

Pour commencer, nous avons besoin de deux autres fonctions d'aide. La première utilise une fonction de mise à jour et l'applique 20 fois à la valeur initiale. La deuxième fonction d'aide visualise la trajectoire de $\mathbf{x}$.

```{.python .input}
#@tab all
def train_2d(trainer, steps=20, f_grad=None):  #@save
    """Optimize a 2D objective function with a customized trainer."""
    # `s1` and `s2` are internal state variables that will be used in Momentum, adagrad, RMSProp
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(steps):
        if f_grad:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2, f_grad)
        else:
            x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print(f'epoch {i + 1}, x1: {float(x1):f}, x2: {float(x2):f}')
    return results

def show_trace_2d(f, results):  #@save
    """Show the trace of 2D variables during optimization."""
    d2l.set_figsize()
    d2l.plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = d2l.meshgrid(d2l.arange(-5.5, 1.0, 0.1),
                          d2l.arange(-3.0, 1.0, 0.1))
    d2l.plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    d2l.plt.xlabel('x1')
    d2l.plt.ylabel('x2')
```

Ensuite, nous observons la trajectoire de la variable d'optimisation $\mathbf{x}$ pour le taux d'apprentissage $\eta = 0.1$. Nous pouvons voir qu'après 20 étapes, la valeur de $\mathbf{x}$ s'approche de son minimum à $[0, 0]$. La progression est assez régulière, bien que plutôt lente.

```{.python .input}
#@tab all
def f_2d(x1, x2):  # Objective function
    return x1 ** 2 + 2 * x2 ** 2

def f_2d_grad(x1, x2):  # Gradient of the objective function
    return (2 * x1, 4 * x2)

def gd_2d(x1, x2, s1, s2, f_grad):
    g1, g2 = f_grad(x1, x2)
    return (x1 - eta * g1, x2 - eta * g2, 0, 0)

eta = 0.1
show_trace_2d(f_2d, train_2d(gd_2d, f_grad=f_2d_grad))
```

## Méthodes adaptatives

Comme nous avons pu le voir sur :numref:`subsec_gd-learningrate` , il est difficile de trouver le taux d'apprentissage $\eta$ "juste". Si nous le choisissons trop petit, nous progressons peu. Si nous le choisissons trop grand, la solution oscille et, dans le pire des cas, elle peut même diverger. Que se passerait-il si nous pouvions déterminer $\eta$ automatiquement ou ne pas avoir à sélectionner un taux d'apprentissage du tout ? 
Les méthodes du second ordre qui examinent non seulement la valeur et le gradient de la fonction objectif
mais aussi sa *courbure* peuvent être utiles dans ce cas. Bien que ces méthodes ne puissent pas être appliquées directement à l'apprentissage profond en raison du coût de calcul, elles fournissent une intuition utile sur la façon de concevoir des algorithmes d'optimisation avancés qui imitent plusieurs des propriétés souhaitables des algorithmes décrits ci-dessous.


### Méthode de Newton

En examinant l'expansion de Taylor d'une certaine fonction $f: \mathbb{R}^d \rightarrow \mathbb{R}$, il n'est pas nécessaire de s'arrêter après le premier terme. En fait, nous pouvons l'écrire comme suit :

$$f(\mathbf{x} + \boldsymbol{\epsilon}) = f(\mathbf{x}) + \boldsymbol{\epsilon}^\top \nabla f(\mathbf{x}) + \frac{1}{2} \boldsymbol{\epsilon}^\top \nabla^2 f(\mathbf{x}) \boldsymbol{\epsilon} + \mathcal{O}(\|\boldsymbol{\epsilon}\|^3).$$ 
 :eqlabel:`gd-hot-taylor` 

 Pour éviter une notation trop lourde, nous définissons $\mathbf{H} \stackrel{\mathrm{def}}{=} \nabla^2 f(\mathbf{x})$ comme étant le Hessien de $f$, qui est une matrice $d \times d$. Pour les petites $d$ et les problèmes simples, $\mathbf{H}$ est facile à calculer. En revanche, pour les réseaux neuronaux profonds, la taille de $\mathbf{H}$ peut être prohibitive, en raison du coût de stockage des entrées de $\mathcal{O}(d^2)$. En outre, il peut être trop coûteux de le calculer par rétropropagation. Pour l'instant, ignorons ces considérations et examinons l'algorithme que nous obtiendrions.

Après tout, le minimum de $f$ satisfait $\nabla f = 0$. 
En suivant les règles de calcul de :numref:`subsec_calculus-grad` ,
en prenant les dérivées de :eqref:`gd-hot-taylor` par rapport à $\boldsymbol{\epsilon}$ et en ignorant les termes d'ordre supérieur, nous obtenons

$$\nabla f(\mathbf{x}) + \mathbf{H} \boldsymbol{\epsilon} = 0 \text{ and hence }
\boldsymbol{\epsilon} = -\mathbf{H}^{-1} \nabla f(\mathbf{x}).$$

C'est-à-dire que nous devons inverser le hessien $\mathbf{H}$ dans le cadre du problème d'optimisation.

À titre d'exemple simple, pour $f(x) = \frac{1}{2} x^2$, nous avons $\nabla f(x) = x$ et $\mathbf{H} = 1$. Par conséquent, pour tout $x$, nous obtenons $\epsilon = -x$. En d'autres termes, une *seule* étape suffit pour converger parfaitement sans avoir besoin d'aucun ajustement ! Hélas, nous avons eu un peu de chance ici : l'expansion de Taylor était exacte depuis $f(x+\epsilon)= \frac{1}{2} x^2 + \epsilon x + \frac{1}{2} \epsilon^2$. 

Voyons ce qui se passe dans d'autres problèmes.
Étant donné une fonction cosinus hyperbolique convexe $f(x) = \cosh(cx)$ pour une certaine constante $c$, nous pouvons voir que
le minimum global à $x=0$ est atteint
après quelques itérations.

```{.python .input}
#@tab all
c = d2l.tensor(0.5)

def f(x):  # Objective function
    return d2l.cosh(c * x)

def f_grad(x):  # Gradient of the objective function
    return c * d2l.sinh(c * x)

def f_hess(x):  # Hessian of the objective function
    return c**2 * d2l.cosh(c * x)

def newton(eta=1):
    x = 10.0
    results = [x]
    for i in range(10):
        x -= eta * f_grad(x) / f_hess(x)
        results.append(float(x))
    print('epoch 10, x:', x)
    return results

show_trace(newton(), f)
```

Considérons maintenant une fonction *non convexe*, telle que $f(x) = x \cos(c x)$ pour une certaine constante $c$. Après tout, notez que dans la méthode de Newton, nous finissons par diviser par le Hessien. Cela signifie que si la dérivée seconde est *négative*, nous pouvons aller dans le sens d'une *augmentation* de la valeur de $f$.
C'est un défaut fatal de l'algorithme. 
Voyons ce qui se passe en pratique.

```{.python .input}
#@tab all
c = d2l.tensor(0.15 * np.pi)

def f(x):  # Objective function
    return x * d2l.cos(c * x)

def f_grad(x):  # Gradient of the objective function
    return d2l.cos(c * x) - c * x * d2l.sin(c * x)

def f_hess(x):  # Hessian of the objective function
    return - 2 * c * d2l.sin(c * x) - x * c**2 * d2l.cos(c * x)

show_trace(newton(), f)
```

C'est une erreur spectaculaire. Comment pouvons-nous le réparer ? Une façon de le faire serait de "réparer" le Hessien en prenant sa valeur absolue à la place. Une autre stratégie consiste à ramener le taux d'apprentissage. Cela semble aller à l'encontre du but recherché, mais pas tout à fait. Le fait de disposer d'informations de second ordre nous permet d'être prudents lorsque la courbure est importante et de prendre des mesures plus longues lorsque la fonction objectif est plus plate. 
Voyons comment cela fonctionne avec un taux d'apprentissage légèrement inférieur, disons $\eta = 0.5$. Comme nous pouvons le constater, nous avons un algorithme assez efficace.

```{.python .input}
#@tab all
show_trace(newton(0.5), f)
```

### Analyse de convergence

Nous analysons uniquement le taux de convergence de la méthode de Newton pour une fonction objectif convexe et trois fois différentiable $f$, où la dérivée seconde est non nulle, c'est-à-dire $f'' > 0$. La preuve multivariée est une extension directe de l'argument unidimensionnel ci-dessous et est omise car elle ne nous aide pas beaucoup en termes d'intuition.

Dénotez par $x^{(k)}$ la valeur de $x$ à l'itération $k^\mathrm{th}$ et laissez $e^{(k)} \stackrel{\mathrm{def}}{=} x^{(k)} - x^*$ être la distance de l'optimalité à l'itération $k^\mathrm{th}$. Par expansion de Taylor, nous avons que la condition $f'(x^*) = 0$ peut être écrite sous la forme

$$0 = f'(x^{(k)} - e^{(k)}) = f'(x^{(k)}) - e^{(k)} f''(x^{(k)}) + \frac{1}{2} (e^{(k)})^2 f'''(\xi^{(k)}),$$ 

 qui tient pour un certain $\xi^{(k)} \in [x^{(k)} - e^{(k)}, x^{(k)}]$. En divisant l'expansion ci-dessus par $f''(x^{(k)})$, on obtient

$$e^{(k)} - \frac{f'(x^{(k)})}{f''(x^{(k)})} = \frac{1}{2} (e^{(k)})^2 \frac{f'''(\xi^{(k)})}{f''(x^{(k)})}.$$ 

 Rappelons que nous avons la mise à jour $x^{(k+1)} = x^{(k)} - f'(x^{(k)}) / f''(x^{(k)})$. 
En introduisant cette équation de mise à jour et en prenant la valeur absolue des deux côtés, nous obtenons

$$\left|e^{(k+1)}\right| = \frac{1}{2}(e^{(k)})^2 \frac{\left|f'''(\xi^{(k)})\right|}{f''(x^{(k)})}.$$ 

 Par conséquent, chaque fois que nous nous trouvons dans une région limitée $\left|f'''(\xi^{(k)})\right| / (2f''(x^{(k)})) \leq c$, nous avons une erreur quadratiquement décroissante 

$$\left|e^{(k+1)}\right| \leq c (e^{(k)})^2.$$ 

 
 En passant, les chercheurs en optimisation appellent cela une convergence *linéaire*, alors qu'une condition telle que $\left|e^{(k+1)}\right| \leq \alpha \left|e^{(k)}\right|$ serait appelée un taux de convergence *constant*.
Notez que cette analyse s'accompagne d'un certain nombre de mises en garde. 
Premièrement, nous n'avons pas vraiment de garantie quant au moment où nous atteindrons la région de convergence rapide. Au contraire, nous savons seulement qu'une fois que nous l'aurons atteinte, la convergence sera très rapide. Deuxièmement, cette analyse exige que $f$ se comporte bien jusqu'aux dérivées d'ordre supérieur. Il s'agit de s'assurer que $f$ n'a pas de propriétés "surprenantes" en ce qui concerne la façon dont il pourrait changer ses valeurs.



### Préconditionnement

Sans surprise, le calcul et le stockage du Hessien complet sont très coûteux. Il est donc souhaitable de trouver des alternatives. Une façon d'améliorer les choses est le *préconditionnement*. Il évite de calculer le Hessien dans son intégralité mais ne calcule que les entrées de la *diagonale*. Cela conduit à des algorithmes de mise à jour de la forme

$$\mathbf{x} \leftarrow \mathbf{x} - \eta \mathrm{diag}(\mathbf{H})^{-1} \nabla f(\mathbf{x}).$$ 

 
 . Bien que cette méthode ne soit pas tout à fait aussi bonne que la méthode de Newton complète, elle est toujours bien meilleure que sa non-utilisation. 
Pour comprendre pourquoi cela peut être une bonne idée, considérons une situation où une variable représente la hauteur en millimètres et l'autre la hauteur en kilomètres. En supposant que pour les deux, l'échelle naturelle est en mètres, nous avons un terrible décalage dans les paramétrages. Heureusement, l'utilisation du préconditionnement permet d'éliminer ce problème. Un préconditionnement efficace avec la descente de gradient revient à sélectionner un taux d'apprentissage différent pour chaque variable (coordonnée du vecteur $\mathbf{x}$).
Comme nous le verrons plus tard, le préconditionnement est à l'origine de certaines innovations dans les algorithmes d'optimisation par descente de gradient stochastique. 


### Descente de gradient avec recherche linéaire

L'un des principaux problèmes de la descente de gradient est que nous risquons de dépasser l'objectif ou de ne pas progresser suffisamment. Une solution simple à ce problème consiste à utiliser la recherche linéaire en conjonction avec la descente de gradient. Autrement dit, nous utilisons la direction donnée par $\nabla f(\mathbf{x})$ et nous effectuons ensuite une recherche binaire pour déterminer quel taux d'apprentissage $\eta$ minimise $f(\mathbf{x} - \eta \nabla f(\mathbf{x}))$.

Cet algorithme converge rapidement (pour une analyse et une preuve, voir par exemple :cite:`Boyd.Vandenberghe.2004` ). Cependant, pour l'apprentissage profond, ce n'est pas aussi faisable, car chaque étape de la recherche linéaire nous obligerait à évaluer la fonction objectif sur l'ensemble des données. Cela est beaucoup trop coûteux à réaliser.

## Résumé

* Les taux d'apprentissage sont importants. S'il est trop élevé, nous divergeons, s'il est trop faible, nous ne progressons pas.
* La descente de gradient peut rester bloquée dans des minima locaux.
* En haute dimension, ajuster le taux d'apprentissage est compliqué.
* Le préconditionnement peut aider à ajuster l'échelle.
* La méthode de Newton est beaucoup plus rapide une fois qu'elle a commencé à fonctionner correctement dans les problèmes convexes.
* Attention à ne pas utiliser la méthode de Newton sans aucun ajustement pour les problèmes non convexes.

## Exercices

1. Expérimentez différents taux d'apprentissage et fonctions objectives pour la descente de gradient.
1. Mettez en œuvre la recherche linéaire pour minimiser une fonction convexe dans l'intervalle $[a, b]$.
   1. Avez-vous besoin de dérivés pour la recherche binaire, c'est-à-dire pour décider s'il faut choisir $[a, (a+b)/2]$ ou $[(a+b)/2, b]$.
 1. Quelle est la vitesse de convergence de l'algorithme ?
   1. Mettez en œuvre l'algorithme et appliquez-le à la minimisation de $\log (\exp(x) + \exp(-2x -3))$.
1. Concevez une fonction objectif définie sur $\mathbb{R}^2$ où la descente du gradient est extrêmement lente. Conseil : modifiez l'échelle des différentes coordonnées.
1. Implémentez la version allégée de la méthode de Newton en utilisant le préconditionnement :
 1. Utilisez la diagonale du Hessian comme préconditionneur.
   1. Utilisez les valeurs absolues de celle-ci plutôt que les valeurs réelles (éventuellement signées).
   1. Appliquez ceci au problème ci-dessus.
1. Appliquez l'algorithme ci-dessus à un certain nombre de fonctions objectives (convexes ou non). Que se passe-t-il si vous faites pivoter les coordonnées de $45$ degrés ?

[Discussions](https://discuss.d2l.ai/t/351)
