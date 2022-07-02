# Distributions
:label:`sec_distributions` 

 Maintenant que nous avons appris à travailler avec les probabilités dans le cadre discret et continu, apprenons à connaître certaines des distributions les plus courantes.  Selon le domaine de l'apprentissage automatique, il se peut que nous devions nous familiariser avec un nombre beaucoup plus important de ces distributions, voire aucune pour certains domaines de l'apprentissage profond.  Il s'agit toutefois d'une bonne liste de base à connaître.  Commençons par importer certaines bibliothèques courantes.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from IPython import display
from math import erf, factorial
import numpy as np
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
from IPython import display
from math import erf, factorial
import torch

torch.pi = torch.acos(torch.zeros(1)) * 2  # Define pi in torch
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from IPython import display
from math import erf, factorial
import tensorflow as tf
import tensorflow_probability as tfp

tf.pi = tf.acos(tf.zeros(1)) * 2  # Define pi in TensorFlow
```

## Bernoulli

Il s'agit de la variable aléatoire la plus simple que l'on rencontre habituellement.  Cette variable aléatoire encode un tirage à pile ou face qui donne $1$ avec une probabilité de $p$ et $0$ avec une probabilité de $1-p$.  Si nous avons une variable aléatoire $X$ avec cette distribution, nous écrirons

$$
X \sim \mathrm{Bernoulli}(p).
$$

La fonction de distribution cumulative est 

$$F(x) = \begin{cases} 0 & x < 0, \\ 1-p & 0 \le x < 1, \\ 1 & x >= 1 . \end{cases}$$
:eqlabel:`eq_bernoulli_cdf`

La fonction de masse de probabilité est tracée ci-dessous.

```{.python .input}
#@tab all
p = 0.3

d2l.set_figsize()
d2l.plt.stem([0, 1], [1 - p, p], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Traçons maintenant la fonction de distribution cumulative :eqref:`eq_bernoulli_cdf` .

```{.python .input}
#@tab mxnet
x = np.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 2, 0.01)

def F(x):
    return 0 if x < 0 else 1 if x > 1 else 1 - p

d2l.plot(x, tf.constant([F(y) for y in x]), 'x', 'c.d.f.')
```

Si $X \sim \mathrm{Bernoulli}(p)$, alors :

* $\mu_X = p$,
* $\sigma_X^2 = p(1-p)$.

Nous pouvons échantillonner un tableau de forme arbitraire à partir d'une variable aléatoire de Bernoulli comme suit.

```{.python .input}
#@tab mxnet
1*(np.random.rand(10, 10) < p)
```

```{.python .input}
#@tab pytorch
1*(torch.rand(10, 10) < p)
```

```{.python .input}
#@tab tensorflow
tf.cast(tf.random.uniform((10, 10)) < p, dtype=tf.float32)
```

## Uniforme discret

La prochaine variable aléatoire couramment rencontrée est un uniforme discret.  Pour notre discussion ici, nous supposerons qu'elle est supportée par les entiers $\{1, 2, \ldots, n\}$, mais tout autre ensemble de valeurs peut être choisi librement.  La signification du mot *uniforme* dans ce contexte est que toutes les valeurs possibles ont la même probabilité.  La probabilité de chaque valeur $i \in \{1, 2, 3, \ldots, n\}$ est $p_i = \frac{1}{n}$.  Nous désignerons une variable aléatoire $X$ avec cette distribution comme suit

$$
X \sim U(n).
$$

La fonction de distribution cumulative est 

$$F(x) = \begin{cases} 0 & x < 1, \\ \frac{k}{n} & k \le x < k+1 \text{ with } 1 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_discrete_uniform_cdf`

Traçons d'abord la fonction de masse de probabilité.

```{.python .input}
#@tab all
n = 5

d2l.plt.stem([i+1 for i in range(n)], n*[1 / n], use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Traçons ensuite la fonction de distribution cumulative :eqref:`eq_discrete_uniform_cdf` .

```{.python .input}
#@tab mxnet
x = np.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else np.floor(x) / n

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else torch.floor(x) / n

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 6, 0.01)

def F(x):
    return 0 if x < 1 else 1 if x > n else tf.floor(x) / n

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Si $X \sim U(n)$, alors :

* $\mu_X = \frac{1+n}{2}$,
* $\sigma_X^2 = \frac{n^2-1}{12}$.

Nous pouvons échantillonner un tableau de forme arbitraire à partir d'une variable aléatoire uniforme discrète comme suit.

```{.python .input}
#@tab mxnet
np.random.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.randint(1, n, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.uniform((10, 10), 1, n, dtype=tf.int32)
```

## Uniforme continu

Voyons maintenant la distribution uniforme continue. L'idée qui sous-tend cette variable aléatoire est que si nous augmentons la valeur de $n$ dans la distribution uniforme discrète, puis que nous la mettons à l'échelle pour qu'elle s'inscrive dans l'intervalle $[a, b]$, nous nous rapprochons d'une variable aléatoire continue qui choisit une valeur arbitraire dans $[a, b]$ avec une probabilité égale.  Nous désignerons cette distribution par

$$
X \sim U(a, b).
$$

La fonction de densité de probabilité est 

$$p(x) = \begin{cases} \frac{1}{b-a} & x \in [a, b], \\ 0 & x \not\in [a, b].\end{cases}$$ 
 :eqlabel:`eq_cont_uniform_pdf` 

 La fonction de distribution cumulative est 

$$F(x) = \begin{cases} 0 & x < a, \\ \frac{x-a}{b-a} & x \in [a, b], \\ 1 & x >= b . \end{cases}$$
:eqlabel:`eq_cont_uniform_cdf`

Traçons d'abord la fonction de densité de probabilité :eqref:`eq_cont_uniform_pdf` .

```{.python .input}
#@tab mxnet
a, b = 1, 3

x = np.arange(0, 4, 0.01)
p = (x > a)*(x < b)/(b - a)

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
a, b = 1, 3

x = torch.arange(0, 4, 0.01)
p = (x > a).type(torch.float32)*(x < b).type(torch.float32)/(b-a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
a, b = 1, 3

x = tf.range(0, 4, 0.01)
p = tf.cast(x > a, tf.float32) * tf.cast(x < b, tf.float32) / (b - a)
d2l.plot(x, p, 'x', 'p.d.f.')
```

Traçons ensuite la fonction de distribution cumulative :eqref:`eq_cont_uniform_cdf` .

```{.python .input}
#@tab mxnet
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, np.array([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, torch.tensor([F(y) for y in x]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def F(x):
    return 0 if x < a else 1 if x > b else (x - a) / (b - a)

d2l.plot(x, [F(y) for y in x], 'x', 'c.d.f.')
```

Si $X \sim U(a, b)$, alors :

* $\mu_X = \frac{a+b}{2}$,
* $\sigma_X^2 = \frac{(b-a)^2}{12}$.

Nous pouvons échantillonner un tableau de forme arbitraire à partir d'une variable aléatoire uniforme comme suit.  Notez qu'elle échantillonne par défaut à partir de $U(0,1)$, donc si nous voulons une plage différente, nous devons la mettre à l'échelle.

```{.python .input}
#@tab mxnet
(b - a) * np.random.rand(10, 10) + a
```

```{.python .input}
#@tab pytorch
(b - a) * torch.rand(10, 10) + a
```

```{.python .input}
#@tab tensorflow
(b - a) * tf.random.uniform((10, 10)) + a
```

## Binomiale

Rendons les choses un peu plus complexes et examinons la variable aléatoire *binomiale*.  Cette variable aléatoire provient de la réalisation d'une séquence de $n$ expériences indépendantes, dont chacune a la probabilité $p$ de réussir, et de la question de savoir combien de réussites nous attendons.

Exprimons cela mathématiquement.  Chaque expérience est une variable aléatoire indépendante $X_i$ où nous utiliserons $1$ pour coder le succès et $0$ pour coder l'échec.  Puisque chaque expérience est un tirage à pile ou face indépendant qui réussit avec la probabilité $p$, nous pouvons dire que $X_i \sim \mathrm{Bernoulli}(p)$.  La variable aléatoire binomiale est donc

$$
X = \sum_{i=1}^n X_i.
$$

Dans ce cas, nous écrirons

$$
X \sim \mathrm{Binomial}(n, p).
$$

Pour obtenir la fonction de distribution cumulative, nous devons remarquer que l'obtention d'exactement $k$ succès peut se produire de $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ façons dont chacune a une probabilité de $p^k(1-p)^{n-k}$ de se produire.  La fonction de distribution cumulative est donc

$$F(x) = \begin{cases} 0 & x < 0, \\ \sum_{m \le k} \binom{n}{m} p^m(1-p)^{n-m}  & k \le x < k+1 \text{ with } 0 \le k < n, \\ 1 & x >= n . \end{cases}$$
:eqlabel:`eq_binomial_cdf`

Commençons par tracer la fonction de masse de probabilité.

```{.python .input}
#@tab mxnet
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = np.array([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = d2l.tensor([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
n, p = 10, 0.2

# Compute binomial coefficient
def binom(n, k):
    comb = 1
    for i in range(min(k, n - k)):
        comb = comb * (n - i) // (i + 1)
    return comb

pmf = tf.constant([p**i * (1-p)**(n - i) * binom(n, i) for i in range(n + 1)])

d2l.plt.stem([i for i in range(n + 1)], pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Traçons ensuite la fonction de distribution cumulative :eqref:`eq_binomial_cdf` .

```{.python .input}
#@tab mxnet
x = np.arange(-1, 11, 0.01)
cmf = np.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 11, 0.01)
cmf = torch.cumsum(pmf, dim=0)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 11, 0.01)
cmf = tf.cumsum(pmf)

def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Si $X \sim \mathrm{Binomial}(n, p)$, alors :

* $\mu_X = np$,
* $\sigma_X^2 = np(1-p)$.

Cela découle de la linéarité de la valeur attendue sur la somme des variables aléatoires de Bernoulli $n$, et du fait que la variance de la somme de variables aléatoires indépendantes est la somme des variances. On peut échantillonner cette somme comme suit.

```{.python .input}
#@tab mxnet
np.random.binomial(n, p, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.binomial.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Binomial(n, p)
m.sample(sample_shape=(10, 10))
```

## Poisson
Réalisons maintenant une expérience de pensée.  Nous nous trouvons à un arrêt de bus et nous voulons savoir combien de bus vont arriver dans la prochaine minute.  Commençons par considérer $X^{(1)} \sim \mathrm{Bernoulli}(p)$ qui est simplement la probabilité qu'un bus arrive dans la fenêtre d'une minute.  Pour les arrêts de bus éloignés d'un centre urbain, cela peut être une assez bonne approximation.  Il se peut que nous ne voyions jamais plus d'un bus en une minute.

Cependant, si nous nous trouvons dans une zone très fréquentée, il est possible, voire probable, que deux bus arrivent.  Nous pouvons modéliser cela en divisant notre variable aléatoire en deux parties pour les 30 premières secondes, ou les 30 secondes suivantes.  Dans ce cas, nous pouvons écrire

$$
X^{(2)} \sim X^{(2)}_1 + X^{(2)}_2,
$$

où $X^{(2)}$ est la somme totale, et $X^{(2)}_i \sim \mathrm{Bernoulli}(p/2)$.  La distribution totale est alors $X^{(2)} \sim \mathrm{Binomial}(2, p/2)$.

Pourquoi s'arrêter là ?  Continuons à diviser cette minute en $n$ parties.  Par le même raisonnement que ci-dessus, nous voyons que

$$X^{(n)} \sim \mathrm{Binomial}(n, p/n).$$ 
 :eqlabel:`eq_eq_poisson_approx` 

 Considérons ces variables aléatoires.  D'après la section précédente, nous savons que :eqref:`eq_eq_poisson_approx` a pour moyenne $\mu_{X^{(n)}} = n(p/n) = p$ et pour variance $\sigma_{X^{(n)}}^2 = n(p/n)(1-(p/n)) = p(1-p/n)$.  Si nous prenons $n \rightarrow \infty$, nous pouvons voir que ces nombres se stabilisent à $\mu_{X^{(\infty)}} = p$, et la variance $\sigma_{X^{(\infty)}}^2 = p$.  Cela indique qu'il *pourrait y avoir* une variable aléatoire que nous pouvons définir dans cette limite de subdivision infinie. 

Cela ne devrait pas trop nous surprendre, puisque dans le monde réel nous pouvons simplement compter le nombre d'arrivées de bus, mais il est agréable de voir que notre modèle mathématique est bien défini.  Cette discussion peut être formalisée comme la *loi des événements rares*.

En suivant attentivement ce raisonnement, nous pouvons arriver au modèle suivant.  Nous dirons que $X \sim \mathrm{Poisson}(\lambda)$ s'il s'agit d'une variable aléatoire qui prend les valeurs $\{0,1,2, \ldots\}$ avec la probabilité

$$p_k = \frac{\lambda^ke^{-\lambda}}{k!}.$$ 
 :eqlabel:`eq_poisson_mass` 

 La valeur $\lambda > 0$ est appelée le *taux* (ou le *paramètre de forme*), et désigne le nombre moyen d'arrivées que nous attendons en une unité de temps. 

Nous pouvons additionner cette fonction de masse de probabilité pour obtenir la fonction de distribution cumulative.

$$F(x) = \begin{cases} 0 & x < 0, \\ e^{-\lambda}\sum_{m = 0}^k \frac{\lambda^m}{m!} & k \le x < k+1 \text{ with } 0 \le k. \end{cases}$$
:eqlabel:`eq_poisson_cdf`

Traçons d'abord la fonction de masse de probabilité :eqref:`eq_poisson_mass` .

```{.python .input}
#@tab mxnet
lam = 5.0

xs = [i for i in range(20)]
pmf = np.array([np.exp(-lam) * lam**k / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
lam = 5.0

xs = [i for i in range(20)]
pmf = torch.tensor([torch.exp(torch.tensor(-lam)) * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
lam = 5.0

xs = [i for i in range(20)]
pmf = tf.constant([tf.exp(tf.constant(-lam)).numpy() * lam**k
                    / factorial(k) for k in xs])

d2l.plt.stem(xs, pmf, use_line_collection=True)
d2l.plt.xlabel('x')
d2l.plt.ylabel('p.m.f.')
d2l.plt.show()
```

Traçons ensuite la fonction de distribution cumulative :eqref:`eq_poisson_cdf` .

```{.python .input}
#@tab mxnet
x = np.arange(-1, 21, 0.01)
cmf = np.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, np.array([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
x = torch.arange(-1, 21, 0.01)
cmf = torch.cumsum(pmf, dim=0)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, torch.tensor([F(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
x = tf.range(-1, 21, 0.01)
cmf = tf.cumsum(pmf)
def F(x):
    return 0 if x < 0 else 1 if x > n else cmf[int(x)]

d2l.plot(x, [F(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Comme nous l'avons vu plus haut, les moyennes et les variances sont particulièrement concises.  Si $X \sim \mathrm{Poisson}(\lambda)$, alors :

* $\mu_X = \lambda$,
* $\sigma_X^2 = \lambda$.

On peut l'échantillonner comme suit.

```{.python .input}
#@tab mxnet
np.random.poisson(lam, size=(10, 10))
```

```{.python .input}
#@tab pytorch
m = torch.distributions.poisson.Poisson(lam)
m.sample((10, 10))
```

```{.python .input}
#@tab tensorflow
m = tfp.distributions.Poisson(lam)
m.sample((10, 10))
```

## Gaussien
Essayons maintenant une expérience différente, mais connexe.  Disons que nous effectuons à nouveau $n$ mesures indépendantes $\mathrm{Bernoulli}(p)$ $X_i$ .  La distribution de la somme de ces mesures est $X^{(n)} \sim \mathrm{Binomial}(n, p)$.  Plutôt que de prendre une limite lorsque $n$ augmente et $p$ diminue, fixons $p$, puis envoyons $n \rightarrow \infty$.  Dans ce cas, $\mu_{X^{(n)}} = np \rightarrow \infty$ et $\sigma_{X^{(n)}}^2 = np(1-p) \rightarrow \infty$, il n'y a donc aucune raison de penser que cette limite devrait être bien définie.

Cependant, tout espoir n'est pas perdu !  Faisons en sorte que la moyenne et la variance se comportent bien en définissant

$$
Y^{(n)} = \frac{X^{(n)} - \mu_{X^{(n)}}}{\sigma_{X^{(n)}}}.
$$

On peut voir que cette distribution a une moyenne de zéro et une variance de un, et il est donc plausible de croire qu'elle convergera vers une certaine distribution limite.  Si nous représentons graphiquement ce à quoi ressemblent ces distributions, nous serons encore plus convaincus que cela fonctionnera.

```{.python .input}
#@tab mxnet
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = np.array([p**i * (1-p)**(n-i) * binom(n, i) for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/np.sqrt(n*p*(1 - p)) for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab pytorch
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = torch.tensor([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/torch.sqrt(torch.tensor(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

```{.python .input}
#@tab tensorflow
p = 0.2
ns = [1, 10, 100, 1000]
d2l.plt.figure(figsize=(10, 3))
for i in range(4):
    n = ns[i]
    pmf = tf.constant([p**i * (1-p)**(n-i) * binom(n, i)
                        for i in range(n + 1)])
    d2l.plt.subplot(1, 4, i + 1)
    d2l.plt.stem([(i - n*p)/tf.sqrt(tf.constant(n*p*(1 - p)))
                  for i in range(n + 1)], pmf,
                 use_line_collection=True)
    d2l.plt.xlim([-4, 4])
    d2l.plt.xlabel('x')
    d2l.plt.ylabel('p.m.f.')
    d2l.plt.title("n = {}".format(n))
d2l.plt.show()
```

Une chose à noter : par rapport au cas de Poisson, nous divisons maintenant par l'écart type, ce qui signifie que nous comprimons les résultats possibles dans des zones de plus en plus petites.  C'est une indication que notre limite ne sera plus discrète, mais plutôt continue.

Une dérivation de ce qui se produit dépasse le cadre de ce document, mais le *théorème central limite* stipule que pour $n \rightarrow \infty$, on obtient la distribution gaussienne (ou parfois la distribution normale).  Plus explicitement, pour toute $a, b$:

$$
\lim_{n \rightarrow \infty} P(Y^{(n)} \in [a, b]) = P(\mathcal{N}(0,1) \in [a, b]),
$$

où l'on dit qu'une variable aléatoire est normalement distribuée avec une moyenne $\mu$ et une variance $\sigma^2$ données, on écrit $X \sim \mathcal{N}(\mu, \sigma^2)$ si $X$ a une densité

$$p_X(x) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-\mu)^2}{2\sigma^2}}.$$ 
 :eqlabel:`eq_gaussian_pdf` 

 Traçons d'abord la fonction de densité de probabilité :eqref:`eq_gaussian_pdf` .

```{.python .input}
#@tab mxnet
mu, sigma = 0, 1

x = np.arange(-3, 3, 0.01)
p = 1 / np.sqrt(2 * np.pi * sigma**2) * np.exp(-(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab pytorch
mu, sigma = 0, 1

x = torch.arange(-3, 3, 0.01)
p = 1 / torch.sqrt(2 * torch.pi * sigma**2) * torch.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

```{.python .input}
#@tab tensorflow
mu, sigma = 0, 1

x = tf.range(-3, 3, 0.01)
p = 1 / tf.sqrt(2 * tf.pi * sigma**2) * tf.exp(
    -(x - mu)**2 / (2 * sigma**2))

d2l.plot(x, p, 'x', 'p.d.f.')
```

Traçons maintenant la fonction de distribution cumulative.  Cela dépasse le cadre de cette annexe, mais la f.d.c. gaussienne n'a pas de formule fermée en termes de fonctions plus élémentaires.  Nous utiliserons `erf` qui fournit un moyen de calculer numériquement cette intégrale.

```{.python .input}
#@tab mxnet
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * np.sqrt(2)))) / 2.0

d2l.plot(x, np.array([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab pytorch
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * torch.sqrt(d2l.tensor(2.))))) / 2.0

d2l.plot(x, torch.tensor([phi(y) for y in x.tolist()]), 'x', 'c.d.f.')
```

```{.python .input}
#@tab tensorflow
def phi(x):
    return (1.0 + erf((x - mu) / (sigma * tf.sqrt(tf.constant(2.))))) / 2.0

d2l.plot(x, [phi(y) for y in x.numpy().tolist()], 'x', 'c.d.f.')
```

Les lecteurs attentifs reconnaîtront certains de ces termes.  En effet, nous avons rencontré cette intégrale dans :numref:`sec_integral_calculus` . En fait, nous avons exactement besoin de ce calcul pour voir que ce $p_X(x)$ a une aire totale de un et est donc une densité valide.

Notre choix de travailler avec des tirages à pile ou face a permis de raccourcir les calculs, mais ce choix n'a rien de fondamental.  En effet, si nous prenons n'importe quelle collection de variables aléatoires indépendantes identiquement distribuées $X_i$, et formons

$$
X^{(N)} = \sum_{i=1}^N X_i.
$$

Alors

$$
\frac{X^{(N)} - \mu_{X^{(N)}}}{\sigma_{X^{(N)}}}
$$

sera approximativement gaussienne.  Des conditions supplémentaires sont nécessaires pour que cela fonctionne, le plus souvent $E[X^4] < \infty$, mais la philosophie est claire.

Le théorème central limite est la raison pour laquelle la gaussienne est fondamentale pour les probabilités, les statistiques et l'apprentissage automatique.  Chaque fois que nous pouvons dire que quelque chose que nous mesurons est une somme de nombreuses petites contributions indépendantes, nous pouvons supposer que la chose mesurée sera proche de la gaussienne. 

Il existe de nombreuses autres propriétés fascinantes des gaussiennes, et nous aimerions en aborder une de plus ici.  La gaussienne est ce que l'on appelle une *distribution à entropie maximale*.  Nous approfondirons la question de l'entropie à l'adresse :numref:`sec_information_theory` , mais tout ce que nous devons savoir à ce stade, c'est qu'il s'agit d'une mesure du caractère aléatoire.  Dans un sens mathématique rigoureux, nous pouvons considérer la gaussienne comme le choix *le plus* aléatoire de variable aléatoire avec une moyenne et une variance fixes.  Ainsi, si nous savons que notre variable aléatoire a une certaine moyenne et une certaine variance, la gaussienne est en quelque sorte le choix de distribution le plus conservateur que nous puissions faire.

Pour clore cette section, rappelons que si $X \sim \mathcal{N}(\mu, \sigma^2)$, alors :

* $\mu_X = \mu$,
* $\sigma_X^2 = \sigma^2$.

Nous pouvons échantillonner à partir de la distribution gaussienne (ou normale standard) comme indiqué ci-dessous.

```{.python .input}
#@tab mxnet
np.random.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab pytorch
torch.normal(mu, sigma, size=(10, 10))
```

```{.python .input}
#@tab tensorflow
tf.random.normal((10, 10), mu, sigma)
```

## Famille exponentielle
:label:`subsec_exponential_family` 

 Une propriété commune à toutes les distributions énumérées ci-dessus est qu'elles appartiennent toutes 
à ce que l'on appelle la *famille exponentielle*. La famille exponentielle 
est un ensemble de distributions dont la densité peut être exprimée sous la forme suivante 
:

$$p(\mathbf{x} | \boldsymbol{\eta}) = h(\mathbf{x}) \cdot \mathrm{exp} \left( \boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) - A(\boldsymbol{\eta}) \right)$$ 
 :eqlabel:`eq_exp_pdf` 

 Cette définition pouvant être un peu subtile, examinons-la de près. 

Tout d'abord, $h(\mathbf{x})$ est connue comme la *mesure sous-jacente* ou la *mesure de base* 
.  Elle peut être considérée comme un choix original de mesure que nous modifions avec notre poids exponentiel sur 
. 

Deuxièmement, nous avons le vecteur $\boldsymbol{\eta} = (\eta_1, \eta_2, ..., \eta_l) \in
\mathbb{R}^l$ appelé les *paramètres naturels* ou *paramètres canoniques*.  Ces
définissent comment la mesure de base sera modifiée.  Les paramètres naturels entrent 
dans la nouvelle mesure en prenant le produit scalaire de ces paramètres contre 
une certaine fonction $T(\cdot)$ de $\mathbf{x}= (x_1, x_2, ..., x_n) \in
\mathbb{R}^n$ et en l'exponentiant. Le vecteur $T(\mathbf{x})= (T_1(\mathbf{x}),
T_2(\mathbf{x}), ..., T_l(\mathbf{x}))$ 
est appelé la *statistique suffisante* pour $\boldsymbol{\eta}$. Ce nom est utilisé car l'information 
représentée par $T(\mathbf{x})$ est suffisante pour calculer la densité de probabilité 
et aucune autre information provenant de l'échantillon $\mathbf{x}$'s 
n'est nécessaire.

Troisièmement, nous avons $A(\boldsymbol{\eta})$, que l'on appelle la fonction *cumulante 
*, qui garantit que la distribution ci-dessus :eqref:`eq_exp_pdf` 
 s'intègre à un, c'est-à-dire,

$$A(\boldsymbol{\eta})  = \log \left[\int h(\mathbf{x}) \cdot \mathrm{exp}
\left(\boldsymbol{\eta}^{\top} \cdot T(\mathbf{x}) \right) d\mathbf{x} \right].$$

Pour être concret, considérons la gaussienne. En supposant que $\mathbf{x}$ est 
une variable univariée, nous avons vu qu'elle avait une densité de

$$
\begin{aligned}
p(x | \mu, \sigma) &= \frac{1}{\sqrt{2 \pi \sigma^2}} \cdot \mathrm{exp} 
\left\{ \frac{-(x-\mu)^2}{2 \sigma^2} \right\} \\
&= \frac{1}{\sqrt{2 \pi}} \cdot \mathrm{exp} \left\{ \frac{\mu}{\sigma^2}x
-\frac{1}{2 \sigma^2} x^2 - \left( \frac{1}{2 \sigma^2} \mu^2
+\log(\sigma) \right) \right\}.
\end{aligned}
$$

Cela correspond à la définition de la famille exponentielle avec :

* *mesure sous-jacente* : $h(x) = \frac{1}{\sqrt{2 \pi}}$,
* *paramètres naturels* $\boldsymbol{\eta} = \begin{bmatrix} \eta_1 \\ \eta_2
\end{bmatrix} = \begin{bmatrix} \frac{\mu}{\sigma^2} \\
\frac{1}{2 \sigma^2} \end{bmatrix}$,
* *statistiques suffisantes* : $T(x) = \begin{bmatrix}x\\-x^2\end{bmatrix}$, et
* *fonction cumulante* $A({\boldsymbol\eta}) = \frac{1}{2 \sigma^2} \mu^2 + \log(\sigma)
= \frac{\eta_1^2}{4 \eta_2} - \frac{1}{2}\log(2 \eta_2)$.

Il convient de noter que le choix exact de chacun des termes ci-dessus est quelque peu 
arbitraire.  En effet, la caractéristique importante est que la distribution peut être 
exprimée sous cette forme, et non la forme exacte elle-même.

Comme nous y faisons allusion dans :numref:`subsec_softmax_and_derivatives` , une technique largement utilisée 
consiste à supposer que la sortie finale $\mathbf{y}$ suit une distribution de la famille exponentielle 
. La famille exponentielle est une famille de distributions courante et 
puissante que l'on rencontre fréquemment en apprentissage automatique.


## Résumé
* Les variables aléatoires de Bernoulli peuvent être utilisées pour modéliser des événements dont l'issue est oui ou non.
* Le modèle des distributions uniformes discrètes sélectionne parmi un ensemble fini de possibilités.
* Les distributions uniformes continues sélectionnent dans un intervalle.
* Les distributions binomiales modélisent une série de variables aléatoires de Bernoulli, et comptent le nombre de réussites.
* Les variables aléatoires de Poisson modélisent l'arrivée d'événements rares.
* Les variables aléatoires gaussiennes modélisent le résultat de l'addition d'un grand nombre de variables aléatoires indépendantes.
* Toutes les distributions ci-dessus appartiennent à la famille exponentielle.

## Exercices

1. Quel est l'écart-type d'une variable aléatoire qui est la différence $X-Y$ de deux variables aléatoires binomiales indépendantes $X, Y \sim \mathrm{Binomial}(16, 1/2)$.
2. Si nous prenons une variable aléatoire de Poisson $X \sim \mathrm{Poisson}(\lambda)$ et considérons $(X - \lambda)/\sqrt{\lambda}$ comme $\lambda \rightarrow \infty$, nous pouvons montrer qu'elle devient approximativement gaussienne.  Pourquoi cela a-t-il un sens ?
3. Quelle est la fonction de masse de probabilité pour une somme de deux variables aléatoires uniformes discrètes sur $n$ éléments ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/417)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1098)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1099)
:end_tab:
