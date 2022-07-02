# Adam
:label:`sec_adam` 

 Dans les discussions qui ont précédé cette section, nous avons rencontré un certain nombre de techniques d'optimisation efficace. Récapitulons-les en détail ici :

* Nous avons vu que :numref:`sec_sgd` est plus efficace que la descente par gradient pour résoudre les problèmes d'optimisation, par exemple, en raison de sa résistance inhérente aux données redondantes. 
* Nous avons vu que :numref:`sec_minibatch_sgd` offre une efficacité supplémentaire significative découlant de la vectorisation, en utilisant de plus grands ensembles d'observations dans un minibatch. C'est la clé de l'efficacité du traitement parallèle multi-machine, multi-GPU et global. 
* :numref:`sec_momentum` a ajouté un mécanisme d'agrégation d'un historique des gradients passés pour accélérer la convergence.
* :numref:`sec_adagrad` a utilisé une mise à l'échelle par coordonnée pour permettre un préconditionnement efficace en termes de calcul. 
* :numref:`sec_rmsprop` a découplé la mise à l'échelle par coordonnée d'un ajustement du taux d'apprentissage. 

Adam :cite:`Kingma.Ba.2014` combine toutes ces techniques en un seul algorithme d'apprentissage efficace. Comme on pouvait s'y attendre, cet algorithme est devenu assez populaire comme l'un des algorithmes d'optimisation les plus robustes et efficaces à utiliser dans l'apprentissage profond. Cependant, il n'est pas exempt de problèmes. En particulier, :cite:`Reddi.Kale.Kumar.2019` montre qu'il existe des situations où Adam peut diverger en raison d'un mauvais contrôle de la variance. Dans un travail de suivi, :cite:`Zaheer.Reddi.Sachan.ea.2018` a proposé un correctif pour Adam, appelé Yogi, qui résout ces problèmes. Nous y reviendrons plus tard. Pour l'instant, passons en revue l'algorithme d'Adam. 

## L'algorithme

L'un des éléments clés d'Adam est qu'il utilise des moyennes mobiles pondérées exponentielles (également connues sous le nom de leaky averaging) pour obtenir une estimation du momentum et du second moment du gradient. C'est-à-dire qu'il utilise les variables d'état

$$\begin{aligned}
    \mathbf{v}_t & \leftarrow \beta_1 \mathbf{v}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \\
    \mathbf{s}_t & \leftarrow \beta_2 \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2.
\end{aligned}$$

Ici, $\beta_1$ et $\beta_2$ sont des paramètres de pondération non négatifs. Les choix courants pour eux sont $\beta_1 = 0.9$ et $\beta_2 = 0.999$. C'est-à-dire que l'estimation de la variance se déplace *beaucoup plus lentement* que le terme d'élan. Notez que si nous initialisons $\mathbf{v}_0 = \mathbf{s}_0 = 0$, nous avons initialement un biais important vers des valeurs plus petites. Ceci peut être résolu en utilisant le fait que $\sum_{i=0}^t \beta^i = \frac{1 - \beta^t}{1 - \beta}$ pour re-normaliser les termes. En conséquence, les variables d'état normalisées sont données par 

$$\hat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_1^t} \text{ and } \hat{\mathbf{s}}_t = \frac{\mathbf{s}_t}{1 - \beta_2^t}.$$ 

 Armés des estimations appropriées, nous pouvons maintenant écrire les équations de mise à jour. Tout d'abord, nous ré-échelonnons le gradient d'une manière très proche de celle de RMSProp pour obtenir

$$\mathbf{g}_t' = \frac{\eta \hat{\mathbf{v}}_t}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}.$$ 

 Contrairement à RMSProp, notre mise à jour utilise le momentum $\hat{\mathbf{v}}_t$ plutôt que le gradient lui-même. De plus, il y a une légère différence cosmétique car le changement d'échelle se fait en utilisant $\frac{1}{\sqrt{\hat{\mathbf{s}}_t} + \epsilon}$ au lieu de $\frac{1}{\sqrt{\hat{\mathbf{s}}_t + \epsilon}}$. Le premier fonctionne sans doute un peu mieux en pratique, d'où la déviation par rapport à RMSProp. En général, nous choisissons $\epsilon = 10^{-6}$ pour un bon compromis entre stabilité et fidélité numériques. 

Nous avons maintenant toutes les pièces en place pour calculer les mises à jour. C'est un peu décevant et nous avons une simple mise à jour de la forme

$$\mathbf{x}_t \leftarrow \mathbf{x}_{t-1} - \mathbf{g}_t'.$$ 

 En examinant la conception d'Adam, son inspiration est claire. Le momentum et l'échelle sont clairement visibles dans les variables d'état. Leur définition assez particulière nous oblige à débaptiser les termes (cela pourrait être corrigé par une condition d'initialisation et de mise à jour légèrement différente). Deuxièmement, la combinaison des deux termes est assez simple, étant donné RMSProp. Enfin, le taux d'apprentissage explicite $\eta$ nous permet de contrôler la longueur du pas pour résoudre les problèmes de convergence. 

## Mise en œuvre 

La mise en œuvre d'Adam à partir de zéro n'est pas très difficile. Par commodité, nous stockons le compteur de pas de temps $t$ dans le dictionnaire `hyperparams`. Au-delà de cela, tout est simple.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
npx.set_np()

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = beta2 * s + (1 - beta2) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

def init_adam_states(feature_dim):
    v_w, v_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    s_w, s_b = d2l.zeros((feature_dim, 1)), d2l.zeros(1)
    return ((v_w, s_w), (v_b, s_b))

def adam(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = beta2 * s + (1 - beta2) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1
```

```{.python .input}
#@tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import tensorflow as tf

def init_adam_states(feature_dim):
    v_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    v_b = tf.Variable(d2l.zeros(1))
    s_w = tf.Variable(d2l.zeros((feature_dim, 1)))
    s_b = tf.Variable(d2l.zeros(1))
    return ((v_w, s_w), (v_b, s_b))

def adam(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(beta2 * s + (1 - beta2) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
```

Nous sommes prêts à utiliser Adam pour entraîner le modèle. Nous utilisons un taux d'apprentissage de $\eta = 0.01$.

```{.python .input}
#@tab all
data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(adam, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

Une mise en œuvre plus concise est directe puisque `adam` est l'un des algorithmes fournis dans le cadre de la bibliothèque d'optimisation Gluon `trainer`. Il suffit donc de passer les paramètres de configuration pour une implémentation dans Gluon.

```{.python .input}
#@tab mxnet
d2l.train_concise_ch11('adam', {'learning_rate': 0.01}, data_iter)
```

```{.python .input}
#@tab pytorch
trainer = torch.optim.Adam
d2l.train_concise_ch11(trainer, {'lr': 0.01}, data_iter)
```

```{.python .input}
#@tab tensorflow
trainer = tf.keras.optimizers.Adam
d2l.train_concise_ch11(trainer, {'learning_rate': 0.01}, data_iter)
```

## Yogi

L'un des problèmes d'Adam est qu'il peut échouer à converger même dans des paramètres convexes lorsque l'estimation du second moment dans $\mathbf{s}_t$ explose. Pour y remédier, :cite:`Zaheer.Reddi.Sachan.ea.2018` a proposé une mise à jour (et une initialisation) raffinée pour $\mathbf{s}_t$. Pour comprendre ce qui se passe, réécrivons la mise à jour d'Adam comme suit :

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \left(\mathbf{g}_t^2 - \mathbf{s}_{t-1}\right).$$ 

 Lorsque $\mathbf{g}_t^2$ a une variance élevée ou que les mises à jour sont rares, $\mathbf{s}_t$ peut oublier trop rapidement les valeurs passées. Une solution possible consiste à remplacer $\mathbf{g}_t^2 - \mathbf{s}_{t-1}$ par $\mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1})$. Maintenant, l'ampleur de la mise à jour ne dépend plus de l'importance de la déviation. Cela donne les mises à jour de Yogi

$$\mathbf{s}_t \leftarrow \mathbf{s}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2 \odot \mathop{\mathrm{sgn}}(\mathbf{g}_t^2 - \mathbf{s}_{t-1}).$$ 

 Les auteurs conseillent en outre d'initialiser le momentum sur un lot initial plus important plutôt que sur une simple estimation ponctuelle initiale. Nous omettons les détails car ils ne sont pas importants pour la discussion et car même sans cela, la convergence reste assez bonne.

```{.python .input}
#@tab mxnet
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        v[:] = beta1 * v + (1 - beta1) * p.grad
        s[:] = s + (1 - beta2) * np.sign(
            np.square(p.grad) - s) * np.square(p.grad)
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:] -= hyperparams['lr'] * v_bias_corr / (np.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab pytorch
def yogi(params, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-3
    for p, (v, s) in zip(params, states):
        with torch.no_grad():
            v[:] = beta1 * v + (1 - beta1) * p.grad
            s[:] = s + (1 - beta2) * torch.sign(
                torch.square(p.grad) - s) * torch.square(p.grad)
            v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
            s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
            p[:] -= hyperparams['lr'] * v_bias_corr / (torch.sqrt(s_bias_corr)
                                                       + eps)
        p.grad.data.zero_()
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

```{.python .input}
#@tab tensorflow
def yogi(params, grads, states, hyperparams):
    beta1, beta2, eps = 0.9, 0.999, 1e-6
    for p, (v, s), grad in zip(params, states, grads):
        v[:].assign(beta1 * v  + (1 - beta1) * grad)
        s[:].assign(s + (1 - beta2) * tf.math.sign(
                   tf.math.square(grad) - s) * tf.math.square(grad))
        v_bias_corr = v / (1 - beta1 ** hyperparams['t'])
        s_bias_corr = s / (1 - beta2 ** hyperparams['t'])
        p[:].assign(p - hyperparams['lr'] * v_bias_corr  
                    / tf.math.sqrt(s_bias_corr) + eps)
    hyperparams['t'] += 1

data_iter, feature_dim = d2l.get_data_ch11(batch_size=10)
d2l.train_ch11(yogi, init_adam_states(feature_dim),
               {'lr': 0.01, 't': 1}, data_iter, feature_dim);
```

## Résumé

* Adam combine les caractéristiques de nombreux algorithmes d'optimisation en une règle de mise à jour assez robuste. 
* Créé sur la base de RMSProp, Adam utilise également EWMA sur le gradient stochastique des minibatchs.
* Adam utilise la correction de biais pour s'adapter à un démarrage lent lors de l'estimation du momentum et d'un second moment. 
* Pour les gradients avec une variance significative, nous pouvons rencontrer des problèmes de convergence. Ils peuvent être corrigés en utilisant des minibatchs plus grands ou en passant à une estimation améliorée pour $\mathbf{s}_t$. Yogi offre une telle alternative. 

## Exercices

1. Ajustez le taux d'apprentissage et observez et analysez les résultats expérimentaux.
1. Pouvez-vous réécrire les mises à jour du momentum et du second moment de telle sorte qu'elles ne nécessitent pas de correction de biais ?
1. Pourquoi devez-vous réduire le taux d'apprentissage $\eta$ au fur et à mesure de la convergence ?
1. Essayez de construire un cas pour lequel Adam diverge et Yogi converge ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/358)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1078)
:end_tab:


:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/1079)
:end_tab:
