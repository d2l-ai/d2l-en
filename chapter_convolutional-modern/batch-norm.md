```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow'])
```

# Normalisation par lots
:label:`sec_batch_norm` 

 l'entrainement de réseaux neuronaux profonds est difficile.
Les faire converger en un temps raisonnable peut s'avérer délicat.
Dans cette section, nous décrivons la *normalisation par lots*, une technique populaire et efficace
qui accélère systématiquement la convergence des réseaux profonds :cite:`Ioffe.Szegedy.2015` .
Avec les blocs résiduels - abordés plus loin dans :numref:`sec_resnet` - la normalisation par lots
a permis aux praticiens d'entraîner régulièrement des réseaux de plus de 100 couches.
Un avantage secondaire (fortuit) de la normalisation par lots est sa régularisation inhérente. 

## entrainement de réseaux profonds

Lorsque nous travaillons avec des données, nous effectuons souvent un prétraitement avant la formation. 
Les choix de prétraitement des données font souvent une énorme différence dans les résultats finaux.
Rappelez-vous notre application des MLP à la prédiction des prix de l'immobilier (:numref:`sec_kaggle_house` ).
Notre première étape, lorsque nous avons travaillé avec des données réelles
, a été de normaliser nos caractéristiques d'entrée afin qu'elles aient 
une moyenne nulle $\mathbf{\mu} = 0$ et une variance unitaire $\mathbf{\Sigma} = \mathbf{1}$ sur plusieurs observations :cite:`friedman1987exploratory` .
Au minimum, il est fréquent de changer l'échelle de manière à ce que la diagonale soit égale à l'unité, c'est-à-dire $\Sigma_{ii} = 1$. 
Une autre stratégie encore consiste à redimensionner les vecteurs à une longueur unitaire, éventuellement à une moyenne nulle *par observation*. 
Cela peut donner de bons résultats, par exemple pour les données de capteurs spatiaux. Ces techniques de prétraitement et bien d'autres sont 
bénéfiques pour garder le problème de l'estimation bien maîtrisé. Voir, par exemple, les articles de :cite:`guyon2008feature` pour un examen des techniques de sélection et d'extraction de caractéristiques.

Intuitivement, cette normalisation joue bien avec nos optimiseurs
puisqu'elle place les paramètres *a priori* à une échelle similaire.
En tant que tel, il est naturel de se demander si une étape de normalisation correspondante *à l'intérieur* d'un réseau profond
ne serait pas bénéfique. Bien que ce ne soit pas tout à fait le raisonnement qui a conduit à l'invention de la normalisation par lot :cite:`Ioffe.Szegedy.2015` , c'est une façon utile de la comprendre, ainsi que sa cousine, la normalisation par couche :cite:`Ba.Kiros.Hinton.2016` , dans un cadre unifié. 

Deuxièmement, pour un MLP ou CNN typique, lors de la formation,
les variables (par exemple, les sorties de la transformation affine dans le MLP)
dans les couches intermédiaires 
peuvent prendre des valeurs avec des amplitudes très variables :
à la fois le long des couches de l'entrée à la sortie, entre les unités de la même couche,
et au fil du temps en raison de nos mises à jour des paramètres du modèle.
Les inventeurs de la normalisation par lots ont postulé de manière informelle
que cette dérive dans la distribution de ces variables pouvait entraver la convergence du réseau.
Intuitivement, nous pourrions conjecturer que si une couche
a des activations variables qui sont 100 fois supérieures à celles d'une autre couche,
cela pourrait nécessiter des ajustements compensatoires dans les taux d'apprentissage. Les solveurs adaptatifs 
tels que AdaGrad :cite:`Duchi.Hazan.Singer.2011` , Adam :cite:`Kingma.Ba.2014` , et Yogi :cite:`Zaheer.Reddi.Sachan.ea.2018` visent à résoudre ce problème du point de vue de l'optimisation. 
L'alternative est d'empêcher le problème de se produire, simplement par une normalisation adaptative.
   
Troisièmement, les réseaux plus profonds sont complexes et ont tendance à être plus facilement surajustés.
Cela signifie que la régularisation devient plus critique. Une technique courante de régularisation est l'injection de bruit 
. Cette technique est connue depuis longtemps, par exemple en ce qui concerne l'injection de bruit pour les entrées 
 :cite:`Bishop.1995` . Elle est également à la base du dropout :numref:`sec_dropout` . Il s'avère que, de manière tout à fait fortuite, la normalisation par lots présente ces trois avantages : prétraitement, stabilité numérique et régularisation. 

La normalisation par lots est appliquée à des couches individuelles ou, en option, à toutes les couches :
Dans chaque itération d'apprentissage,
nous normalisons d'abord les entrées (de la normalisation par lots)
en soustrayant leur moyenne et
en les divisant par leur écart type,
où les deux sont estimés sur la base des statistiques du mini-batch actuel.
Ensuite, nous appliquons un coefficient d'échelle et un décalage pour récupérer les degrés de liberté perdus 
. C'est précisément en raison de cette *normalisation* basée sur les statistiques du *lot*
que la *normalisation du lot* tire son nom.

Notez que si nous essayions d'appliquer la normalisation par lot avec des minibatchs de taille 1,
nous ne pourrions rien apprendre.
En effet, après avoir soustrait les moyennes,
chaque unité cachée prendrait la valeur 0.
Comme vous pouvez le deviner, puisque nous consacrons une section entière à la normalisation par lots,
avec des minibatchs suffisamment grands, l'approche s'avère efficace et stable.
L'une des conclusions à retenir ici est que lors de l'application de la normalisation par lot,
le choix de la taille du lot est
encore plus important que sans normalisation par lot, ou du moins, 
une calibration appropriée est nécessaire car nous pourrions l'ajuster.

Formellement, en désignant par $\mathbf{x} \in \mathcal{B}$ une entrée de normalisation par lot ($\mathrm{BN}$)
qui provient d'un mini-lot $\mathcal{B}$,
la normalisation par lot transforme $\mathbf{x}$
 selon l'expression suivante :

$$\mathrm{BN}(\mathbf{x}) = \boldsymbol{\gamma} \odot \frac{\mathbf{x} - \hat{\boldsymbol{\mu}}_\mathcal{B}}{\hat{\boldsymbol{\sigma}}_\mathcal{B}} + \boldsymbol{\beta}.$$ 
 :eqlabel:`eq_batchnorm` 

 Dans :eqref:`eq_batchnorm` ,
$\hat{\boldsymbol{\mu}}_\mathcal{B}$ est la moyenne de l'échantillon
et $\hat{\boldsymbol{\sigma}}_\mathcal{B}$ est l'écart type de l'échantillon du mini-lot $\mathcal{B}$.
Après application de la normalisation,
le minibatch résultant
a une moyenne nulle et une variance unitaire. 
Le choix de la variance unitaire
(par rapport à un autre nombre magique) est un choix arbitraire. Nous récupérons ce degré de liberté
en incluant un paramètre d'échelle *
* $\boldsymbol{\gamma}$ et un paramètre de décalage * $\boldsymbol{\beta}$
 qui ont la même forme que $\mathbf{x}$. Ces deux paramètres sont des paramètres que 
doit apprendre dans le cadre de l'apprentissage du modèle.

Par conséquent, les amplitudes des variables
pour les couches intermédiaires ne peuvent pas diverger pendant l'apprentissage
car la normalisation par lots les centre activement et les remet à l'échelle
à une moyenne et une taille données (via $\hat{\boldsymbol{\mu}}_\mathcal{B}$ et ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$).
L'expérience pratique confirme que, comme nous y avons fait allusion lors de la discussion sur la remise à l'échelle des caractéristiques, la normalisation par lots semble permettre des taux d'apprentissage plus agressifs.
Formellement, 
nous calculons $\hat{\boldsymbol{\mu}}_\mathcal{B}$ et ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ dans :eqref:`eq_batchnorm` comme suit :

$$\hat{\boldsymbol{\mu}}_\mathcal{B} = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} \mathbf{x}
\text{ and }
\hat{\boldsymbol{\sigma}}_\mathcal{B}^2 = \frac{1}{|\mathcal{B}|} \sum_{\mathbf{x} \in \mathcal{B}} (\mathbf{x} - \hat{\boldsymbol{\mu}}_{\mathcal{B}})^2 + \epsilon.$$

Notez que nous ajoutons une petite constante $\epsilon > 0$
 à l'estimation de la variance
pour nous assurer que nous ne tentons jamais une division par zéro,
même dans les cas où l'estimation de la variance empirique pourrait être très faible ou même disparaître, ce qui entraînerait une division par zéro.
Les estimations $\hat{\boldsymbol{\mu}}_\mathcal{B}$ et ${\hat{\boldsymbol{\sigma}}_\mathcal{B}}$ contrecarrent le problème de mise à l'échelle
en utilisant des estimations bruitées de la moyenne et de la variance.
Vous pourriez penser que ce caractère bruité devrait constituer un problème.
Or, il s'avère qu'elle est en fait bénéfique.

Il s'agit d'un thème récurrent dans l'apprentissage profond.
Pour des raisons qui ne sont pas encore bien caractérisées sur le plan théorique,
diverses sources de bruit dans l'optimisation
conduisent souvent à un apprentissage plus rapide et à moins de surajustement :
cette variation semble agir comme une forme de régularisation.
:cite:`Teye.Azizpour.Smith.2018` et :cite:`Luo.Wang.Shao.ea.2018` 
 relient les propriétés de la normalisation par lots aux prieurs et pénalités bayésiens respectivement.
En particulier, cela permet d'éclaircir l'énigme
de savoir pourquoi la normalisation par lots fonctionne mieux pour des minilots de taille modérée dans la gamme $50 \sim 100$. 
Cette taille particulière de minibatch semble injecter juste la "bonne quantité" de bruit par couche : un 
minibatch plus grand régularise moins en raison des estimations plus stables, tandis que les minibatchs minuscules 
détruisent le signal utile en raison de la variance élevée. En explorant cette direction plus avant, en considérant d'autres types 
de prétraitement et de filtrage, on peut encore aboutir à d'autres types de régularisation efficaces. 

En fixant un modèle entraîné, on pourrait penser
que nous préférerions utiliser l'ensemble des données
pour estimer la moyenne et la variance.
Une fois l'entrainement terminée, pourquoi voudrions-nous que
la même image soit classée différemment,
en fonction du lot dans lequel elle se trouve ?
Pendant la formation, un tel calcul exact est infaisable
car les variables intermédiaires
pour tous les exemples de données
changent à chaque fois que nous mettons à jour notre modèle.
Cependant, une fois le modèle formé,
nous pouvons calculer les moyennes et les variances
des variables de chaque couche sur la base de l'ensemble des données.
Il s'agit en effet d'une pratique standard pour les modèles
employant la normalisation par lots
. Les couches de normalisation par lots fonctionnent donc différemment
en mode *formation* (normalisation par statistiques de minibatchs)
et en mode *prédiction* (normalisation par statistiques de jeux de données). 
Sous cette forme, ils ressemblent beaucoup au comportement de la régularisation par abandon de :numref:`sec_dropout` ,
où le bruit n'est injecté que pendant l'apprentissage. 


## Couches de normalisation par lots

Nous sommes maintenant prêts à examiner comment la normalisation par lots fonctionne en pratique.
Les implémentations de la normalisation par lots pour les couches entièrement connectées
et les couches convolutionnelles sont légèrement différentes.
L'une des principales différences entre la normalisation par lots et les autres couches
est que, comme la normalisation par lots opère sur un minilot complet à la fois,
nous ne pouvons pas simplement ignorer la dimension du lot
comme nous l'avons fait précédemment lors de l'introduction d'autres couches.

### Couches entièrement connectées

Lors de l'application de la normalisation par lots aux couches entièrement connectées,
l'article original insérait la normalisation par lots après la transformation affine
et avant la fonction d'activation non linéaire. 
Des applications ultérieures ont expérimenté l'insertion de la normalisation par lots juste après les fonctions d'activation :cite:`Ioffe.Szegedy.2015` .
En désignant l'entrée de la couche entièrement connectée par $\mathbf{x}$,
la transformation affine
par $\mathbf{W}\mathbf{x} + \mathbf{b}$ (avec le paramètre de poids $\mathbf{W}$ et le paramètre de biais $\mathbf{b}$),
et la fonction d'activation par $\phi$,
nous pouvons exprimer le calcul d'une sortie de couche entièrement connectée
avec normalisation par lots $\mathbf{h}$ comme suit :

$$\mathbf{h} = \phi(\mathrm{BN}(\mathbf{W}\mathbf{x} + \mathbf{b}) ).$$

Rappelons que la moyenne et la variance sont calculées
sur le *même* minibatch 
sur lequel la transformation est appliquée.

### Couches convolutionnelles

De même, avec les couches convolutionnelles,
nous pouvons appliquer la normalisation par lot après la convolution
et avant la fonction d'activation non linéaire. La principale différence avec la normalisation par lot 
dans les couches entièrement connectées est que nous appliquons l'opération sur une base par canal 
*sur tous les emplacements*. Ceci est compatible avec notre hypothèse d'invariance de traduction 
qui a conduit aux convolutions : nous avons supposé que l'emplacement spécifique d'un motif 
dans une image n'était pas critique pour la compréhension.

Supposons que nos minibatchs contiennent $m$ exemples
et que pour chaque canal,
la sortie de la convolution a une hauteur $p$ et une largeur $q$.
Pour les couches convolutionnelles, nous effectuons chaque normalisation de lot
sur les éléments $m \cdot p \cdot q$ par canal de sortie simultanément.
Ainsi, nous collectons les valeurs sur tous les emplacements spatiaux
lors du calcul de la moyenne et de la variance
et par conséquent 
appliquons les mêmes moyenne et variance
dans un canal donné
pour normaliser la valeur à chaque emplacement spatial.
Chaque canal possède ses propres paramètres d'échelle et de décalage,
qui sont tous deux des scalaires.

Notez que dans le contexte des convolutions, la normalisation des lots est bien définie même pour 
des minis lots de taille 1 : après tout, nous avons tous les emplacements d'une image à moyenner. Par conséquent, la moyenne et la variance de 
sont bien définies, même si ce n'est que pour une seule observation. Cette considération 
a conduit :cite:`Ba.Kiros.Hinton.2016` à introduire la notion de *norme de couche*. Elle fonctionne exactement comme 
une norme de lot, mais elle est appliquée à une image à la fois. Il existe des cas où la normalisation des couches améliore la précision d'un modèle 
. Nous passons les détails et recommandons au lecteur intéressé de consulter l'article original 
. 

### Normalisation par lots pendant la prédiction

Comme nous l'avons mentionné précédemment, la normalisation par lots se comporte généralement différemment
en mode entrainement et en mode prédiction.
Tout d'abord, le bruit dans la moyenne et la variance de l'échantillon
provenant de l'estimation de chacun sur des minis lots
n'est plus souhaitable une fois que nous avons formé le modèle.
Deuxièmement, nous n'avons pas forcément le luxe
de calculer les statistiques de normalisation par lot.
Par exemple,
nous pouvons avoir besoin d'appliquer notre modèle pour faire une prédiction à la fois.

En général, après la formation, nous utilisons l'ensemble des données
pour calculer des estimations stables des statistiques des variables
et les fixer ensuite au moment de la prédiction.
Par conséquent, la normalisation par lot se comporte différemment pendant l'entrainement et au moment du test.
Rappelons que le dropout présente également cette caractéristique.

## (**Implémentation à partir de zéro**)

Pour voir comment la normalisation par lots fonctionne en pratique, nous en implémentons une à partir de zéro ci-dessous.

```{.python .input}
%%tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, np, npx, init
from mxnet.gluon import nn
npx.set_np()

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `autograd` to determine whether the current mode is training mode or
    # prediction mode
    if not autograd.is_training():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / np.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(axis=0)
            var = ((X - mean) ** 2).mean(axis=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(axis=(0, 2, 3), keepdims=True)
            var = ((X - mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / np.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean, moving_var
```

```{.python .input}
%%tab pytorch
from d2l import torch as d2l
import torch
from torch import nn

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum):
    # Use `is_grad_enabled` to determine whether the current mode is training
    # mode or prediction mode
    if not torch.is_grad_enabled():
        # If it is prediction mode, directly use the mean and variance
        # obtained by moving average
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            # When using a fully connected layer, calculate the mean and
            # variance on the feature dimension
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # When using a two-dimensional convolutional layer, calculate the
            # mean and variance on the channel dimension (axis=1). Here we
            # need to maintain the shape of `X`, so that the broadcasting
            # operation can be carried out later
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        # In training mode, the current mean and variance are used for the
        # standardization
        X_hat = (X - mean) / torch.sqrt(var + eps)
        # Update the mean and variance using moving average
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    Y = gamma * X_hat + beta  # Scale and shift
    return Y, moving_mean.data, moving_var.data
```

```{.python .input}
%%tab tensorflow
from d2l import tensorflow as d2l
import tensorflow as tf

def batch_norm(X, gamma, beta, moving_mean, moving_var, eps):
    # Compute reciprocal of square root of the moving variance elementwise
    inv = tf.cast(tf.math.rsqrt(moving_var + eps), X.dtype)
    # Scale and shift
    inv *= gamma
    Y = X * inv + (beta - moving_mean * inv)
    return Y
```

Nous pouvons maintenant [**créer une couche `BatchNorm` appropriée.**]
Notre couche maintiendra les paramètres appropriés
pour l'échelle `gamma` et le décalage `beta`,
qui seront tous deux mis à jour au cours de la formation.
De plus, notre couche maintiendra
les moyennes mobiles des moyennes et des variances
pour une utilisation ultérieure pendant la prédiction du modèle.

En laissant de côté les détails algorithmiques,
notez le modèle de conception qui sous-tend notre mise en œuvre de la couche.
En général, nous définissons les mathématiques dans une fonction distincte, par exemple `batch_norm`.
Nous intégrons ensuite cette fonctionnalité dans une couche personnalisée,
dont le code traite principalement des questions de comptabilité,
telles que le déplacement des données vers le bon contexte de dispositif,
l'allocation et l'initialisation de toutes les variables requises,
le suivi des moyennes mobiles (ici pour la moyenne et la variance), et ainsi de suite.
Ce modèle permet une séparation nette entre les mathématiques et le code passe-partout.
Notez également que, pour des raisons de commodité,
nous ne nous sommes pas souciés de déduire automatiquement la forme de l'entrée ici,
et nous devons donc spécifier le nombre de caractéristiques tout au long du processus.
À l'heure actuelle, tous les cadres d'apprentissage profond modernes offrent une détection automatique de la taille et de la forme dans les API de normalisation de lot de haut niveau 
(en pratique, nous utiliserons cette fonction).

```{.python .input}
%%tab mxnet
class BatchNorm(nn.Block):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims, **kwargs):
        super().__init__(**kwargs)
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.params.get('gamma', shape=shape, init=init.One())
        self.beta = self.params.get('beta', shape=shape, init=init.Zero())
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = np.zeros(shape)
        self.moving_var = np.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `var_mouvante` to the device where `X` is located
        if self.moving_mean.ctx != X.ctx:
            self.moving_mean = self.moving_mean.copyto(X.ctx)
            self.moving_var = self.moving_var.copyto(X.ctx)
        # Save the updated `moyenne mobile` and `variable_mobile`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma.data(), self.beta.data(), self.moving_mean,
            self.moving_var, eps=1e-12, momentum=0.9)
        return Y
```

```{.python .input}
%%tab pytorch
class BatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, X):
        # If `X` is not on the main memory, copy `moyenne mobile` and
        # `variable mobile` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moyenne mobile` and `variable mobile`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.9)
        return Y
```

```{.python .input}
%%tab tensorflow
class BatchNorm(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)

    def build(self, input_shape):
        weight_shape = [input_shape[-1], ]
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively
        self.gamma = self.add_weight(name='gamma', shape=weight_shape,
            initializer=tf.initializers.ones, trainable=True)
        self.beta = self.add_weight(name='beta', shape=weight_shape,
            initializer=tf.initializers.zeros, trainable=True)
        # The variables that are not model parameters are initialized to 0
        self.moving_mean = self.add_weight(name='moving_mean',
            shape=weight_shape, initializer=tf.initializers.zeros,
            trainable=False)
        self.moving_variance = self.add_weight(name='moving_variance',
            shape=weight_shape, initializer=tf.initializers.ones,
            trainable=False)
        super(BatchNorm, self).build(input_shape)

    def assign_moving_average(self, variable, value):
        momentum = 0.9
        delta = variable * momentum + value * (1 - momentum)
        return variable.assign(delta)

    @tf.function
    def call(self, inputs, training):
        if training:
            axes = list(range(len(inputs.shape) - 1))
            batch_mean = tf.reduce_mean(inputs, axes, keepdims=True)
            batch_variance = tf.reduce_mean(tf.math.squared_difference(
                inputs, tf.stop_gradient(batch_mean)), axes, keepdims=True)
            batch_mean = tf.squeeze(batch_mean, axes)
            batch_variance = tf.squeeze(batch_variance, axes)
            mean_update = self.assign_moving_average(
                self.moving_mean, batch_mean)
            variance_update = self.assign_moving_average(
                self.moving_variance, batch_variance)
            self.add_update(mean_update)
            self.add_update(variance_update)
            mean, variance = batch_mean, batch_variance
        else:
            mean, variance = self.moving_mean, self.moving_variance
        output = batch_norm(inputs, moving_mean=mean, moving_var=variance,
            beta=self.beta, gamma=self.gamma, eps=1e-5)
        return output
```

Notez que nous avons utilisé la variable `momentum` pour régir l'agrégation sur les estimations passées de la moyenne et de la variance. Il s'agit d'une appellation quelque peu erronée, car elle n'a absolument rien à voir avec le terme *momentum* d'optimisation dans :numref:`sec_momentum` . Néanmoins, c'est le nom communément adopté pour ce terme et, par respect pour la convention de dénomination de l'API, nous utilisons également le même nom de variable dans notre code.

## [**Applying Batch Normalization in LeNet**]

Pour voir comment appliquer `BatchNorm` dans son contexte,
ci-dessous l'applique à un modèle LeNet traditionnel (:numref:`sec_lenet` ).
Rappelons que la normalisation par lots est appliquée
après les couches convolutionnelles ou les couches entièrement connectées
mais avant les fonctions d'activation correspondantes.

```{.python .input}
%%tab all
class BNLeNetScratch(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2), nn.Dense(120),
                BatchNorm(120, num_dims=2), nn.Activation('sigmoid'),
                nn.Dense(84), BatchNorm(84, num_dims=2),
                nn.Activation('sigmoid'), nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), BatchNorm(6, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), BatchNorm(16, num_dims=4),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120),
                BatchNorm(120, num_dims=2), nn.Sigmoid(), nn.LazyLinear(84),
                BatchNorm(84, num_dims=2), nn.Sigmoid(),
                nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                BatchNorm(), tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84), BatchNorm(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

Comme précédemment, nous allons [**entraîner notre réseau sur le jeu de données Fashion-MNIST**].
Ce code est pratiquement identique à celui utilisé lors du premier entraînement de LeNet (:numref:`sec_lenet` ).

```{.python .input}
%%tab mxnet, pytorch
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNetScratch(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNetScratch(lr=0.5)
    trainer.fit(model, data)
```

Regardons [**le paramètre d'échelle `gamma`
 et le paramètre de décalage `beta`**] appris
de la première couche de normalisation par lots.

```{.python .input}
%%tab mxnet
model.net[1].gamma.data().reshape(-1,), model.net[1].beta.data().reshape(-1,)
```

```{.python .input}
%%tab pytorch
model.net[1].gamma.reshape((-1,)), model.net[1].beta.reshape((-1,))
```

```{.python .input}
%%tab tensorflow
tf.reshape(model.net.layers[1].gamma, (-1,)), tf.reshape(
    model.net.layers[1].beta, (-1,))
```

## [**Mise en œuvre concise**]

Par rapport à la classe `BatchNorm`,
que nous venons de définir nous-mêmes,
nous pouvons utiliser directement la classe `BatchNorm` définie dans les API de haut niveau du cadre d'apprentissage profond.
Le code semble pratiquement identique
à notre implémentation ci-dessus, sauf que nous n'avons plus besoin de fournir des arguments supplémentaires pour qu'elle obtienne les bonnes dimensions.

```{.python .input}
%%tab all
class BNLeNet(d2l.Classifier):
    def __init__(self, lr=0.1, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        if tab.selected('mxnet'):
            self.net = nn.Sequential()
            self.net.add(
                nn.Conv2D(6, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5), nn.BatchNorm(),
                nn.Activation('sigmoid'),
                nn.AvgPool2D(pool_size=2, strides=2),
                nn.Dense(120), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(84), nn.BatchNorm(), nn.Activation('sigmoid'),
                nn.Dense(num_classes))
            self.initialize()
        if tab.selected('pytorch'):
            self.net = nn.Sequential(
                nn.LazyConv2d(6, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.LazyConv2d(16, kernel_size=5), nn.LazyBatchNorm2d(),
                nn.Sigmoid(), nn.AvgPool2d(kernel_size=2, stride=2),
                nn.Flatten(), nn.LazyLinear(120), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(84), nn.LazyBatchNorm1d(),
                nn.Sigmoid(), nn.LazyLinear(num_classes))
        if tab.selected('tensorflow'):
            self.net = tf.keras.models.Sequential([
                tf.keras.layers.Conv2D(filters=6, kernel_size=5,
                                       input_shape=(28, 28, 1)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Conv2D(filters=16, kernel_size=5),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
                tf.keras.layers.Flatten(), tf.keras.layers.Dense(120),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(84),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation('sigmoid'),
                tf.keras.layers.Dense(num_classes)])
```

Ci-dessous, nous [**utilisons les mêmes hyperparamètres pour entraîner notre modèle.**]
Notez que, comme d'habitude, la variante API de haut niveau s'exécute beaucoup plus rapidement
car son code a été compilé en C++ ou CUDA
alors que notre implémentation personnalisée doit être interprétée par Python.

```{.python .input}
%%tab mxnet, pytorch
trainer = d2l.Trainer(max_epochs=10, num_gpus=1)
data = d2l.FashionMNIST(batch_size=128)
model = BNLeNet(lr=0.1)
if tab.selected('pytorch'):
    model.apply_init([next(iter(data.get_dataloader(True)))[0]], d2l.init_cnn)
trainer.fit(model, data)
```

```{.python .input}
%%tab tensorflow
trainer = d2l.Trainer(max_epochs=10)
data = d2l.FashionMNIST(batch_size=128)
with d2l.try_gpu():
    model = BNLeNet(lr=0.5)
    trainer.fit(model, data)
```

## Discussion

Intuitivement, la normalisation des lots est censée
rendre le paysage de l'optimisation plus lisse. 
Cependant, nous devons prendre soin de faire la distinction entre
les intuitions spéculatives et les véritables explications
des phénomènes que nous observons lors de l'entraînement des modèles profonds.
Rappelons que nous ne savons même pas pourquoi les réseaux neuronaux profonds plus simples
(MLP et CNN classiques)
généralisent bien en premier lieu.
Même avec le dropout et la décroissance des poids,
ils restent si flexibles que leur capacité à généraliser à des données inédites
nécessite probablement des garanties de généralisation beaucoup plus raffinées en théorie de l'apprentissage.

Dans l'article original proposant la normalisation par lots,
les auteurs, en plus de présenter un outil puissant et utile,
ont offert une explication de son fonctionnement :
en réduisant le *décalage interne des covariables*.
On peut supposer que par *internal covariate shift*, les auteurs
entendaient quelque chose comme l'intuition exprimée ci-dessus - la notion
que la distribution des valeurs des variables change
au cours de la formation.
Cependant, cette explication pose deux problèmes :
i) Cette dérive est très différente du *décalage des covariables*,
rendant le nom erroné.
ii) L'explication offre une intuition sous-spécifiée
mais laisse la question de savoir *pourquoi précisément cette technique fonctionne*
une question ouverte souhaitant une explication rigoureuse.
Tout au long de ce livre, nous cherchons à transmettre les intuitions que les praticiens
utilisent pour guider leur développement de réseaux de neurones profonds.
Cependant, nous pensons qu'il est important
de séparer ces intuitions directrices
des faits scientifiques établis.
À terme, lorsque vous maîtriserez cette matière
et que vous commencerez à rédiger vos propres articles de recherche
, vous voudrez être en mesure de délimiter clairement
entre les affirmations techniques et les intuitions.

Après le succès de la normalisation par lots,
son explication en termes de *décalage interne des covariables*
a fait surface à plusieurs reprises dans les débats de la littérature technique
et dans un discours plus large sur la manière de présenter la recherche en apprentissage automatique.
Dans un discours mémorable prononcé lors de l'acceptation du prix Test of Time
à la conférence NeurIPS 2017,
Ali Rahimi a utilisé *internal covariate shift*
comme point central d'un argument assimilant
la pratique moderne de l'apprentissage profond à l'alchimie.
Par la suite, l'exemple a été repris en détail
dans un document de synthèse décrivant
les tendances troublantes de l'apprentissage automatique :cite:`Lipton.Steinhardt.2018` .
D'autres auteurs
ont proposé d'autres explications pour le succès de la normalisation par lots,
certains affirmant que le succès de la normalisation par lots survient malgré un comportement
qui est, à certains égards, opposé à ceux revendiqués dans l'article original :cite:`Santurkar.Tsipras.Ilyas.ea.2018` .

Nous notons que le *décalage interne des covariables*
n'est pas plus digne de critique que n'importe laquelle des
milliers d'affirmations aussi vagues
faites chaque année dans la littérature technique de l'apprentissage automatique.
Il est probable que sa résonance en tant que point focal de ces débats
est due à sa large reconnaissabilité par le public cible.
La normalisation par lots s'est avérée être une méthode indispensable,
appliquée dans presque tous les classificateurs d'images déployés,
ce qui a valu à l'article qui a introduit cette technique
des dizaines de milliers de citations. Nous pensons cependant que les principes directeurs 
de régularisation par injection de bruit, d'accélération par remise à l'échelle et enfin de prétraitement
pourraient bien conduire à d'autres inventions de couches et de techniques à l'avenir. 

D'un point de vue plus pratique, plusieurs aspects de la normalisation par lots méritent d'être rappelés : 
* Pendant l'entrainement du modèle, la normalisation par lots ajuste en permanence la sortie intermédiaire du réseau 
 en utilisant la moyenne et l'écart type du minilot, de sorte que les valeurs 
 de la sortie intermédiaire de chaque couche du réseau neuronal sont plus stables.
* La normalisation des lots pour les couches entièrement connectées et les couches convolutionnelles est légèrement différente. En fait, 
 pour les couches convolutives, la normalisation des couches peut parfois être utilisée comme alternative. 
* Comme une couche d'abandon, les couches de normalisation par lots ont des comportements différents 
 en mode entrainement et en mode prédiction.
* La normalisation par lots est utile pour la régularisation et l'amélioration de la convergence dans l'optimisation. D'autre part, 
 la motivation initiale de réduction du décalage interne des covariables ne semble pas être une explication valable.

## Exercices

1. Peut-on supprimer le paramètre de biais de la couche entièrement connectée ou de la couche convolutive avant la normalisation par lots ? Pourquoi ?
1. Comparez les taux d'apprentissage de LeNet avec et sans normalisation par lots.
   1. Tracez l'augmentation de la précision de validation.
   1. Quelle taille pouvez-vous donner au taux d'apprentissage avant que l'optimisation n'échoue dans les deux cas ?
1. Avons-nous besoin d'une normalisation par lot dans chaque couche ? Faites-en l'expérience ?
1. Implémentez une version "allégée" de la normalisation par lots qui supprime uniquement la moyenne, ou alternativement une version qui 
 supprime uniquement la variance. Comment se comporte-t-elle ?
1. Fixez les paramètres `beta` et `gamma`, et observez et analysez les résultats.
1. Pouvez-vous remplacer l'abandon par la normalisation par lots ? Comment le comportement change-t-il ?
1. Consultez la documentation en ligne pour `BatchNorm` à partir des API de haut niveau pour voir 
 d'autres cas d'utilisation. 
1. Recherchez des idées : pensez à d'autres transformations de normalisation que vous pouvez appliquer :
   1. Pouvez-vous appliquer la transformée intégrale de probabilité ? 
    1. Pouvez-vous utiliser une estimation de covariance de rang complet ? Pourquoi pas ?
   1. Une compression de sparsification agit-elle comme un régularisateur ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/83)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/84)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/330)
:end_tab:
