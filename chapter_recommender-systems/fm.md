# Factorization Machines

Factorization machines (FM) :cite:`Rendle.2010`, proposé par Steffen Rendle en 2010, est un algorithme supervisé qui peut être utilisé pour des tâches de classification, de régression et de classement. Il s'est rapidement fait remarquer et est devenu une méthode populaire et impactante pour faire des prédictions et des recommandations. En particulier, il s'agit d'une généralisation du modèle de régression linéaire et du modèle de factorisation matricielle. De plus, il rappelle les machines à vecteurs de support avec un noyau polynomial. Les points forts des machines à factoriser par rapport à la régression linéaire et à la factorisation matricielle sont les suivants : (1) elle peut modéliser les interactions entre les variables $\chi$-way, où $\chi$ est le nombre d'ordre polynomial et est généralement fixé à deux. (2) Un algorithme d'optimisation rapide associé aux machines à factoriser peut réduire le temps de calcul polynomial à une complexité linéaire, ce qui le rend extrêmement efficace, en particulier pour les entrées éparses de haute dimension.  Pour ces raisons, les machines à factoriser sont largement utilisées dans la publicité moderne et les recommandations de produits. Les détails techniques et les implémentations sont décrits ci-dessous.


## Machines de factorisation à 2 voies

Formellement, laissez $x \in \mathbb{R}^d$ désigner les vecteurs de caractéristiques d'un échantillon, et $y$ désigner l'étiquette correspondante qui peut être une étiquette à valeur réelle ou une étiquette de classe telle que la classe binaire "clic/non-clic". Le modèle pour une machine à factoriser de degré deux est défini comme suit :

$$
\hat{y}(x) = \mathbf{w}_0 + \sum_{i=1}^d \mathbf{w}_i x_i + \sum_{i=1}^d\sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j
$$

où $\mathbf{w}_0 \in \mathbb{R}$ est le biais global ; $\mathbf{w} \in \mathbb{R}^d$ désigne les poids de la i-ième variable ; $\mathbf{V} \in \mathbb{R}^{d\times k}$ représente les incorporations de caractéristiques ; $\mathbf{v}_i$ représente la ligne $i^\mathrm{th}$ de $\mathbf{V}$; $k$ est la dimensionnalité des facteurs latents ; $\langle\cdot, \cdot \rangle$ est le produit scalaire de deux vecteurs. $\langle \mathbf{v}_i, \mathbf{v}_j \rangle$ modélise l'interaction entre la caractéristique $i^\mathrm{th}$ et $j^\mathrm{th}$. Certaines interactions de caractéristiques peuvent être facilement comprises et peuvent donc être conçues par des experts. Cependant, la plupart des autres interactions de caractéristiques sont cachées dans les données et difficiles à identifier. La modélisation automatique des interactions entre les fonctionnalités peut donc réduire considérablement les efforts d'ingénierie des fonctionnalités. Il est évident que les deux premiers termes correspondent au modèle de régression linéaire et que le dernier terme est une extension du modèle de factorisation matricielle. Si la caractéristique $i$ représente un article et la caractéristique $j$ représente un utilisateur, le troisième terme est exactement le produit scalaire entre l'utilisateur et les incorporations d'articles. Il est intéressant de noter que FM peut également être généralisé à des ordres supérieurs (degré &gt; 2). Néanmoins, la stabilité numérique pourrait affaiblir la généralisation.


## Un critère d'optimisation efficace

L'optimisation des machines de factorisation par une méthode directe conduit à une complexité de $\mathcal{O}(kd^2)$ car toutes les interactions par paire doivent être calculées. Pour résoudre ce problème d'inefficacité, nous pouvons réorganiser le troisième terme de FM, ce qui pourrait réduire considérablement le coût de calcul, conduisant à une complexité linéaire en temps ($\mathcal{O}(kd)$).  La reformulation du terme d'interaction par paire est la suivante :

$$
\begin{aligned}
&\sum_{i=1}^d \sum_{j=i+1}^d \langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j \\
 &= \frac{1}{2} \sum_{i=1}^d \sum_{j=1}^d\langle\mathbf{v}_i, \mathbf{v}_j\rangle x_i x_j - \frac{1}{2}\sum_{i=1}^d \langle\mathbf{v}_i, \mathbf{v}_i\rangle x_i x_i \\
 &= \frac{1}{2} \big (\sum_{i=1}^d \sum_{j=1}^d \sum_{l=1}^k\mathbf{v}_{i, l} \mathbf{v}_{j, l} x_i x_j - \sum_{i=1}^d \sum_{l=1}^k \mathbf{v}_{i, l} \mathbf{v}_{i, l} x_i x_i \big)\\
 &=  \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i) (\sum_{j=1}^d \mathbf{v}_{j, l}x_j) - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2 \big ) \\
 &= \frac{1}{2} \sum_{l=1}^k \big ((\sum_{i=1}^d \mathbf{v}_{i, l} x_i)^2 - \sum_{i=1}^d \mathbf{v}_{i, l}^2 x_i^2)
 \end{aligned}
$$

Avec cette reformulation, la complexité du modèle est considérablement réduite. De plus, pour les caractéristiques éparses, seuls les éléments non nuls doivent être calculés, de sorte que la complexité globale est linéaire par rapport au nombre de caractéristiques non nulles.

Pour apprendre le modèle FM, nous pouvons utiliser la perte MSE pour les tâches de régression, la perte d'entropie croisée pour les tâches de classification et la perte BPR pour les tâches de classement. Les optimiseurs standard tels que la descente de gradient stochastique et Adam sont viables pour l'optimisation.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Implémentation du modèle
Le code suivant implémente les machines de factorisation. Il est clair de voir que FM consiste en un bloc de régression linéaire et un bloc d'interaction efficace des caractéristiques. Nous appliquons une fonction sigmoïde sur le score final puisque nous traitons la prédiction du CTR comme une tâche de classification.

```{.python .input  n=2}
#@tab mxnet
class FM(nn.Block):
    def __init__(self, field_dims, num_factors):
        super(FM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)

    def forward(self, x):
        square_of_sum = np.sum(self.embedding(x), axis=1) ** 2
        sum_of_square = np.sum(self.embedding(x) ** 2, axis=1)
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True)
        x = npx.sigmoid(x)
        return x
```

## Charger l'ensemble de données publicitaires
Nous utilisons l'enveloppeur de données CTR de la dernière section pour charger l'ensemble de données publicitaires en ligne.

```{.python .input  n=3}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
```

## Train the Model
Ensuite, nous formons le modèle. Le taux d'apprentissage est fixé à 0,02 et la taille d'intégration est fixée à 20 par défaut. L'optimiseur `Adam` et la perte `SigmoidBinaryCrossEntropyLoss` sont utilisés pour l'entraînement du modèle.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
net = FM(train_data.field_dims, num_factors=20)
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.02, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

## Résumé

* FM est un cadre général qui peut être appliqué à une variété de tâches telles que la régression, la classification et le classement.
* L'interaction/le croisement des caractéristiques est important pour les tâches de prédiction et l'interaction à deux voies peut être modélisée efficacement avec FM.

## Exercices

* Pouvez-vous tester FM sur d'autres jeux de données tels que Avazu, MovieLens et Criteo ?
* Variez la taille d'intégration pour vérifier son impact sur la performance, pouvez-vous observer un schéma similaire à celui de la factorisation matricielle ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/406)
:end_tab:
