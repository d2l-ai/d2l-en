# AutoRec : Prédiction des notes avec des autoencodeurs

Bien que le modèle de factorisation matricielle atteigne une performance décente sur la tâche de prédiction de notation, il s'agit essentiellement d'un modèle linéaire. Ainsi, de tels modèles ne sont pas capables de capturer les relations complexes non linéaires et complexes qui peuvent être prédictives des préférences des utilisateurs. Dans cette section, nous présentons un modèle de filtrage collaboratif par réseau neuronal non linéaire, AutoRec :cite:`Sedhain.Menon.Sanner.ea.2015`. Il identifie le filtrage collaboratif (FC) avec une architecture d'autoencodeur et vise à intégrer des transformations non linéaires dans le FC sur la base d'un feedback explicite. Il a été prouvé que les réseaux neuronaux sont capables d'approximer n'importe quelle fonction continue, ce qui les rend aptes à traiter les limites de la factorisation matricielle et à en enrichir l'expressivité.

D'une part, AutoRec a la même structure qu'un autoencodeur qui consiste en une couche d'entrée, une couche cachée et une couche de reconstruction (sortie).  Un autoencodeur est un réseau neuronal qui apprend à copier son entrée sur sa sortie afin de coder les entrées dans les représentations cachées (et généralement de faible dimension). Dans AutoRec, au lieu d'intégrer explicitement les utilisateurs/articles dans un espace de faible dimension, il utilise la colonne/rangée de la matrice d'interaction comme entrée, puis reconstruit la matrice d'interaction dans la couche de sortie.

D'autre part, AutoRec diffère d'un auto-codeur traditionnel : plutôt que d'apprendre les représentations cachées, AutoRec se concentre sur l'apprentissage/la reconstruction de la couche de sortie. Il utilise une matrice d'interaction partiellement observée comme entrée, dans le but de reconstruire une matrice d'évaluation complète. Pendant ce temps, les entrées manquantes de l'entrée sont remplies dans la couche de sortie par reconstruction dans le but de faire des recommandations.

Il existe deux variantes d'AutoRec : basée sur l'utilisateur et basée sur l'élément. Par souci de concision, nous ne présentons ici que l'AutoRec basé sur les éléments. L'AutoRec basé sur l'utilisateur peut être dérivé en conséquence.


## Modèle

Soit $\mathbf{R}_{*i}$ la colonne $i^\mathrm{th}$ de la matrice d'évaluation, où les évaluations inconnues sont définies comme des zéros par défaut. L'architecture neuronale est définie comme suit

$$
h(\mathbf{R}_{*i}) = f(\mathbf{W} \cdot g(\mathbf{V} \mathbf{R}_{*i} + \mu) + b)
$$

où $f(\cdot)$ et $g(\cdot)$ représentent les fonctions d'activation, $\mathbf{W}$ et $\mathbf{V}$ sont les matrices de poids, $\mu$ et $b$ sont les biais. Soit $h( \cdot )$ pour désigner l'ensemble du réseau d'AutoRec. La sortie $h(\mathbf{R}_{*i})$ est la reconstruction de la colonne $i^\mathrm{th}$ de la matrice d'évaluation.

La fonction objective suivante vise à minimiser l'erreur de reconstruction :

$$
\underset{\mathbf{W},\mathbf{V},\mu, b}{\mathrm{argmin}} \sum_{i=1}^M{\parallel \mathbf{R}_{*i} - h(\mathbf{R}_{*i})\parallel_{\mathcal{O}}^2} +\lambda(\| \mathbf{W} \|_F^2 + \| \mathbf{V}\|_F^2)
$$

où $\| \cdot \|_{\mathcal{O}}$ signifie que seule la contribution des évaluations observées est prise en compte, c'est-à-dire que seuls les poids qui sont associés aux entrées observées sont mis à jour pendant la rétropropagation.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx

npx.set_np()
```

## Implémentation du modèle

Un auto-codeur typique consiste en un encodeur et un décodeur. L'encodeur projette l'entrée vers des représentations cachées et le décodeur fait correspondre la couche cachée à la couche de reconstruction. Nous suivons cette pratique et créons l'encodeur et le décodeur avec des couches entièrement connectées. L'activation de l'encodeur est fixée à `sigmoid` par défaut et aucune activation n'est appliquée au décodeur. Le Dropout est inclus après la transformation de l'encodage pour réduire l'over-fitting. Les gradients des entrées non observées sont masqués pour garantir que seules les évaluations observées contribuent au processus d'apprentissage du modèle.

```{.python .input  n=2}
#@tab mxnet
class AutoRec(nn.Block):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Dense(num_hidden, activation='sigmoid',
                                use_bias=True)
        self.decoder = nn.Dense(num_users, use_bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        hidden = self.dropout(self.encoder(input))
        pred = self.decoder(hidden)
        if autograd.is_training():  # Mask the gradient during training
            return pred * np.sign(input)
        else:
            return pred
```

## Réimplémentation de l'évaluateur

L'entrée et la sortie ayant été modifiées, nous devons réimplémenter la fonction d'évaluation, tout en continuant à utiliser RMSE comme mesure de précision.

```{.python .input  n=3}
#@tab mxnet
def evaluator(network, inter_matrix, test_data, devices):
    scores = []
    for values in inter_matrix:
        feat = gluon.utils.split_and_load(values, devices, even_split=False)
        scores.extend([network(i).asnumpy() for i in feat])
    recons = np.array([item for sublist in scores for item in sublist])
    # Calculate the test RMSE
    rmse = np.sqrt(np.sum(np.square(test_data - np.sign(test_data) * recons))
                   / np.sum(np.sign(test_data)))
    return float(rmse)
```

## entrainement et évaluation du modèle

Maintenant, formons et évaluons AutoRec sur le jeu de données MovieLens. Nous pouvons clairement voir que le RMSE de test est inférieur au modèle de factorisation matricielle, ce qui confirme l'efficacité des réseaux neuronaux dans la tâche de prédiction de classement.

```{.python .input  n=4}
#@tab mxnet
devices = d2l.try_all_gpus()
# Load the MovieLens 100K dataset
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items)
_, _, _, train_inter_mat = d2l.load_data_ml100k(train_data, num_users,
                                                num_items)
_, _, _, test_inter_mat = d2l.load_data_ml100k(test_data, num_users,
                                               num_items)
train_iter = gluon.data.DataLoader(train_inter_mat, shuffle=True,
                                   last_batch="rollover", batch_size=256,
                                   num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(np.array(train_inter_mat), shuffle=False,
                                  last_batch="keep", batch_size=1024,
                                  num_workers=d2l.get_dataloader_workers())
# Model initialization, training, and evaluation
net = AutoRec(500, num_users)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 25, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
d2l.train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices, evaluator, inter_mat=test_inter_mat)
```

## Résumé

* Nous pouvons encadrer l'algorithme de factorisation matricielle avec des autoencodeurs, tout en intégrant des couches non linéaires et une régularisation de type dropout.
* Les expériences sur le jeu de données MovieLens 100K montrent qu'AutoRec atteint des performances supérieures à la factorisation matricielle.



## Exercices

* Faites varier la dimension cachée d'AutoRec pour voir son impact sur les performances du modèle.
* Essayez d'ajouter plus de couches cachées. Cela permet-il d'améliorer les performances du modèle ?
* Pouvez-vous trouver une meilleure combinaison des fonctions d'activation du décodeur et de l'encodeur ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/401)
:end_tab:
