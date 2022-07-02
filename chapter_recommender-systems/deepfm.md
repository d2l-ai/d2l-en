# Deep Factorization Machines

L'apprentissage de combinaisons de caractéristiques efficaces est essentiel au succès de la tâche de prédiction du taux de clics. Les machines à factoriser modélisent les interactions des caractéristiques dans un paradigme linéaire (par exemple, les interactions bilinéaires). Cela est souvent insuffisant pour les données du monde réel où les structures inhérentes de croisement de caractéristiques sont généralement très complexes et non linéaires. Pire encore, les interactions de second ordre sont généralement utilisées en pratique dans les machines à factoriser. La modélisation de combinaisons de caractéristiques de plus haut degré avec des machines à factoriser est possible en théorie, mais elle n'est généralement pas adoptée en raison de l'instabilité numérique et de la complexité de calcul élevée.

Une solution efficace consiste à utiliser des réseaux neuronaux profonds. Les réseaux neuronaux profonds sont puissants dans l'apprentissage de la représentation des caractéristiques et ont le potentiel d'apprendre des interactions de caractéristiques sophistiquées. Il est donc naturel d'intégrer les réseaux neuronaux profonds aux machines à factoriser. L'ajout de couches de transformation non linéaires aux machines à factoriser leur donne la capacité de modéliser à la fois des combinaisons de caractéristiques d'ordre inférieur et des combinaisons de caractéristiques d'ordre supérieur. De plus, les structures non linéaires inhérentes aux entrées peuvent également être capturées par les réseaux neuronaux profonds. Dans cette section, nous présenterons un modèle représentatif appelé machine à factoriser profonde (DeepFM) :cite:`Guo.Tang.Ye.ea.2017` qui combine la FM et les réseaux neuronaux profonds.


## Architectures du modèle

DeepFM se compose d'un composant FM et d'un composant profond qui sont intégrés dans une structure parallèle. La composante FM est la même que les machines de factorisation à deux voies qui sont utilisées pour modéliser les interactions de caractéristiques d'ordre inférieur. Le composant profond est un MLP qui est utilisé pour capturer les interactions de caractéristiques d'ordre supérieur et les non-linéarités. Ces deux composantes partagent les mêmes entrées/embeddings et leurs sorties sont additionnées pour former la prédiction finale. Il convient de souligner que l'esprit de DeepFM ressemble à celui de l'architecture Wide &amp; Deep qui peut capturer à la fois la mémorisation et la généralisation. L'avantage de DeepFM par rapport au modèle Wide \&amp; Deep est qu'il réduit l'effort d'ingénierie manuelle des caractéristiques en identifiant automatiquement les combinaisons de caractéristiques.

Par souci de concision, nous omettons la description du composant FM et désignons la sortie par $\hat{y}^{(FM)}$. Le lecteur est invité à se reporter à la dernière section pour plus de détails. Soit $\mathbf{e}_i \in \mathbb{R}^{k}$, le vecteur de caractéristiques latentes du champ $i^\mathrm{th}$.  L'entrée de la composante profonde est la concaténation des enchâssements denses de tous les champs qui sont recherchés avec l'entrée des caractéristiques catégorielles éparses, désignée par :

$$
\mathbf{z}^{(0)}  = [\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_f],
$$

où $f$ est le nombre de champs.  Elle est ensuite introduite dans le réseau neuronal suivant :

$$
\mathbf{z}^{(l)}  = \alpha(\mathbf{W}^{(l)}\mathbf{z}^{(l-1)} + \mathbf{b}^{(l)}),
$$

où $\alpha$ est la fonction d'activation. $\mathbf{W}_{l}$ et $\mathbf{b}_{l}$ sont le poids et le biais de la couche $l^\mathrm{th}$. Soit $y_{DNN}$ la sortie de la prédiction. La prédiction finale de DeepFM est la somme des sorties de FM et de DNN. Nous avons donc

$$
\hat{y} = \sigma(\hat{y}^{(FM)} + \hat{y}^{(DNN)}),
$$

où $\sigma$ est la fonction sigmoïde. L'architecture de DeepFM est illustrée ci-dessous.
![Illustration of the DeepFM model](../img/rec-deepfm.svg)

Il est intéressant de noter que DeepFM n'est pas la seule façon de combiner des réseaux neuronaux profonds avec FM. Nous pouvons également ajouter des couches non linéaires sur les interactions de caractéristiques :cite:`He.Chua.2017` .

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import init, gluon, np, npx
from mxnet.gluon import nn
import os

npx.set_np()
```

## Implémentation de DeepFM
L'implémentation de DeepFM est similaire à celle de FM. Nous gardons la partie FM inchangée et utilisons un bloc MLP avec `relu` comme fonction d'activation. Le Dropout est également utilisé pour régulariser le modèle. Le nombre de neurones du MLP peut être ajusté avec l'hyperparamètre `mlp_dims`.

```{.python .input  n=2}
#@tab mxnet
class DeepFM(nn.Block):
    def __init__(self, field_dims, num_factors, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        num_inputs = int(sum(field_dims))
        self.embedding = nn.Embedding(num_inputs, num_factors)
        self.fc = nn.Embedding(num_inputs, 1)
        self.linear_layer = nn.Dense(1, use_bias=True)
        input_dim = self.embed_output_dim = len(field_dims) * num_factors
        self.mlp = nn.Sequential()
        for dim in mlp_dims:
            self.mlp.add(nn.Dense(dim, 'relu', True, in_units=input_dim))
            self.mlp.add(nn.Dropout(rate=drop_rate))
            input_dim = dim
        self.mlp.add(nn.Dense(in_units=input_dim, units=1))

    def forward(self, x):
        embed_x = self.embedding(x)
        square_of_sum = np.sum(embed_x, axis=1) ** 2
        sum_of_square = np.sum(embed_x ** 2, axis=1)
        inputs = np.reshape(embed_x, (-1, self.embed_output_dim))
        x = self.linear_layer(self.fc(x).sum(1)) \
            + 0.5 * (square_of_sum - sum_of_square).sum(1, keepdims=True) \
            + self.mlp(inputs)
        x = npx.sigmoid(x)
        return x
```

## Formation et évaluation du modèle
Le processus de chargement des données est le même que celui de FM. Nous définissons le composant MLP de DeepFM comme un réseau dense à trois couches avec une structure pyramidale (30-20-10). Tous les autres hyperparamètres restent les mêmes que ceux de FM.

```{.python .input  n=4}
#@tab mxnet
batch_size = 2048
data_dir = d2l.download_extract('ctr')
train_data = d2l.CTRDataset(os.path.join(data_dir, 'train.csv'))
test_data = d2l.CTRDataset(os.path.join(data_dir, 'test.csv'),
                           feat_mapper=train_data.feat_mapper,
                           defaults=train_data.defaults)
field_dims = train_data.field_dims
train_iter = gluon.data.DataLoader(
    train_data, shuffle=True, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
test_iter = gluon.data.DataLoader(
    test_data, shuffle=False, last_batch='rollover', batch_size=batch_size,
    num_workers=d2l.get_dataloader_workers())
devices = d2l.try_all_gpus()
net = DeepFM(field_dims, num_factors=10, mlp_dims=[30, 20, 10])
net.initialize(init.Xavier(), ctx=devices)
lr, num_epochs, optimizer = 0.01, 30, 'adam'
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {'learning_rate': lr})
loss = gluon.loss.SigmoidBinaryCrossEntropyLoss()
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
```

Comparé à FM, DeepFM converge plus rapidement et obtient de meilleures performances.

## Résumé

* L'intégration de réseaux neuronaux à FM lui permet de modéliser des interactions complexes et de haut niveau.
* DeepFM surpasse le FM original sur le jeu de données de la publicité.

## Exercices

* Varier la structure du MLP pour vérifier son impact sur les performances du modèle.
* Changez le jeu de données pour Criteo et comparez-le avec le modèle FM original.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/407)
:end_tab:
