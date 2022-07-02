# Facteurisation matricielle

La factorisation matricielle :cite:`Koren.Bell.Volinsky.2009` est un algorithme bien établi dans la littérature sur les systèmes de recommandation. La première version du modèle de factorisation matricielle est proposée par Simon Funk dans un célèbre article intitulé [blog
post](https://sifter.org/~)simon/journal/20061211.html) dans lequel il décrivait l'idée de factoriser la matrice d'interaction. Elle est ensuite devenue largement connue grâce au concours Netflix qui s'est tenu en 2006. À cette époque, Netflix, une société de streaming de médias et de location de vidéos, a annoncé un concours visant à améliorer les performances de son système de recommandation. La meilleure équipe capable d'améliorer de 10 % les performances du système de référence de Netflix (c'est-à-dire Cinematch) remporterait un prix d'un million de dollars américains.  Ce concours a attiré
beaucoup d'attention dans le domaine de la recherche sur les systèmes de recommandation. Par la suite, le grand prix a été remporté par l'équipe Pragmatic Chaos de BellKor, une équipe combinée de BellKor, Pragmatic Theory et BigChaos (vous n'avez pas besoin de vous soucier de ces algorithmes maintenant). Bien que le score final soit le résultat d'une solution d'ensemble (c'est-à-dire une combinaison de nombreux algorithmes), l'algorithme de factorisation matricielle a joué un rôle essentiel dans le mélange final. Le rapport technique de la solution du Grand Prix Netflix :cite:`Toscher.Jahrer.Bell.2009` fournit une introduction détaillée au modèle adopté. Dans cette section, nous allons plonger dans les détails du modèle de factorisation matricielle et de son implémentation.


## Le modèle de factorisation matricielle

La factorisation matricielle est une classe de modèles de filtrage collaboratif. Plus précisément, le modèle factorise la matrice d'interaction entre l'utilisateur et l'article (par exemple, la matrice d'évaluation) en un produit de deux matrices de rang inférieur, capturant la structure de rang inférieur des interactions entre l'utilisateur et l'article.

Soit $\mathbf{R} \in \mathbb{R}^{m \times n}$ la matrice d'interaction avec $m$ utilisateurs et $n$ éléments, et les valeurs de $\mathbf{R}$ représentent les évaluations explicites. L'interaction utilisateur-article sera factorisée en une matrice latente d'utilisateur $\mathbf{P} \in \mathbb{R}^{m \times k}$ et une matrice latente d'article $\mathbf{Q} \in \mathbb{R}^{n \times k}$, où $k \ll m, n$ est la taille du facteur latent. Soit $\mathbf{p}_u$ la ligne $u^\mathrm{th}$ de $\mathbf{P}$ et $\mathbf{q}_i$ la ligne $i^\mathrm{th}$ de $\mathbf{Q}$. Pour un élément donné $i$, les éléments de $\mathbf{q}_i$ mesurent la mesure dans laquelle l'élément possède les caractéristiques telles que les genres et les langues d'un film. Pour un utilisateur donné $u$, les éléments de $\mathbf{p}_u$ mesurent le degré d'intérêt de l'utilisateur pour les caractéristiques correspondantes des articles. Ces facteurs latents peuvent mesurer des dimensions évidentes comme celles mentionnées dans ces exemples ou être complètement ininterprétables. Les évaluations prédites peuvent être estimées par

$$\hat{\mathbf{R}} = \mathbf{PQ}^\top$$ 

 où $\hat{\mathbf{R}}\in \mathbb{R}^{m \times n}$ est la matrice des évaluations prédites qui a la même forme que $\mathbf{R}$. Un problème majeur de cette règle de prédiction est que les biais des utilisateurs et des articles ne peuvent pas être modélisés. Par exemple, certains utilisateurs ont tendance à donner des notes plus élevées ou certains articles obtiennent toujours des notes plus basses en raison de leur qualité inférieure. Ces biais sont courants dans les applications du monde réel. Pour capturer ces biais, des termes de biais spécifiques aux utilisateurs et aux éléments sont introduits. Plus précisément, l'évaluation prédite que l'utilisateur $u$ donne à l'élément $i$ est calculée comme suit

$$
\hat{\mathbf{R}}_{ui} = \mathbf{p}_u\mathbf{q}^\top_i + b_u + b_i
$$

Ensuite, nous entraînons le modèle de factorisation matricielle en minimisant l'erreur quadratique moyenne entre les notes prédites et les notes réelles.  La fonction objectif est définie comme suit :

$$
\underset{\mathbf{P}, \mathbf{Q}, b}{\mathrm{argmin}} \sum_{(u, i) \in \mathcal{K}} \| \mathbf{R}_{ui} -
\hat{\mathbf{R}}_{ui} \|^2 + \lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )
$$

où $\lambda$ désigne le taux de régularisation. Le terme de régularisation $\lambda (\| \mathbf{P} \|^2_F + \| \mathbf{Q}
\|^2_F + b_u^2 + b_i^2 )$ est utilisé pour éviter le surajustement en pénalisant l'amplitude des paramètres. Les paires $(u, i)$ pour lesquelles $\mathbf{R}_{ui}$ est connu sont stockées dans l'ensemble
$\mathcal{K}=\{(u, i) \mid \mathbf{R}_{ui} \text{ is known}\}$ . Les paramètres du modèle peuvent être appris à l'aide d'un algorithme d'optimisation, tel que la descente de gradient stochastique et Adam.

Une illustration intuitive du modèle de factorisation matricielle est présentée ci-dessous :

![Illustration of matrix factorization model](../img/rec-mf.svg) 

 Dans le reste de cette section, nous allons expliquer la mise en œuvre de la factorisation matricielle et entraîner le modèle sur le jeu de données MovieLens.

```{.python .input  n=2}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
npx.set_np()
```

## Mise en œuvre du modèle

Tout d'abord, nous mettons en œuvre le modèle de factorisation matricielle décrit ci-dessus. Les facteurs latents des utilisateurs et des éléments peuvent être créés à l'aide de `nn.Embedding`. `input_dim` est le nombre d'éléments/utilisateurs et (`output_dim`) est la dimension des facteurs latents ($k$).  Nous pouvons également utiliser `nn.Embedding` pour créer les biais utilisateur/item en fixant la valeur de `output_dim` à un. Dans la fonction `forward`, les identifiants des utilisateurs et des articles sont utilisés pour rechercher les embeddings.

```{.python .input  n=4}
#@tab mxnet
class MF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(input_dim=num_users, output_dim=num_factors)
        self.Q = nn.Embedding(input_dim=num_items, output_dim=num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

    def forward(self, user_id, item_id):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = (P_u * Q_i).sum(axis=1) + np.squeeze(b_u) + np.squeeze(b_i)
        return outputs.flatten()
```

### Mesures d'évaluation

Nous implémentons ensuite la mesure RMSE (root-mean-square error), qui est couramment utilisée pour mesurer les différences entre les notes prédites par le modèle et les notes réellement observées (vérité terrain) :cite:`Gunawardana.Shani.2015` . La RMSE est définie comme suit

$$
\mathrm{RMSE} = \sqrt{\frac{1}{|\mathcal{T}|}\sum_{(u, i) \in \mathcal{T}}(\mathbf{R}_{ui} -\hat{\mathbf{R}}_{ui})^2}
$$

où $\mathcal{T}$ est l'ensemble constitué de paires d'utilisateurs et d'éléments que vous souhaitez évaluer. $|\mathcal{T}|$ est la taille de cet ensemble. Nous pouvons utiliser la fonction RMSE fournie par `mx.metric`.

```{.python .input  n=3}
#@tab mxnet
def evaluator(net, test_iter, devices):
    rmse = mx.metric.RMSE()  # Get the RMSE
    rmse_list = []
    for idx, (users, items, ratings) in enumerate(test_iter):
        u = gluon.utils.split_and_load(users, devices, even_split=False)
        i = gluon.utils.split_and_load(items, devices, even_split=False)
        r_ui = gluon.utils.split_and_load(ratings, devices, even_split=False)
        r_hat = [net(u, i) for u, i in zip(u, i)]
        rmse.update(labels=r_ui, preds=r_hat)
        rmse_list.append(rmse.get()[1])
    return float(np.mean(np.array(rmse_list)))
```

## Formation et évaluation du modèle


 Dans la fonction de formation, nous adoptons la perte $\ell_2$ avec décroissance du poids. Le mécanisme de décroissance du poids a le même effet que la régularisation $\ell_2$.

```{.python .input  n=4}
#@tab mxnet
#@save
def train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                        devices=d2l.try_all_gpus(), evaluator=None,
                        **kwargs):
    timer = d2l.Timer()
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 2],
                            legend=['train loss', 'test RMSE'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            timer.start()
            input_data = []
            values = values if isinstance(values, list) else [values]
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            train_feat = input_data[:-1] if len(values) > 1 else input_data
            train_label = input_data[-1]
            with autograd.record():
                preds = [net(*t) for t in zip(*train_feat)]
                ls = [loss(p, s) for p, s in zip(preds, train_label)]
            [l.backward() for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean() / len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        if len(kwargs) > 0:  # It will be used in section AutoRec
            test_rmse = evaluator(net, test_iter, kwargs['inter_mat'],
                                  devices)
        else:
            test_rmse = evaluator(net, test_iter, devices)
        train_l = l / (i + 1)
        animator.add(epoch + 1, (train_l, test_rmse))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test RMSE {test_rmse:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Enfin, rassemblons tous les éléments et entraînons le modèle. Ici, nous fixons la dimension du facteur latent à 30.

```{.python .input  n=5}
#@tab mxnet
devices = d2l.try_all_gpus()
num_users, num_items, train_iter, test_iter = d2l.split_and_load_ml100k(
    test_ratio=0.1, batch_size=512)
net = MF(30, num_users, num_items)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.002, 20, 1e-5, 'adam'
loss = gluon.loss.L2Loss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_recsys_rating(net, train_iter, test_iter, loss, trainer, num_epochs,
                    devices, evaluator)
```

Ci-dessous, nous utilisons le modèle entraîné pour prédire la note qu'un utilisateur (ID 20) pourrait donner à un élément (ID 30).

```{.python .input  n=6}
#@tab mxnet
scores = net(np.array([20], dtype='int', ctx=devices[0]),
             np.array([30], dtype='int', ctx=devices[0]))
scores
```

## Résumé

* Le modèle de factorisation matricielle est largement utilisé dans les systèmes de recommandation.  Il peut être utilisé pour prédire les notes qu'un utilisateur pourrait donner à un article.
* Nous pouvons implémenter et entraîner la factorisation matricielle pour les systèmes de recommandation.


## Exercices

* Faites varier la taille des facteurs latents. Comment la taille des facteurs latents influence-t-elle la performance du modèle ?
* Essayez différents optimiseurs, taux d'apprentissage et taux de décroissance des poids.
* Vérifiez les notes prédites par les autres utilisateurs pour un film spécifique.


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/400)
:end_tab:
