# Filtrage collaboratif neuronal pour le classement personnalisé

Cette section va au-delà de la rétroaction explicite, en présentant le cadre du filtrage collaboratif neuronal (NCF) pour la recommandation avec rétroaction implicite. La rétroaction implicite est omniprésente dans les systèmes de recommandation. Les actions telles que les clics, les achats et les montres sont des réactions implicites courantes, faciles à collecter et indicatives des préférences des utilisateurs. Le modèle que nous allons présenter, intitulé NeuMF :cite:`He.Liao.Zhang.ea.2017`, abréviation de "neural matrix factorization" (factorisation matricielle neuronale), a pour but de traiter la tâche de classement personnalisé avec un retour implicite. Ce modèle tire parti de la flexibilité et de la non-linéarité des réseaux neuronaux pour remplacer les produits scalaires de la factorisation matricielle, afin d'améliorer l'expressivité du modèle. Plus précisément, ce modèle est structuré avec deux sous-réseaux comprenant la factorisation matricielle généralisée (GMF) et le MLP et modélise les interactions de deux voies au lieu de simples produits de points. Les sorties de ces deux réseaux sont concaténées pour le calcul final des scores de prédiction. Contrairement à la tâche de prédiction de notation dans AutoRec, ce modèle génère une liste de recommandations classées pour chaque utilisateur sur la base du feedback implicite. Nous utiliserons la perte de classement personnalisé introduite dans la dernière section pour entraîner ce modèle.

## Le modèle NeuMF

Comme mentionné précédemment, NeuMF fusionne deux sous-réseaux. Le GMF est une version réseau neuronal générique de la factorisation matricielle où l'entrée est le produit par éléments des facteurs latents de l'utilisateur et de l'article. Il se compose de deux couches neuronales :

$$
\mathbf{x} = \mathbf{p}_u \odot \mathbf{q}_i \\
\hat{y}_{ui} = \alpha(\mathbf{h}^\top \mathbf{x}),
$$

où $\odot$ désigne le produit Hadamard des vecteurs. $\mathbf{P} \in \mathbb{R}^{m \times k}$ et $\mathbf{Q} \in \mathbb{R}^{n \times k}$ correspondent respectivement aux matrices latentes de l'utilisateur et de l'élément. $\mathbf{p}_u \in \mathbb{R}^{ k}$ est la ligne $u^\mathrm{th}$ de $P$ et $\mathbf{q}_i \in \mathbb{R}^{ k}$ est la ligne $i^\mathrm{th}$ de $Q$. $\alpha$ et $h$ désignent la fonction d'activation et le poids de la couche de sortie. $\hat{y}_{ui}$ est le score de prédiction que l'utilisateur $u$ pourrait donner à l'élément $i$.

Un autre composant de ce modèle est le MLP. Pour enrichir la flexibilité du modèle, le sous-réseau MLP ne partage pas les incorporations d'utilisateurs et d'articles avec le GMF. Il utilise la concaténation des encastrements d'utilisateurs et d'éléments comme entrée. Grâce aux connexions complexes et aux transformations non linéaires, il est capable d'estimer les interactions complexes entre les utilisateurs et les éléments. Plus précisément, le sous-réseau MLP est défini comme suit :

$$
\begin{aligned}
z^{(1)} &= \phi_1(\mathbf{U}_u, \mathbf{V}_i) = \left[ \mathbf{U}_u, \mathbf{V}_i \right] \\
\phi^{(2)}(z^{(1)})  &= \alpha^1(\mathbf{W}^{(2)} z^{(1)} + b^{(2)}) \\
&... \\
\phi^{(L)}(z^{(L-1)}) &= \alpha^L(\mathbf{W}^{(L)} z^{(L-1)} + b^{(L)})) \\
\hat{y}_{ui} &= \alpha(\mathbf{h}^\top\phi^L(z^{(L-1)}))
\end{aligned}
$$

où $\mathbf{W}^*, \mathbf{b}^*$ et $\alpha^*$ désignent la matrice de poids, le vecteur de biais et la fonction d'activation. $\phi^*$ désigne la fonction de la couche correspondante. $\mathbf{z}^*$ désigne la sortie de la couche correspondante.

Pour fusionner les résultats du GMF et du MLP, au lieu d'une simple addition, NeuMF concatène les avant-dernières couches de deux sous-réseaux pour créer un vecteur de caractéristiques qui peut être transmis aux couches suivantes. Ensuite, les sorties sont projetées avec la matrice $\mathbf{h}$ et une fonction d'activation sigmoïde. La couche de prédiction est formulée comme suit :
$$
\hat{y}_{ui} = \sigma(\mathbf{h}^\top[\mathbf{x}, \phi^L(z^{(L-1)})]).
$$

La figure suivante illustre l'architecture du modèle NeuMF.

![Illustration of the NeuMF model](../img/rec-neumf.svg)

```{.python .input  n=1}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import autograd, gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## Implémentation du modèle
Le code suivant implémente le modèle NeuMF. Il consiste en un modèle de factorisation matricielle généralisée et un MLP avec différents vecteurs d'intégration d'utilisateurs et d'éléments. La structure du MLP est contrôlée par le paramètre `nums_hiddens`. ReLU est utilisé comme fonction d'activation par défaut.

```{.python .input  n=2}
#@tab mxnet
class NeuMF(nn.Block):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens,
                 **kwargs):
        super(NeuMF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)
        self.mlp = nn.Sequential()
        for num_hiddens in nums_hiddens:
            self.mlp.add(nn.Dense(num_hiddens, activation='relu',
                                  use_bias=True))
        self.prediction_layer = nn.Dense(1, activation='sigmoid', use_bias=False)

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp = self.mlp(np.concatenate([p_mlp, q_mlp], axis=1))
        con_res = np.concatenate([gmf, mlp], axis=1)
        return self.prediction_layer(con_res)
```

## Jeu de données personnalisé avec échantillonnage négatif

Pour la perte de classement par paire, une étape importante est l'échantillonnage négatif. Pour chaque utilisateur, les éléments avec lesquels un utilisateur n'a pas interagi sont des éléments candidats (entrées non observées). La fonction suivante prend l'identité des utilisateurs et les éléments candidats en entrée, et échantillonne les éléments négatifs de manière aléatoire pour chaque utilisateur à partir de l'ensemble des éléments candidats de cet utilisateur. Pendant la phase d'apprentissage, le modèle s'assure que les éléments qu'un utilisateur aime sont classés plus haut que les éléments qu'il n'aime pas ou avec lesquels il n'a pas interagi.

```{.python .input  n=3}
#@tab mxnet
class PRDataset(gluon.data.Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
```

## Evaluateur
Dans cette section, nous adoptons la stratégie de division par le temps pour construire les ensembles d'entraînement et de test. Deux mesures d'évaluation sont utilisées pour évaluer l'efficacité du modèle, à savoir le taux de réussite à une position donnée $\ell$ ($\text{Hit}@\ell$) et l'aire sous la courbe ROC (AUC).  Le taux de réussite à une position donnée $\ell$ pour chaque utilisateur indique si l'élément recommandé est inclus dans la première liste classée $\ell$. La définition formelle est la suivante

$$
\text{Hit}@\ell = \frac{1}{m} \sum_{u \in \mathcal{U}} \textbf{1}(rank_{u, g_u} <= \ell),
$$

où $\textbf{1}$ désigne une fonction indicatrice qui est égale à un si l'élément de vérité terrain est classé dans la liste supérieure $\ell$, sinon elle est égale à zéro. $rank_{u, g_u}$ désigne le classement de l'élément de vérité terrain $g_u$ de l'utilisateur $u$ dans la liste de recommandation (le classement idéal est 1). $m$ est le nombre d'utilisateurs. $\mathcal{U}$ est l'ensemble des utilisateurs.

La définition de l'AUC est la suivante :

$$
\text{AUC} = \frac{1}{m} \sum_{u \in \mathcal{U}} \frac{1}{|\mathcal{I} \backslash S_u|} \sum_{j \in I \backslash S_u} \textbf{1}(rank_{u, g_u} < rank_{u, j}),
$$

où $\mathcal{I}$ est l'ensemble des éléments. $S_u$ est les éléments candidats de l'utilisateur $u$. Notez que de nombreux autres protocoles d'évaluation tels que la précision, le rappel et le gain cumulé actualisé normalisé (NDCG) peuvent également être utilisés.

La fonction suivante calcule le taux de réussite et l'AUC pour chaque utilisateur.

```{.python .input  n=4}
#@tab mxnet
#@save
def hit_and_auc(rankedlist, test_matrix, k):
    hits_k = [(idx, val) for idx, val in enumerate(rankedlist[:k])
              if val in set(test_matrix)]
    hits_all = [(idx, val) for idx, val in enumerate(rankedlist)
                if val in set(test_matrix)]
    max = len(rankedlist) - 1
    auc = 1.0 * (max - hits_all[0][0]) / max if len(hits_all) > 0 else 0
    return len(hits_k), auc
```

Ensuite, le taux de réussite global et l'AUC sont calculés comme suit.

```{.python .input  n=5}
#@tab mxnet
#@save
def evaluate_ranking(net, test_input, seq, candidates, num_users, num_items,
                     devices):
    ranked_list, ranked_items, hit_rate, auc = {}, {}, [], []
    all_items = set([i for i in range(num_users)])
    for u in range(num_users):
        neg_items = list(all_items - set(candidates[int(u)]))
        user_ids, item_ids, x, scores = [], [], [], []
        [item_ids.append(i) for i in neg_items]
        [user_ids.append(u) for _ in neg_items]
        x.extend([np.array(user_ids)])
        if seq is not None:
            x.append(seq[user_ids, :])
        x.extend([np.array(item_ids)])
        test_data_iter = gluon.data.DataLoader(
            gluon.data.ArrayDataset(*x), shuffle=False, last_batch="keep",
            batch_size=1024)
        for index, values in enumerate(test_data_iter):
            x = [gluon.utils.split_and_load(v, devices, even_split=False)
                 for v in values]
            scores.extend([list(net(*t).asnumpy()) for t in zip(*x)])
        scores = [item for sublist in scores for item in sublist]
        item_scores = list(zip(item_ids, scores))
        ranked_list[u] = sorted(item_scores, key=lambda t: t[1], reverse=True)
        ranked_items[u] = [r[0] for r in ranked_list[u]]
        temp = hit_and_auc(ranked_items[u], test_input[u], 50)
        hit_rate.append(temp[0])
        auc.append(temp[1])
    return np.mean(np.array(hit_rate)), np.mean(np.array(auc))
```

## entrainement et évaluation du modèle

La fonction de entrainement est définie ci-dessous. Nous entraînons le modèle par paire.

```{.python .input  n=6}
#@tab mxnet
#@save
def train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices, evaluator,
                  candidates, eval_step=1):
    timer, hit_rate, auc = d2l.Timer(), 0, 0
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['test hit rate', 'test AUC'])
    for epoch in range(num_epochs):
        metric, l = d2l.Accumulator(3), 0.
        for i, values in enumerate(train_iter):
            input_data = []
            for v in values:
                input_data.append(gluon.utils.split_and_load(v, devices))
            with autograd.record():
                p_pos = [net(*t) for t in zip(*input_data[:-1])]
                p_neg = [net(*t) for t in zip(*input_data[:-2],
                                              input_data[-1])]
                ls = [loss(p, n) for p, n in zip(p_pos, p_neg)]
            [l.backward(retain_graph=False) for l in ls]
            l += sum([l.asnumpy() for l in ls]).mean()/len(devices)
            trainer.step(values[0].shape[0])
            metric.add(l, values[0].shape[0], values[0].size)
            timer.stop()
        with autograd.predict_mode():
            if (epoch + 1) % eval_step == 0:
                hit_rate, auc = evaluator(net, test_iter, test_seq_iter,
                                          candidates, num_users, num_items,
                                          devices)
                animator.add(epoch + 1, (hit_rate, auc))
    print(f'train loss {metric[0] / metric[1]:.3f}, '
          f'test hit rate {float(hit_rate):.3f}, test AUC {float(auc):.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(devices)}')
```

Maintenant, nous pouvons charger le jeu de données MovieLens 100k et entraîner le modèle. Étant donné que l'ensemble de données MovieLens ne contient que des évaluations, avec quelques pertes de précision, nous binarisons ces évaluations en zéros et en uns. Si un utilisateur a évalué un élément, nous considérons le retour implicite comme un, sinon comme zéro. L'action de noter un élément peut être considérée comme une forme de rétroaction implicite.  Ici, nous divisons l'ensemble de données dans le mode `seq-aware` où les derniers éléments interceptés par les utilisateurs sont laissés de côté pour le test.

```{.python .input  n=11}
#@tab mxnet
batch_size = 1024
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_iter = gluon.data.DataLoader(
    PRDataset(users_train, items_train, candidates, num_items ), batch_size,
    True, last_batch="rollover", num_workers=d2l.get_dataloader_workers())
```

Nous créons et initialisons ensuite le modèle. Nous utilisons un MLP à trois couches avec une taille cachée constante de 10.

```{.python .input  n=8}
#@tab mxnet
devices = d2l.try_all_gpus()
net = NeuMF(10, num_users, num_items, nums_hiddens=[10, 10, 10])
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
```

Le code suivant entraîne le modèle.

```{.python .input  n=12}
#@tab mxnet
lr, num_epochs, wd, optimizer = 0.01, 10, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})
train_ranking(net, train_iter, test_iter, loss, trainer, None, num_users,
              num_items, num_epochs, devices, evaluate_ranking, candidates)
```

## Résumé

* L'ajout de la non-linéarité au modèle de factorisation matricielle est bénéfique pour améliorer la capacité et l'efficacité du modèle.
* NeuMF est une combinaison de la factorisation matricielle et d'un MLP. Le MLP prend en entrée la concaténation des embeddings utilisateur et élément.

## Exercices

* Varier la taille des facteurs latents. Quel est l'impact de la taille des facteurs latents sur la performance du modèle ?
* Varier les architectures (par exemple, le nombre de couches, le nombre de neurones de chaque couche) du MLP pour vérifier son impact sur la performance.
* Essayez différents optimiseurs, taux d'apprentissage et taux de décroissance des poids.
* Essayez d'utiliser la perte charnière définie dans la dernière section pour optimiser ce modèle.

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/403)
:end_tab:
