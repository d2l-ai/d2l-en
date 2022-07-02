# Systèmes de recommandation tenant compte de la séquence

Dans les sections précédentes, nous avons abstrait la tâche de recommandation comme un problème de remplissage de matrice sans tenir compte des comportements à court terme des utilisateurs. Dans cette section, nous présentons un modèle de recommandation qui prend en compte les journaux d'interaction des utilisateurs ordonnés de manière séquentielle.  Il s'agit d'un recommandeur sensible à la séquence :cite:`Quadrana.Cremonesi.Jannach.2018` où l'entrée est une liste ordonnée et souvent horodatée des actions passées de l'utilisateur.  Un certain nombre d'ouvrages récents ont démontré l'utilité d'incorporer de telles informations pour modéliser les schémas comportementaux temporels des utilisateurs et découvrir leur dérive d'intérêt.

Le modèle que nous allons présenter, Caser :cite:`Tang.Wang.2018` , abréviation de convolutional sequence embedding recommendation model, adopte des réseaux neuronaux à convolution pour capturer les influences dynamiques des activités récentes des utilisateurs. Le composant principal de Caser se compose d'un réseau convolutif horizontal et d'un réseau convolutif vertical, visant à découvrir les modèles de séquence au niveau de l'union et au niveau du point, respectivement.  Le modèle au niveau du point indique l'impact d'un seul élément de la séquence historique sur l'élément cible, tandis que le modèle au niveau de l'union implique les influences de plusieurs actions précédentes sur la cible suivante. Par exemple, l'achat simultané de lait et de beurre entraîne une probabilité plus élevée d'acheter de la farine que l'achat d'un seul de ces produits. En outre, les intérêts généraux des utilisateurs, ou leurs préférences à long terme, sont également modélisés dans les dernières couches entièrement connectées, ce qui permet une modélisation plus complète des intérêts des utilisateurs. Les détails du modèle sont décrits comme suit.

## Architectures du modèle

Dans un système de recommandation sensible à la séquence, chaque utilisateur est associé à une séquence de certains éléments de l'ensemble des éléments. Soit $S^u = (S_1^u, ... S_{|S_u|}^u)$ dénote la séquence ordonnée. L'objectif de Caser est de recommander des articles en tenant compte des goûts généraux de l'utilisateur ainsi que de ses intentions à court terme. Supposons que nous prenions en considération les éléments précédents $L$, une matrice d'incorporation qui représente les interactions précédentes pour le pas de temps $t$ peut être construite :

$$
\mathbf{E}^{(u, t)} = [ \mathbf{q}_{S_{t-L}^u} , ..., \mathbf{q}_{S_{t-2}^u}, \mathbf{q}_{S_{t-1}^u} ]^\top,
$$

où $\mathbf{Q} \in \mathbb{R}^{n \times k}$ représente les incorporations d'éléments et $\mathbf{q}_i$ désigne la ligne de $i^\mathrm{th}$. $\mathbf{E}^{(u, t)} \in \mathbb{R}^{L \times k}$ peut être utilisé pour déduire l'intérêt transitoire de l'utilisateur $u$ au pas de temps $t$. Nous pouvons considérer la matrice d'entrée $\mathbf{E}^{(u, t)}$ comme une image qui est l'entrée des deux composantes convolutionnelles suivantes.

La couche convolutive horizontale comporte $d$ filtres horizontaux $\mathbf{F}^j \in \mathbb{R}^{h \times k}, 1 \leq j \leq d, h = \{1, ..., L\}$, et la couche convolutive verticale comporte $d'$ filtres verticaux $\mathbf{G}^j \in \mathbb{R}^{ L \times 1}, 1 \leq j \leq d'$. Après une série d'opérations de convolution et de pool, nous obtenons les deux sorties :

$$
\mathbf{o} = \text{HConv}(\mathbf{E}^{(u, t)}, \mathbf{F}) \\
\mathbf{o}'= \text{VConv}(\mathbf{E}^{(u, t)}, \mathbf{G}) ,
$$

où $\mathbf{o} \in \mathbb{R}^d$ est la sortie du réseau convolutif horizontal et $\mathbf{o}' \in \mathbb{R}^{kd'}$ est la sortie du réseau convolutif vertical. Pour simplifier, nous omettons les détails des opérations de convolution et de pool. Elles sont concaténées et introduites dans une couche de réseau neuronal entièrement connecté pour obtenir des représentations de plus haut niveau.

$$
\mathbf{z} = \phi(\mathbf{W}[\mathbf{o}, \mathbf{o}']^\top + \mathbf{b}),
$$

où $\mathbf{W} \in \mathbb{R}^{k \times (d + kd')}$ est la matrice de poids et $\mathbf{b} \in \mathbb{R}^k$ est le biais. Le vecteur appris $\mathbf{z} \in \mathbb{R}^k$ est la représentation de l'intention à court terme de l'utilisateur.

Enfin, la fonction de prédiction combine le goût à court terme et le goût général des utilisateurs, qui est défini comme suit :

$$
\hat{y}_{uit} = \mathbf{v}_i \cdot [\mathbf{z}, \mathbf{p}_u]^\top + \mathbf{b}'_i,
$$

où $\mathbf{V} \in \mathbb{R}^{n \times 2k}$ est une autre matrice d'intégration d'élément. $\mathbf{b}' \in \mathbb{R}^n$ est le biais spécifique à l'élément. $\mathbf{P} \in \mathbb{R}^{m \times k}$ est la matrice d'intégration de l'utilisateur pour les goûts généraux des utilisateurs. $\mathbf{p}_u \in \mathbb{R}^{ k}$ est la ligne $u^\mathrm{th}$ de $P$ et $\mathbf{v}_i \in \mathbb{R}^{2k}$ est la ligne $i^\mathrm{th}$ de $\mathbf{V}$.

Le modèle peut être appris avec BPR ou la perte Hinge. L'architecture de Caser est présentée ci-dessous :

![Illustration of the Caser Model](../img/rec-caser.svg) 

 Nous commençons par importer les bibliothèques nécessaires.

```{.python .input  n=3}
#@tab mxnet
from d2l import mxnet as d2l
from mxnet import gluon, np, npx
from mxnet.gluon import nn
import mxnet as mx
import random

npx.set_np()
```

## Implémentation du modèle
Le code suivant implémente le modèle de Caser. Il se compose d'une couche convolutive verticale, d'une couche convolutive horizontale et d'une couche entièrement connectée.

```{.python .input  n=4}
#@tab mxnet
class Caser(nn.Block):
    def __init__(self, num_factors, num_users, num_items, L=5, d=16,
                 d_prime=4, drop_ratio=0.05, **kwargs):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.d_prime, self.d = d_prime, d
        # Vertical convolution layer
        self.conv_v = nn.Conv2D(d_prime, (L, 1), in_channels=1)
        # Horizontal convolution layer
        h = [i + 1 for i in range(L)]
        self.conv_h, self.max_pool = nn.Sequential(), nn.Sequential()
        for i in h:
            self.conv_h.add(nn.Conv2D(d, (i, num_factors), in_channels=1))
            self.max_pool.add(nn.MaxPool1D(L - i + 1))
        # Fully connected layer
        self.fc1_dim_v, self.fc1_dim_h = d_prime * num_factors, d * len(h)
        self.fc = nn.Dense(in_units=d_prime * num_factors + d * L,
                           activation='relu', units=num_factors)
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id, seq, item_id):
        item_embs = np.expand_dims(self.Q(seq), 1)
        user_emb = self.P(user_id)
        out, out_h, out_v, out_hs = None, None, None, []
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = out_v.reshape(out_v.shape[0], self.fc1_dim_v)
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = np.squeeze(npx.relu(conv(item_embs)), axis=3)
                t = maxp(conv_out)
                pool_out = np.squeeze(t, axis=2)
                out_hs.append(pool_out)
            out_h = np.concatenate(out_hs, axis=1)
        out = np.concatenate([out_v, out_h], axis=1)
        z = self.fc(self.dropout(out))
        x = np.concatenate([z, user_emb], axis=1)
        q_prime_i = np.squeeze(self.Q_prime(item_id))
        b = np.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res
```

## Ensemble de données séquentielles avec échantillonnage négatif
Pour traiter les données d'interaction séquentielles, nous devons réimplémenter la classe `Dataset`. Le code suivant crée une nouvelle classe de jeu de données nommée `SeqDataset`. Dans chaque échantillon, il fournit l'identité de l'utilisateur, les éléments qu'il a interagis précédemment avec $L$ sous forme de séquence et le prochain élément qu'il interagit comme cible. La figure suivante illustre le processus de chargement des données pour un utilisateur. Supposons que cet utilisateur ait aimé 9 films, nous organisons ces neuf films par ordre chronologique. Le dernier film est laissé de côté en tant qu'élément test. Pour les huit films restants, nous pouvons obtenir trois échantillons d'entraînement, chaque échantillon contenant une séquence de cinq films ($L=5$) et son élément suivant comme élément cible. Les échantillons négatifs sont également inclus dans l'ensemble de données personnalisé.

![Illustration of the data generation process](../img/rec-seq-data.svg)

```{.python .input  n=5}
#@tab mxnet
class SeqDataset(gluon.data.Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items,
                 candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)),
                                   key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids.asnumpy(), candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(sum([c - L if c >= L + 1 else 1 for c
                                in np.array([len(i[1]) for i in temp])]))
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype='int32')
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, - step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size:i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]):stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self):
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx],
                neg[i])
```

## Charger l'ensemble de données MovieLens 100K

Ensuite, nous lisons et divisons l'ensemble de données MovieLens 100K en mode séquentiel et chargeons les données d'entraînement avec le dataloader séquentiel implémenté ci-dessus.

```{.python .input  n=6}
#@tab mxnet
TARGET_NUM, L, batch_size = 1, 5, 4096
df, num_users, num_items = d2l.read_data_ml100k()
train_data, test_data = d2l.split_data_ml100k(df, num_users, num_items,
                                              'seq-aware')
users_train, items_train, ratings_train, candidates = d2l.load_data_ml100k(
    train_data, num_users, num_items, feedback="implicit")
users_test, items_test, ratings_test, test_iter = d2l.load_data_ml100k(
    test_data, num_users, num_items, feedback="implicit")
train_seq_data = SeqDataset(users_train, items_train, L, num_users,
                            num_items, candidates)
train_iter = gluon.data.DataLoader(train_seq_data, batch_size, True,
                                   last_batch="rollover",
                                   num_workers=d2l.get_dataloader_workers())
test_seq_iter = train_seq_data.test_seq
train_seq_data[0]
```

La structure des données d'entraînement est présentée ci-dessus. Le premier élément est l'identité de l'utilisateur, la liste suivante indique les cinq derniers éléments que cet utilisateur a aimés, et le dernier élément est l'élément que cet utilisateur a aimé après les cinq éléments.

## Former le modèle
Maintenant, nous allons former le modèle. Nous utilisons les mêmes paramètres que NeuMF, y compris le taux d'apprentissage, l'optimiseur et $k$, dans la dernière section afin que les résultats soient comparables.

```{.python .input  n=7}
#@tab mxnet
devices = d2l.try_all_gpus()
net = Caser(10, num_users, num_items, L)
net.initialize(ctx=devices, force_reinit=True, init=mx.init.Normal(0.01))
lr, num_epochs, wd, optimizer = 0.04, 8, 1e-5, 'adam'
loss = d2l.BPRLoss()
trainer = gluon.Trainer(net.collect_params(), optimizer,
                        {"learning_rate": lr, 'wd': wd})

d2l.train_ranking(net, train_iter, test_iter, loss, trainer, test_seq_iter,
                  num_users, num_items, num_epochs, devices,
                  d2l.evaluate_ranking, candidates, eval_step=1)
```

## Résumé
* L'inférence des intérêts à court et à long terme d'un utilisateur peut rendre plus efficace la prédiction du prochain élément qu'il a préféré.
* Les réseaux neuronaux convolutifs peuvent être utilisés pour capturer les intérêts à court terme des utilisateurs à partir d'interactions séquentielles.

## Exercices

* Effectuez une étude d'ablation en supprimant l'un des réseaux convolutifs horizontaux et verticaux, quel composant est le plus important ?
* Faites varier l'hyperparamètre $L$. Des interactions historiques plus longues apportent-elles une meilleure précision ?
* Outre la tâche de recommandation basée sur la séquence que nous avons présentée ci-dessus, il existe un autre type de tâche de recommandation basée sur la séquence appelée recommandation basée sur la session :cite:`Hidasi.Karatzoglou.Baltrunas.ea.2015` . Pouvez-vous expliquer les différences entre ces deux tâches ?


:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/404)
:end_tab:
