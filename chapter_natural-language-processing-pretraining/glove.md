# Word Embedding with Global Vectors (GloVe)
:label:`sec_glove` 

 
 Les cooccurrences mot-mot 
dans des fenêtres de contexte
peuvent être porteuses d'informations sémantiques riches.
Par exemple,
dans un grand corpus
le mot "solide" est
plus susceptible de cooccurber
avec "glace" qu'avec "vapeur",
mais le mot "gaz"
cooccure probablement avec "vapeur"
plus fréquemment que "glace".
En outre, les statistiques du corpus global

 de ces cooccurrences
peuvent être précalculées :
cela peut conduire à une formation plus efficace.
Afin d'exploiter les informations statistiques
de l'ensemble du corpus
pour l'intégration des mots,
reprenons d'abord
le modèle de saut de programme dans :numref:`subsec_skip-gram` ,
mais en l'interprétant
à l'aide des statistiques globales du corpus
telles que les comptes de cooccurrence.

## Skip-Gram avec les statistiques globales du corpus
:label:`subsec_skipgram-global` 

 En désignant par $q_{ij}$
 la probabilité conditionnelle
$P(w_j\mid w_i)$ 
 du mot $w_j$ étant donné le mot $w_i$
 dans le modèle skip-gram,
nous avons

$$q_{ij}=\frac{\exp(\mathbf{u}_j^\top \mathbf{v}_i)}{ \sum_{k \in \mathcal{V}} \text{exp}(\mathbf{u}_k^\top \mathbf{v}_i)},$$ 

 où 
pour tout index $i$
 les vecteurs $\mathbf{v}_i$ et $\mathbf{u}_i$
 représentent le mot $w_i$
 comme le mot central et le mot de contexte,
respectivement, et $\mathcal{V} = \{0, 1, \ldots, |\mathcal{V}|-1\}$ 
 est l'ensemble d'index du vocabulaire.

Considérons le mot $w_i$
 qui peut apparaître plusieurs fois
dans le corpus.
Dans l'ensemble du corpus,
tous les mots de contexte
où $w_i$ est pris comme leur mot central
forment un *multiset* $\mathcal{C}_i$
 d'indices de mots
qui *permet des instances multiples du même élément*.
Pour tout élément,
son nombre d'instances est appelé sa *multiplicité*.
Pour illustrer avec un exemple,
supposons que le mot $w_i$ apparaît deux fois dans le corpus
et les indices des mots de contexte
qui prennent $w_i$ comme mot central
dans les deux fenêtres de contexte
sont 
$k, j, m, k$ et $k, l, k, j$.
Ainsi, le multiset $\mathcal{C}_i = \{j, j, k, k, k, k, l, m\}$, où 
multiplicités des éléments $j, k, l, m$
 sont 2, 4, 1, 1, respectivement.

Désignons maintenant la multiplicité de l'élément $j$ dans le multiset
 $\mathcal{C}_i$ par $x_{ij}$.
Il s'agit du nombre global de cooccurrences 
du mot $w_j$ (en tant que mot de contexte)
et du mot $w_i$ (en tant que mot central)
dans la même fenêtre de contexte
dans l'ensemble du corpus.
En utilisant ces statistiques globales du corpus,
la fonction de perte du modèle de saut de programme 
est équivalente à

$$-\sum_{i\in\mathcal{V}}\sum_{j\in\mathcal{V}} x_{ij} \log\,q_{ij}.$$ 
 :eqlabel:`eq_skipgram-x_ij` 

 Nous désignons en outre par
$x_i$ 
 le nombre de tous les mots de contexte
dans les fenêtres de contexte
où $w_i$ apparaît comme leur mot central,
qui est équivalent à $|\mathcal{C}_i|$.
Si l'on laisse $p_{ij}$
 être la probabilité conditionnelle
$x_{ij}/x_i$ de générer
le mot contextuel $w_j$ étant donné le mot central $w_i$,
:eqref:`eq_skipgram-x_ij` 
 peut être réécrit comme suit :

$$-\sum_{i\in\mathcal{V}} x_i \sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}.$$ 
 :eqlabel:`eq_skipgram-p_ij` 

 In :eqref:`eq_skipgram-p_ij` , $-\sum_{j\in\mathcal{V}} p_{ij} \log\,q_{ij}$ calcule
l'entropie croisée 
de
la distribution conditionnelle $p_{ij}$
 des statistiques globales du corpus
et
la distribution conditionnelle
 $q_{ij}$ 
 des prédictions du modèle.
Cette perte
est également pondérée par $x_i$ comme expliqué ci-dessus.
La minimisation de la fonction de perte dans 
:eqref:`eq_skipgram-p_ij` 
 permettra à
la distribution conditionnelle prédite
de se rapprocher de
la distribution conditionnelle
des statistiques globales du corpus.


Bien qu'elle soit couramment utilisée
pour mesurer la distance
entre des distributions de probabilité,
la fonction de perte d'entropie croisée peut ne pas être un bon choix ici. 
D'une part, comme nous l'avons mentionné dans :numref:`sec_approx_train` , 
le coût d'une normalisation correcte $q_{ij}$
 résulte de la somme sur l'ensemble du vocabulaire,
ce qui peut être coûteux en termes de calcul.
D'autre part, 
un grand nombre d'événements rares 
d'un grand corpus
sont souvent modélisés par la perte d'entropie croisée
pour être assignés avec
trop de poids.

## Le modèle GloVe

Dans cette optique,
le modèle *GloVe* apporte trois modifications
au modèle de saut de programme basé sur la perte au carré :cite:`Pennington.Socher.Manning.2014` :

1. Utiliser des variables $p'_{ij}=x_{ij}$ et $q'_{ij}=\exp(\mathbf{u}_j^\top \mathbf{v}_i)$ 
 qui ne sont pas des distributions de probabilité
et prendre le logarithme des deux, de sorte que le terme de perte au carré est $\left(\log\,p'_{ij} - \log\,q'_{ij}\right)^2 = \left(\mathbf{u}_j^\top \mathbf{v}_i - \log\,x_{ij}\right)^2$.
2. Ajouter deux paramètres scalaires du modèle pour chaque mot $w_i$: le biais du mot central $b_i$ et le biais du mot contextuel $c_i$.
3. Remplacez le poids de chaque terme de perte par la fonction de poids $h(x_{ij})$, où $h(x)$ est croissant dans l'intervalle $[0, 1]$.

En mettant tout cela ensemble, la formation de GloVe consiste à minimiser la fonction de perte suivante :

$$\sum_{i\in\mathcal{V}} \sum_{j\in\mathcal{V}} h(x_{ij}) \left(\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j - \log\,x_{ij}\right)^2.$$ 
 :eqlabel:`eq_glove-loss` 

 Pour la fonction de poids, un choix suggéré est : 
$h(x) = (x/c) ^\alpha$ (par exemple $\alpha = 0.75$) si $x < c$ (par exemple, $c = 100$) ; sinon $h(x) = 1$.
Dans ce cas,
parce que $h(0)=0$,
le terme de perte au carré pour tout $x_{ij}=0$ peut être omis
pour des raisons d'efficacité de calcul.
Par exemple,
lorsque vous utilisez la descente de gradient stochastique par minilots pour l'apprentissage, 
à chaque itération
nous échantillonnons de manière aléatoire un minilots de *non-zéro* $x_{ij}$ 
 pour calculer les gradients
et mettre à jour les paramètres du modèle. 
Notez que ces non-zéros $x_{ij}$ sont des statistiques de corpus global précalculées 
;
ainsi, le modèle est appelé GloVe
pour *Global Vectors*.

Il convient de souligner que
si le mot $w_i$ apparaît dans la fenêtre de contexte du mot 
 $w_j$ , alors *vice versa*. 
Par conséquent, $x_{ij}=x_{ji}$. 
Contrairement au mot2vec
qui correspond à la probabilité conditionnelle asymétrique
$p_{ij}$ ,
GloVe correspond à la probabilité conditionnelle symétrique $\log \, x_{ij}$.
Par conséquent, le vecteur mot central et
le vecteur mot contextuel de tout mot sont mathématiquement équivalents dans le modèle GloVe. 
Cependant, en pratique, en raison de valeurs d'initialisation différentes,
le même mot peut encore obtenir des valeurs différentes
dans ces deux vecteurs après l'entraînement :
GloVe les additionne en tant que vecteur de sortie.



## Interpréter GloVe à partir du rapport des probabilités de cooccurrence


 Nous pouvons également interpréter le modèle GloVe d'un autre point de vue. 
En utilisant la même notation que dans 
:numref:`subsec_skipgram-global` ,
, laissez $p_{ij} \stackrel{\mathrm{def}}{=} P(w_j \mid w_i)$ être la probabilité conditionnelle de générer le mot contextuel $w_j$ étant donné $w_i$ comme mot central dans le corpus. 
:numref:`tab_glove`
liste plusieurs probabilités de cooccurrence
avec les mots "ice" et "steam"
et leurs ratios basés sur les statistiques d'un grand corpus.


 : Probabilités de cooccurrence mot-mot et leurs ratios à partir d'un grand corpus (adapté du tableau 1 dans :cite:`Pennington.Socher.Manning.2014`:)


|$w_k$=|solid|gas|water|fashion|
|:--|:-|:-|:-|:-|
|$p_1=P(w_k\mid \text{ice})$|0.00019|0.000066|0.003|0.000017|
|$p_2=P(w_k\mid\text{steam})$|0.000022|0.00078|0.0022|0.000018|
|$p_1/p_2$|8.9|0.085|1.36|0.96|
:label:`tab_glove` 

 
 Nous pouvons observer ce qui suit à partir de :numref:`tab_glove` :

* Pour un mot $w_k$ qui est lié à "ice" mais non lié à "steam", tel que $w_k=\text{solid}$, nous nous attendons à un plus grand rapport de probabilités de cooccurrence, tel que 8.9.
* Pour un mot $w_k$ qui est lié à "vapeur" mais non lié à "glace", comme $w_k=\text{gas}$, nous nous attendons à un plus petit rapport de probabilités de cooccurrence, comme 0.085.
* Pour un mot $w_k$ qui est lié à la fois à "glace" et à "vapeur", comme $w_k=\text{water}$, nous nous attendons à un rapport de probabilités de cooccurrence proche de 1, comme 1,36.
* Pour un mot $w_k$ qui n'est pas lié à la fois à "glace" et à "vapeur", tel que $w_k=\text{fashion}$, nous nous attendons à un rapport de probabilités de cooccurrence proche de 1, tel que 0,96.



 
 On peut voir que le rapport
des probabilités de cooccurrence
peut intuitivement exprimer
la relation entre les mots. 
Ainsi, nous pouvons concevoir une fonction
de trois vecteurs de mots
pour ajuster ce rapport.
Pour le rapport des probabilités de cooccurrence
${p_{ij}}/{p_{ik}}$ 
 avec $w_i$ étant le mot central
et $w_j$ et $w_k$ étant les mots du contexte,
nous voulons ajuster ce rapport
en utilisant une fonction $f$:

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) \approx \frac{p_{ij}}{p_{ik}}.$$ 
 :eqlabel:`eq_glove-f` 

 Parmi les nombreuses conceptions possibles pour $f$,
nous ne faisons qu'un choix raisonnable dans ce qui suit.
Puisque le rapport des probabilités de cooccurrence
est un scalaire,
nous exigeons que
$f$ soit une fonction scalaire, telle que
$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = f\left((\mathbf{u}_j - \mathbf{u}_k)^\top {\mathbf{v}}_i\right)$ . 
En permutant les indices de mots
$j$ et $k$ dans :eqref:`eq_glove-f` ,
, il faut que
$f(x)f(-x)=1$ ,
donc une possibilité est $f(x)=\exp(x)$,
c'est-à-dire 

$$f(\mathbf{u}_j, \mathbf{u}_k, {\mathbf{v}}_i) = \frac{\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right)}{\exp\left(\mathbf{u}_k^\top {\mathbf{v}}_i\right)} \approx \frac{p_{ij}}{p_{ik}}.$$ 

 Choisissons maintenant
$\exp\left(\mathbf{u}_j^\top {\mathbf{v}}_i\right) \approx \alpha p_{ij}$ ,
où $\alpha$ est une constante.
Puisque $p_{ij}=x_{ij}/x_i$, après avoir pris le logarithme des deux côtés, nous obtenons $\mathbf{u}_j^\top {\mathbf{v}}_i \approx \log\,\alpha + \log\,x_{ij} - \log\,x_i$. 
Nous pouvons utiliser des termes de biais supplémentaires pour ajuster $- \log\, \alpha + \log\, x_i$, tels que le biais du mot central $b_i$ et le biais du mot contextuel $c_j$:

$$\mathbf{u}_j^\top \mathbf{v}_i + b_i + c_j \approx \log\, x_{ij}.$$ 
 :eqlabel:`eq_glove-square` 

 En mesurant l'erreur quadratique de
:eqref:`eq_glove-square` avec les poids,
on obtient la fonction de perte GloVe dans
:eqref:`eq_glove-loss` .



## Résumé

* Le modèle de saut de programme peut être interprété à l'aide de statistiques globales du corpus telles que les comptes de cooccurrence mot-mot.
* La perte d'entropie croisée peut ne pas être un bon choix pour mesurer la différence de deux distributions de probabilité, en particulier pour un grand corpus. GloVe utilise la perte au carré pour ajuster les statistiques globales précalculées du corpus.
* Le vecteur mot central et le vecteur mot contextuel sont mathématiquement équivalents pour tout mot dans GloVe.
* GloVe peut être interprété à partir du rapport des probabilités de co-occurrence mot-mot.


## Exercices

1. Si les mots $w_i$ et $w_j$ cooccurrent dans la même fenêtre de contexte, comment pouvons-nous utiliser leur distance dans la séquence de texte pour redéfinir la méthode de calcul de la probabilité conditionnelle $p_{ij}$? Indice : voir la section 4.2 de l'article de GloVe :cite:`Pennington.Socher.Manning.2014` .
1. Pour tout mot, son biais de mot central et son biais de mot contextuel sont-ils mathématiquement équivalents dans GloVe ? Pourquoi ?


[Discussions](https://discuss.d2l.ai/t/385)
