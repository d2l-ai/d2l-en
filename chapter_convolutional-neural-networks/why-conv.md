# From Fully Connected Layers to Convolutions
:label:`sec_why-conv`

To this day,
the models that we have discussed so far
remain appropriate options
when we are dealing with tabular data.
By tabular, we mean that the data consist
of rows corresponding to examples
and columns corresponding to features.
With tabular data, we might anticipate
that the patterns we seek could involve
interactions among the features,
but we do not assume any structure *a priori*
concerning how the features interact.

Sometimes, we truly lack knowledge to guide
the construction of craftier architectures.
In these cases, an MLP
may be the best that we can do.
However, for high-dimensional perceptual data,
such structure-less networks can grow unwieldy.

For instance, let's return to our running example
of distinguishing cats from dogs.
Say that we do a thorough job in data collection,
collecting an annotated dataset of one-megapixel photographs.
This means that each input to the network has one million dimensions.
Even an aggressive reduction to one thousand hidden dimensions
would require a fully connected layer
characterized by $10^6 \times 10^3 = 10^9$ parameters.
Unless we have lots of GPUs, a talent
for distributed optimization,
and an extraordinary amount of patience,
learning the parameters of this network
may turn out to be infeasible.

A careful reader might object to this argument
on the basis that one megapixel resolution may not be necessary.
However, while we might be able
to get away with one hundred thousand pixels,
our hidden layer of size 1000 grossly underestimates
the number of hidden units that it takes
to learn good representations of images,
so a practical system will still require billions of parameters.
Moreover, learning a classifier by fitting so many parameters
might require collecting an enormous dataset.
And yet today both humans and computers are able
to distinguish cats from dogs quite well,
seemingly contradicting these intuitions.
That is because images exhibit rich structure
that can be exploited by humans
and machine learning models alike.
Convolutional neural networks (CNNs) are one creative way
that machine learning has embraced for exploiting
some of the known structure in natural images.


## Invariance

Imagine that we want to detect an object in an image.
It seems reasonable that whatever method
we use to recognize objects should not be overly concerned
with the precise location of the object in the image.
Ideally, our system should exploit this knowledge.
Pigs usually do not fly and planes usually do not swim.
Nonetheless, we should still recognize
a pig were one to appear at the top of the image.
We can draw some inspiration here
from the children's game "Where's Waldo"
(depicted in :numref:`img_waldo`).
The game consists of a number of chaotic scenes
bursting with activities.
Waldo shows up somewhere in each,
typically lurking in some unlikely location.
The reader's goal is to locate him.
Despite his characteristic outfit,
this can be surprisingly difficult,
due to the large number of distractions.
However, *what Waldo looks like*
does not depend upon *where Waldo is located*.
We could sweep the image with a Waldo detector
that could assign a score to each patch,
indicating the likelihood that the patch contains Waldo. 
In fact, many object detection and segmentation algorithms 
are based on this approach :cite:`Long.Shelhamer.Darrell.2015`. 
CNNs systematize this idea of *spatial invariance*,
exploiting it to learn useful representations
with fewer parameters.

![An image of the "Where's Waldo" game.](../img/where-wally-walker-books.jpg)
:width:`400px`
:label:`img_waldo`

We can now make these intuitions more concrete 
by enumerating a few desiderata to guide our design
of a neural network architecture suitable for computer vision:

1. In the earliest layers, our network
   should respond similarly to the same patch,
   regardless of where it appears in the image. This principle is called *translation invariance* (or *translation equivariance*).
1. The earliest layers of the network should focus on local regions,
   without regard for the contents of the image in distant regions. This is the *locality* principle.
   Eventually, these local representations can be aggregated
   to make predictions at the whole image level.
1. As we proceed, deeper layers should be able to capture longer-range features of the 
   image, in a way similar to higher level vision in nature. 

Let's see how this translates into mathematics.


## Constraining the MLP

To start off, we can consider an MLP
with two-dimensional images $\mathbf{X}$ as inputs
and their immediate hidden representations
$\mathbf{H}$ similarly represented as matrices (they are two-dimensional tensors in code), where both $\mathbf{X}$ and $\mathbf{H}$ have the same shape.
Let that sink in.
We now conceive of not only the inputs but
also the hidden representations as possessing spatial structure.

Let $[\mathbf{X}]_{i, j}$ and $[\mathbf{H}]_{i, j}$ denote the pixel
at location $(i,j)$
in the input image and hidden representation, respectively.
Consequently, to have each of the hidden units
receive input from each of the input pixels,
we would switch from using weight matrices
(as we did previously in MLPs)
to representing our parameters
as fourth-order weight tensors $\mathsf{W}$.
Suppose that $\mathbf{U}$ contains biases,
we could formally express the fully connected layer as

$$\begin{aligned} \left[\mathbf{H}\right]_{i, j} &= [\mathbf{U}]_{i, j} + \sum_k \sum_l[\mathsf{W}]_{i, j, k, l}  [\mathbf{X}]_{k, l}\\ &=  [\mathbf{U}]_{i, j} +
\sum_a \sum_b [\mathsf{V}]_{i, j, a, b}  [\mathbf{X}]_{i+a, j+b}.\end{aligned}$$

The switch from $\mathsf{W}$ to $\mathsf{V}$ is entirely cosmetic for now
since there is a one-to-one correspondence
between coefficients in both fourth-order tensors.
We simply re-index the subscripts $(k, l)$
such that $k = i+a$ and $l = j+b$.
In other words, we set $[\mathsf{V}]_{i, j, a, b} = [\mathsf{W}]_{i, j, i+a, j+b}$.
The indices $a$ and $b$ run over both positive and negative offsets,
covering the entire image.
For any given location ($i$, $j$) in the hidden representation $[\mathbf{H}]_{i, j}$,
we compute its value by summing over pixels in $x$,
centered around $(i, j)$ and weighted by $[\mathsf{V}]_{i, j, a, b}$. Before we carry on, let's consider the total number of parameters required for a *single* layer in this parametrization: a $1000 \times 1000$ image (1 megapixel) is mapped to a $1000 \times 1000$ hidden representation. This requires $10^{12}$ parameters, far beyond what computers currently can handle.  

### Translation Invariance

Now let's invoke the first principle
established above: translation invariance :cite:`Zhang.ea.1988`.
This implies that a shift in the input $\mathbf{X}$
should simply lead to a shift in the hidden representation $\mathbf{H}$.
This is only possible if $\mathsf{V}$ and $\mathbf{U}$ do not actually depend on $(i, j)$. As such,
we have $[\mathsf{V}]_{i, j, a, b} = [\mathbf{V}]_{a, b}$ and $\mathbf{U}$ is a constant, say $u$.
As a result, we can simplify the definition for $\mathbf{H}$:

$$[\mathbf{H}]_{i, j} = u + \sum_a\sum_b [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$


This is a *convolution*!
We are effectively weighting pixels at $(i+a, j+b)$
in the vicinity of location $(i, j)$ with coefficients $[\mathbf{V}]_{a, b}$
to obtain the value $[\mathbf{H}]_{i, j}$.
Note that $[\mathbf{V}]_{a, b}$ needs many fewer coefficients than $[\mathsf{V}]_{i, j, a, b}$ since it
no longer depends on the location within the image. Consequently, the number of parameters required is no longer $10^{12}$ but a much more reasonable $4 \cdot 10^6$: we still have the dependency on $a, b \in (-1000, 1000)$. In short, we have made significant progress. Time-delay neural networks (TDNNs) are some of the first examples to exploit this idea :cite:`Waibel.Hanazawa.Hinton.ea.1989`.

###  Locality

Now let's invoke the second principle: locality.
As motivated above, we believe that we should not have
to look very far away from location $(i, j)$
in order to glean relevant information
to assess what is going on at $[\mathbf{H}]_{i, j}$.
This means that outside some range $|a|> \Delta$ or $|b| > \Delta$,
we should set $[\mathbf{V}]_{a, b} = 0$.
Equivalently, we can rewrite $[\mathbf{H}]_{i, j}$ as

$$[\mathbf{H}]_{i, j} = u + \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} [\mathbf{V}]_{a, b}  [\mathbf{X}]_{i+a, j+b}.$$
:eqlabel:`eq_conv-layer`

This reduces the number of parameters from $4 \cdot 10^6$ to $4 \Delta^2$, where $\Delta$ is typically smaller than $10$. As such, we reduced the number of parameters by another 4 orders of magnitude. Note that :eqref:`eq_conv-layer`, in a nutshell, is what is called a *convolutional layer*. 
*Convolutional neural networks* (CNNs)
sont une famille particulière de réseaux neuronaux qui contiennent des couches convolutionnelles.
Dans la communauté des chercheurs en apprentissage profond,
$\mathbf{V}$ est appelé un *noyau de convolution*,
un *filtre*, ou simplement les *poids* de la couche qui sont des paramètres apprenables.

Alors qu'auparavant, nous aurions pu avoir besoin de milliards de paramètres
pour représenter une seule couche dans un réseau de traitement d'images,
nous n'en avons plus besoin que de quelques centaines, sans
altérer la dimensionnalité des entrées ou des représentations cachées
.
Le prix à payer pour cette réduction drastique des paramètres
est que nos caractéristiques sont désormais invariantes par rapport à la traduction
et que notre couche ne peut incorporer que des informations locales,
lors de la détermination de la valeur de chaque activation cachée.
Tout apprentissage dépend de l'imposition d'un biais inductif.
Lorsque ce biais correspond à la réalité,
nous obtenons des modèles efficaces en termes d'échantillonnage
qui se généralisent bien aux données non vues.
Mais bien sûr, si ces biais ne correspondent pas à la réalité,
, par exemple si les images s'avéraient ne pas être invariantes par rapport à la traduction,
nos modèles pourraient même avoir du mal à s'adapter à nos données d'apprentissage.

Cette réduction spectaculaire des paramètres nous amène à notre dernier desideratum, 
, à savoir que les couches plus profondes doivent représenter des aspects plus importants et plus complexes 
d'une image. Ceci peut être réalisé en entrelaçant les non-linéarités et les couches convolutives 
de manière répétée. 

## Convolutions

Rappelons brièvement pourquoi :eqref:`eq_conv-layer` est appelé une convolution. 
En mathématiques, la *convolution* entre deux fonctions :cite:`Rudin.1973` ,
dis $f, g: \mathbb{R}^d \to \mathbb{R}$ est définie comme

$$(f * g)(\mathbf{x}) = \int f(\mathbf{z}) g(\mathbf{x}-\mathbf{z}) d\mathbf{z}.$$ 

 C'est-à-dire que nous mesurons le chevauchement entre $f$ et $g$
 lorsqu'une fonction est "retournée" et décalée par $\mathbf{x}$.
Lorsque nous avons des objets discrets, l'intégrale se transforme en une somme.
Par exemple, pour les vecteurs de
, l'ensemble des vecteurs infinis de dimension infinie sommables au carré
avec un indice courant sur $\mathbb{Z}$, nous obtenons la définition suivante :

$$(f * g)(i) = \sum_a f(a) g(i-a).$$ 

 Pour les tenseurs bidimensionnels, nous avons une somme correspondante
avec des indices $(a, b)$ pour $f$ et $(i-a, j-b)$ pour $g$, respectivement :

$$(f * g)(i, j) = \sum_a\sum_b f(a, b) g(i-a, j-b).$$ 
 :eqlabel:`eq_2d-conv-discrete` 

 Cela ressemble à :eqref:`eq_conv-layer` , avec une différence majeure.
Au lieu d'utiliser $(i+a, j+b)$, nous utilisons la différence.
Notez cependant que cette distinction est surtout cosmétique
puisque nous pouvons toujours faire correspondre la notation entre
:eqref:`eq_conv-layer` et :eqref:`eq_2d-conv-discrete` .
Notre définition originale dans :eqref:`eq_conv-layer` décrit plus correctement
une *corrélation croisée*.
Nous y reviendrons dans la section suivante.


## Canaux
:label:`subsec_why-conv-channels` 

 Revenons à notre détecteur Waldo, voyons à quoi cela ressemble.
La couche convolutive choisit des fenêtres d'une taille donnée
et pondère les intensités en fonction du filtre $\mathsf{V}$, comme le montre :numref:`fig_waldo_mask` .
Nous pourrions chercher à apprendre un modèle de telle sorte que
partout où la "waldoness" est la plus élevée,
nous devrions trouver un pic dans les représentations de la couche cachée.

![Detect Waldo.](../img/waldo-mask.jpg)
:width:`400px`
:label:`fig_waldo_mask`

Cette approche pose un seul problème.
Jusqu'à présent, nous avons ignoré béatement que les images se composent
de 3 canaux : rouge, vert et bleu. 
En somme, les images ne sont pas des objets bidimensionnels
mais plutôt des tenseurs du troisième ordre,
caractérisés par une hauteur, une largeur et un canal,
par exemple, avec une forme $1024 \times 1024 \times 3$ pixels. 
Alors que les deux premiers axes concernent les relations spatiales,
le troisième peut être considéré comme attribuant
une représentation multidimensionnelle à chaque emplacement de pixel.
Nous indexons donc $\mathsf{X}$ comme $[\mathsf{X}]_{i, j, k}$.
Le filtre convolutif doit s'adapter en conséquence.
Au lieu de $[\mathbf{V}]_{a,b}$, nous avons maintenant $[\mathsf{V}]_{a,b,c}$.

De plus, tout comme notre entrée consiste en un tenseur du troisième ordre,
il s'avère être une bonne idée de formuler de la même manière
nos représentations cachées comme des tenseurs du troisième ordre $\mathsf{H}$.
En d'autres termes, plutôt que d'avoir une seule représentation cachée
correspondant à chaque emplacement spatial,
nous voulons un vecteur entier de représentations cachées
correspondant à chaque emplacement spatial.
On peut considérer que les représentations cachées comprennent
un certain nombre de grilles bidimensionnelles empilées les unes sur les autres.
Comme dans les entrées, on les appelle parfois des *canaux*.
Elles sont aussi parfois appelées *cartes de caractéristiques*,
car chacune fournit un ensemble spatialisé
de caractéristiques apprises à la couche suivante.
Intuitivement, on peut imaginer qu'au niveau des couches inférieures, plus proches des entrées,
certains canaux pourraient être spécialisés dans la reconnaissance des bords tandis que
d'autres pourraient reconnaître les textures.

Pour prendre en charge plusieurs canaux à la fois dans les entrées ($\mathsf{X}$) et les représentations cachées ($\mathsf{H}$),
nous pouvons ajouter une quatrième coordonnée à $\mathsf{V}$: $[\mathsf{V}]_{a, b, c, d}$.
En mettant tout ensemble, nous avons :

$$[\mathsf{H}]_{i,j,d} = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c [\mathsf{V}]_{a, b, c, d} [\mathsf{X}]_{i+a, j+b, c},$$ 
 :eqlabel:`eq_conv-layer-channels` 

 où $d$ indexe les canaux de sortie dans les représentations cachées $\mathsf{H}$. La couche convolutive suivante prendra en entrée un tenseur de troisième ordre, $\mathsf{H}$.
De manière plus générale,
:eqref:`eq_conv-layer-channels` est
la définition d'une couche convolutive pour des canaux multiples, où $\mathsf{V}$ est un noyau ou un filtre de la couche.

Il reste encore de nombreuses opérations à traiter.
Par exemple, nous devons trouver comment combiner toutes les représentations cachées
en une seule sortie, par exemple pour savoir s'il y a un Waldo *n'importe où* dans l'image.
Nous devons également décider comment calculer les choses efficacement,
comment combiner plusieurs couches,
des fonctions d'activation appropriées,
et comment faire des choix de conception raisonnables
pour produire des réseaux efficaces en pratique.
Nous aborderons ces questions dans le reste du chapitre.

## Résumé et discussion

Dans cette section, nous avons dérivé la structure des réseaux de neurones convolutifs à partir des premiers principes. Bien qu'il ne soit pas clair si c'est ce qui a conduit à l'invention des CNN, il est satisfaisant de savoir qu'ils constituent le *bon* choix lorsqu'on applique des principes raisonnables à la manière dont les algorithmes de traitement d'images et de vision par ordinateur devraient fonctionner, au moins aux niveaux inférieurs. En particulier, l'invariance de translation dans les images implique que tous les patchs d'une image seront traités de la même manière. La localité signifie que seul un petit voisinage de pixels sera utilisé pour calculer les représentations cachées correspondantes. Certaines des premières références aux CNN se trouvent sous la forme du Neocognitron :cite:`Fukushima.1982` . 

Un deuxième principe que nous avons rencontré dans notre raisonnement est de savoir comment réduire le nombre de paramètres dans une classe de fonctions sans limiter son pouvoir expressif, du moins, lorsque certaines hypothèses sur le modèle se vérifient. Nous avons constaté une réduction spectaculaire de la complexité grâce à cette restriction, transformant des problèmes infaisables sur le plan informatique et statistique en modèles traitables. 

L'ajout de canaux nous a permis de récupérer une partie de la complexité perdue en raison des restrictions imposées au noyau convolutif par la localité et l'invariance de traduction. Notez que les canaux sont un ajout tout à fait naturel au-delà du rouge, du vert et du bleu. De nombreuses images satellites 
, notamment pour l'agriculture et la météorologie, comportent des dizaines voire des centaines de canaux, 
générant plutôt des images hyperspectrales. Elles rapportent des données sur de nombreuses longueurs d'onde différentes. Dans ce qui suit, nous verrons comment utiliser efficacement les convolutions pour manipuler la dimensionnalité des images sur lesquelles elles opèrent, comment passer de représentations basées sur l'emplacement à des représentations basées sur les canaux et comment traiter efficacement un grand nombre de catégories. 

## Exercices

1. Supposons que la taille du noyau de convolution soit $\Delta = 0$.
  Montrez que dans ce cas, le noyau de convolution
 met en œuvre un MLP indépendamment pour chaque ensemble de canaux. Cela conduit au réseau en réseau 
 architectures :cite:`Lin.Chen.Yan.2013` . 
1. Les données audio sont souvent représentées comme une séquence unidimensionnelle. 
    1. Quand voudriez-vous imposer la localité et l'invariance de translation pour l'audio ? 
    1. Dérivez les opérations de convolution pour l'audio.
   1. Pouvez-vous traiter l'audio en utilisant les mêmes outils que la vision par ordinateur ? Indice : utilisez le spectrogramme.
1. Pourquoi l'invariance de traduction n'est-elle pas une bonne idée après tout ? Donnez un exemple. 
1. Pensez-vous que les couches convolutionnelles pourraient également être applicables aux données textuelles ?
  Quels problèmes pouvez-vous rencontrer avec la langue ?
1. Que se passe-t-il avec les convolutions lorsqu'un objet se trouve à la limite d'une image. 
1. Prouvez que la convolution est symétrique, c'est-à-dire $f * g = g * f$.
1. Prouvez le théorème de convolution, c'est-à-dire $f * g = \mathcal{F}^{-1}\left[\mathcal{F}[f] \cdot \mathcal{F}[g]\right]$. 
   Pouvez-vous l'utiliser pour accélérer les convolutions ? 

[Discussions](https://discuss.d2l.ai/t/64)
