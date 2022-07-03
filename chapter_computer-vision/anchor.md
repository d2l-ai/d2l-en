# Boîtes d'ancrage
:label:`sec_anchor` 

 
 Les algorithmes de détection d'objets échantillonnent généralement
un grand nombre de régions dans l'image d'entrée, déterminent si ces régions contiennent
des objets intéressants et ajustent les limites
des régions de manière à prédire plus précisément les boîtes d'ancrage
*ground-truth bounding boxes*
des objets.
Différents modèles peuvent adopter
différents schémas d'échantillonnage des régions. 
Nous présentons ici l'une de ces méthodes :
elle génère plusieurs boîtes de délimitation avec des échelles et des rapports d'aspect variables centrées sur chaque pixel. 
Ces boîtes de délimitation sont appelées *boîtes d'ancrage*.
Nous allons concevoir un modèle de détection d'objets
basé sur les boîtes d'ancrage dans :numref:`sec_ssd` .

Tout d'abord, modifions la précision d'impression
pour obtenir des résultats plus concis.

```{.python .input}
#@tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import gluon, image, np, npx

np.set_printoptions(2)  # Simplify printing accuracy
npx.set_np()
```

```{.python .input}
#@tab pytorch
%matplotlib inline
from d2l import torch as d2l
import torch

torch.set_printoptions(2)  # Simplify printing accuracy
```

## Génération de boîtes d'ancrage multiples

Supposons que l'image d'entrée ait une hauteur de $h$ et une largeur de $w$. 
Nous générons des boîtes d'ancrage de formes différentes centrées sur chaque pixel de l'image.
Supposons que l'*échelle* soit $s\in (0, 1]$ et que
le *aspect ratio* (rapport entre la largeur et la hauteur) soit $r > 0$. 
Ensuite, [**la largeur et la hauteur de la boîte d'ancrage sont respectivement $ws\sqrt{r}$ et $hs/\sqrt{r}$.**]
Notez que lorsque la position centrale est donnée, une boîte d'ancrage de largeur et de hauteur connues est déterminée.

Pour générer plusieurs boîtes d'ancrage de formes différentes,
définissons une série d'échelles
$s_1,\ldots, s_n$ et 
une série de rapports d'aspect $r_1,\ldots, r_m$.
En utilisant toutes les combinaisons de ces échelles et rapports d'aspect avec chaque pixel comme centre,
l'image d'entrée aura un total de $whnm$ boîtes d'ancrage. Bien que ces boîtes d'ancrage puissent couvrir toutes les boîtes de délimitation de la vérité du sol
, la complexité de calcul est facilement trop élevée.
En pratique,
nous ne pouvons (**considérer que les combinaisons
contenant**) $s_1$ ou $r_1$:

(**$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$**)

C'est-à-dire que le nombre de boîtes d'ancrage centrées sur le même pixel est $n+m-1$. Pour l'ensemble de l'image d'entrée, nous générerons un total de $wh(n+m-1)$ boîtes d'ancrage.

La méthode ci-dessus de génération de boîtes d'ancrage est mise en œuvre dans la fonction suivante `multibox_prior`. Nous spécifions l'image d'entrée, une liste d'échelles et une liste de rapports d'aspect, puis cette fonction renvoie toutes les boîtes d'ancrage.

```{.python .input}
#@tab mxnet
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.ctx, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, ctx=device)
    ratio_tensor = d2l.tensor(ratios, ctx=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y-axis
    steps_w = 1.0 / in_width  # Scaled steps in x-axis

    # Generate all center points for the anchor boxes
    center_h = (d2l.arange(in_height, ctx=device) + offset_h) * steps_h
    center_w = (d2l.arange(in_width, ctx=device) + offset_w) * steps_w
    shift_x, shift_y = d2l.meshgrid(center_w, center_h)
    shift_x, shift_y = shift_x.reshape(-1), shift_y.reshape(-1)

    # Generate `boîtes_par_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = np.concatenate((size_tensor * np.sqrt(ratio_tensor[0]),
                        sizes[0] * np.sqrt(ratio_tensor[1:]))) \
                        * in_height / in_width  # Handle rectangular inputs
    h = np.concatenate((size_tensor / np.sqrt(ratio_tensor[0]),
                        sizes[0] / np.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = np.tile(np.stack((-w, -h, w, h)).T,
                                   (in_height * in_width, 1)) / 2

    # Each center point will have `boîtes_par_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boîtes_par_pixel` repeats
    out_grid = d2l.stack([shift_x, shift_y, shift_x, shift_y],
                         axis=1).repeat(boxes_per_pixel, axis=0)
    output = out_grid + anchor_manipulations
    return np.expand_dims(output, axis=0)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_prior(data, sizes, ratios):
    """Generate anchor boxes with different shapes centered on each pixel."""
    in_height, in_width = data.shape[-2:]
    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios - 1)
    size_tensor = d2l.tensor(sizes, device=device)
    ratio_tensor = d2l.tensor(ratios, device=device)
    # Offsets are required to move the anchor to the center of a pixel. Since
    # a pixel has height=1 and width=1, we choose to offset our centers by 0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # Scaled steps in y axis
    steps_w = 1.0 / in_width  # Scaled steps in x axis

    # Generate all center points for the anchor boxes
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # Generate `boîtes_par_pixel` number of heights and widths that are later
    # used to create anchor box corner coordinates (xmin, xmax, ymin, ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))\
                   * in_height / in_width  # Handle rectangular inputs
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # Divide by 2 to get half height and half width
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2

    # Each center point will have `boîtes_par_pixel` number of anchor boxes, so
    # generate a grid of all anchor box centers with `boîtes_par_pixel` repeats
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
```

Nous pouvons voir que [**la forme de la variable de boîte d'ancrage retournée `Y`**] est
(taille du lot, nombre de boîtes d'ancrage, 4).

```{.python .input}
#@tab mxnet
img = image.imread('../img/catdog.jpg').asnumpy()
h, w = img.shape[:2]

print(h, w)
X = np.random.uniform(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

```{.python .input}
#@tab pytorch
img = d2l.plt.imread('../img/catdog.jpg')
h, w = img.shape[:2]

print(h, w)
X = torch.rand(size=(1, 3, h, w))  # Construct input data
Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape
```

Après avoir changé la forme de la variable de boîte d'ancrage `Y` en (hauteur d'image, largeur d'image, nombre de boîtes d'ancrage centrées sur le même pixel, 4),
nous pouvons obtenir toutes les boîtes d'ancrage centrées sur une position de pixel spécifiée.
Dans ce qui suit,
nous [**accédons à la première boîte d'ancrage centrée sur (250, 250)**]. Elle comporte quatre éléments : les coordonnées de l'axe $(x, y)$ dans le coin supérieur gauche et les coordonnées de l'axe $(x, y)$ dans le coin inférieur droit de la boîte d'ancrage.
Les valeurs des coordonnées des deux axes
sont divisées par la largeur et la hauteur de l'image, respectivement ; la plage est donc comprise entre 0 et 1.

```{.python .input}
#@tab all
boxes = Y.reshape(h, w, 5, 4)
boxes[250, 250, 0, :]
```

Afin de [**montrer toutes les boîtes d'ancrage centrées sur un pixel de l'image**],
nous définissons la fonction suivante `show_bboxes` pour dessiner plusieurs boîtes englobantes sur l'image.

```{.python .input}
#@tab all
#@save
def show_bboxes(axes, bboxes, labels=None, colors=None):
    """Show bounding boxes."""

    def make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj

    labels = make_list(labels)
    colors = make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(d2l.numpy(bbox), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))
```

Comme nous venons de le voir, les valeurs des coordonnées des axes $x$ et $y$ dans la variable `boxes` ont été divisées par la largeur et la hauteur de l'image, respectivement.
Lorsque nous dessinons des boîtes d'ancrage,
nous devons rétablir leurs valeurs de coordonnées d'origine ;
nous définissons donc la variable `bbox_scale` ci-dessous. 
Maintenant, nous pouvons dessiner toutes les boîtes d'ancrage centrées sur (250, 250) dans l'image.
Comme vous pouvez le voir, la boîte d'ancrage bleue avec une échelle de 0,75 et un rapport d'aspect de 1 bien
entoure le chien dans l'image.

```{.python .input}
#@tab all
d2l.set_figsize()
bbox_scale = d2l.tensor((w, h, w, h))
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1', 's=0.75, r=2',
             's=0.75, r=0.5'])
```

## [**Intersection over Union (IoU)**]

Nous venons de mentionner qu'une boîte d'ancrage entoure "bien" le chien dans l'image.
Si la boîte d'ancrage de l'objet est connue, comment peut-on quantifier le "bien" entouré ?
Intuitivement, nous pouvons mesurer la similarité entre
la boîte d'ancrage et la boîte limite de vérité.
Nous savons que l'indice *Jaccard* peut mesurer la similarité entre deux ensembles. Étant donné les ensembles $\mathcal{A}$ et $\mathcal{B}$, leur indice de Jaccard est la taille de leur intersection divisée par la taille de leur union :

$$J(\mathcal{A},\mathcal{B}) = \frac{\left|\mathcal{A} \cap \mathcal{B}\right|}{\left| \mathcal{A} \cup \mathcal{B}\right|}.$$ 

 
 En fait, nous pouvons considérer la surface des pixels de n'importe quelle boîte englobante comme un ensemble de pixels. 
De cette façon, nous pouvons mesurer la similarité de deux boîtes englobantes par l'indice de Jaccard de leurs ensembles de pixels. Pour deux boîtes englobantes, nous désignons généralement leur indice de Jaccard par l'expression *intersection sur union* (*IoU*), qui est le rapport entre leur zone d'intersection et leur zone d'union, comme indiqué sur :numref:`fig_iou` .
L'intervalle d'un IoU est compris entre 0 et 1 :
0 signifie que deux boîtes englobantes ne se chevauchent pas du tout,
tandis que 1 indique que les deux boîtes englobantes sont égales.

![IoU is the ratio of the intersection area to the union area of two bounding boxes.](../img/iou.svg)
:label:`fig_iou`

Dans le reste de cette section, nous utiliserons IoU pour mesurer la similarité entre les boîtes d'ancrage et les boîtes de délimitation de la vérité du sol, ainsi qu'entre différentes boîtes d'ancrage.
Étant donné deux listes de boîtes d'ancrage ou de boîtes englobantes,
le tableau suivant `box_iou` calcule leur IoU par paire
sur ces deux listes.

```{.python .input}
#@tab mxnet
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boîtes1`, `boîtes2`, `zones1`, `zones2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clip(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

```{.python .input}
#@tab pytorch
#@save
def box_iou(boxes1, boxes2):
    """Compute pairwise IoU across two lists of anchor or bounding boxes."""
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) *
                              (boxes[:, 3] - boxes[:, 1]))
    # Shape of `boxes1`, `boxes2`, `areas1`, `areas2`: (no. of boxes1, 4),
    # (no. of boxes2, 4), (no. of boxes1,), (no. of boxes2,)
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    # Shape of `inter_upperlefts`, `inter_lowerrights`, `inters`: (no. of
    # boxes1, no. of boxes2, 2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    inter_lowerrights = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # Shape of `inter_areas` and `union_areas`: (no. of boxes1, no. of boxes2)
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas
```

## Étiquetage des boîtes d'ancrage dans les données d'entraînement
:label:`subsec_labeling-anchor-boxes` 

 
 Dans un ensemble de données d'entraînement,
nous considérons chaque boîte d'ancrage comme un exemple d'entraînement.
Afin d'entraîner un modèle de détection d'objets,
nous avons besoin d'étiquettes de *classe* et de *décalage* pour chaque boîte d'ancrage,
où la première est
la classe de l'objet pertinent pour la boîte d'ancrage
et la seconde est le décalage
de la boîte d'ancrage de référence par rapport à la boîte d'ancrage.
Pendant la prédiction,
pour chaque image
nous générons plusieurs boîtes d'ancrage,
nous prédisons les classes et les décalages pour toutes les boîtes d'ancrage,
nous ajustons leurs positions en fonction des décalages prédits pour obtenir les boîtes limites prédites,
et finalement nous ne produisons que les boîtes limites prédites 
qui satisfont certains critères.


Comme nous le savons, un ensemble d'entraînement à la détection d'objets
est fourni avec des étiquettes pour
les emplacements des *boîtes englobantes de terrain*
et les classes des objets qui les entourent.
Pour étiqueter une *boîte d'ancrage* générée,
fait référence à l'emplacement et à la classe étiquetés
de sa *boîte d'ancrage de terrain assignée* la plus proche de la boîte d'ancrage.
Dans ce qui suit,
nous décrivons un algorithme permettant d'attribuer
aux boîtes d'ancrage les boîtes limites les plus proches de la vérité du terrain. 

### [**Assigning Ground-Truth Bounding Boxes to Anchor Boxes**]

Given an image,
suppose that the anchor boxes are $A_1, A_2, \ldots, A_{n_a}$ and the ground-truth bounding boxes are $B_1, B_2, \ldots, B_{n_b}$, where $n_a \geq n_b$.
Let's define a matrix $\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$, whose element $x_{ij}$ in the $i^\mathrm{th}$ row and $j^\mathrm{th}$ column is the IoU of the anchor box $A_i$ and the ground-truth bounding box $B_j$. The algorithm consists of the following steps:

1. Find the largest element in matrix $\mathbf{X}$ and denote its row and column indices as $i_1$ and $j_1$, respectively. Then the ground-truth bounding box $B_{j_1}$ is assigned to the anchor box $A_{i_1}$. This is quite intuitive because $A_{i_1}$ and $B_{j_1}$ are the closest among all the pairs of anchor boxes and ground-truth bounding boxes. After the first assignment, discard all the elements in the ${i_1}^\mathrm{th}$ row and the ${j_1}^\mathrm{th}$ column in matrix $\mathbf{X}$. 
1. Find the largest of the remaining elements in matrix $\mathbf{X}$ and denote its row and column indices as $i_2$ and $j_2$, respectively. We assign ground-truth bounding box $B_{j_2}$ to anchor box $A_{i_2}$ and discard all the elements in the ${i_2}^\mathrm{th}$ row and the ${j_2}^\mathrm{th}$ column in matrix $\mathbf{X}$.
1. At this point, elements in two rows and two columns in  matrix $\mathbf{X}$ have been discarded. We proceed until all elements in $n_b$ columns in matrix $\mathbf{X}$ are discarded. At this time, we have assigned a ground-truth bounding box to each of $n_b$ anchor boxes.
1. Only traverse through the remaining $n_a - n_b$ anchor boxes. For example, given any anchor box $A_i$, find the ground-truth bounding box $B_j$ with the largest IoU with $A_i$ throughout the $i^\mathrm{th}$ row of matrix $\mathbf{X}$, and assign $B_j$ to $A_i$ only if this IoU is greater than a predefined threshold.

Let's illustrate the above algorithm using a concrete
example.
As shown in :numref:`fig_anchor_label` (left)en supposant que la valeur maximale de la matrice $\mathbf{X}$ est $x_{23}$, nous assignons la boîte limite de vérité de base $B_3$ à la boîte d'ancrage $A_2$.
Ensuite, nous éliminons tous les éléments de la ligne 2 et de la colonne 3 de la matrice, nous trouvons la plus grande valeur de $x_{71}$ dans les éléments restants (zone ombrée) et nous attribuons la boîte limite de vérité fondamentale $B_1$ à la boîte d'ancrage $A_7$. 
Ensuite, comme indiqué dans :numref:`fig_anchor_label` (au milieu), éliminez tous les éléments de la ligne 7 et de la colonne 1 de la matrice, trouvez le plus grand $x_{54}$ dans les éléments restants (zone ombrée) et affectez la boîte limite de vérité de base $B_4$ à la boîte d'ancrage $A_5$. 
Enfin, comme indiqué sur :numref:`fig_anchor_label` (à droite), éliminez tous les éléments de la ligne 5 et de la colonne 4 de la matrice, trouvez le plus grand $x_{92}$ dans les éléments restants (zone ombrée) et attribuez la boîte limite de vérité de base $B_2$ à la boîte d'ancrage $A_9$.
Ensuite, il suffit de parcourir
les autres boîtes d'ancrage $A_1, A_3, A_4, A_6, A_8$ et de déterminer s'il faut leur attribuer des boîtes limites de vérité fondamentale en fonction du seuil.

![Assigning ground-truth bounding boxes to anchor boxes.](../img/anchor-label.svg)
:label:`fig_anchor_label`

Cet algorithme est mis en œuvre dans la fonction suivante `assign_anchor_to_bbox`.

```{.python .input}
#@tab mxnet
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = np.full((num_anchors,), -1, dtype=np.int32, ctx=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = np.max(jaccard, axis=1), np.argmax(jaccard, axis=1)
    anc_i = np.nonzero(max_ious >= iou_threshold)[0]
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = np.full((num_anchors,), -1)
    row_discard = np.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = np.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).astype('int32')
        anc_idx = (max_idx / num_gt_boxes).astype('int32')
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

```{.python .input}
#@tab pytorch
#@save
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """Assign closest ground-truth bounding boxes to anchor boxes."""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # Element x_ij in the i-th row and j-th column is the IoU of the anchor
    # box i and the ground-truth bounding box j
    jaccard = box_iou(anchors, ground_truth)
    # Initialize the tensor to hold the assigned ground-truth bounding box for
    # each anchor
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long,
                                  device=device)
    # Assign ground-truth bounding boxes according to the threshold
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)  # Find the largest IoU
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map
```

### Étiquetage des classes et des décalages

Nous pouvons maintenant étiqueter la classe et le décalage de chaque boîte d'ancrage. Supposons qu'une boîte d'ancrage $A$ se voit attribuer
une boîte limite de vérité du sol $B$. 
D'une part,
la classe de la boîte d'ancrage $A$ sera étiquetée
comme celle de $B$.
D'autre part,
le décalage de la boîte d'ancrage $A$ 
 sera étiqueté en fonction de la position relative 
entre
les coordonnées centrales de $B$ et $A$
 ainsi que de la taille relative entre
ces deux boîtes.
Étant donné les positions et les tailles variables
des différentes boîtes de l'ensemble de données,
nous pouvons appliquer des transformations
à ces positions et tailles relatives
qui peuvent conduire à 
des décalages
plus uniformément distribués et plus faciles à ajuster.
Nous décrivons ici une transformation courante.
[**Étant donné les coordonnées centrales de $A$ et $B$ comme $(x_a, y_a)$ et $(x_b, y_b)$, 
leurs largeurs comme $w_a$ et $w_b$, 
et leurs hauteurs comme $h_a$ et $h_b$, respectivement. 
Nous pouvons étiqueter le décalage de $A$ comme

$$\left( \frac{ \frac{x_b - x_a}{w_a} - \mu_x }{\sigma_x},
\frac{ \frac{y_b - y_a}{h_a} - \mu_y }{\sigma_y},
\frac{ \log \frac{w_b}{w_a} - \mu_w }{\sigma_w},
\frac{ \log \frac{h_b}{h_a} - \mu_h }{\sigma_h}\right),$$
**]
où les valeurs par défaut des constantes sont $\mu_x = \mu_y = \mu_w = \mu_h = 0, \sigma_x=\sigma_y=0.1$, et $\sigma_w=\sigma_h=0.2$.
Cette transformation est mise en œuvre ci-dessous dans la fonction `offset_boxes`.

```{.python .input}
#@tab all
#@save
def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """Transform for anchor box offsets."""
    c_anc = d2l.box_corner_to_center(anchors)
    c_assigned_bb = d2l.box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * d2l.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = d2l.concat([offset_xy, offset_wh], axis=1)
    return offset
```

Si une boîte d'ancrage n'est pas associée à une boîte de délimitation de la vérité du sol, nous étiquetons simplement la classe de la boîte d'ancrage comme "arrière-plan".
Les boîtes d'ancrage dont la classe est l'arrière-plan sont souvent appelées boîtes d'ancrage *négatives*,
et les autres sont appelées boîtes d'ancrage *positives*.
Nous implémentons la fonction suivante `multibox_target`
 pour [**étiqueter les classes et les décalages pour les boîtes d'ancrage**] (l'argument `anchors` ) en utilisant des boîtes d'encombrement conformes à la réalité (l'argument `labels` ).
Cette fonction définit la classe d'arrière-plan à zéro et incrémente de un l'indice entier d'une nouvelle classe.

```{.python .input}
#@tab mxnet
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.ctx, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = np.tile((np.expand_dims((anchors_bbox_map >= 0),
                                            axis=-1)), (1, 4)).astype('int32')
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = d2l.zeros(num_anchors, dtype=np.int32, ctx=device)
        assigned_bb = d2l.zeros((num_anchors, 4), dtype=np.float32,
                                ctx=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = np.nonzero(anchors_bbox_map >= 0)[0]
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].astype('int32') + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = d2l.stack(batch_offset)
    bbox_mask = d2l.stack(batch_mask)
    class_labels = d2l.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_target(anchors, labels):
    """Label anchor boxes using ground-truth bounding boxes."""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        label = labels[i, :, :]
        anchors_bbox_map = assign_anchor_to_bbox(
            label[:, 1:], anchors, device)
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(
            1, 4)
        # Initialize class labels and assigned bounding box coordinates with
        # zeros
        class_labels = torch.zeros(num_anchors, dtype=torch.long,
                                   device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32,
                                  device=device)
        # Label classes of anchor boxes using their assigned ground-truth
        # bounding boxes. If an anchor box is not assigned any, we label its
        # class as background (the value remains zero)
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # Offset transformation
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
```

### Un exemple

Illustrons l'étiquetage des boîtes d'ancrage
par un exemple concret.

$(x, y)$ Nous définissons des boîtes d'ancrage véridiques pour le chien et le chat dans l'image chargée,
où le premier élément est la classe (0 pour le chien et 1 pour le chat) et les quatre autres éléments sont les coordonnées de l'axe
du coin supérieur gauche et du coin inférieur droit
(la plage est comprise entre 0 et 1). 
Nous construisons également cinq boîtes d'ancrage à étiqueter
en utilisant les coordonnées de
au coin supérieur gauche et au coin inférieur droit :
$A_0, \ldots, A_4$ (l'indice commence à 0).
Ensuite, nous [**traçons ces boîtes limites de vérité du sol 
et ces boîtes d'ancrage 
dans l'image.**]

```{.python .input}
#@tab all
ground_truth = d2l.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = d2l.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])

fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4']);
```

À l'aide de la fonction `multibox_target` définie ci-dessus,
nous pouvons [**étiqueter les classes et les décalages
de ces boîtes d'ancrage sur la base de
les boîtes limites de vérité du sol**] pour le chien et le chat.
Dans cet exemple, les indices de
les classes d'arrière-plan, de chien et de chat
sont 0, 1 et 2, respectivement. 
Nous ajoutons ci-dessous une dimension pour des exemples de boîtes d'ancrage et de boîtes limites de vérité fondamentale.

```{.python .input}
#@tab mxnet
labels = multibox_target(np.expand_dims(anchors, axis=0),
                         np.expand_dims(ground_truth, axis=0))
```

```{.python .input}
#@tab pytorch
labels = multibox_target(anchors.unsqueeze(dim=0),
                         ground_truth.unsqueeze(dim=0))
```

Il y a trois éléments dans le résultat retourné, tous au format tenseur.
Le troisième élément contient les classes étiquetées des boîtes d'ancrage d'entrée.

Analysons les étiquettes de classe renvoyées ci-dessous en fonction de la position des boîtes d'ancrage
et des boîtes de délimitation de la vérité du sol dans l'image.
Premièrement, parmi toutes les paires de boîtes d'ancrage
et de boîtes limites de vérité,
l'IoU de la boîte d'ancrage $A_4$ et la boîte limite de vérité du chat sont les plus grandes. 
Ainsi, la classe de $A_4$ est étiquetée comme étant le chat.
Si l'on exclut les paires 
contenant $A_4$ ou la boîte englobante de vérité du chat, parmi les autres 
, la paire de la boîte d'ancrage $A_1$ et de la boîte englobante de vérité du chien a le plus grand IoU.
La classe de $A_1$ est donc étiquetée comme étant celle du chien.
Ensuite, nous devons parcourir les trois autres boîtes d'ancrage non étiquetées : $A_0$, $A_2$, et $A_3$.
Pour $A_0$,
la classe de la boîte d'ancrage de vérité du sol avec le plus grand IoU est le chien,
mais l'IoU est inférieur au seuil prédéfini (0.5),
la classe est donc étiquetée comme étant l'arrière-plan ;
pour $A_2$,
la classe de la boîte englobante de vérité avec la plus grande IoU est le chat et l'IoU dépasse le seuil, la classe est donc étiquetée comme étant le chat ;
pour $A_3$,
la classe de la boîte englobante de vérité avec la plus grande IoU est le chat, mais la valeur est inférieure au seuil, la classe est donc étiquetée comme étant l'arrière-plan.

```{.python .input}
#@tab all
labels[2]
```

Le deuxième élément renvoyé est une variable de masque de la forme (taille du lot, quatre fois le nombre de boîtes d'ancrage).
Tous les quatre éléments de la variable de masque 
correspondent aux quatre valeurs de décalage de chaque boîte d'ancrage.
Comme nous ne nous soucions pas de la détection de l'arrière-plan, les décalages de
de cette classe négative ne devraient pas affecter la fonction objective.
Grâce à des multiplications par éléments, les zéros de la variable de masque filtreront les décalages de classe négative avant de calculer la fonction objectif.

```{.python .input}
#@tab all
labels[1]
```

Le premier élément retourné contient les quatre valeurs de décalage étiquetées pour chaque boîte d'ancrage.
Notez que les décalages des boîtes d'ancrage de classe négative sont étiquetés comme des zéros.

```{.python .input}
#@tab all
labels[0]
```

## Prédiction de boîtes englobantes avec suppression non maximale
:label:`subsec_predicting-bounding-boxes-nms` 

 Pendant la prédiction,
nous générons plusieurs boîtes d'ancrage pour l'image et prédisons les classes et les décalages pour chacune d'elles.
Une *bounding box* prédite
est ainsi obtenue en fonction de 
une boîte d'ancrage avec son décalage prédit.
Nous implémentons ci-dessous la fonction `offset_inverse`
 qui prend en entrée les boîtes d'ancrage et les prédictions de décalage
et [**applique des transformations de décalage inverses pour
renvoyer les coordonnées prédites de la boîte englobante**].

```{.python .input}
#@tab all
#@save
def offset_inverse(anchors, offset_preds):
    """Predict bounding boxes based on anchor boxes with predicted offsets."""
    anc = d2l.box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = d2l.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = d2l.concat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = d2l.box_center_to_corner(pred_bbox)
    return predicted_bbox
```

Lorsqu'il existe de nombreuses boîtes d'ancrage,
de nombreuses boîtes englobantes prédites similaires (avec un chevauchement important)
 
 peuvent potentiellement être produites pour entourer le même objet.
Pour simplifier la sortie,
nous pouvons fusionner les boîtes englobantes prédites similaires
qui appartiennent au même objet
en utilisant la *suppression non maximale* (NMS).

Voici comment fonctionne la suppression non maximale.
Pour une boîte englobante prédite $B$,
le modèle de détection d'objets calcule la vraisemblance prédite
pour chaque classe.
En désignant par $p$ la probabilité prédite la plus élevée,
la classe correspondant à cette probabilité est la classe prédite pour $B$.
Plus précisément, nous appelons $p$ la *confiance* (score) de la boîte englobante prédite $B$.
Sur la même image,
toutes les boîtes englobantes sans arrière-plan prédites 
sont triées par confiance en ordre décroissant
pour générer une liste $L$.
Ensuite, nous manipulons la liste triée $L$ selon les étapes suivantes :

1. Sélectionnez la boîte englobante prédite $B_1$ avec la confiance la plus élevée dans $L$ comme base et supprimez de $L$ toutes les boîtes englobantes prédites sans base dont l'IoU avec $B_1$ dépasse un seuil prédéfini $\epsilon$. À ce stade, $L$ conserve la boîte englobante prédite ayant le degré de confiance le plus élevé, mais élimine les autres boîtes qui lui sont trop similaires. En bref, celles dont le score de confiance n'est pas maximal sont supprimées.
1. Sélectionnez la boîte englobante prédite $B_2$ avec le deuxième degré de confiance le plus élevé à partir de $L$ comme autre base et supprimez toutes les boîtes englobantes prédites non basées dont l'IoU avec $B_2$ dépasse $\epsilon$ à partir de $L$.
1. Répétez le processus ci-dessus jusqu'à ce que toutes les boîtes englobantes prédites de $L$ aient été utilisées comme base. À ce stade, l'IoU de toute paire de boîtes englobantes prédites dans $L$ est inférieur au seuil $\epsilon$; ainsi, aucune paire n'est trop similaire entre elles. 
1. Affichez toutes les boîtes englobantes prédites dans la liste $L$.

[**La fonction suivante `nms` trie les scores de confiance par ordre décroissant et renvoie leurs indices.**]

```{.python .input}
#@tab mxnet
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = scores.argsort()[::-1]
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = np.nonzero(iou <= iou_threshold)[0]
        B = B[inds + 1]
    return np.array(keep, dtype=np.int32, ctx=boxes.ctx)
```

```{.python .input}
#@tab pytorch
#@save
def nms(boxes, scores, iou_threshold):
    """Sort confidence scores of predicted bounding boxes."""
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # Indices of predicted bounding boxes that will be kept
    while B.numel() > 0:
        i = B[0]
        keep.append(i)
        if B.numel() == 1: break
        iou = box_iou(boxes[i, :].reshape(-1, 4),
                      boxes[B[1:], :].reshape(-1, 4)).reshape(-1)
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return d2l.tensor(keep, device=boxes.device)
```

Nous définissons la fonction suivante `multibox_detection`
 pour [**appliquer une suppression non maximale
à la prédiction des boîtes englobantes**].
Ne vous inquiétez pas si vous trouvez l'implémentation
un peu compliquée : nous montrerons comment cela fonctionne
avec un exemple concret juste après l'implémentation.

```{.python .input}
#@tab mxnet
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.ctx, cls_probs.shape[0]
    anchors = np.squeeze(anchors, axis=0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = np.max(cls_prob[1:], 0), np.argmax(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`garder` indices and set the class to background
        all_idx = np.arange(num_anchors, dtype=np.int32, ctx=device)
        combined = d2l.concat((keep, all_idx))
        unique, counts = np.unique(combined, return_counts=True)
        non_keep = unique[counts == 1]
        all_id_sorted = d2l.concat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted].astype('float32')
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = d2l.concat((np.expand_dims(class_id, axis=1),
                                np.expand_dims(conf, axis=1),
                                predicted_bb), axis=1)
        out.append(pred_info)
    return d2l.stack(out)
```

```{.python .input}
#@tab pytorch
#@save
def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """Predict bounding boxes using non-maximum suppression."""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_prob[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)
        # Find all non-`garder` indices and set the class to background
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # Here `pos_threshold` is a threshold for positive (non-background)
        # predictions
        below_min_idx = (conf < pos_threshold)
        class_id[below_min_idx] = -1
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat((class_id.unsqueeze(1),
                               conf.unsqueeze(1),
                               predicted_bb), dim=1)
        out.append(pred_info)
    return d2l.stack(out)
```

Appliquons maintenant [**les implémentations ci-dessus
à un exemple concret avec quatre boîtes d'ancrage**].
Pour simplifier, nous supposons que les décalages prédits de
sont tous des zéros.
Cela signifie que les boîtes de délimitation prédites sont des boîtes d'ancrage. 
Pour chaque classe parmi le fond, le chien et le chat,
nous définissons également sa vraisemblance prédite.

```{.python .input}
#@tab all
anchors = d2l.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                      [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
offset_preds = d2l.tensor([0] * d2l.size(anchors))
cls_probs = d2l.tensor([[0] * 4,  # Predicted background likelihood 
                      [0.9, 0.8, 0.7, 0.1],  # Predicted dog likelihood 
                      [0.1, 0.2, 0.3, 0.9]])  # Predicted cat likelihood
```

Nous pouvons [**tracer ces boîtes englobantes prédites avec leur confiance sur l'image.**]

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
show_bboxes(fig.axes, anchors * bbox_scale,
            ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])
```

Nous pouvons maintenant invoquer la fonction `multibox_detection`
 pour effectuer une suppression non maximale,
où le seuil est fixé à 0,5.
Notez que nous ajoutons
une dimension pour les exemples dans l'entrée tensorielle.

Nous pouvons voir que [**la forme du résultat retourné**] est
(taille du lot, nombre de boîtes d'ancrage, 6).
Les six éléments de la dimension la plus intérieure
donnent les informations de sortie pour la même boîte d'ancrage prédite.
Le premier élément est l'indice de classe prédit, qui commence à 0 (0 est chien et 1 est chat). La valeur -1 indique l'arrière-plan ou la suppression en cas de suppression non maximale.
Le deuxième élément est la confiance de la boîte de délimitation prédite.
Les quatre éléments restants sont les coordonnées sur l'axe $(x, y)$ du coin supérieur gauche et 
du coin inférieur droit de la boîte englobante prédite, respectivement (la plage est comprise entre 0 et 1).

```{.python .input}
#@tab mxnet
output = multibox_detection(np.expand_dims(cls_probs, axis=0),
                            np.expand_dims(offset_preds, axis=0),
                            np.expand_dims(anchors, axis=0),
                            nms_threshold=0.5)
output
```

```{.python .input}
#@tab pytorch
output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)
output
```

Après avoir supprimé les boîtes englobantes prédites
de classe -1, 
nous pouvons [**produire la boîte englobante prédite finale
conservée par suppression non maximale**].

```{.python .input}
#@tab all
fig = d2l.plt.imshow(img)
for i in d2l.numpy(output[0]):
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [d2l.tensor(i[2:]) * bbox_scale], label)
```

Dans la pratique, nous pouvons supprimer les boîtes englobantes prédites ayant un faible niveau de confiance avant même d'effectuer la suppression non maximale, ce qui réduit le calcul de cet algorithme.
Nous pouvons également post-traiter le résultat de la suppression non maximale, par exemple, en ne conservant que les résultats
avec une confiance plus élevée
dans le résultat final.


## Résumé

* Nous générons des boîtes d'ancrage de différentes formes centrées sur chaque pixel de l'image.
* L'intersection sur l'union (IoU), également connue sous le nom d'indice de Jaccard, mesure la similarité de deux boîtes englobantes. Il s'agit du rapport entre leur zone d'intersection et leur zone d'union.
* Dans un ensemble d'apprentissage, nous avons besoin de deux types d'étiquettes pour chaque boîte d'ancrage. L'un est la classe de l'objet pertinent pour la boîte d'ancrage et l'autre est le décalage de la boîte d'ancrage de référence par rapport à la boîte d'ancrage.
* Pendant la prédiction, nous pouvons utiliser la suppression non maximale (NMS) pour éliminer les boîtes de délimitation prédites similaires, simplifiant ainsi la sortie.


## Exercices

1. Modifiez les valeurs de `sizes` et `ratios` dans la fonction `multibox_prior`. Quels sont les changements apportés aux boîtes d'ancrage générées ?
1. Construisez et visualisez deux boîtes englobantes avec un IoU de 0,5. Comment se chevauchent-elles ?
1. Modifiez la variable `anchors` dans :numref:`subsec_labeling-anchor-boxes` et :numref:`subsec_predicting-bounding-boxes-nms` . Comment les résultats changent-ils ?
1. La suppression non maximale est un algorithme gourmand qui supprime les boîtes englobantes prédites en les *supprimant*. Est-il possible que certaines de ces boîtes supprimées soient réellement utiles ? Comment peut-on modifier cet algorithme pour supprimer *mollement* ? Vous pouvez vous référer à Soft-NMS :cite:`Bodla.Singh.Chellappa.ea.2017` .
1. Plutôt que d'être fabriquée à la main, la suppression non-maximale peut-elle être apprise ?

:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/370)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/1603)
:end_tab:
