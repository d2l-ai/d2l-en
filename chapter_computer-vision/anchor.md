# Anchor-based Detection
:label:`sec_anchor`

Object detection algorithms usually sample a large number of regions in the input image, determine whether these regions contain objects of interest, and adjust the edges of the regions so as to predict the ground-truth bounding box of the target more accurately. The locations of these regions are often called anchor boxes. 

The way how each algorithm samples anchors boxes and predicts based on these boxes varies. In this chapter, we will illustrate the basic ideas based on Faster-RCNN :cite:`Ren.He.Girshick.ea.2015` and SSD :cite:`Liu.Anguelov.Erhan.ea.2016`, which will be described in details later. 

First, import the packages or modules required for this section.

```{.python .input  n=11}
%matplotlib inline
import d2l
from mxnet import image, np, npx

npx.set_np()
```

## Generating Anchor Boxes

Assume that the input image has a height of $h$ and width of $w$. One way to generate anchor boxes is that, for each pixel in the image, we sample multiple boxes centered on this pixel while have different sizes and aspect ratios. 
For an anchor box, assume the size is $s\in (0, 1]$, the aspect ratio is $r > 0$, then the width and height of the anchor box are $\min(w, h)s\sqrt{r}$ and $\min(w, h)s/\sqrt{r}$, respectively. 

Then we need to choose a set of sizes $s_1,\ldots, s_n$ and a set of aspect ratios $r_1,\ldots, r_m$. We can use a combination of all sizes and aspect ratios with each pixel as the center, the input image will have a total of $whnm$ anchor boxes. Although these anchor boxes may cover all ground-truth bounding boxes, the computational complexity is often excessive since we need to predict for every anchor. 

We can reduce the size by removing the anchor boxes if they are out of bounds. For example, any anchor box with height or width larger than 1 is out of bounds when centered at the top-left corner. Faster-RCNN adopted this approach. But note that it results in the number of anchors boxes centered at each pixel varies.

Another way is limiting the size and aspect ratio combinations. For example, in SSD, the following combination list is used: 

$$(s_1, r_1), (s_1, r_2), \ldots, (s_1, r_m), (s_2, r_1), (s_3, r_1), \ldots, (s_n, r_1).$$

That is, the number of anchor boxes centered on the same pixel is $n+m-1$. For the entire input image, we will generate a total of $wh(n+m-1)$ anchor boxes.

We implement the above sampling strategy in `generate_anchors`. Note that the returned anchors are projected to $[0,1]$, namely a coordinate is presented by $(x/w, y/h)$.

```{.python .input  n=2}
# Saved in the d2l package for later use
def generate_anchors(height, width, sizes, ratios):
    # n + m - 1 ratios and widths
    s = np.array([sizes[0]]*len(ratios) + sizes[1:])
    r = np.array(ratios + [ratios[0]]*(len(sizes)-1))
    w = min(width, height) / width * s * r ** 0.5
    h = min(width, height) / height * s * r ** -0.5
    # all center points
    x, y = np.meshgrid(np.arange(0.5/width, 1.0, 1.0/width), 
                       np.arange(0.5/height, 1.0, 1.0/height))
    # generate n + m - 1 anchors boxes for each center point
    w, x = np.meshgrid(w.reshape(-1), x.reshape(-1))
    h, y = np.meshgrid(h.reshape(-1), y.reshape(-1))
    anchors = np.stack([a.reshape(-1) for a in (x, y, w, h)])
    return d2l.box_center_to_corner(anchors)
```

Let's visualize the anchors boxes generated for a pixel on the cat face with 3 sizes and 3 aspect ratios. You can see that these anchor boxes cover different scale objects, and large tall or flat objects.

```{.python .input  n=3}
img = image.imread(d2l.download('catdog')).asnumpy()
h, w, _ = img.shape
anchors = generate_anchors(h, w, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
anchors = anchors.reshape((4, h, w, -1))[:,h//2,2*w//3,:]
scale = np.array([w, h, w, h])
d2l.set_figsize((3.5, 2.5))
fig = d2l.plt.imshow(img)
d2l.show_boxes(fig.axes, anchors.T * scale)
```

## Predicting based on Anchor Boxes

In prediction, we examine every anchor box. If we believe an anchor box $a$ contains an object we are interested, then we predict the actual bounding box of this object, namely the ground truth bounding box $g$. We often predict the offset from $a$ to $g$ as a regression problem. Despite it's straightforward to choose $g-a$ as the offset, it often leads to unstable prediction. We often use the following transformation to  make the offset distribution more uniform and easier to fit. 

Assume the center-width-height presentations of $a$ is $(x_a, y_a, w_a, h_a)$, same $(x_g, y_g, w_g, h_g)$ for $g$. 
Then the offset is set to be 

$$\left( \sigma_x\frac{x_g - x_a}{w_a},\,
\sigma_y\frac{y_g - y_a}{h_a},\,
\sigma_w\log \frac{w_g}{w_a},\,
\sigma_h\log \frac{h_g}{h_a}\right),$$

where $\sigma_x, \sigma_y, \sigma_w$ and $\sigma_h$ are used to scale each coordinate properly. 

The following `anchor_to_gt_offset` function implements this transformation, and `anchor_plus_offset` applies the offsets to the anchor boxes, with  $\sigma_x = \sigma_y = 10$ and $\sigma_w = \sigma_h = 5$.

```{.python .input  n=4}
# Saved in the d2l package for later use
def anchor_to_gt_offset(anchors, ground_truths):
    """Return the offset from anchor boxes to ground_truth boxes"""
    a = d2l.box_corner_to_center(anchors)
    g = d2l.box_corner_to_center(ground_truths)
    return np.stack((10*(g[0]-a[0])/a[2], 10*(g[1]-a[1])/a[3],
                     5*np.log(g[2]/a[2]), 5*np.log(g[3]/a[3])))

# Saved in the d2l package for later use
def anchor_plus_offset(anchors, offsets):
    """Apply offset to anchors to predict ground truth boxes"""
    a = d2l.box_corner_to_center(anchors)
    x_g, y_g = a[0] + offsets[0] * a[2] / 10, a[1] + offsets[1] * a[3] / 10
    w_g, h_g = a[2] * np.exp(offsets[2] / 5), a[3] * np.exp(offsets[3] / 5)
    return d2l.box_center_to_corner(np.stack((x_g, y_g, w_g, h_g)))
```

Let's check the offsets from the anchors we visualized before to the cat's bounding box.

```{.python .input  n=12}
np.set_printoptions(2)
cat_box = np.array([400, 112, 655, 493])
offsets = anchor_to_gt_offset(anchors, cat_box/scale)
offsets
```

Apply these offsets on the anchors to obtain predicted bounding boxes, which should be identical to the ground truth.

```{.python .input  n=13}
anchor_plus_offset(anchors, offsets).T - cat_box/scale
```

Projects

```{.python .input  n=14}
# Saved in the d2l package for later use
def project_box(boxes):
    """Project boxes into [0,1]"""
    return np.stack((np.maximum(boxes[0], 0),
                     np.maximum(boxes[1], 0),
                     np.minimum(boxes[2], 1),
                     np.minimum(boxes[3], 1)))
```

## Non-maximum suppression

Two identical shape anchor boxes centered on adjacent pixels are hight overlapped. By the way we generate anchor boxes, multiple anchors may response to the same object. Therefore, we may get many similar predicted bounding boxes for the same object. We often need to remove duplicated predictions. A commonly used deduplication method is called non-maximum suppression (NMS).


Let's take a look at how NMS works. For a prediction bounding box $b$, we assume there is a score $s$, which is often the predicted probability for a particular object category. Given boxes $b_1, \ldots, b_n$ with their scores $s_1, \ldots, s_n$, we first initialize $B=\{1,\ldots, n\}$. Next we pick the box with the highest score in $B$, $i=\arg\max_{i\in B}s_i$. and keep it. Then remove all boxes with IoU to the $i$-th box larger than the threshold $\theta$, $B=B\setminus \{j:\mathrm{iou}(b_i, b_j)\ge\theta\}$. Of course we need to remove $i$ from $B$ as well. We repeat these steps until $B$ is empty, and return the kept box indices. Function `nms` implements this algorithm.

```{.python .input  n=98}
# Saved in the d2l package for later use
def nms(scores, boxes, iou_threshold, use_numpy=False):
    """Non-maximum suppression
    
    scores : shape (n,) 
    boxes : shape (4, n)
    use_numpy : fallback to numpy, will remove later
    """
    if use_numpy:
        scores, boxes = scores.asnumpy(), boxes.asnumpy()
    # sorting scores by the descending order and return their indices 
    B = scores.argsort()[::-1]
    keep = []  # boxes indices that will be kept
    while B.size > 0:
        i = B[0]
        keep.append(i)
        if B.size == 1: break  #FIXME, unnecessary
        iou = d2l.iou(boxes[:,i], boxes[:,B[1:]], use_numpy)
        inds = (iou <= iou_threshold).nonzero()[0]
        B = B[inds + 1]
    return np.array(keep)
```

Let construct multiple bounding boxes with scores and visualize them.

```{.python .input  n=20}
dog_box = np.array([60, 45, 378, 516])
boxes = np.array([[80, 245, 328, 420],[120, 125, 500, 540],cat_box,dog_box]) 
scores = np.array([.8, .4, .3, .9])
d2l.show_boxes(d2l.plt.imshow(img).axes, boxes, [str(i) for i in scores])
```

After NMS, we can find that two boxes overlapped with `dog_box` are removed because their scores are lower. And `cat_box` also is kept despite that its score is the lowest as it doesn't overlap with `dog_box`.

```{.python .input  n=21}
keep = nms(scores, boxes.T, 0.2)
d2l.show_boxes(d2l.plt.imshow(img).axes, boxes[keep], [str(i) for i in scores[keep]])
```

## Labeling Anchor Boxes


Now let's look at training. During training, each anchor box is treated as a training example. As these anchor boxes are generated, we need to assign labels to them for training. The label has two parts. First, we mark if an anchor box contains an object, or only has background, or should be ignored. Second, if containing an object, we compute the offset from this anchor box to the ground-truth bounding box if this object.


In FPN, we assign three categories to each anchor box: $1$ for positive anchors that containing an object, $0$ negative anchors that have no object, $-1$ for invalid anchors. Given an image, assume it has ground-truth bounding boxes $g_1, \ldots, g_n$, and anchor boxes $a_1, \ldots, a_m$. For each ground-truth $g_i$, we mark $a_j$ as containing an object if the IoU between $g_i$ and $a_j$ is larger than 0.7. If no such $g_i$ exists, we mark the one has the largest IoU to $g_i$. An anchor box $a_j$ is marked as only containing background if there is no ground-truth $g_i$ such as the IoU between $a_j$ and $g_i$ is greater than $0.3$. The rest anchor boxes are marked as invalid. If we generated too much positive or negative anchor boxes for each image, we can randomly sample some of them.

Function `rpn_targets` implements the above labeling strategy, where we keep all positive anchors while sample 100 negative anchors for each image.

```{.python .input  n=91}
# Saved in the d2l package for later use
def label_anchors(ground_truth, anchors, pos_iou, neg_iou, neg_pos_ratio):
    """Assign labels to anchors

    ground_truth : (batch_size, #objects_per_image, 5)
    anchors : (4, #anchors)
    pos_iou : iou threshold (>) for a positve anchor
    neg_iou : iou threshold (<=) for a negative anchor
    neg_pos_ratio : #neg / #pos, subsampling negatives when necessary
    """
    num_anchors, batch_size = anchors.shape[1], ground_truth.shape[0]
    cls_labels = - np.ones((batch_size, num_anchors))
    box_offsets = np.zeros((batch_size, num_anchors, 4))
    for i, label in enumerate(ground_truth):
        label = label[label[:,0]>=0, :]  # Ignore invalid labels
        neg = np.zeros((len(label), num_anchors), dtype='bool')  # Fix me...
        #print(neg)
        for j, y in enumerate(label):
            iou = d2l.iou(y[1:], anchors)
            pos = iou > pos_iou
            k = int(iou.argmax()) # In case pos is all False
            pos[k] = True  
            cls_labels[i, pos] = y[0]+1  # Positive anchors
            # fixme: i][pos -> i, pos
            box_offsets[i][pos, :] = d2l.anchor_to_gt_offset(
                anchors[:,pos].reshape((4,-1)), # fixme, no reshape is needed
                y[1:].reshape((-1,1))).T
            neg[j][iou < neg_iou] = True
            #print((iou < neg_iou).nonzero()[0])
            neg[j][k] = True
        # Randomly sample negative
        n_pos = int((cls_labels[i]>0).sum())
        neg = neg.all(axis=0).nonzero()[0]        
        np.random.shuffle(neg)
        n_neg = min(len(neg), int(n_pos*neg_pos_ratio))
        if n_neg > 0:  # fixme, no needed
            cls_labels[i, neg[:n_neg]] = 0
    return cls_labels, box_offsets
```

```{.python .input  n=95}
gt = np.zeros((1,2,5))
gt[0,0,1:], gt[0,1,0], gt[0,1,1:] = cat_box/scale, 1, dog_box/scale
anchors = generate_anchors(8, 8, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
cls_labels, box_labels = label_anchors(gt, anchors, 0.6, 0.5, 5)
[int((cls_labels[0]==i).sum()) for i in [0,1,2]] 
```

```{.python .input  n=96}
axes = d2l.show_images([np.array(img)]*3, 1, 3, scale=2.5)
for ax, cls in zip(axes, [0, 1,2]):
    boxes = anchors[:,(cls_labels[0]==cls).nonzero()[0]].T*scale
    d2l.show_boxes(ax, boxes)
```

## Summary

* We generate multiple anchor boxes with different sizes and aspect ratios, centered on each pixel.
* During prediction, we predict the offset from an anchor box to a ground-truth bounding box.
* We can use non-maximum suppression (NMS) to remove similar prediction bounding boxes, thereby simplifying the results.

## Exercises

1. Change the `sizes` and `ratios` values and observe the changes to the generated anchor boxes.
1. Compute the offsets from all generated anchors boxes to the ground-truth `dog_box` and `cat_box` to observe the value distributions. 
1. Increase widht and height when generate box 
1. Cat only has one, investigate the reason, and how to improve it.
1. Modify `generate_anchors` so that out-of-bounds anchor boxes are cropped so they are within bounds. 

## [Discussions](https://discuss.mxnet.io/t/2445)

![](../img/qr_anchor.svg)
