# Semantic Segmentation and the Dataset
:label:`sec_semantic_segmentation`

In our discussion of object detection issues in previous sections, we only used rectangular bounding boxes to label and predict objects in images. In this section, we will look at semantic segmentation, which attempts to segment images into regions with different semantic categories. These semantic regions label and predict objects at the pixel level. :numref:`fig_segmentation` shows a semantically-segmented image, with areas labeled "dog", "cat", and "background". As you can see, compared to object detection, semantic segmentation labels areas with pixel-level borders, for significantly greater precision.

![Semantically-segmented image, with areas labeled "dog", "cat", and "background". ](../img/segmentation.svg)
:label:`fig_segmentation`


## Image Segmentation and Instance Segmentation

In the computer vision field, there are two important methods related to semantic segmentation: image segmentation and instance segmentation. Here, we will distinguish these concepts from semantic segmentation as follows:

* Image segmentation divides an image into several constituent regions. This method generally uses the correlations between pixels in an image. During training, labels are not needed for image pixels. However, during prediction, this method cannot ensure that the segmented regions have the semantics we want. If we input the image in 9.10, image segmentation might divide the dog into two regions, one covering the dog's mouth and eyes where black is the prominent color and the other covering the rest of the dog where yellow is the prominent color.
* Instance segmentation is also called simultaneous detection and segmentation. This method attempts to identify the pixel-level regions of each object instance in an image. In contrast to semantic segmentation, instance segmentation not only distinguishes semantics, but also different object instances. If an image contains two dogs, instance segmentation will distinguish which pixels belong to which dog.


## Reading Pascal VOC2012 Semantic Segmentation Dataset

We already described the VOC dataset for object detection in :numref:`sec_detection_dataset`. This same dataset has a semantic segmentation task, which is one of widely used datasets in semantic segmentation. To better explore this task, we must first import the package or module needed for the experiment.

```{.python .input  n=3}
%matplotlib inline
import d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

We can reuse the functions we developed in :numref:`sec_detection_dataset`. But the segmentation task needs a different way to load the pixel level labels. Let's first download and extract this datasets.

```{.python .input  n=4}
voc_dir = '../data/VOCdevkit/VOC2012'  # fixme, remove this one
#voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

The training images are still in the `JPEGImages` folder as for the detection task, which can be read by the `read_voc_images` function. While the labels are saved as png files in the `SegmentationClass`, we need a new function to load them.

```{.python .input  n=13}
# Saved in the d2l package for later use
def read_voc_seg_labels(voc_dir, image_set):
    """Read images specified in the list of files."""
    with open('%s/ImageSets/%s' % (voc_dir, image_set), 'r') as f:
        lines = f.read().split('\n')
    labels = []
    for fn in lines:
        if fn:
            labels.append(image.imread(
                '%s/SegmentationClass/%s.png' % (voc_dir, fn)))
    return labels
```

The set of images for training and test are stored in the `ImageSets/Segmentation` folder. Now we load the images and labels for training.

```{.python .input  n=15}
image_set = 'Segmentation/train.txt'
images, _ = d2l.read_voc_images(voc_dir, [image_set])
labels = read_voc_seg_labels(voc_dir, image_set)
```

We draw the first five input images and their labels. In the label images, white represents borders and black represents the background. Other colors correspond to different categories.

```{.python .input  n=17}
n = 5
imgs = images[0:n] + labels[0:n]
d2l.set_figsize()
d2l.show_images(imgs, 2, n);
```

Remember this dataset has 20 object classes, with a background class. We list each RGB color value for each category, and then defined

in the labels, then we can easily find the category index for each pixel in the labels.

```{.python .input  n=25}
# Saved in the d2l package for later use
VOC_SEG_COLORS = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]]

# Saved in the d2l package for later use
def build_colormap(colors=VOC_SEG_COLORS):
    """Build an colormap that maps RGB colors to indices."""
    colormap = np.zeros(256 ** 3, dtype='int32') 
    for i, color in enumerate(colors):
        colormap[(color[0]*256 + color[1])*256 + color[2]] = i
    return colormap

# Saved in the d2l package for later use
def colors_to_indices(colors, colormap):
    """Map an RGB color to an index."""
    colors = colors.astype('int32')  # convert from uint8 to int32
    return colormap[(colors[:,:,0]*256+colors[:,:,1])*256 + colors[:,:,2]]
```

For example, in the first example image, the category index for the front part of the airplane is 1 and the index for the background is 0.

```{.python .input  n=26}
y = colors_to_indices(labels[0], build_colormap())
y[105:115, 130:140], d2l.VOC_CLASSES[1]
```

## Semantic Segmentation Dataset Class

In  image classification and object detection, we resized images to make them fit the input shape of the model. In semantic segmentation, this method would require us to re-map the predicted pixel categories back to the original-size input image. It would be very difficult to do this precisely, especially in segmented regions with different semantics. To avoid this problem, we crop the images to set dimensions and do not scale them. Specifically, we use the random cropping method used in image augmentation to crop the same region from input images and their labels.


```{.python .input}
# Saved in the d2l package for later use
def segmentation_rand_crop(feature, label, height, width):
    """Randomly crop for both feature and label images."""
    feature, rect = image.random_crop(feature, (width, height))
    label = image.fixed_crop(label, *rect)
    return feature, label

imgs = []
for _ in range(n):
    imgs += segmentation_rand_crop(images[0], labels[0], 200, 300)
d2l.show_images(imgs[::2] + imgs[1::2], 2, n);
```

Next, as we did in :numref:`sec_detection_dataset`, we inherit the `gluon.data.Dataset` class to create a customized semantic segmentation dataset class. Note that we removed images that are smaller than the desired size as we cannot upscale images. 

```{.python .input  n=9}
# Saved in the d2l package for later use
class SegmentationDataset(gluon.data.Dataset):
    """A customized dataset to load VOC dataset."""
    def __init__(self, images, labels, height, width):
        self.height, self.width = height, width
        self.colormap = build_colormap()
        remove_small_images = lambda imgs: [img for img in imgs if (
            img.shape[1] >= width and img.shape[0] >= height)]        
        self.features = [(img.astype('float32')/255-d2l.RGB_MEAN)/d2l.RGB_STD 
                         for img in remove_small_images(images)]
        self.labels = remove_small_images(labels)
    
    def __getitem__(self, idx):
        feature, label = segmentation_rand_crop(
            self.features[idx], self.labels[idx], self.height, self.width)
        return (feature.transpose(2, 0, 1),
                colors_to_indices(label, self.colormap))

    def __len__(self):
        return len(self.features)
```

Let's construct a data loader to read the first batch. 

```{.python .input}
train_ds = SegmentationDataset(images, labels, 320, 480)
train_iter = gluon.data.DataLoader(train_ds, 10)
for x, y in train_iter:
    break
x.shape, y.shape
```

### Putting All Things Together

Finally, we define a function `load_data_voc_segmentation` that  downloads and loads this dataset, and then returns the data loaders.

```{.python .input  n=12}
# Saved in the d2l package for later use
def load_data_voc_segmentation(batch_size, height, width):
    """Download and load the VOC2012 semantic segmentation dataset."""
    def load_data(dataset):
        image_set = 'Segmentation/'+dataset+'.txt'
        images, _ = d2l.read_voc_images(voc_dir, [image_set])
        labels = read_voc_seg_labels(voc_dir, image_set)
        return images, labels
    voc_dir = '../data/VOCdevkit/VOC2012'  # fixme, remove this one
    #voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
    train_ds = SegmentationDataset(*load_data('train'), height, width)
    test_ds = SegmentationDataset(*load_data('val'), height, width)
    train_iter, test_iter = [gluon.data.DataLoader(
        ds, batch_size, shuffle=shuffle, last_batch='discard',
        num_workers=d2l.get_dataloader_workers()) for ds, shuffle in
                             [[train_ds, True], [test_ds, False]]]
    return train_iter, test_iter
```

## Summary

* Semantic segmentation looks at how images can be segmented into regions with different semantic categories.
* In the semantic segmentation field, one important dataset is Pascal VOC2012.
* Because the input images and labels in semantic segmentation have a one-to-one correspondence at the pixel level, we randomly crop them to a fixed size, rather than scaling them.

## Exercises

1. Recall the content we covered in :numref:`sec_image_augmentation`. Which of the image augmentation methods used in image classification would be hard to use in semantic segmentation?


## [Discussions](https://discuss.mxnet.io/t/2448)

![](../img/qr_semantic-segmentation-and-dataset.svg)
