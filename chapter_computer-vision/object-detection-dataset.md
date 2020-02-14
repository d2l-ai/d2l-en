# The Object Detection Dataset (VOC)
:numref:`sec_detection_dataset`

[Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) is one the widely used dataset in object detection. In this chapter, we will describe how to download and read this dataset.

```{.python .input  n=4}
%matplotlib inline
import d2l
from mxnet import gluon, image, np, npx
import os

npx.set_np()
```

## Downloading the Dataset

The whole dataset is contained in the `VOCtrainval_11-May-2012.tar` that can be downloaded from the website. The archive is about 2 GB, so it will take some time to download. The original site might be unstable, here we download the data from a mirror site.  After you decompress the archive, the dataset is located in the `../data/VOCdevkit/VOC2012`.

```{.python .input  n=5}
# Saved in the d2l package for later use
d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
                           '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

voc_dir = '../data/VOCdevkit/VOC2012'  # fixme, remove this one
#voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
```

## Reading the Dataset

This dataset contains 20 objects in four categories: person, animal, vehicle and indoor, with an additional background class.

```{.python .input  n=6}
# Saved in the d2l package for later use
VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

For each class, the training and validation sets splitting can be found in the `ImageSets/Main` folder. Each file contains the image filenames with `1` and `-1` labels. All images are in the `JPEGImages` folder. The following function `read_voc_images` loads the images into memory by giving the image set filename.

```{.python .input  n=9}
# Saved in the d2l package for later use
def read_voc_images(voc_dir, image_sets):
    """Read images specified in the list of files."""
    lines = []
    for fn in image_sets:
        image_set_fname = '%s/ImageSets/%s' % (voc_dir, fn)
        with open(image_set_fname, 'r') as f:
            lines.extend(f.read().split('\n'))
    # only keep positive instances
    image_fns, images = [], []
    for l in lines:
        items = l.split(' ')
        if l and (len(items) == 1 or items[-1] == '1'):
            image_fns.append(items[0])
    for fn in image_fns:
        images.append(image.imread('%s/JPEGImages/%s.jpg' % (voc_dir, fn)))
    return images, image_fns
```

Let's read all images in the training set for the cat class, and compute their average image shape.

```{.python .input  n=10}
images, image_fns = read_voc_images(voc_dir, ['Main/cat_train.txt'])
sum([np.array(img.shape) for img in images])/len(images)
```

The labels for each image are stored as a xml file in the `Annotations` folder. Each labeled object in the image is represented as an "object" element, which contains class name ("name" element) and bounding box ("bndbox" element) with the two-corner representation. Function `read_voc_bboxes` reads the bounding boxes of a set of classes. It returns a list of ndarrays, the shape of the $i$-the ndarray is $(n, 5)$, where $n$ is the number of labeled objects in the $i$-th image, and each row contains the class index and the corresponding bounding box.

```{.python .input  n=11}
# Saved in the d2l package for later use
def read_voc_detection_labels(voc_dir, image_fns, classes):
    import xml.etree.cElementTree as et
    labels = []
    class_to_idx = dict([(cls, i) for i, cls in enumerate(classes)])
    for fn in image_fns:
        xml_fn = '%s/Annotations/%s.xml' % (voc_dir, fn)
        root = et.parse(xml_fn).getroot()
        label = []
        for obj in root.iter('object'):
            cls = obj.find('name').text.strip().lower()
            if cls not in class_to_idx: continue
            box = [float(obj.find('bndbox').find(name).text) - 1
                   for name in ('xmin', 'ymin', 'xmax', 'ymax')]
            label.append([class_to_idx[cls]]+box)
        labels.append(np.array(label))
    return labels
```

Let's read all ground-truth bonding boxes for the cat class.

```{.python .input  n=12}
labels = read_voc_detection_labels(voc_dir, image_fns, ['cat'])
len(labels), labels[0]
```

Visualize the first 10 images with their ground truth bounding boxes. Note that the $7$-th image contains two bounding boxes.

```{.python .input  n=13}
axes = d2l.show_images(images[:10], 2, 5, scale=2)
for ax, label in zip(axes, labels[:10]):
    d2l.show_boxes(ax, label[:,1:5], colors=['w'])
```

## Object Detection Dataset Class

Next we create a subclass of `gluon.data.Dataset` to return preprocessed examples. We can apply the augmentations described in :numref:`sec_image_augmentation` straightforwardly except for random cropping. When cropping an image, we need to adjust the bounding boxes as well. For simplicity, here we just resize images to the same size, normalize the RGB channels, and randomly flip the image.


As mentioned before, an image may contain more than one bounding boxes and therefore the label shape varies. To batch these labels, we find the image with most bounding boxes, denoted by $n$ the size, then pad the label array into $(n,5)$. The first elements of the padded rows are set to $-1$, referring to an invalid class.

```{.python .input  n=14}
# Saved in the d2l package for later use
class DetectionDataset(gluon.data.Dataset):
    """A customized dataset to load VOC dataset."""
    def __init__(self, images, labels, height, width):
        n = max([l.shape[0] for l in labels])  # max #objects per image
        self.labels = []
        for label, img in zip(labels, images):
            # Project bounding boxes to [0, 1]
            h, w, _ = img.shape
            label[:,1:5] /= np.array([w, h, w, h])
            k = label.shape[0]
            if k < n:  # Pad invalid (-1) labels
                label = np.concatenate([label, np.zeros((n-k, 5))], axis=0)
                label[k:,0] = -1
            self.labels.append(label)
        # Resize and then normalize RGB channels
        self.features = [(image.imresize(img, width, height).astype(
            'float32')/255-d2l.RGB_MEAN)/d2l.RGB_STD for img in images]

    def __getitem__(self, idx):
        feature, label = self.features[idx], self.labels[idx]
        # Flip both image and bounding box horizontally in random.
        if float(np.random.uniform()) > 0.5:
            feature = feature[:, ::-1, :]
            label = np.stack([label[:,0], 1-label[:,3], label[:,2],
                              1-label[:,1], label[:,4]], axis=1)
        return feature.transpose(2, 0, 1), label

    def __len__(self):
        return len(self.features)
```

Let's construct a data loader to read the first batch. Comparing reading image classification data, the label shape changes from `(batch_size, )` to `(batch_size, n, 5)`.

```{.python .input  n=15}
train_ds = DetectionDataset(images, labels, 400, 470)
train_iter = gluon.data.DataLoader(train_ds, 10)
for x, y in train_iter:
    break
x.shape, y.shape
```

We can visualize the images as before.

```{.python .input  n=20}
imgs = [img.transpose(1, 2, 0)*d2l.RGB_STD+d2l.RGB_MEAN for img in x]
axes = d2l.show_images(imgs, 2, 5, scale=2)
for ax, label in zip(axes, y):
    h, w, _ = imgs[0].shape
    d2l.show_boxes(ax, label[:,1:5]*np.array([w,h,w,h]), colors=['w'])
```

## Putting All Things Together

Finally, we define a function `load_data_voc_detection` that downloads and loads this dataset, and then returns the data loaders. As this dataset contains more than 17,000 images, in default we only read the "cat" and the "dog" classes to accelerate the following trainings.

```{.python .input  n=25}
# Saved in the d2l package for later use
def load_data_voc_detection(batch_size, height, width, classes=['cat','dog']):
    """Load the pikachu dataset."""
    def load_data(dataset):
        txt_fns = ['Main/%s_%s.txt'%(cls, dataset) for cls in classes]
        images, image_fns = read_voc_images(voc_dir, txt_fns)
        labels = read_voc_detection_labels(voc_dir, image_fns, classes)
        return images, labels
    #voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012') 
    voc_dir = '../data/VOCdevkit/VOC2012/' #FIXME
    train_ds = DetectionDataset(*load_data('train'), height, width)
    test_ds = DetectionDataset(*load_data('val'), height, width)
    train_iter, test_iter = [gluon.data.DataLoader(
        ds, batch_size, shuffle=shuffle, last_batch='discard',
        num_workers=d2l.get_dataloader_workers()) for ds, shuffle in
                             [[train_ds, True], [test_ds, False]]]
    return train_iter, test_iter
```

## Summary

1. Pascal VOC2012 is a popular dataset for object detection.
1. We need to update the ground-truth bounding boxes when transforming images, such as flipping and cropping.
1. We pad with invalid bounding boxes to batch the labels.


## Exercises

1. Add random cropping into `DetectionDataset`. You may need to try more than one time if a bounding box is cropped too much. You can refer to how we do random crop for semantic detection in :numref:`sec_semantic_segmentation`. 
1. Loading the whole dataset with 20 classes is time consuming. Change `DetectionDataset` so that it reads the example when needed by `__getitem__`.
