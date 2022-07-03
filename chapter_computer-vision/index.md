# Vision par ordinateur
:label:`chap_cv` 

Qu'il s'agisse de diagnostic médical, de véhicules à conduite autonome, de surveillance par caméra ou de filtres intelligents, de nombreuses applications dans le domaine de la vision par ordinateur sont étroitement liées à notre vie actuelle et future. 
Ces dernières années, l'apprentissage profond a été
le pouvoir transformateur pour faire progresser les performances des systèmes de vision par ordinateur.
On peut dire que les applications de vision par ordinateur les plus avancées sont presque inséparables de l'apprentissage profond.
Dans cette optique, ce chapitre se concentre sur le domaine de la vision par ordinateur et étudie les méthodes et les applications qui ont récemment exercé une influence dans les milieux universitaires et industriels.


Dans :numref:`chap_cnn` et :numref:`chap_modern_cnn`, nous avons étudié différents réseaux de neurones convolutifs
couramment utilisés en vision par ordinateur et nous les avons appliqués
à des tâches simples de classification d'images. 
Au début de ce chapitre, nous décrirons
deux méthodes qui 
peuvent améliorer la généralisation du modèle, à savoir *l'augmentation de l'image* et *le réglage fin*,
et nous les appliquerons à la classification d'images. 
Étant donné que les réseaux neuronaux profonds peuvent représenter efficacement les images à plusieurs niveaux,
ces représentations en couches ont été utilisées avec succès 
dans diverses tâches de vision par ordinateur telles que la *détection d'objets*, la *segmentation sémantique* et le *transfert de style*. 
En suivant l'idée clé d'exploiter les représentations en couches dans la vision par ordinateur,
nous commencerons par les principaux composants et techniques de détection d'objets. Ensuite, nous montrerons comment utiliser les *réseaux entièrement convolutifs* pour la segmentation sémantique des images. Puis nous expliquerons comment utiliser les techniques de transfert de style pour générer des images comme la couverture de ce livre.
Enfin, nous concluons ce chapitre
en appliquant les matériaux de ce chapitre et de plusieurs chapitres précédents sur deux ensembles de données de référence populaires en vision par ordinateur.

```toc
:maxdepth: 2

image-augmentation
fine-tuning
bounding-box
anchor
multiscale-object-detection
object-detection-dataset
ssd
rcnn
semantic-segmentation-and-dataset
transposed-conv
fcn
neural-style
kaggle-cifar10
kaggle-dog
```

