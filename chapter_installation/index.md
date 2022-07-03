# Installation
:label:`chap_installation` 

Afin d'être opérationnel,
nous aurons besoin d'un environnement pour exécuter Python,
le Jupyter Notebook, les bibliothèques pertinentes,
et le code nécessaire pour exécuter le livre lui-même.

## Installation de Miniconda

Votre option la plus simple est d'installer
[Miniconda](https://conda.io/en/latest/miniconda.html) .
Notez que la version Python 3.x est requise.
Vous pouvez sauter les étapes suivantes
si conda est déjà installé sur votre machine.

Visitez le site web de Miniconda et déterminez
la version appropriée pour votre système
en fonction de votre version Python 3.x et de l'architecture de votre machine.
Supposons que votre version de Python soit 3.8
(notre version testée).
Si vous utilisez macOS,
vous téléchargeriez le script bash
dont le nom contient les chaînes "MacOSX",
navigueriez vers l'emplacement de téléchargement,
et exécuteriez l'installation comme suit :

```bash
# The file name is subject to changes
sh Miniconda3-py38_4.10.3-MacOSX-x86_64.sh -b
```


Un utilisateur de Linux
téléchargerait le fichier
dont le nom contient les chaînes "Linux"
et exécuterait ce qui suit à l'emplacement du téléchargement :

```bash
# The file name is subject to changes
sh Miniconda3-py38_4.10.3-Linux-x86_64.sh -b
```


Ensuite, initialisez le shell afin que nous puissions exécuter directement `conda`.

```bash
~/miniconda3/bin/conda init
```


Ensuite, fermez et rouvrez votre shell actuel.
Vous devriez être en mesure de créer
un nouvel environnement comme suit :

```bash
conda create --name d2l python=3.8 -y
```


Maintenant, nous pouvons activer l'environnement `d2l`:

```bash
conda activate d2l
```


## Installation du cadre d'apprentissage profond et du paquet `d2l`

Avant d'installer un cadre d'apprentissage profond,
veuillez d'abord vérifier si
vous disposez ou non de GPU appropriés sur votre machine
(les GPU qui alimentent l'écran
sur un ordinateur portable standard ne sont pas pertinents pour nos objectifs).
Par exemple,
si votre ordinateur est équipé de GPU NVIDIA et a installé [CUDA](https://developer.nvidia.com/cuda-downloads),
alors vous êtes prêt.
Si votre machine n'abrite pas de GPU,
il n'y a pas lieu de s'inquiéter pour l'instant.
Votre processeur fournit plus qu'assez de puissance
pour vous faire passer les premiers chapitres.
Rappelez-vous simplement que vous voudrez accéder aux GPU
avant d'exécuter des modèles plus importants.


:begin_tab:`mxnet`

Pour installer une version de MXNet compatible avec les GPU,
nous devons savoir quelle version de CUDA vous avez installée.
Vous pouvez le vérifier en exécutant `nvcc --version`
ou `cat /usr/local/cuda/version.txt`.
Supposons que vous ayez installé CUDA 10.1,
puis exécutez la commande suivante :

```bash
# For macOS and Linux users
pip install mxnet-cu101==1.7.0

# For Windows users
pip install mxnet-cu101==1.7.0 -f https://dist.mxnet.io/python
```


Vous pouvez changer les derniers chiffres en fonction de votre version de CUDA, par exemple, `cu100` pour
CUDA 10.0 et `cu90` pour CUDA 9.0.


Si votre machine ne possède pas de GPU NVIDIA 
ou CUDA,
vous pouvez installer la version CPU
comme suit :

```bash
pip install mxnet==1.7.0.post1
```


:end_tab:


:begin_tab:`pytorch`

Vous pouvez installer PyTorch avec le support CPU ou GPU comme suit :

```bash
pip install torch==1.8.1
pip install torchvision==0.9.1
```


:end_tab:

:begin_tab:`tensorflow`
Vous pouvez installer TensorFlow avec le support du CPU ou du GPU comme suit :

```bash
pip install tensorflow==2.3.1
pip install tensorflow-probability==0.11.1
```


:end_tab:


L'étape suivante consiste à installer
le paquetage `d2l` que nous avons développé
afin d'encapsuler
les fonctions et classes fréquemment utilisées
que vous trouverez tout au long de ce livre :

```bash
pip install d2l
```


### Téléchargement et exécution du code

Ensuite, nous devons télécharger le code de ce livre.
Vous pouvez cliquer sur l'onglet "Notebooks"
en haut de n'importe quelle page HTML de ce livre
pour télécharger et décompresser le code.
Sinon, si vous disposez de `unzip`
 (sinon, exécutez `sudo apt-get install unzip`):

:begin_tab :`mxnet`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd mxnet
```


:end_tab:


:begin_tab:`pytorch`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd pytorch
```


:end_tab:

:begin_tab:`tensorflow`

```bash
mkdir d2l-en && cd d2l-en
curl https://d2l.ai/d2l-en.zip -o d2l-en.zip
unzip d2l-en.zip && rm d2l-en.zip
cd tensorflow
```
:end_tab:


Nous pouvons maintenant démarrer le serveur Jupyter Notebook en exécutant :

```bash
jupyter notebook
```


À ce stade, vous pouvez ouvrir http://localhost:8888
(il se peut qu'il se soit déjà ouvert automatiquement) dans votre navigateur Web.
Ensuite, nous pouvons exécuter le code pour chaque section du livre.
Veuillez toujours exécuter `conda activate d2l`
pour activer l'environnement d'exécution
avant d'exécuter le code du livre
ou de mettre à jour le cadre d'apprentissage profond ou le paquet `d2l`.
Pour quitter l'environnement,
exécutez `conda deactivate`.



:begin_tab:`mxnet`
[Discussions](https://discuss.d2l.ai/t/23)
:end_tab:

:begin_tab:`pytorch`
[Discussions](https://discuss.d2l.ai/t/24)
:end_tab:

:begin_tab:`tensorflow`
[Discussions](https://discuss.d2l.ai/t/436)
:end_tab:
