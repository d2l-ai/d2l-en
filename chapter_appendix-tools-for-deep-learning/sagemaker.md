# Utilisation d'Amazon SageMaker
:label:`sec_sagemaker` 

Les applications d'apprentissage profond
peuvent demander tellement de ressources informatiques
qu'elles dépassent facilement
ce que votre machine locale peut offrir.
Les services de cloud computing
vous permettent d'exécuter 
le code intensif en GPU de ce livre
plus facilement
en utilisant des ordinateurs plus puissants.
Cette section présente 
comment utiliser Amazon SageMaker
pour exécuter le code de ce livre.

## S'inscrire

Tout d'abord, nous devons créer un compte sur https://aws.amazon.com/.
Pour plus de sécurité,
il est recommandé d'utiliser une authentification à deux facteurs 
.
C'est également une bonne idée de
configurer une facturation détaillée et des alertes de dépenses pour
éviter toute surprise,
par exemple, 
en cas d'oubli d'arrêter les instances en cours.
Après vous être connecté à votre compte AWS, 
o à votre [console](http://console.aws.amazon.com/) et recherchez "Amazon SageMaker" (voir :numref:`fig_sagemaker` ), 
puis cliquez dessus pour ouvrir le panneau SageMaker.

![Search for and open the SageMaker panel.](../img/sagemaker.png)
:width:`300px`
:label:`fig_sagemaker`

## Création d'une instance SageMaker

Ensuite, créons une instance de notebook comme décrit dans :numref:`fig_sagemaker-create`.

![Create a SageMaker instance.](../img/sagemaker-create.png)
:width:`400px`
:label:`fig_sagemaker-create`

SageMaker propose plusieurs instances [instance types](https://aws.amazon.com/sagemaker/pricing/instance-types/) dont la puissance de calcul et le prix varient.
Lors de la création d'une instance de notebook,
nous pouvons spécifier son nom et son type.
Dans :numref:`fig_sagemaker-create-2`, nous choisissons `ml.p3.2xlarge`: avec un GPU Tesla V100 et un CPU à 8 cœurs, cette instance est suffisamment puissante pour la majeure partie du livre.

![Choose the instance type.](../img/sagemaker-create-2.png)
:width:`400px`
:label:`fig_sagemaker-create-2`

:begin_tab:`mxnet`
Le livre entier au format ipynb pour être exécuté avec SageMaker est disponible sur https://github.com/d2l-ai/d2l-en-sagemaker. Nous pouvons spécifier l'URL de ce dépôt GitHub (:numref:`fig_sagemaker-create-3` ) pour permettre à SageMaker de le cloner lors de la création de l'instance.
:end_tab:

:begin_tab:`pytorch`
Le livre entier au format ipynb pour l'exécution avec SageMaker est disponible à l'adresse https://github.com/d2l-ai/d2l-pytorch-sagemaker. Nous pouvons spécifier l'URL du dépôt GitHub (:numref:`fig_sagemaker-create-3` ) pour permettre à SageMaker de le cloner lors de la création de l'instance.
:end_tab:

:begin_tab:`tensorflow`
Le livre entier au format ipynb pour l'exécution avec SageMaker est disponible à l'adresse https://github.com/d2l-ai/d2l-tensorflow-sagemaker. Nous pouvons spécifier l'URL du dépôt GitHub (:numref:`fig_sagemaker-create-3` ) pour permettre à SageMaker de le cloner lors de la création de l'instance.
:end_tab:

![Specify the GitHub repository.](../img/sagemaker-create-3.png)
:width:`400px`
:label:`fig_sagemaker-create-3`

## Exécution et arrêt d'une instance

La création d'une instance
peut prendre quelques minutes.
Lorsque l'instance est prête,
cliquez sur le lien " Open Jupyter " situé à côté (:numref:`fig_sagemaker-open` ) pour pouvoir
éditer et exécuter tous les notebooks Jupyter
de ce livre sur cette instance
(similaire aux étapes de :numref:`sec_jupyter` ).

![Open Jupyter on the created SageMaker instance.](../img/sagemaker-open.png)
:width:`400px`
:label:`fig_sagemaker-open`


Après avoir terminé votre travail,
n'oubliez pas d'arrêter l'instance pour éviter que 
ne soit facturé davantage (:numref:`fig_sagemaker-stop` ).

![Stop a SageMaker instance.](../img/sagemaker-stop.png)
:width:`300px`
:label:`fig_sagemaker-stop`

## Mise à jour des notebooks

:begin_tab:`mxnet` 
Les notebooks de ce livre open-source seront régulièrement mis à jour dans le dépôt [d2l-ai/d2l-en-sagemaker](https://github.com/d2l-ai/d2l-en-sagemaker)
sur GitHub.
Pour mettre à jour la dernière version,
vous pouvez ouvrir un terminal sur l'instance SageMaker (:numref:`fig_sagemaker-terminal` ).
:end_tab:

:begin_tab:`pytorch`
Les notebooks de ce livre open-source seront régulièrement mis à jour dans le dépôt [d2l-ai/d2l-pytorch-sagemaker](https://github.com/d2l-ai/d2l-pytorch-sagemaker)
sur GitHub.
Pour mettre à jour la dernière version,
vous pouvez ouvrir un terminal sur l'instance SageMaker (:numref:`fig_sagemaker-terminal` ).
:end_tab:


:begin_tab:`tensorflow`
Les notebooks de ce livre open-source seront régulièrement mis à jour dans le dépôt [d2l-ai/d2l-tensorflow-sagemaker](https://github.com/d2l-ai/d2l-tensorflow-sagemaker)
sur GitHub.
Pour mettre à jour la dernière version,
vous pouvez ouvrir un terminal sur l'instance SageMaker (:numref:`fig_sagemaker-terminal` ).
:end_tab:


![Open a terminal on the SageMaker instance.](../img/sagemaker-terminal.png)
:width:`300px`
:label:`fig_sagemaker-terminal`

Vous pouvez valider vos modifications locales avant de récupérer les mises à jour du dépôt distant. 
Sinon, il suffit de supprimer toutes vos modifications locales
avec les commandes suivantes dans le terminal:

:begin_tab :`mxnet`

```bash
cd SageMaker/d2l-en-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`pytorch`

```bash
cd SageMaker/d2l-pytorch-sagemaker/
git reset --hard
git pull
```


:end_tab:

:begin_tab:`tensorflow`

```bash
cd SageMaker/d2l-tensorflow-sagemaker/
git reset --hard
git pull
```


:end_tab:

## Résumé

* Nous pouvons créer une instance de notebook en utilisant Amazon SageMaker pour exécuter le code intensif en GPU de ce livre.
* Nous pouvons mettre à jour les notebooks via le terminal sur l'instance Amazon SageMaker.


## Exercices


 1. Modifiez et exécutez toute section qui nécessite un GPU en utilisant Amazon SageMaker.
1. Ouvrez un terminal pour accéder au répertoire local qui héberge tous les notebooks de ce livre.


[Discussions](https://discuss.d2l.ai/t/422)
