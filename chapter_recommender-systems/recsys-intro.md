# Aperçu des systèmes de recommandation



 Au cours de la dernière décennie, Internet s'est transformé en une plate-forme de services en ligne à grande échelle, ce qui a profondément changé notre façon de communiquer, de lire les nouvelles, d'acheter des produits et de regarder des films.  Dans le même temps, le nombre sans précédent d'articles (nous utilisons le terme *article* pour désigner les films, les nouvelles, les livres et les produits) proposés en ligne nécessite un système capable de nous aider à découvrir les articles que nous préférons. Les systèmes de recommandation sont donc de puissants outils de filtrage de l'information qui peuvent faciliter les services personnalisés et offrir une expérience sur mesure aux utilisateurs individuels. En bref, les systèmes de recommandation jouent un rôle central dans l'utilisation de la richesse des données disponibles pour rendre les choix plus faciles à gérer. De nos jours, les systèmes de recommandation sont au cœur de plusieurs fournisseurs de services en ligne tels qu'Amazon, Netflix et YouTube. Rappelez-vous l'exemple des livres d'apprentissage profond recommandés par Amazon sur :numref:`subsec_recommender_systems` . Les avantages de l'utilisation des systèmes de recommandation sont doubles : D'une part, ils peuvent réduire considérablement les efforts des utilisateurs pour trouver des articles et atténuer le problème de la surcharge d'informations. D'autre part, il peut ajouter de la valeur commerciale aux fournisseurs de services en ligne
et constitue une importante source de revenus.  Ce chapitre présente les concepts fondamentaux, les modèles classiques et les avancées récentes de l'apprentissage profond dans le domaine des systèmes de recommandation, ainsi que des exemples mis en œuvre.

![Illustration of the Recommendation Process](../img/rec-intro.svg)


## Filtrage collaboratif

Nous commençons notre voyage par le concept important des systèmes de recommandation : le filtrage collaboratif
(CF), qui a été inventé par le système Tapestry :cite:`Goldberg.Nichols.Oki.ea.1992` , faisant référence à " des personnes qui collaborent pour s'aider mutuellement à effectuer le processus de filtrage afin de traiter les grandes quantités d'e-mails et de messages postés dans les groupes de discussion ". Ce terme a été enrichi de plusieurs sens. Dans un sens large, il s'agit du processus de filtrage
pour trouver des informations ou des modèles à l'aide de techniques impliquant la collaboration entre plusieurs utilisateurs, agents et sources de données. La FC a de nombreuses formes et de nombreuses méthodes de FC ont été proposées depuis son avènement. 

Dans l'ensemble, les techniques de CF peuvent être classées en trois catégories : CF basée sur la mémoire, CF basée sur un modèle et leurs hybrides :cite:`Su.Khoshgoftaar.2009` . Les techniques représentatives de la FC basée sur la mémoire sont la FC basée sur le plus proche voisin, comme la FC basée sur l'utilisateur et la FC basée sur l'élément :cite:`Sarwar.Karypis.Konstan.ea.2001` .  Les modèles de facteurs latents tels que la factorisation matricielle sont des exemples de FC basée sur un modèle.  La FC basée sur la mémoire est limitée dans le traitement des données éparses et à grande échelle, car elle calcule les valeurs de similarité sur la base d'éléments communs.  Les méthodes basées sur des modèles deviennent plus populaires grâce à leur
meilleure capacité à traiter la rareté et l'évolutivité.  De nombreuses approches de CF basées sur des modèles peuvent être étendues aux réseaux neuronaux, ce qui permet d'obtenir des modèles plus flexibles et plus évolutifs grâce à l'accélération des calculs dans l'apprentissage profond :cite:`Zhang.Yao.Sun.ea.2019` .  En général, la CF n'utilise que les données d'interaction utilisateur-article pour faire des prédictions et des recommandations. Outre la CF, les systèmes de recommandation basés sur le contenu et le contexte sont également utiles pour intégrer le contenu DeepL des articles/utilisateurs et les signaux contextuels tels que les horodatages et les emplacements.  Évidemment, nous pouvons avoir besoin d'ajuster les types/structures de modèles lorsque différentes données d'entrée sont disponibles.



## Feedback explicite et feedback implicite

Pour connaître les préférences des utilisateurs, le système doit recueillir leurs commentaires.  Le retour d'information peut être explicite ou implicite :cite:`Hu.Koren.Volinsky.2008` . Par exemple, [IMDB](https://www.imdb.com/) recueille des évaluations par étoiles allant de une à dix étoiles pour les films. YouTube fournit les boutons "pouce en haut" et "pouce en bas" pour que les utilisateurs puissent indiquer leurs préférences.  Il est évident que la collecte de commentaires explicites exige que les utilisateurs indiquent leurs intérêts de manière proactive.  Néanmoins, le retour d'information explicite n'est pas toujours facile à obtenir, car de nombreux utilisateurs peuvent être réticents à évaluer des produits. En revanche, le feedback implicite est souvent disponible puisqu'il s'agit principalement de modéliser des comportements implicites tels que les clics des utilisateurs. Ainsi, de nombreux systèmes de recommandation sont centrés sur la rétroaction implicite qui reflète indirectement l'opinion de l'utilisateur en observant son comportement.  Il existe diverses formes de rétroaction implicite, notamment l'historique des achats, l'historique de navigation, les montres et même les mouvements de la souris. Par exemple, un utilisateur qui a acheté de nombreux livres du même auteur aime probablement cet auteur.   Notez que la rétroaction implicite est intrinsèquement bruyante.  Nous ne pouvons que *deviner* leurs préférences et leurs véritables motivations. Un utilisateur qui a regardé un film n'indique pas nécessairement une opinion positive de ce film.



## Tâches de recommandation

Un certain nombre de tâches de recommandation ont été étudiées au cours des dernières décennies.  En fonction du domaine d'application, on trouve la recommandation de films, de nouvelles, de centres d'intérêt, etc. :cite:`Ye.Yin.Lee.ea.2011` .  Il est également possible de différencier les tâches en fonction des types de retour d'information et des données d'entrée. Par exemple, la tâche de prédiction des notes vise à prédire les notes explicites. La recommandation Top-$n$ (classement des articles) classe tous les articles pour chaque utilisateur personnellement sur la base des commentaires implicites. Si les informations d'horodatage sont également incluses, nous pouvons construire une recommandation tenant compte de la séquence :cite:`Quadrana.Cremonesi.Jannach.2018` .  Une autre tâche populaire est la prédiction du taux de clics, qui est également basée sur le feedback implicite, mais diverses caractéristiques catégorielles peuvent être utilisées. La recommandation pour les nouveaux utilisateurs et la recommandation de nouveaux éléments aux utilisateurs existants sont appelées recommandation de démarrage à froid :cite:`Schein.Popescul.Ungar.ea.2002` .



## Résumé

* Les systèmes de recommandation sont importants pour les utilisateurs individuels et les industries. Le filtrage collaboratif est un concept clé de la recommandation.
* Il existe deux types de feedbacks : le feedback implicite et le feedback explicite.  Un certain nombre de tâches de recommandation ont été explorées au cours de la dernière décennie.

## Exercices

1. Pouvez-vous expliquer comment les systèmes de recommandation influencent votre vie quotidienne ?
2. Quelles tâches de recommandation intéressantes peuvent être étudiées, selon vous ?

[Discussions](https://discuss.d2l.ai/t/398)
