# ClimaCity Paris

## Analyse de données de mobilite urbaine avec Apache Spark et PySpark

### Licence Professionnelle / Master -- Module "Traitement de donnees massives"

Projet individuel ou en binome -- 3 jours de formation

---

## Sommaire

1. [Contexte du projet](#1-contexte-du-projet)
   - [Presentation generale](#11-presentation-generale)
   - [Le pitch narratif](#12-le-pitch-narratif)
2. [Objectifs pedagogiques](#2-objectifs-pedagogiques)
   - [Perimetre fonctionnel](#21-perimetre-fonctionnel)
   - [Jalons par journee](#22-jalons-par-journee)
3. [Architecture des donnees](#3-architecture-des-donnees)
   - [Sources de donnees](#31-sources-de-donnees)
   - [Schema cible](#32-schema-cible)
4. [Contraintes du projet](#4-contraintes-du-projet)
5. [Etapes suggerees pour la resolution](#5-etapes-suggerees-pour-la-resolution)
   - [Etape 1 -- Exploration batch avec RDD et DataFrame](#etape-1----exploration-batch-avec-rdd-et-dataframe-jour-1)
   - [Etape 2 -- Analyse SQL et traitement en flux](#etape-2----analyse-sql-et-traitement-en-flux-jour-2)
   - [Etape 3 -- Machine learning et optimisation](#etape-3----machine-learning-et-optimisation-jour-3)
6. [Rendu final](#6-rendu-final)
7. [Criteres d'evaluation](#7-criteres-devaluation)
8. [Conseils et bonnes pratiques](#8-conseils-et-bonnes-pratiques)
9. [Ressources et sources de donnees](#9-ressources-et-sources-de-donnees)
10. [Bibliographie](#10-bibliographie)

---

## 1. Contexte du projet

### 1.1 Presentation generale

Ce projet constitue le fil rouge d'une formation de trois jours a Apache Spark et PySpark. Il mobilise l'ensemble des competences introduites pendant la formation, depuis la manipulation des RDD jusqu'au deploiement de modeles predictifs avec MLlib, en passant par Spark SQL et le traitement de flux avec Structured Streaming.

Le projet prend appui sur des donnees reelles et librement accessibles portant sur le systeme Velib' Metropole et sur les observations meteorologiques de la station Paris-Montsouris. Il ne requiert aucune infrastructure distribuee : tout s'execute sur une machine locale, en mode Spark standalone, dans un environnement Jupyter Lab.

L'objectif n'est pas de produire un systeme de production finalisé, mais de parcourir de facon coherente et motivee l'ensemble du "spectre Spark", en s'appuyant sur un cas d'usage suffisamment riche pour justifier chacune des briques technologiques abordees.

### 1.2 Le pitch narratif

Paris exploite aujourd'hui l'un des plus grands reseaux de velos en libre-service au monde. Velib' Metropole compte environ 1 400 stations reparties sur 55 communes de la Metropole du Grand Paris, pour une superficie desservie d'environ 400 km². En 2024, le service enregistrait pres de 50 millions de trajets annuels. Ce reseau repose sur une infrastructure de capteurs permanents qui remontent, chaque minute, l'etat de chaque station : nombre de velos mecaniques disponibles, nombre de velos electriques disponibles, nombre de bornettes libres. Ces donnees sont publiees en temps reel sous licence ouverte, conformement a la reglementation europeenne sur les donnees de mobilite.

Cet afflux continu de donnees, croisant mobilite urbaine, conditions meteorologiques et qualite de l'air, represente exactement le type de probleme que les architectures de traitement de donnees massives ont ete concues pour resoudre. La question est simple en apparence : peut-on anticiper, pour les prochaines heures, quelles stations seront en situation de tension -- trop vides pour emprunter un velo, trop pleines pour en deposer un ? Mais sa resolution reclame une chaine de traitement complete, de l'ingestion brute a la prediction, en passant par la mise en qualite, l'analyse exploratoire, le requetage SQL et l'apprentissage automatique.

C'est precisement ce a quoi vous etes conviés pendant ces trois jours. Votre equipe joue le role d'un binome de data engineers recrutes par une startup de mobilite parisienne. Votre mission est de construire, brique par brique, une plateforme de donnees capable de :

- ingerer et consolider plusieurs annees d'historique de disponibilite des stations,
- croiser ces donnees avec des observations meteorologiques horaires,
- repondre a des questions analytiques precisees par l'equipe metier,
- traiter un flux simule de mises a jour en temps quasi-reel,
- predire la disponibilite future d'une station en fonction de son historique et des conditions exterieures.

Chaque demi-journee debloque une nouvelle capacite de la plateforme et introduit un nouveau module de Spark. Le soir du troisieme jour, vous disposerez d'un notebook Jupyter reproductible couvrant l'ensemble de la chaine.

---

## 2. Objectifs pedagogiques

### 2.1 Perimetre fonctionnel

A l'issue des trois jours, chaque etudiant ou binome devra avoir realise les elements suivants :

- **Charger et manipuler des donnees avec les RDD.** Lire un fichier CSV compresse, appliquer des transformations elementaires (`map`, `filter`, `flatMap`, `reduceByKey`), comprendre la semantique de l'evaluation paresseuse (lazy evaluation) et la distinction entre transformations et actions.

- **Construire et enrichir des DataFrames.** Creer un DataFrame depuis un RDD, inferer ou specifier explicitement un schema, appliquer des operations de nettoyage (valeurs nulles, outliers, mauvais types), joindre le jeu de donnees Velib' avec la table meteorologique horaire.

- **Utiliser Spark SQL pour l'analyse exploratoire.** Enregistrer des vues temporaires, rediger des requetes SQL exploitant les fonctions de fenetrage (`WINDOW OVER`), identifier les stations les plus tendues par tranche horaire et par condition meteorologique.

- **Ecrire et lire en format Delta Lake.** Persistre les donnees transformees au format Delta, effectuer une requete de type time-travel, realiser une operation `MERGE INTO` pour simuler une mise a jour incrementale.

- **Mettre en oeuvre Structured Streaming.** Connecter Spark a un flux simule de mises a jour de stations (source fichier ou micro-service fourni), calculer des agregations sur des fenetres glissantes, ecrire le resultat en mode append dans une table Delta, et declencher une alerte sur depassement de seuil.

- **Entrainer un modele predictif avec MLlib.** Construire un Pipeline Spark (`VectorAssembler`, `StandardScaler`, estimateur), entrainer un modele de regression gradient-boosted (`GBTRegressor`) pour predire le taux d'occupation d'une station, evaluer le modele avec validation croisee, sauvegarder et recharger le Pipeline.

- **Appliquer un algorithme de clustering.** Regrouper les stations par profil d'usage avec K-Means, choisir le nombre de clusters par la methode du coude, visualiser le resultat sur une carte avec Folium.

- **Identifier et corriger des goulets d'etranglement.** Lire le DAG et les statistiques de stages dans le Spark UI, comprendre l'impact d'un shuffle, remedier a un data skew par repartitionnement, mesurer le gain du `.cache()` sur un calcul iteratif.

- **Suivre des experiences avec MLflow.** Logger les hyperparametres et les metriques de chaque run, comparer les runs dans l'interface MLflow, charger le meilleur modele enregistre.

### 2.2 Jalons par journee

| Journee | Matin | Apres-midi |
|---------|-------|------------|
| Jour 1 | RDD : chargement, transformations, actions, Spark UI (niveau 1) | DataFrame : nettoyage, jointure meteo, persistance, `.cache()` |
| Jour 2 | Spark SQL : fenetrage, agregations, Delta Lake (ecriture, time-travel, MERGE) | Structured Streaming : flux simule, fenetres glissantes, alertes, sink Delta |
| Jour 3 | MLlib : Pipeline, GBTRegressor, CrossValidator, K-Means, Folium | MLflow, Spark UI (niveau 2), optimisation, bilan d'architecture |

---

## 3. Architecture des donnees

### 3.1 Sources de donnees

Trois sources de donnees sont utilisees dans le projet. Elles sont toutes libres et accessibles sans authentification prealable.

**Source 1 -- Historique de disponibilite Velib'**

Le depot GitHub `lovasoa/historique-velib-opendata` collecte automatiquement, toutes les quinze minutes, la disponibilite de l'ensemble des stations Velib' depuis decembre 2019. Les donnees sont disponibles sous forme d'archives compressees au format CSV. Un snapshot typique contient, pour chaque station et chaque horodatage, le nombre de velos mecaniques disponibles, le nombre de velos electriques disponibles et le nombre de bornettes libres.

Pour les besoins du cours, un extrait pre-prepare couvrant les annees 2022 et 2023 (environ 12 millions de lignes) est distribue aux etudiants au format Parquet partitione par mois. Ce volume est suffisant pour que les differences de performance entre Pandas et Spark soient perceptibles sur une machine de classe courante.

**Source 2 -- Observations meteorologiques Paris-Montsouris**

Meteo-France publie sur `data.gouv.fr` l'ensemble de ses observations de surface (reseau SYNOP) depuis 1996, sous Licence Ouverte, au format CSV toutes les trois heures. La station Paris-Montsouris (identifiant 75114001) fournit temperature, humidite relative, precipitation, vitesse du vent et etat du ciel.

En complement, l'API Open-Meteo permet de telecharger des donnees horaires reconstructes pour Paris depuis 1940, sans cle ni inscription. Un script de collecte est fourni dans le depot du cours.

**Source 3 -- Flux temps reel Velib' (GBFS)**

L'API GBFS de Velib' Metropole expose, sans authentification, la disponibilite actuelle de toutes les stations, mise a jour chaque minute. Cette source alimente la partie Structured Streaming du Jour 2. Un micro-service de simulation est fourni aux etudiants : il rejoue en accelere les donnees historiques et ecrit des fichiers JSON dans un repertoire surveille par Spark, ce qui evite toute dependance a une connexion Internet pendant la seance.

### 3.2 Schema cible

Le schema consolide produit a l'issue du Jour 1 sert de socle pour l'ensemble des jours suivants.

**Table `disponibilite`**

| Colonne | Type | Description |
|---------|------|-------------|
| `station_id` | `IntegerType` | Identifiant unique de la station |
| `nom_station` | `StringType` | Denomination commerciale |
| `arrondissement` | `IntegerType` | Arrondissement ou code commune |
| `latitude` | `DoubleType` | Coordonnee geographique |
| `longitude` | `DoubleType` | Coordonnee geographique |
| `horodatage` | `TimestampType` | Date et heure de l'observation (UTC) |
| `velos_mecaniques` | `IntegerType` | Nombre de velos mecaniques disponibles |
| `velos_electriques` | `IntegerType` | Nombre de velos electriques disponibles |
| `bornettes_libres` | `IntegerType` | Nombre d'emplacements libres |
| `capacite_totale` | `IntegerType` | Capacite nominale de la station |
| `taux_occupation` | `DoubleType` | `(capacite - bornettes_libres) / capacite` |

**Table `meteo_horaire`**

| Colonne | Type | Description |
|---------|------|-------------|
| `horodatage` | `TimestampType` | Heure de l'observation (UTC) |
| `temperature_c` | `DoubleType` | Temperature en degres Celsius |
| `humidite_pct` | `DoubleType` | Humidite relative (%) |
| `precipitation_mm` | `DoubleType` | Precipitation sur l'heure (mm) |
| `vitesse_vent_kmh` | `DoubleType` | Vitesse du vent (km/h) |
| `est_pluie` | `BooleanType` | `precipitation_mm > 0.5` |

---

## 4. Contraintes du projet

Les contraintes suivantes s'appliquent a toutes les equipes. Leur non-respect est pris en compte dans l'evaluation.

**1. Execution en mode local**

Tout le code doit s'executer sans modification sur une machine unique, sans cluster. La session Spark est initialisee avec `SparkSession.builder.master("local[*]")`. Aucune dependance a un service exterieur (base de donnees, broker Kafka en production, API tierce en direct) n'est autorisee pendant la seance.

**2. Reproductibilite**

Le notebook final doit s'executer de la premiere a la derniere cellule sans erreur, sur une machine disposant de l'environnement conda fourni. Les chemins vers les donnees doivent utiliser des variables de configuration declarees en debut de notebook et non des chemins absolus en dur.

**3. Coherence de la chaine**

Le projet doit couvrir l'integralite de la chaine : ingestion brute -> nettoyage -> SQL -> streaming -> ML -> evaluation. Un rendu qui couvrirait uniquement certains modules sans les articuler entre eux ne satisfera pas les criteres de coherence.

**4. Usage explicite des API Spark**

Chaque operation significative doit utiliser l'API Spark native (RDD, DataFrame, Spark SQL, MLlib). Le recours a Pandas est tolere uniquement pour la visualisation finale (Folium, Plotly) et pour des operations sur des sous-ensembles de moins de 10 000 lignes collectes via `.toPandas()`. Tout traitement massif effectue hors de Spark sera signale.

**5. Documentation du code**

Chaque cellule de notebook doit etre precedee d'une cellule Markdown expliquant brievement l'intention de l'operation. Les fonctions Python utilitaires (parsing, nettoyage, generation de features) doivent comporter une docstring au format Google (`Args`, `Returns`, `Example`).

**6. Gestion des performances**

Le notebook doit inclure, pour au moins deux operations critiques (une jointure et un calcul iteratif), une mesure comparative avec et sans `.cache()` ou `.persist()`, commentee a partir des informations lues dans le Spark UI.

---

## 5. Etapes suggerees pour la resolution

Les etapes ci-dessous constituent une demarche recommandee, non un cahier des charges rigide. Les equipes sont libres d'adapter leur approche, sous reserve de respecter les contraintes et d'atteindre les objectifs de chaque journee.

### Etape 1 -- Exploration batch avec RDD et DataFrame (Jour 1)

**Matin -- API RDD**

- Initialiser une `SparkSession` et un `SparkContext`. Observer l'interface web du Spark UI.
- Charger un fichier CSV compresse directement via `sc.textFile()`. Analyser le schema brut.
- Appliquer une chaine de transformations pour compter, par station, le nombre de snapshots disponibles sur l'annee 2022, en utilisant uniquement des RDD (`map`, `filter`, `reduceByKey`).
- Comparer le temps d'execution avec un equivalent Pandas sur le meme fichier. Commenter la difference (overhead JVM, parallelisme, volume).
- Lire le DAG produit dans le Spark UI. Identifier les stages et les shuffles.

**Apres-midi -- API DataFrame**

- Creer un DataFrame depuis les RDD produits le matin, puis depuis les fichiers Parquet distribues. Comparer les schemas inferes.
- Nettoyer les donnees : traitement des valeurs nulles, correction des outliers (stations avec une capacite nulle ou negative), cast des types, calcul de `taux_occupation`.
- Charger la table meteorologique. La mettre en cache (`persist(StorageLevel.MEMORY_AND_DISK)`). Realiser la jointure temporelle avec la table de disponibilite (jointure sur l'heure tronquee a la quinzaine de minutes la plus proche).
- Ecrire le DataFrame consolide en format Parquet partitionne par annee et par mois.
- Mesurer le gain du cache sur la jointure repete en fin d'apres-midi.

### Etape 2 -- Analyse SQL et traitement en flux (Jour 2)

**Matin -- Spark SQL et Delta Lake**

- Enregistrer les tables `disponibilite` et `meteo_horaire` comme vues temporaires. Rediger des requetes SQL ad hoc pour repondre aux questions metier suivantes :
  - Quelles sont les dix stations les plus souvent en rupture (zero velo disponible) entre 8 h et 10 h, les jours de semaine, hors jours feries ?
  - La pluie reduit-elle statistiquement le nombre de deplacements (indicateur de proxy : baisse du taux d'occupation) ? De combien de points en moyenne ?
  - Quelle est la station dont la disponibilite presente la plus forte saisonnalite intra-journaliere ?
- Utiliser des fonctions de fenetrage (`LAG`, `LEAD`, moyenne mobile sur 3 heures) pour construire des features temporelles.
- Ecrire les resultats en format Delta Lake. Effectuer une requete time-travel (`VERSION AS OF 0`). Simuler une correction de donnees avec `MERGE INTO`.

**Apres-midi -- Structured Streaming**

- Lancer le micro-service de simulation fourni. Observer les fichiers JSON produits dans le repertoire de sortie.
- Creer une source de streaming (`readStream`) sur ce repertoire, avec un schema explicite.
- Calculer le nombre de velos disponibles par arrondissement sur une fenetre glissante de 10 minutes, avancant toutes les 2 minutes. Ecrire le resultat en mode append dans une table Delta.
- Ajouter un filtre d'alerte : lorsqu'une station voit son nombre de velos disponibles tomber a zero pour deux fenetres consecutives, ecrire un evenement dans une table `alertes`.
- Tester la robustesse du pipeline a la donnee tardive (late data) en introduisant artificiellement un retard dans le simulateur.

### Etape 3 -- Machine learning et optimisation (Jour 3)

**Matin -- MLlib**

- Construire un jeu de features a partir des donnees consolidees : heure de la journee, jour de la semaine, indicateur week-end, indicateur pluie, temperature, taux d'occupation a `t-1` et `t-4` (features retardees).
- Assembler un Pipeline Spark : `VectorAssembler` -> `StandardScaler` -> `GBTRegressor` (variable cible : `taux_occupation` a l'heure suivante).
- Entrainer le pipeline sur les donnees 2022. Evaluer sur les donnees 2023. Calculer le RMSE et le R².
- Optimiser les hyperparametres (`maxDepth`, `numTrees`) par `CrossValidator` avec 3 folds. Logger chaque run dans MLflow.
- Sauvegarder le meilleur pipeline avec `model.save()`. Le recharger et l'appliquer sur un nouveau batch.
- Appliquer K-Means (k de 3 a 8) sur un vecteur de features de station (profil moyen d'occupation par heure de la journee). Choisir k par la methode du coude (inertia). Visualiser les clusters sur une carte Folium coloree par cluster.

**Apres-midi -- Optimisation et bilan**

- Identifier dans le Spark UI un stage lent du Jour 2. Analyser les causes (skew, absence de partitionnement, broadcast manque).
- Comparer une jointure standard avec une broadcast join sur la table meteorologique. Mesurer le gain en temps et en shuffles.
- Reconfigurer le nombre de partitions pour les calculs iteratifs du clustering. Observer l'effet sur le Spark UI.
- Bilan d'architecture : quand Spark vaut-il le cout de son overhead ? Illustrer avec les mesures prises pendant les trois jours.
- Finaliser et nettoyer le notebook. Verifier la reproductibilite bout en bout.

---

## 6. Rendu final

Le rendu se compose d'un element unique, depose sur la plateforme du cours avant la date limite communiquee par l'equipe pedagogique.

**Notebook Jupyter reproductible**

Le notebook est organise en sections correspondant aux trois journees. Il doit s'executer de la premiere a la derniere cellule sans intervention manuelle, a condition de disposer de l'environnement conda fourni et du repertoire de donnees a l'emplacement configure en debut de notebook.

Le notebook doit imperativement contenir :

- Une cellule de configuration initiale declarant les chemins, les parametres Spark et les seeds aleatoires.
- Des cellules Markdown de contexte avant chaque bloc de code significatif.
- Les mesures de performance explicitement mentionnees dans les contraintes (comparaisons avec/sans cache, analyse du Spark UI).
- Les visualisations Folium et Plotly (ou equivalents) pour les resultats du clustering et des predictions.
- La capture d'ecran ou l'export HTML de l'interface MLflow montrant les runs et le modele champion.

La structure recommandee du notebook est la suivante :

```
00_configuration.ipynb     -- Parametres globaux, chemins, verification de l'environnement
01_rdd_exploration.ipynb   -- Jour 1 matin
02_dataframe_pipeline.ipynb -- Jour 1 apres-midi
03_spark_sql_delta.ipynb   -- Jour 2 matin
04_streaming.ipynb         -- Jour 2 apres-midi
05_mllib_prediction.ipynb  -- Jour 3 matin
06_optimisation_bilan.ipynb -- Jour 3 apres-midi
```

Les notebooks peuvent etre soumis comme un seul fichier fusionne ou comme un repertoire archive (`.zip`).

---

## 7. Criteres d'evaluation

L'evaluation porte sur cinq dimensions. La note finale est la moyenne ponderee des cinq notes.

| Dimension | Ce qui est evalue | Poids |
|-----------|-------------------|-------|
| Couverture technique | Presence et correction des six modules Spark (RDD, DataFrame, SQL, Delta, Streaming, MLlib). Chaque module manquant ou non fonctionnel penalise. | 35 % |
| Qualite du code et documentation | Clarte des cellules Markdown, presence des docstrings, absence de chemins en dur, configuration centralisee, lisibilite generale. | 20 % |
| Pertinence des analyses | Les requetes SQL repondent effectivement aux questions posees. Les features du modele ML sont justifiees. Les resultats sont commentes et interpretes. | 20 % |
| Performances et optimisation | Mesures documentees, analyse du Spark UI, au moins une optimisation demonstree et chiffree. | 15 % |
| Reproductibilite | Le notebook s'execute sans erreur sur la machine de l'evaluateur avec l'environnement fourni. | 10 % |

**Points de bonus** (jusqu'a +2 points sur la note finale) :

- Integration d'une source de qualite de l'air (OpenAQ ou Atmo Ile-de-France) comme feature supplementaire dans le modele ML (+1 point).
- Mise en place d'un modele de serie temporelle alternatif avec MLlib (`FMRegressor` ou features ARIMA manuelles) et comparaison chiffree avec le GBTRegressor (+1 point).

---

## 8. Conseils et bonnes pratiques

**Organisation du travail**

- Travailler en binome avec une separation claire des roles : un membre pilote le clavier (driver), l'autre relit, questionne et documente. Alterner les roles a chaque demi-journee.
- Ne pas attendre la fin de chaque journee pour tester le notebook bout en bout. Un notebook qui ne s'execute pas lineairement a 17 h est une source de stress evitable.
- Commenter les anomalies rencontrees dans les donnees (valeurs aberrantes, discontinuites) : elles font partie du travail et seront appréciees a l'evaluation.

**Bonnes pratiques techniques**

- Fixer le seed Spark en debut de session (`spark.sparkContext.setCheckpointDir(...)`, `seed=42` dans les appels MLlib) pour garantir la reproductibilite des resultats ML.
- Partitionner les lectures des fichiers Parquet explicitement (`repartition(8)`) plutot que de laisser Spark choisir. Observer l'effet sur le nombre de taches dans le Spark UI.
- Ne pas appeler `.collect()` ou `.toPandas()` sur un DataFrame de plusieurs millions de lignes. Toujours filtrer ou echantillonner avant de ramener des donnees en local.
- Stopper la `SparkSession` proprement en fin de chaque notebook (`spark.stop()`), surtout sur les machines avec peu de RAM.
- Utiliser `spark.sql.shuffle.partitions` (par defaut 200) et l'ajuster a la baisse (par exemple 8 ou 16) pour les volumes du projet, afin d'eviter des centaines de taches inutiles lors des shuffles.

**Pieges a eviter**

- Ne pas confondre une transformation (qui ne declenche rien) avec une action (qui declenche le calcul). Lire le DAG dans le Spark UI apres chaque action pour comprendre ce qui s'est reellement passe.
- Ne pas appliquer `.cache()` systematiquement : mettre en cache un DataFrame non reutilise consomme de la memoire inutilement et peut degrader les performances globales.
- Ne pas ignorer le Spark UI. Il est la principale source d'information pour comprendre ce que Spark fait reellement et pourquoi un calcul est lent.
- En Structured Streaming, ne pas modifier le schema d'une source apres le demarrage du flux sans reinitialiser le checkpoint. Spark leve une exception et cela peut etre destabilisant en seance.

---

## 9. Ressources et sources de donnees

**Donnees Velib' et mobilite**

- Velib' Metropole -- API GBFS (temps reel, sans authentification) : `https://www.velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole`
- Historique de disponibilite Velib' (mises a jour toutes les 15 min depuis decembre 2019) : `https://github.com/lovasoa/historique-velib-opendata`
- Localisation des stations Velib' (Paris Open Data) : `https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/`
- Donnees de mobilite Ile-de-France : `https://data.iledefrance-mobilites.fr/`
- Point d'Acces National aux donnees de transport (PAN) : `https://transport.data.gouv.fr/`

**Donnees meteorologiques**

- Open-Meteo -- API historique gratuite, sans cle, granularite horaire depuis 1940 : `https://open-meteo.com/`
- Observations SYNOP Meteo-France (data.gouv.fr) -- archives depuis 1996, toutes les 3 heures, format CSV : `https://www.data.gouv.fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques/`
- Portail Open Data Meteo-France : `https://meteo.data.gouv.fr/`
- Infoclimat -- donnees historiques de stations (usage non commercial, compte requis) : `https://www.infoclimat.fr/opendata/`

**Qualite de l'air (bonus)**

- OpenAQ -- donnees mondiales de qualite de l'air en temps reel et historique, API gratuite : `https://openaq.org/`
- Atmo Ile-de-France -- donnees de qualite de l'air pour la region parisienne : `https://data-airparif.iledefrance.fr/`

**Documentation Spark et outils**

- Documentation officielle Apache Spark 3.x : `https://spark.apache.org/docs/latest/`
- Guide PySpark API Reference : `https://spark.apache.org/docs/latest/api/python/`
- Documentation Delta Lake : `https://docs.delta.io/latest/`
- Documentation MLflow : `https://mlflow.org/docs/latest/`
- Documentation Folium (visualisation cartographique) : `https://python-visualization.github.io/folium/`
- Open-Meteo Python SDK : `https://pypi.org/project/openmeteo-requests/`

**Environnement technique**

L'environnement conda requis est distribue avec le depot du cours. Il contient notamment :

- `pyspark >= 3.5`
- `delta-spark` (version compatible avec la version de PySpark)
- `mlflow`
- `folium`
- `plotly`
- `openmeteo-requests`
- `jupyterlab`

---

## 10. Bibliographie

**Apache Spark -- Fondements et architecture**

Zaharia, M., Chowdhury, M., Das, T., Dave, A., Ma, J., McCauly, M., Franklin, M. J., Shenker, S., & Stoica, I. (2012). Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing. *Proceedings of the 9th USENIX Symposium on Networked Systems Design and Implementation (NSDI '12)*, 15-28.

Zaharia, M., Xin, R. S., Wendell, P., Das, T., Armbrust, M., Dave, A., Meng, X., Rosen, J., Venkataraman, S., Franklin, M. J., Ghodsi, A., Gonzalez, J., Shenker, S., & Stoica, I. (2016). Apache Spark: A Unified Engine for Big Data Processing. *Communications of the ACM*, 59(11), 56-65. `https://doi.org/10.1145/2934664`

Armbrust, M., Das, T., Torres, J., Yavuz, B., Zeng, S., Xin, R., Ghodsi, A., Stoica, I., & Zaharia, M. (2018). Structured Streaming: A Declarative API for Real-Time Applications in Apache Spark. *Proceedings of the 2018 International Conference on Management of Data (SIGMOD '18)*. `https://doi.org/10.1145/3183713.3190664`

**Delta Lake et architectures lakehouse**

Armbrust, M., Ghodsi, A., Xin, R., & Zaharia, M. (2021). Lakehouse: A New Generation of Open Platforms that Unify Data Warehousing and Advanced Analytics. *Proceedings of the 11th Conference on Innovative Data Systems Research (CIDR '21)*.

**MLlib et apprentissage automatique distribue**

Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., Freeman, J., Tsai, D. B., Amde, M., Owen, S., Xin, D., Xin, R., Franklin, M. J., Zadeh, R., Zaharia, M., & Smola, A. (2016). MLlib: Machine Learning in Apache Spark. *Journal of Machine Learning Research*, 17(34), 1-7. `https://jmlr.org/papers/v17/15-237.html`

**Mobilite urbaine et donnees de velos en libre-service**

Borgnat, P., Abry, P., Flandrin, P., Robardet, C., Rouquier, J.-B., & Fleury, E. (2011). Shared Bicycles in a City: A Signal Processing and Data Analysis Perspective. *Advances in Complex Systems*, 14(3), 415-438. `https://doi.org/10.1142/S0219525911002950`

Caulfield, B., O'Mahony, M., Brasil, W., & Pallot, P. (2017). Understanding How Cyclists Plan Their Routes, An Analysis of Revealed Preference GPS Data. *Transportation Research Part A: Policy and Practice*, 101, 174-185. `https://doi.org/10.1016/j.tra.2017.05.008`

Faghih-Imani, A., & Eluru, N. (2016). Incorporating the Impact of Spatio-Temporal Interactions on Bicycle Sharing System Demand: A Case Study of New York Citibike System. *Journal of Transport Geography*, 54, 218-227. `https://doi.org/10.1016/j.jtrangeo.2016.06.008`

**Series temporelles et prediction de demande**

Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*. `https://doi.org/10.1145/2939672.2939785`

Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *The Annals of Statistics*, 29(5), 1189-1232.

**Ouvrages de reference**

Chambers, B., & Zaharia, M. (2018). *Spark: The Definitive Guide. Big Data Processing Made Simple*. O'Reilly Media.

Perrin, J. (2020). *PySpark in Action*. Manning Publications.

Damji, J. S., Wenig, B., Das, T., & Lee, D. (2020). *Learning Spark: Lightning-Fast Data Analytics* (2nd ed.). O'Reilly Media.

**Donnees ouvertes et reglementation**

Syndicat Autolib' Velib' Metropole (2024). *Velib' Metropole -- Donnees Open Data GBFS*. Publie sous licence Open Database License (ODbL). `https://www.velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole`

Meteo-France (2024). *Donnees d'observation des principales stations meteorologiques*. Publie sous Licence Ouverte / Open Licence v2.0 (Etalab). `https://www.data.gouv.fr/datasets/donnees-d-observation-des-principales-stations-meteorologiques/`

Zippenbom, P. (2019-2025). *historique-velib-opendata -- Historique des donnees d'occupation de stations Velib'*. Depot GitHub, licence ODbL. `https://github.com/lovasoa/historique-velib-opendata`

Zippenbom, P. (2022). *API Open-Meteo -- Historical Weather Data*. Licence CC BY 4.0. `https://open-meteo.com/`
