# Module 1 — Introduction à Spark et à l'écosystème Big Data

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : notions de base en Python, algorithmique et systèmes distribués

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Expliquer les limites des traitements de données classiques face au Big Data
- Décrire l'architecture interne de Spark et le rôle de ses composants principaux
- Distinguer Spark de Hadoop/MapReduce et justifier les gains de performance
- Identifier les bibliothèques de l'écosystème Spark et leurs cas d'usage
- Comprendre le positionnement de PySpark comme interface Python de Spark

---

## 1. Contexte : le Big Data et ses limites avec les outils classiques

### 1.1 L'explosion des données

Depuis le début des années 2010, le volume de données produites à l'échelle mondiale a connu une croissance exponentielle. Quelques ordres de grandeur pour fixer les idées :

| Source | Volume approximatif |
|---|---|
| Twitter/X | ~500 millions de tweets par jour |
| YouTube | ~500 heures de vidéo uploadées par minute |
| Capteurs IoT | des milliards d'événements par seconde à l'échelle mondiale |
| Transactions bancaires | des dizaines de milliards par jour dans le monde |

Les caractéristiques du Big Data sont classiquement résumées par les **3 V** (auxquels on ajoute souvent un 4e et un 5e) :

- **Volume** : des téraoctets, voire des pétaoctets de données
- **Vitesse** (*Velocity*) : données produites et à traiter en temps quasi-réel
- **Variété** : données structurées (SQL), semi-structurées (JSON, XML), non structurées (texte, image, audio)
- **Véracité** (*Veracity*) : qualité et fiabilité des données incertaines
- **Valeur** (*Value*) : la donnée brute doit être transformée pour créer de la valeur

### 1.2 Pourquoi les outils classiques ne suffisent plus

Un serveur classique dispose de ressources limitées (RAM, CPU). Traiter 10 To de logs sur une seule machine avec Pandas soulève des problèmes immédiats :

- **Mémoire insuffisante** : Pandas charge tout le jeu de données en RAM
- **Temps de traitement prohibitif** : un seul processeur ne peut pas paralléliser
- **Pas de tolérance aux pannes** : si la machine tombe, le calcul est perdu

La solution naturelle consiste à **distribuer** le traitement sur un **cluster** de plusieurs machines. C'est précisément l'objectif des frameworks de calcul distribué.

---

## 2. Histoire et philosophie de Spark

### 2.1 De Hadoop à Spark

**Apache Hadoop** (2006) a été le premier framework open-source populaire de calcul distribué. Il repose sur deux piliers :

- **HDFS** (*Hadoop Distributed File System*) : système de fichiers distribué, redondant et tolérant aux pannes
- **MapReduce** : modèle de programmation distribué en deux étapes — `Map` (transformation locale) et `Reduce` (agrégation globale)

Hadoop MapReduce souffrait cependant d'une limitation majeure : **chaque étape de calcul implique une lecture et une écriture sur disque**. Pour des pipelines comportant de nombreuses étapes (machine learning itératif, par exemple), le coût en I/O devient rédhibitoire.

```
Hadoop MapReduce (pipeline en 3 étapes) :
  Disque → Map → Disque → Map → Disque → Reduce → Disque
              ↑ écriture/lecture à chaque étape ↑
```

**Apache Spark** est né en 2009 dans le laboratoire AMPLab de l'Université de Berkeley, avec une idée centrale : **conserver les données intermédiaires en mémoire** plutôt que de les persister sur disque entre chaque étape.

```
Spark (même pipeline) :
  Disque → [Map → Map → Reduce] → Disque
                  ↑ tout en RAM ↑
```

Spark est devenu un projet Apache en 2013 et est aujourd'hui le framework de référence pour le traitement de données à grande échelle.

### 2.2 Les principes philosophiques de Spark

| Principe | Description |
|---|---|
| **Calcul en mémoire** | Les données intermédiaires sont stockées en RAM, pas sur disque |
| **Lazy evaluation** | Les transformations ne sont pas exécutées immédiatement, mais planifiées |
| **Tolérance aux pannes** | Les données perdues peuvent être recalculées grâce au lignage (*lineage*) |
| **API unifiée** | Un seul framework pour le batch, le streaming, le SQL, le ML et les graphes |
| **Polyglotte** | APIs officielles en Scala, Java, Python et R |

### 2.3 Performances comparées

Spark est généralement **10 à 100 fois plus rapide** que Hadoop MapReduce :

- **10x** sur des traitements sur disque (grâce à l'optimisation des I/O)
- **100x** sur des traitements en mémoire (grâce au cache en RAM)

Ces chiffres proviennent des benchmarks originaux de l'AMPLab et ont été largement confirmés par la communauté.

---

## 3. L'architecture de Spark

Comprendre l'architecture de Spark est indispensable pour écrire du code performant et diagnostiquer les problèmes.

### 3.1 Vue d'ensemble

Un cluster Spark est composé de trois types d'entités :

```
┌─────────────────────────────────────────────────────┐
│                   APPLICATION                       │
│                                                     │
│   ┌──────────────┐         ┌───────────────────┐   │
│   │    DRIVER    │◄───────►│  CLUSTER MANAGER  │   │
│   │  (programme  │         │ (Standalone, YARN,│   │
│   │  principal)  │         │   Mesos, K8s)     │   │
│   └──────┬───────┘         └───────────────────┘   │
│          │                                          │
│    ┌─────┴──────────────────────────────┐           │
│    ▼             ▼              ▼        │           │
│  Worker        Worker         Worker     │           │
│ ┌────────┐   ┌────────┐   ┌────────┐   │           │
│ │Executor│   │Executor│   │Executor│   │           │
│ │ Task   │   │ Task   │   │ Task   │   │           │
│ │ Task   │   │ Task   │   │ Task   │   │           │
│ └────────┘   └────────┘   └────────┘   │           │
└─────────────────────────────────────────────────────┘
```

### 3.2 Le Driver

Le **Driver** est le programme principal de l'application Spark. C'est lui qui :

- Contient la fonction `main()` de l'application
- Crée la `SparkSession` (point d'entrée unique depuis Spark 2.x)
- Construit le **plan d'exécution** (DAG) à partir des transformations décrites
- Coordonne les Executors et collecte les résultats finaux

> ⚠️ **Point clé** : le Driver tourne sur une seule machine. Toute opération qui ramène des données vers le Driver (comme `collect()`) peut saturer sa mémoire si le volume est trop important.

### 3.3 Le Cluster Manager

Le **Cluster Manager** gère les ressources (CPU, RAM) du cluster. Spark supporte plusieurs gestionnaires :

| Gestionnaire | Cas d'usage typique |
|---|---|
| **Standalone** | Clusters Spark dédiés, simple à configurer |
| **YARN** | Environnements Hadoop existants |
| **Apache Mesos** | Multi-framework (en déclin) |
| **Kubernetes** | Déploiements cloud-native modernes |
| **Local** | Développement et tests sur une seule machine |

### 3.4 Les Executors

Les **Executors** sont les processus de calcul qui s'exécutent sur les nœuds *workers* du cluster. Chaque executor :

- Reçoit des **tâches** (*tasks*) à exécuter depuis le Driver
- Stocke les données en mémoire (cache) ou sur disque
- Renvoie les résultats au Driver

Le calcul est découpé en **Jobs → Stages → Tasks** :

- **Job** : déclenché par une action (`count()`, `collect()`, etc.)
- **Stage** : ensemble de tâches sans *shuffle* réseau entre elles
- **Task** : unité de travail minimale, s'applique à une partition de données

### 3.5 Le DAG (*Directed Acyclic Graph*)

Quand vous écrivez des transformations Spark, aucun calcul ne se produit immédiatement. Spark construit un **graphe orienté acyclique (DAG)** représentant l'ensemble des opérations à effectuer. Ce DAG n'est exécuté que lorsqu'une **action** est déclenchée.

```
RDD/DF initial
      │
      ▼ filter()       ← transformation (lazy)
      │
      ▼ map()          ← transformation (lazy)
      │
      ▼ groupBy()      ← transformation (lazy)
      │
      ▼ count()        ← ACTION → déclenche l'exécution du DAG
```

Cette approche permet à Spark d'**optimiser le plan d'exécution** avant de lancer les calculs (réordonnancement des opérations, fusion de filtres, etc.).

---

## 4. L'écosystème Spark

Spark est bien plus qu'un moteur de calcul : c'est une **plateforme unifiée** composée de plusieurs bibliothèques spécialisées.

```
┌─────────────────────────────────────────────────────────┐
│                      SPARK CORE                         │
│         (RDD, scheduling, mémoire, I/O, shuffle)        │
├──────────────┬──────────────┬──────────────┬────────────┤
│  Spark SQL   │    MLlib     │   GraphX     │ Structured │
│  DataFrames  │ (Machine     │  (Graphes)   │ Streaming  │
│  Datasets    │  Learning)   │  GraphFrames │            │
└──────────────┴──────────────┴──────────────┴────────────┘
```

### 4.1 Spark Core

Spark Core est le moteur de base. Il gère :
- Le scheduling des tâches
- La gestion de la mémoire et du stockage
- La tolérance aux pannes
- L'API RDD (*Resilient Distributed Dataset*)

### 4.2 Spark SQL

Spark SQL permet d'interroger des données structurées avec du **SQL standard** ou via l'API DataFrame/Dataset. Il embarque le **Catalyst Optimizer**, un optimiseur de requêtes qui transforme automatiquement les requêtes en plans d'exécution efficaces.

```python
# Requête SQL
spark.sql("SELECT name, age FROM users WHERE age > 25").show()

# API DataFrame équivalente
df.filter(df.age > 25).select("name", "age").show()
```

### 4.3 MLlib

MLlib est la bibliothèque de **machine learning distribué** de Spark. Elle fournit :

- Des algorithmes classiques : régression, classification, clustering, ALS (recommandation)
- Des outils de feature engineering : TF-IDF, Word2Vec, normalisation
- Des **Pipelines ML** : chaînes de transformations reproductibles

### 4.4 GraphX / GraphFrames

GraphX est la bibliothèque de **traitement de graphes** (API Scala). GraphFrames est son équivalent pour PySpark. Ils permettent d'implémenter des algorithmes comme PageRank, la détection de communautés ou le calcul de composantes connexes.

### 4.5 Structured Streaming

Structured Streaming permet de traiter des **flux de données en temps réel** en utilisant la même API que les DataFrames batch. Les sources supportées incluent Kafka, des fichiers, des sockets TCP, etc.

---

## 5. PySpark : Python + Spark

### 5.1 Qu'est-ce que PySpark ?

PySpark est l'interface officielle Python pour Apache Spark. Il permet d'écrire des programmes Spark en Python tout en bénéficiant de la puissance du moteur distribué Spark (écrit en Scala/JVM).

### 5.2 Architecture PySpark

PySpark repose sur **Py4J**, une bibliothèque qui crée un pont entre la JVM (où tourne Spark) et l'interpréteur Python.

```
┌────────────────────────┐      ┌──────────────────────────┐
│   Processus Python     │      │       JVM (Spark)         │
│                        │      │                           │
│  Code PySpark          │      │  SparkContext             │
│  spark.read(...)  ─────┼─────►│  SparkSession             │
│  df.filter(...)        │      │  DataFrame API (Scala)    │
│  df.show()        ◄────┼──────│  Résultats                │
│                        │ Py4J │                           │
└────────────────────────┘      └──────────────────────────┘
```

### 5.3 Pourquoi PySpark pour l'IA ?

| Avantage | Détail |
|---|---|
| **Écosystème Python** | Compatibilité avec NumPy, Pandas, scikit-learn, PyTorch... |
| **Pandas API on Spark** | `pyspark.pandas` : API quasi-identique à Pandas |
| **Pandas UDFs** | Fonctions Python vectorisées exécutées directement sur les Executors |
| **MLlib + Python** | Accès à tous les algorithmes ML distribués depuis Python |
| **Notebooks** | Intégration native avec Jupyter, Databricks, Google Colab |

### 5.4 Premier programme PySpark

```python
from pyspark.sql import SparkSession

# Créer une SparkSession (point d'entrée unique)
spark = SparkSession.builder \
    .appName("MonPremierProgramme") \
    .master("local[*]") \       # local[*] = tous les cœurs disponibles
    .getOrCreate()

# Lire un fichier CSV
df = spark.read.option("header", True).csv("data/utilisateurs.csv")

# Afficher le schéma
df.printSchema()

# Filtrer et afficher
df.filter(df["age"] > 25).select("nom", "age").show()

# Toujours fermer la session en fin de programme
spark.stop()
```

---

## 6. Spark vs les alternatives

| Critère | Spark | Pandas | Dask | Ray |
|---|---|---|---|---|
| Scalabilité | Très haute (PB) | Faible (RAM machine) | Haute | Haute |
| Calcul distribué | Natif | Non | Oui | Oui |
| SQL | Spark SQL | Limité | Limité | Non |
| ML distribué | MLlib | scikit-learn | Non natif | Ray AIR |
| Streaming | Structured Streaming | Non | Limité | Non |
| Courbe d'apprentissage | Élevée | Faible | Moyenne | Moyenne |
| Cas d'usage idéal | Big Data, pipelines ETL, ML à grande échelle | Prototypage, petits datasets | Données > RAM, même API Pandas | ML distribué, RL |

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Big Data** | Volume, Vitesse, Variété — les outils classiques ne suffisent plus |
| **Spark vs Hadoop** | Spark garde les données en RAM → 10-100x plus rapide |
| **Driver** | Chef d'orchestre — coordonne mais ne calcule pas |
| **Executors** | Calculent réellement les données, en parallèle |
| **Lazy evaluation** | Les transformations sont planifiées, pas exécutées immédiatement |
| **DAG** | Plan d'exécution optimisé, déclenché par une action |
| **Écosystème** | Spark SQL, MLlib, GraphX, Structured Streaming |
| **PySpark** | Python + Py4J → accès complet à Spark depuis Python |

---

## Exercices

### Exercice 1 — Réflexion (10 min)
> Une entreprise de e-commerce veut analyser 5 To de logs de clics utilisateurs pour construire un modèle de recommandation. Pourquoi Pandas ne convient-il pas ? Quels composants Spark seraient impliqués dans cette solution ?

### Exercice 2 — Schéma d'architecture (15 min)
> Dessinez le schéma d'un cluster Spark avec 1 Driver, 1 Cluster Manager et 4 Workers (chacun avec 2 Executors). Indiquez le sens des flux d'information lors de l'exécution d'un `count()`.

### Exercice 3 — Mise en pratique (30 min)
> Dans un notebook Jupyter ou Google Colab :
> 1. Installez PySpark (`pip install pyspark`)
> 2. Créez une `SparkSession` en mode local
> 3. Créez un DataFrame à partir d'une liste Python de dictionnaires
> 4. Affichez son schéma, filtrez une colonne et comptez les lignes
> 5. Observez le DAG dans la Spark UI (port 4040)

---

## Pour aller plus loin

- 📖 **Documentation officielle** : [spark.apache.org/docs/latest](https://spark.apache.org/docs/latest/)
- 📖 **Databricks Academy** : cours gratuits PySpark sur [academy.databricks.com](https://academy.databricks.com)
- 📄 **Article fondateur** : *Resilient Distributed Datasets: A Fault-Tolerant Abstraction for In-Memory Cluster Computing* — Zaharia et al., NSDI 2012
- 🎥 **Conférence** : Spark Summit talks (YouTube, chaîne Databricks)

---

*Module suivant → **Module 2 : Installation et prise en main***
