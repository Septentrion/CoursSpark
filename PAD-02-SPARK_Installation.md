# Module 2 — Installation et prise en main

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 1 — Introduction à Spark et à l'écosystème Big Data

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Installer et configurer un environnement PySpark fonctionnel en local
- Utiliser les outils en ligne de commande de Spark (`pyspark`, `spark-submit`, `spark-shell`)
- Créer et configurer une `SparkSession` adaptée à différents contextes
- Naviguer dans la Spark UI pour monitorer et diagnostiquer une application
- Exécuter ses premiers programmes PySpark dans un notebook Jupyter

---

## 1. Prérequis système

Avant d'installer Spark, il faut s'assurer que les dépendances suivantes sont présentes sur la machine.

### 1.1 Java (JVM)

Spark est écrit en Scala et tourne sur la **JVM (Java Virtual Machine)**. Java est donc une dépendance obligatoire, même pour PySpark.

```bash
# Vérifier la version de Java installée
java -version
# Résultat attendu : openjdk version "11.x.x" ou "17.x.x"
```

Spark 3.x supporte officiellement **Java 8, 11 et 17**. Java 11 est recommandé pour sa stabilité.

**Installation sur Linux/macOS (via SDKMAN) :**
```bash
curl -s "https://get.sdkman.io" | bash
sdk install java 11.0.21-tem
```

**Installation sur Windows :**  
Télécharger le JDK depuis [adoptium.net](https://adoptium.net) et définir la variable d'environnement `JAVA_HOME`.

### 1.2 Python

PySpark supporte **Python 3.8 à 3.12**. Il est fortement recommandé d'utiliser un environnement virtuel.

```bash
# Vérifier la version de Python
python --version   # ou python3 --version
# Résultat attendu : Python 3.9.x ou supérieur

# Créer un environnement virtuel dédié (bonne pratique)
python -m venv .venv-spark
source .venv-spark/bin/activate       # Linux/macOS
.venv-spark\Scripts\activate          # Windows
```

### 1.3 Variables d'environnement essentielles

Spark et PySpark s'appuient sur plusieurs variables d'environnement. Elles peuvent être définies dans le fichier de configuration du shell (`~/.bashrc`, `~/.zshrc`) ou dans un fichier `.env`.

| Variable | Rôle | Exemple de valeur |
|---|---|---|
| `JAVA_HOME` | Chemin vers le JDK | `/usr/lib/jvm/java-11-openjdk` |
| `SPARK_HOME` | Chemin vers Spark (si installé manuellement) | `/opt/spark-3.5.0` |
| `PYSPARK_PYTHON` | Interpréteur Python utilisé par les Executors | `python3` |
| `PYSPARK_DRIVER_PYTHON` | Interpréteur Python utilisé par le Driver | `python3` |
| `HADOOP_HOME` | Nécessaire sur Windows pour les utilitaires Hadoop | `C:\hadoop` |

```bash
# Exemple de configuration dans ~/.bashrc
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export SPARK_HOME=/opt/spark-3.5.0
export PATH=$PATH:$SPARK_HOME/bin
export PYSPARK_PYTHON=python3
```

---

## 2. Installation de PySpark

Il existe deux méthodes principales d'installation. Le choix dépend du contexte d'utilisation.

### 2.1 Méthode 1 — via pip (recommandée pour le développement local)

C'est la méthode la plus simple. Elle installe Spark et PySpark directement depuis le Python Package Index.

```bash
pip install pyspark==3.5.0

# Avec des dépendances utiles pour le développement
pip install pyspark==3.5.0 jupyter findspark
```

> ✅ **Avantage** : rapide, sans configuration manuelle de Spark  
> ⚠️ **Limite** : moins flexible pour des configurations avancées de cluster

### 2.2 Méthode 2 — installation manuelle de Spark

Cette méthode est utile pour configurer un cluster réel ou pour avoir un contrôle fin sur les options de déploiement.

```bash
# 1. Télécharger Spark depuis le site officiel
wget https://downloads.apache.org/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz

# 2. Décompresser l'archive
tar -xvzf spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 /opt/spark-3.5.0

# 3. Configurer les variables d'environnement
export SPARK_HOME=/opt/spark-3.5.0
export PATH=$PATH:$SPARK_HOME/bin

# 4. Installer le package Python PySpark correspondant
pip install pyspark==3.5.0
```

### 2.3 Méthode 3 — environnements cloud et notebooks managés

Dans un contexte professionnel ou de TP, on utilise souvent des environnements préconfigurés :

| Environnement | Avantage | Accès |
|---|---|---|
| **Google Colab** | Gratuit, PySpark installable en 1 commande | colab.research.google.com |
| **Databricks Community Edition** | Cluster Spark managé, notebooks intégrés | community.cloud.databricks.com |
| **Amazon EMR** | Cluster AWS, intégration S3 native | aws.amazon.com/emr |
| **Azure HDInsight** | Cluster Azure avec Spark préinstallé | azure.microsoft.com |

**Installation sur Google Colab :**
```python
# À exécuter dans la première cellule du notebook
!pip install pyspark
import pyspark
print(pyspark.__version__)
```

### 2.4 Vérification de l'installation

```bash
# Tester que PySpark est correctement installé
python -c "import pyspark; print(pyspark.__version__)"
# Résultat attendu : 3.5.0

# Tester que la SparkSession se crée sans erreur
python -c "
from pyspark.sql import SparkSession
spark = SparkSession.builder.master('local').appName('test').getOrCreate()
print('Spark version :', spark.version)
spark.stop()
"
```

---

## 3. La ligne de commande Spark

Spark fournit plusieurs outils en ligne de commande, chacun avec un rôle bien précis.

### 3.1 `pyspark` — Shell interactif Python

`pyspark` ouvre un **REPL (Read-Eval-Print Loop)** Python avec une `SparkSession` pré-initialisée. C'est l'outil idéal pour l'exploration interactive de données.

```bash
# Lancer le shell PySpark en local (tous les cœurs disponibles)
pyspark --master local[*]

# Avec des ressources limitées (2 cœurs, 4 Go de RAM pour l'Executor)
pyspark --master local[2] --driver-memory 2g --executor-memory 4g
```

Une fois dans le shell, les variables `spark` (SparkSession) et `sc` (SparkContext) sont automatiquement disponibles :

```python
# Dans le shell pyspark
>>> spark
<pyspark.sql.session.SparkSession object at 0x...>

>>> sc
<SparkContext master=local[*] appName=PySparkShell>

>>> df = spark.range(10)
>>> df.show()
+---+
| id|
+---+
|  0|
|  1|
...
```

### 3.2 `spark-shell` — Shell interactif Scala

`spark-shell` est l'équivalent de `pyspark` mais pour **Scala**. Même si on travaille en Python, il est utile de le connaître pour lire la documentation officielle Spark, majoritairement écrite en Scala.

```bash
spark-shell --master local[*]
```

```scala
// Dans le shell Scala
scala> val df = spark.range(10)
scala> df.show()
```

### 3.3 `spark-submit` — Soumission de jobs en production

`spark-submit` est la commande pour **exécuter un script Spark en production** sur un cluster. C'est l'équivalent d'un `python mon_script.py` mais pour Spark.

**Syntaxe générale :**
```bash
spark-submit \
  --master <url-du-cluster> \
  --deploy-mode <client|cluster> \
  --num-executors <N> \
  --executor-cores <N> \
  --executor-memory <Ng> \
  --driver-memory <Ng> \
  --conf spark.sql.shuffle.partitions=200 \
  mon_script.py [arguments...]
```

**Exemples concrets :**

```bash
# Mode local (développement)
spark-submit --master local[*] mon_script.py

# Sur un cluster YARN (Hadoop)
spark-submit \
  --master yarn \
  --deploy-mode cluster \
  --num-executors 10 \
  --executor-memory 8g \
  --executor-cores 4 \
  pipeline_etl.py --date 2024-01-15

# Sur un cluster Standalone
spark-submit \
  --master spark://master-node:7077 \
  --executor-memory 4g \
  mon_script.py
```

**Différence `client` vs `cluster` pour `--deploy-mode` :**

| Mode | Driver tourne sur... | Usage typique |
|---|---|---|
| `client` (défaut) | La machine qui lance `spark-submit` | Développement, débogage (logs visibles directement) |
| `cluster` | Un nœud du cluster | Production (le terminal peut être fermé) |

### 3.4 `spark-sql` — Shell SQL interactif

`spark-sql` permet d'exécuter des requêtes SQL directement en ligne de commande, sans écrire de code Python ou Scala.

```bash
spark-sql --master local[*]
```

```sql
-- Dans le shell spark-sql
spark-sql> CREATE TABLE IF NOT EXISTS users USING CSV
           OPTIONS (path '/data/users.csv', header 'true');

spark-sql> SELECT age, COUNT(*) as nb FROM users GROUP BY age ORDER BY nb DESC;
```

### 3.5 Récapitulatif des outils CLI

| Commande | Langage | Usage |
|---|---|---|
| `pyspark` | Python | Exploration interactive, développement |
| `spark-shell` | Scala | Exploration interactive, lecture de docs |
| `spark-submit` | Python/Scala/R | Exécution de jobs en production |
| `spark-sql` | SQL | Requêtes SQL ad hoc |

---

## 4. La SparkSession — point d'entrée unique

Depuis Spark 2.0, la **`SparkSession`** est le point d'entrée unique pour toutes les fonctionnalités de Spark. Elle remplace et unifie les anciens `SparkContext`, `SQLContext` et `HiveContext`.

### 4.1 Création d'une SparkSession

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NomDeLApplication") \         # Nom visible dans la Spark UI
    .master("local[*]") \                    # URL du cluster
    .config("spark.sql.shuffle.partitions", "8") \   # Paramètre de configuration
    .config("spark.driver.memory", "2g") \  # Mémoire allouée au Driver
    .getOrCreate()                           # Crée ou récupère une session existante
```

> 💡 **`getOrCreate()`** est important : si une session existe déjà dans le contexte (notebook Databricks, par exemple), elle sera réutilisée plutôt qu'une nouvelle ne soit créée.

### 4.2 Les URL de master les plus courantes

```python
# Mode local — 1 thread (pas de parallélisme)
.master("local")

# Mode local — N threads
.master("local[4]")        # 4 threads
.master("local[*]")        # Tous les cœurs disponibles

# Cluster Standalone
.master("spark://192.168.1.10:7077")

# YARN
.master("yarn")

# Kubernetes
.master("k8s://https://k8s-api-server:6443")
```

### 4.3 Configurations importantes

Les paramètres de configuration Spark peuvent être définis à la création de la session ou chargés depuis un fichier `spark-defaults.conf`.

```python
spark = SparkSession.builder \
    .appName("MonApp") \
    .master("local[*]") \
    # Nombre de partitions après un shuffle (défaut : 200, trop élevé en local)
    .config("spark.sql.shuffle.partitions", "8") \
    # Activation de l'AQE (Adaptive Query Execution) — Spark 3.x
    .config("spark.sql.adaptive.enabled", "true") \
    # Activation du broadcast join automatique (seuil en octets)
    .config("spark.sql.autoBroadcastJoinThreshold", "10mb") \
    # Répertoire temporaire pour les shuffles
    .config("spark.local.dir", "/tmp/spark-temp") \
    .getOrCreate()
```

### 4.4 Accéder au SparkContext depuis la SparkSession

Le `SparkContext` (bas niveau, API RDD) est accessible via l'attribut `.sparkContext` :

```python
sc = spark.sparkContext
print(sc.version)          # Version de Spark
print(sc.master)           # URL du master
print(sc.defaultParallelism)  # Nombre de partitions par défaut
```

### 4.5 Fermer une session

```python
# Toujours fermer la session en fin de script pour libérer les ressources
spark.stop()
```

> ⚠️ Dans un notebook, ne pas appeler `spark.stop()` entre chaque cellule — uniquement en toute fin de session.

---

## 5. La Spark UI — interface de monitoring

La **Spark UI** est une interface web automatiquement disponible dès qu'une application Spark est lancée. C'est l'outil de diagnostic incontournable.

### 5.1 Accès

| Contexte | URL par défaut |
|---|---|
| Application en cours | `http://localhost:4040` |
| Plusieurs applications simultanées | `4040`, `4041`, `4042`... |
| Spark History Server (jobs terminés) | `http://localhost:18080` |

### 5.2 Les onglets principaux

**Jobs**
- Liste tous les jobs déclenchés par les actions (`count()`, `collect()`, etc.)
- Pour chaque job : durée, nombre de stages, statut (réussi / échoué)

**Stages**
- Détaille les stages de chaque job
- Permet de voir les tâches lentes (*stragglers*) ou échouées
- Informations clés : durée, volume de données lu/écrit, shuffle read/write

**Storage**
- Affiche les RDD/DataFrames mis en cache (`persist()`, `cache()`)
- Mémoire utilisée vs disponible

**Environment**
- Toutes les variables d'environnement et paramètres de configuration actifs

**Executors**
- Ressources consommées par chaque executor (CPU, RAM, tâches complétées)
- Utile pour détecter les déséquilibres de charge (*data skew*)

**SQL / DataFrame**
- Plans d'exécution des requêtes Spark SQL et DataFrames
- Vue logique et physique — très utile pour l'optimisation

### 5.3 Lire un plan d'exécution

```python
df = spark.read.csv("data/ventes.csv", header=True, inferSchema=True)
df_filtre = df.filter(df["montant"] > 1000).groupBy("region").sum("montant")

# Afficher le plan d'exécution dans le terminal
df_filtre.explain()

# Plan détaillé (avec optimisations Catalyst)
df_filtre.explain(mode="extended")
```

Sortie typique de `explain()` :
```
== Physical Plan ==
*(2) HashAggregate(keys=[region#12], functions=[sum(montant#8)])
+- Exchange hashpartitioning(region#12, 8)    ← shuffle réseau
   +- *(1) HashAggregate(keys=[region#12], functions=[partial_sum(montant#8)])
      +- *(1) Filter (montant#8 > 1000)
         +- FileScan csv [region#12, montant#8]
```

---

## 6. Utilisation dans un notebook Jupyter

### 6.1 Configuration initiale

```bash
# Installation des dépendances
pip install pyspark jupyter findspark
```

```python
# Cellule 1 : initialisation (si Spark n'est pas dans le PATH)
import findspark
findspark.init()   # Localise automatiquement SPARK_HOME

# Cellule 2 : création de la SparkSession
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("NotebookPySpark") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Afficher la version pour confirmer
print(f"PySpark version : {spark.version}")
```

### 6.2 Configurer PySpark pour ouvrir Jupyter automatiquement

On peut configurer la variable `PYSPARK_DRIVER_PYTHON` pour que la commande `pyspark` ouvre directement Jupyter :

```bash
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser --port=8888"

# Lancer Jupyter avec PySpark préconfiguré
pyspark --master local[*]
```

### 6.3 Affichage des DataFrames dans Jupyter

```python
# Affichage classique (texte brut)
df.show(5)

# Affichage formaté en HTML dans le notebook (via Pandas)
df.limit(5).toPandas()   # ⚠️ Uniquement pour de petits volumes !

# Afficher le schéma
df.printSchema()
```

### 6.4 Bonnes pratiques en notebook

```python
# ✅ Toujours limiter les données affichées
df.show(20)           # Affiche 20 lignes maximum
df.limit(100).show()  # Sécurisé même pour de gros DataFrames

# ✅ Utiliser toPandas() uniquement sur des petits sous-ensembles
df.filter(...).limit(1000).toPandas()

# ❌ Ne jamais faire cela sur un gros DataFrame
df.toPandas()      # Peut saturer la mémoire du Driver !
df.collect()       # Même problème
```

---

## 7. Premier programme complet

Voici un programme PySpark complet qui illustre l'ensemble des notions du module.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# ─── 1. Initialisation ───────────────────────────────────────────────────────
spark = SparkSession.builder \
    .appName("AnalyseVentes") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")   # Réduire les logs INFO

# ─── 2. Création d'un DataFrame depuis une liste Python ──────────────────────
data = [
    ("Alice",   "Paris",  1500.0),
    ("Bob",     "Lyon",    800.0),
    ("Claire",  "Paris",  2200.0),
    ("David",   "Lyon",   1100.0),
    ("Emma",    "Nantes",  950.0),
    ("François","Paris",  3000.0),
]
colonnes = ["nom", "ville", "chiffre_affaires"]

df = spark.createDataFrame(data, schema=colonnes)

# ─── 3. Exploration ──────────────────────────────────────────────────────────
print("=== Schéma ===")
df.printSchema()

print("=== Aperçu ===")
df.show()

print(f"Nombre de lignes : {df.count()}")
print(f"Nombre de partitions : {df.rdd.getNumPartitions()}")

# ─── 4. Transformations ──────────────────────────────────────────────────────
# Filtrer les commerciaux avec un CA > 1000€
df_top = df.filter(F.col("chiffre_affaires") > 1000)

# Ajouter une colonne calculée
df_top = df_top.withColumn("ca_ht", F.col("chiffre_affaires") / 1.2)

# Agréger par ville
df_par_ville = df_top.groupBy("ville") \
    .agg(
        F.sum("chiffre_affaires").alias("ca_total"),
        F.count("*").alias("nb_commerciaux"),
        F.avg("chiffre_affaires").alias("ca_moyen")
    ) \
    .orderBy(F.col("ca_total").desc())

print("=== CA par ville (top vendeurs) ===")
df_par_ville.show()

# ─── 5. Afficher le plan d'exécution ─────────────────────────────────────────
print("=== Plan d'exécution ===")
df_par_ville.explain()

# ─── 6. Fermeture ────────────────────────────────────────────────────────────
spark.stop()
```

**Sortie attendue :**
```
=== CA par ville (top vendeurs) ===
+------+---------+--------------+--------+
| ville|ca_total |nb_commerciaux|ca_moyen|
+------+---------+--------------+--------+
| Paris|  6700.0 |             3|  2233.3|
|  Lyon|  1100.0 |             1|  1100.0|
+------+---------+--------------+--------+
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Prérequis** | Java (JVM) obligatoire, même pour PySpark ; Python 3.8+ recommandé |
| **Installation pip** | `pip install pyspark` — méthode la plus simple en développement |
| **`pyspark`** | Shell interactif Python, `spark` et `sc` disponibles d'emblée |
| **`spark-submit`** | Outil de production — soumission de jobs sur cluster |
| **SparkSession** | Point d'entrée unique depuis Spark 2.x — toujours via `.builder.getOrCreate()` |
| **`local[*]`** | Mode de développement, utilise tous les cœurs de la machine |
| **Spark UI** | `localhost:4040` — indispensable pour le monitoring et le diagnostic |
| **Notebooks** | Intégration naturelle avec Jupyter — attention à `toPandas()` sur gros volumes |

---

## Exercices

### Exercice 1 — Installation et vérification (20 min)
> 1. Installer PySpark dans un environnement virtuel Python
> 2. Vérifier l'installation via le shell Python
> 3. Lancer `pyspark --master local[*]` et créer un DataFrame de 100 lignes avec `spark.range(100)`
> 4. Ouvrir la Spark UI et identifier le job correspondant au `count()` lancé sur ce DataFrame

### Exercice 2 — spark-submit (20 min)
> 1. Écrire le programme de l'exemple complet (section 7) dans un fichier `analyse_ventes.py`
> 2. L'exécuter avec `spark-submit --master local[*] analyse_ventes.py`
> 3. Identifier dans la Spark UI combien de jobs et de stages ont été créés

### Exercice 3 — Configuration et plan d'exécution (30 min)
> 1. Créer une SparkSession avec `spark.sql.shuffle.partitions` à 4
> 2. Lire un fichier CSV (de votre choix) et effectuer un `groupBy` + `count()`
> 3. Appeler `.explain(mode="extended")` sur le résultat et identifier :
>    - L'étape de scan du fichier
>    - Les étapes d'agrégation partielle et globale
>    - L'échange réseau (*shuffle*)

### Exercice 4 — Notebook Jupyter (30 min)
> 1. Lancer Jupyter Notebook avec PySpark configuré
> 2. Reproduire le programme de la section 7 dans un notebook
> 3. Afficher le résultat final avec `.toPandas()` pour un rendu tabulaire
> 4. Documenter chaque cellule avec des cellules Markdown explicatives

---

## Pour aller plus loin

- 📖 **Configuration Spark** : [spark.apache.org/docs/latest/configuration.html](https://spark.apache.org/docs/latest/configuration.html)
- 📖 **spark-submit** : [spark.apache.org/docs/latest/submitting-applications.html](https://spark.apache.org/docs/latest/submitting-applications.html)
- 🛠️ **Databricks Community Edition** : environnement Spark managé gratuit, idéal pour les TP
- 📄 **spark-defaults.conf** : fichier de configuration global dans `$SPARK_HOME/conf/`
- 🎥 **Spark UI expliquée** : [youtu.be/BVMIT7WO-5I](https://youtu.be/BVMIT7WO-5I) (Databricks)

---

*Module précédent → **Module 1 : Introduction à Spark***  
*Module suivant → **Module 3 : Les structures de données Spark***
