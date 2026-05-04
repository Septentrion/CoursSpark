# Tutoriel — Apache Spark en ligne de commande

> Ce tutoriel couvre l'installation de Spark et l'utilisation complète de ses outils CLI :  
> `spark-shell`, `pyspark`, `spark-submit`, `spark-sql` et les utilitaires associés.  
> **Public cible** : utilisateurs à l'aise avec le terminal Linux/macOS ou Windows (WSL).

---

## Table des matières

1. [Prérequis et installation](#1-prérequis-et-installation)
2. [Structure d'un répertoire Spark](#2-structure-dun-répertoire-spark)
3. [Variables d'environnement](#3-variables-denvironnement)
4. [Le shell interactif Python — `pyspark`](#4-le-shell-interactif-python--pyspark)
5. [Le shell interactif Scala — `spark-shell`](#5-le-shell-interactif-scala--spark-shell)
6. [Le shell SQL — `spark-sql`](#6-le-shell-sql--spark-sql)
7. [La soumission de jobs — `spark-submit`](#7-la-soumission-de-jobs--spark-submit)
8. [Gestion des dépendances](#8-gestion-des-dépendances)
9. [Configuration avancée](#9-configuration-avancée)
10. [Monitoring et logs](#10-monitoring-et-logs)
11. [Exemples complets de bout en bout](#11-exemples-complets-de-bout-en-bout)
12. [Référence rapide des options](#12-référence-rapide-des-options)

---

## 1. Prérequis et installation

### 1.1 Java (JVM) — prérequis obligatoire

Spark est écrit en Scala et tourne sur la JVM. Java est donc **indispensable**, même pour PySpark.

```bash
# Vérifier si Java est installé
java -version
# Résultat attendu : openjdk version "11.x.x" ou "17.x.x"

# Ubuntu / Debian
sudo apt update
sudo apt install -y openjdk-11-jdk

# macOS (via Homebrew)
brew install openjdk@11
echo 'export PATH="/opt/homebrew/opt/openjdk@11/bin:$PATH"' >> ~/.zshrc

# CentOS / RHEL / Fedora
sudo dnf install java-11-openjdk-devel

# Windows : télécharger le JDK depuis https://adoptium.net
# Puis ajouter C:\Program Files\Eclipse Adoptium\jdk-11...\bin au PATH

# Vérifier la version après installation
java -version
javac -version
echo $JAVA_HOME     # Doit pointer vers le répertoire du JDK
```

### 1.2 Python — pour PySpark

```bash
# Vérifier la version (Python 3.8 à 3.12 supporté)
python3 --version

# Ubuntu / Debian
sudo apt install -y python3 python3-pip python3-venv

# macOS
brew install python@3.11

# Créer un environnement virtuel dédié (fortement recommandé)
python3 -m venv ~/envs/spark-env
source ~/envs/spark-env/bin/activate
```

### 1.3 Téléchargement et installation de Spark

**Méthode A — Installation manuelle (recommandée pour le contrôle total)**

```bash
# 1. Télécharger Spark depuis le site officiel
# Choisir la version la plus récente avec Hadoop pré-bundlé
SPARK_VERSION="3.5.0"
HADOOP_VERSION="3"

wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/\
spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

# 2. Vérifier l'intégrité (optionnel mais recommandé)
wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/\
spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz.sha512
sha512sum -c spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz.sha512

# 3. Décompresser et installer
tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
sudo mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark

# 4. Créer un lien symbolique (facilite les mises à jour futures)
sudo ln -sf /opt/spark /opt/spark-current
```

**Méthode B — Via pip (pour le développement Python uniquement)**

```bash
# Installation dans l'environnement virtuel
pip install pyspark==3.5.0

# Vérification
python -c "import pyspark; print(pyspark.__version__)"
# → 3.5.0

# Trouver où PySpark a installé Spark
python -c "import pyspark; print(pyspark.__file__)"
# → /home/user/envs/spark-env/lib/python3.11/site-packages/pyspark/__init__.py
```

**Méthode C — Via conda**

```bash
conda create -n spark-env python=3.11
conda activate spark-env
conda install -c conda-forge pyspark openjdk
```

### 1.4 Vérification de l'installation

```bash
# Si installation manuelle :
/opt/spark/bin/spark-shell --version
# Si via pip :
python -m pyspark --version

# Dans les deux cas, la sortie doit ressembler à :
# Welcome to
#       ____              __
#      / __/__  ___ _____/ /__
#     _\ \/ _ \/ _ `/ __/  '_/
#    /___/ .__/\_,_/_/ /_/\_\   version 3.5.0
#       /_/
```

---

## 2. Structure d'un répertoire Spark

Comprendre la structure d'une installation Spark est utile pour la configuration.

```
/opt/spark/
├── bin/                    ← Commandes CLI (spark-shell, pyspark, spark-submit...)
│   ├── pyspark
│   ├── spark-shell
│   ├── spark-sql
│   ├── spark-submit
│   ├── spark-class
│   └── spark-daemon.sh
├── conf/                   ← Fichiers de configuration
│   ├── spark-defaults.conf.template
│   ├── spark-env.sh.template
│   ├── log4j2.properties.template
│   └── workers.template
├── jars/                   ← JARs Spark (le moteur lui-même)
├── python/                 ← Code source PySpark
│   └── pyspark/
├── sbin/                   ← Scripts de gestion du cluster Standalone
│   ├── start-master.sh
│   ├── start-worker.sh
│   └── stop-all.sh
├── examples/               ← Exemples officiels (très utiles !)
│   ├── src/main/python/
│   └── src/main/scala/
├── data/                   ← Données de test
│   └── mllib/
└── logs/                   ← Logs du cluster Standalone
```

---

## 3. Variables d'environnement

### 3.1 Configuration essentielle

Ajouter ces lignes à `~/.bashrc` (Bash) ou `~/.zshrc` (Zsh) :

```bash
# ─── Java ────────────────────────────────────────────────────────────────────
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
# macOS avec Homebrew :
# export JAVA_HOME=/opt/homebrew/opt/openjdk@11

# ─── Spark ───────────────────────────────────────────────────────────────────
export SPARK_HOME=/opt/spark
export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin

# ─── Python pour PySpark ─────────────────────────────────────────────────────
# Interpréteur Python utilisé par les Executors
export PYSPARK_PYTHON=python3
# Interpréteur Python utilisé par le Driver (doit être le même que PYSPARK_PYTHON)
export PYSPARK_DRIVER_PYTHON=python3

# ─── Mémoire du Driver (utile en local) ──────────────────────────────────────
export SPARK_DRIVER_MEMORY=2g

# ─── Logs (optionnel — réduire la verbosité) ─────────────────────────────────
export SPARK_LOG_DIR=/tmp/spark-logs

# Appliquer sans redémarrer le terminal
source ~/.bashrc   # ou source ~/.zshrc
```

### 3.2 Variables optionnelles mais utiles

```bash
# Pour les installations Windows (accès aux utilitaires Hadoop natifs)
export HADOOP_HOME=/opt/hadoop
export PATH=$PATH:$HADOOP_HOME/bin

# Pour les connexions S3
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG...
export AWS_DEFAULT_REGION=eu-west-1

# Désactiver IPv6 si des problèmes réseau surviennent
export SPARK_LOCAL_IP=127.0.0.1

# Répertoire temporaire pour les shuffles (préférer un disque rapide)
export SPARK_LOCAL_DIRS=/tmp/spark-temp

# Pour Jupyter avec PySpark
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --no-browser"
```

### 3.3 Fichier `spark-env.sh`

Ce fichier est chargé automatiquement par les scripts Spark. Il est utile pour les déploiements en cluster.

```bash
# Copier le template
cp $SPARK_HOME/conf/spark-env.sh.template $SPARK_HOME/conf/spark-env.sh

# Éditer le fichier
cat >> $SPARK_HOME/conf/spark-env.sh << 'EOF'
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
export PYSPARK_PYTHON=python3
export SPARK_WORKER_CORES=4
export SPARK_WORKER_MEMORY=8g
export SPARK_DAEMON_MEMORY=1g
EOF
```

---

## 4. Le shell interactif Python — `pyspark`

`pyspark` ouvre un REPL Python avec une `SparkSession` et un `SparkContext` pré-initialisés, accessibles via les variables `spark` et `sc`.

### 4.1 Syntaxe générale

```bash
pyspark [OPTIONS]
```

### 4.2 Lancement de base

```bash
# Mode local — 1 thread (débogage, pas de parallélisme)
pyspark --master local

# Mode local — tous les cœurs disponibles (recommandé en développement)
pyspark --master local[*]

# Mode local — 4 cœurs exactement
pyspark --master local[4]

# Sur un cluster Standalone
pyspark --master spark://mon-master:7077

# Sur un cluster YARN (depuis un nœud du cluster)
pyspark --master yarn

# Sur Kubernetes
pyspark --master k8s://https://k8s-api:6443
```

### 4.3 Options mémoire et ressources

```bash
# Mémoire allouée au Driver (le processus Python principal)
pyspark --master local[*] \
        --driver-memory 4g

# Mémoire allouée à chaque Executor (sur cluster)
pyspark --master spark://master:7077 \
        --executor-memory 8g \
        --executor-cores 4 \
        --num-executors 10

# Mémoire Driver + limitation des Executors en local
pyspark --master local[4] \
        --driver-memory 2g \
        --conf spark.executor.memory=2g
```

### 4.4 Options de configuration inline

```bash
# Passer des paramètres de configuration directement
pyspark --master local[*] \
        --conf spark.sql.shuffle.partitions=8 \
        --conf spark.sql.adaptive.enabled=true \
        --conf spark.driver.maxResultSize=2g \
        --conf spark.ui.port=4041 \
        --conf spark.eventLog.enabled=true \
        --conf spark.eventLog.dir=/tmp/spark-events
```

### 4.5 Lancer directement avec Jupyter

```bash
# Option 1 : via les variables d'environnement
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --port=8888 --no-browser"
pyspark --master local[*]
# → Jupyter s'ouvre, spark et sc sont disponibles dans les cellules

# Option 2 : lancer Jupyter normalement et initialiser PySpark dans une cellule
jupyter notebook
# Dans une cellule :
# import findspark; findspark.init()
# from pyspark.sql import SparkSession
# spark = SparkSession.builder.master("local[*]").getOrCreate()
```

### 4.6 Session interactive — exemples d'utilisation

Une fois dans le shell `pyspark` :

```python
# ── Les variables disponibles d'emblée ────────────────────────────────────────
spark    # SparkSession
sc       # SparkContext
sql      # Raccourci pour spark.sql()

# ── Vérifications ─────────────────────────────────────────────────────────────
print(spark.version)                    # "3.5.0"
print(sc.master)                        # "local[*]"
print(sc.defaultParallelism)            # 8 (selon le nombre de cœurs)
print(sc.getConf().getAll())            # Toutes les configurations actives

# ── Créer un RDD ──────────────────────────────────────────────────────────────
rdd = sc.parallelize(range(1, 101))
print(rdd.count())                      # 100
print(rdd.filter(lambda x: x%2==0).sum())  # 2550

# ── Créer un DataFrame ────────────────────────────────────────────────────────
df = spark.createDataFrame([
    ("Alice", 30, 45000.0),
    ("Bob",   25, 35000.0),
    ("Claire",40, 60000.0),
], ["nom", "age", "salaire"])
df.show()
df.printSchema()

# ── Lire un fichier ───────────────────────────────────────────────────────────
df_csv = spark.read.option("header","true").csv("/path/to/file.csv")
df_json = spark.read.json("/path/to/data/*.json")

# ── Spark SQL ─────────────────────────────────────────────────────────────────
df.createOrReplaceTempView("employes")
spark.sql("SELECT avg(salaire) FROM employes WHERE age > 30").show()

# ── Commandes shell depuis PySpark ────────────────────────────────────────────
import os
os.system("ls /tmp/data/")

# ── Quitter ───────────────────────────────────────────────────────────────────
quit()    # ou Ctrl+D
exit()
```

### 4.7 Historique et autocomplétion

```bash
# PySpark utilise le REPL Python standard avec readline
# → Flèche haut/bas : historique des commandes
# → Tab : autocomplétion des méthodes Spark

# L'historique est sauvegardé dans :
cat ~/.python_history

# Pour activer l'autocomplétion améliorée (ptpython)
pip install ptpython
export PYSPARK_DRIVER_PYTHON=ptpython
pyspark --master local[*]
```

---

## 5. Le shell interactif Scala — `spark-shell`

`spark-shell` est l'équivalent de `pyspark` pour le langage **Scala**. Même si vous travaillez principalement en Python, connaître `spark-shell` est utile pour lire la documentation officielle et les exemples Spark, majoritairement en Scala.

### 5.1 Lancement

```bash
# Mode local
spark-shell --master local[*]

# Avec configuration
spark-shell --master local[*] \
            --driver-memory 2g \
            --conf spark.sql.shuffle.partitions=8
```

### 5.2 Variables disponibles

```scala
// Disponibles d'emblée dans spark-shell :
spark    // SparkSession
sc       // SparkContext

// Vérifier la version
spark.version                           // res0: String = 3.5.0
sc.master                               // res1: String = local[*]
```

### 5.3 Exemples d'utilisation

```scala
// ── RDD ───────────────────────────────────────────────────────────────────────
val rdd = sc.parallelize(1 to 100)
rdd.count()                             // 100
rdd.filter(_ % 2 == 0).sum()           // 2550.0

// ── DataFrame ─────────────────────────────────────────────────────────────────
val df = spark.createDataFrame(Seq(
    ("Alice", 30, 45000.0),
    ("Bob",   25, 35000.0),
    ("Claire",40, 60000.0),
)).toDF("nom", "age", "salaire")

df.show()
df.printSchema()
df.filter($"age" > 28).select("nom","salaire").show()

// ── Spark SQL ─────────────────────────────────────────────────────────────────
df.createOrReplaceTempView("employes")
spark.sql("SELECT avg(salaire) FROM employes WHERE age > 30").show()

// ── Lire un fichier Parquet ───────────────────────────────────────────────────
val dfParquet = spark.read.parquet("/chemin/vers/data.parquet")
dfParquet.explain()

// ── Charger un script Scala depuis le shell ───────────────────────────────────
:load /chemin/vers/mon_script.scala

// ── Importer une bibliothèque ─────────────────────────────────────────────────
import org.apache.spark.sql.functions._
val df2 = df.withColumn("salaire_k", round(col("salaire") / 1000, 1))

// ── Afficher le plan d'exécution ──────────────────────────────────────────────
df2.groupBy("age").avg("salaire").explain(true)

// ── Quitter ───────────────────────────────────────────────────────────────────
:quit    // ou Ctrl+D
```

### 5.4 Commandes méta du REPL Scala

```scala
:help           // Afficher l'aide
:history        // Historique des commandes
:paste          // Mode collage multi-lignes (terminer avec Ctrl+D)
:type expr      // Afficher le type d'une expression
:imports        // Lister les imports actifs
:reset          // Réinitialiser le REPL (efface les définitions)
:load fichier   // Charger et exécuter un fichier .scala
:quit           // Quitter
```

---

## 6. Le shell SQL — `spark-sql`

`spark-sql` est un REPL SQL permettant d'exécuter des requêtes SQL directement sur des fichiers ou des tables, sans écrire de code Python/Scala.

### 6.1 Lancement

```bash
# Mode local
spark-sql --master local[*]

# Avec configuration
spark-sql --master local[*] \
          --conf spark.sql.shuffle.partitions=8 \
          --conf spark.sql.adaptive.enabled=true

# Avec une base de données Hive
spark-sql --master yarn \
          --conf spark.sql.warehouse.dir=/user/hive/warehouse

# Exécuter un fichier SQL directement (sans shell interactif)
spark-sql --master local[*] \
          -f /chemin/vers/mes_requetes.sql

# Exécuter une requête inline
spark-sql --master local[*] \
          -e "SELECT current_date()"
```

### 6.2 Fonctionnalités interactives

```sql
-- ── Créer une table temporaire depuis un CSV ──────────────────────────────────
CREATE TABLE IF NOT EXISTS ventes
USING CSV
OPTIONS (
    path '/data/ventes.csv',
    header 'true',
    inferSchema 'true',
    sep ';'
);

-- ── Créer une table temporaire depuis un Parquet ──────────────────────────────
CREATE TABLE IF NOT EXISTS transactions
USING PARQUET
LOCATION '/data/transactions/';

-- ── Requêtes SQL standard ─────────────────────────────────────────────────────
SELECT ville, COUNT(*) AS nb, ROUND(SUM(montant), 2) AS ca
FROM ventes
WHERE montant > 0
GROUP BY ville
ORDER BY ca DESC;

-- ── Fonctions de fenêtre ──────────────────────────────────────────────────────
SELECT
    nom,
    ville,
    montant,
    RANK() OVER (PARTITION BY ville ORDER BY montant DESC) AS rang,
    ROUND(AVG(montant) OVER (PARTITION BY ville), 2) AS moy_ville
FROM ventes;

-- ── CREATE TABLE AS SELECT (CTAS) ────────────────────────────────────────────
CREATE TABLE top_ventes
USING PARQUET
AS
SELECT * FROM ventes WHERE montant > 1000;

-- ── Inspecter le catalogue ────────────────────────────────────────────────────
SHOW DATABASES;
SHOW TABLES;
DESCRIBE ventes;
DESCRIBE EXTENDED ventes;

-- ── Plan d'exécution ──────────────────────────────────────────────────────────
EXPLAIN SELECT ville, SUM(montant) FROM ventes GROUP BY ville;
EXPLAIN EXTENDED SELECT ville, SUM(montant) FROM ventes GROUP BY ville;

-- ── Variables de configuration ────────────────────────────────────────────────
SET spark.sql.shuffle.partitions=16;
SET spark.sql.adaptive.enabled=true;
SET;    -- Afficher toutes les configurations actives

-- ── Quitter ───────────────────────────────────────────────────────────────────
quit;
-- ou Ctrl+D
```

### 6.3 Exécuter un fichier `.sql`

```bash
# Créer le fichier SQL
cat > /tmp/analyse_ventes.sql << 'EOF'
-- Configuration
SET spark.sql.shuffle.partitions=8;

-- Créer la table source
CREATE OR REPLACE TEMP VIEW ventes
USING CSV
OPTIONS (path '/data/ventes.csv', header 'true', inferSchema 'true');

-- Analyse 1 : CA par ville
SELECT ville,
       COUNT(*) AS nb_ventes,
       ROUND(SUM(montant), 2) AS ca_total,
       ROUND(AVG(montant), 2) AS ca_moyen
FROM ventes
GROUP BY ville
ORDER BY ca_total DESC;

-- Analyse 2 : Évolution mensuelle
SELECT mois,
       ROUND(SUM(montant), 2) AS ca,
       ROUND(SUM(SUM(montant)) OVER (ORDER BY mois), 2) AS ca_cumule
FROM ventes
GROUP BY mois
ORDER BY mois;
EOF

# Exécuter le fichier
spark-sql --master local[*] -f /tmp/analyse_ventes.sql

# Rediriger la sortie
spark-sql --master local[*] -f /tmp/analyse_ventes.sql > /tmp/resultats.txt 2>/dev/null
```

---

## 7. La soumission de jobs — `spark-submit`

`spark-submit` est **l'outil de production** de Spark. Il permet d'exécuter des scripts Python, Scala ou Java sur n'importe quel cluster Spark.

### 7.1 Syntaxe générale

```bash
spark-submit \
    --master <url-master> \
    --deploy-mode <client|cluster> \
    [OPTIONS] \
    <script.py ou fichier.jar> \
    [arguments-du-script...]
```

### 7.2 Options principales

```bash
# ─── Master / Mode de déploiement ────────────────────────────────────────────
--master local[*]              # Local (développement)
--master spark://host:7077     # Cluster Standalone
--master yarn                  # YARN (Hadoop)
--master k8s://https://host    # Kubernetes
--deploy-mode client           # Driver sur la machine qui lance la commande
--deploy-mode cluster          # Driver sur le cluster (production)

# ─── Ressources ──────────────────────────────────────────────────────────────
--driver-memory 2g             # Mémoire du Driver
--executor-memory 4g           # Mémoire par Executor
--executor-cores 2             # Cœurs CPU par Executor
--num-executors 10             # Nombre d'Executors (YARN/Standalone)
--total-executor-cores 40      # Total de cœurs (Standalone uniquement)

# ─── Identification de l'application ─────────────────────────────────────────
--name "Mon Application"       # Nom affiché dans la Spark UI

# ─── Configuration ────────────────────────────────────────────────────────────
--conf clé=valeur              # Paramètre Spark (répétable)
--properties-file fichier.conf # Fichier de propriétés

# ─── Dépendances ─────────────────────────────────────────────────────────────
--jars a.jar,b.jar             # JARs supplémentaires
--packages org:artifact:vers   # Packages Maven (téléchargement auto)
--py-files module.py,lib.zip   # Fichiers Python supplémentaires
--files data.csv,config.yml    # Fichiers à distribuer aux Executors
--archives app.zip             # Archives à extraire sur les Executors

# ─── Queue et priorité (YARN) ────────────────────────────────────────────────
--queue production             # Queue YARN
--principal user@REALM         # Kerberos principal
--keytab /path/to/user.keytab  # Fichier Kerberos keytab
```

### 7.3 Exemples d'utilisation

**Script Python simple :**

```bash
# Créer un script de démonstration
cat > /tmp/word_count.py << 'EOF'
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

def main(input_path, output_path):
    spark = SparkSession.builder \
        .appName("WordCount") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.text(input_path)

    word_count = df \
        .select(F.explode(F.split(F.col("value"), r"\W+")).alias("word")) \
        .filter(F.length(F.col("word")) > 2) \
        .withColumn("word", F.lower(F.col("word"))) \
        .groupBy("word") \
        .count() \
        .orderBy(F.col("count").desc())

    word_count.write.mode("overwrite").csv(output_path, header=True)
    print(f"Top 10 mots : ")
    word_count.show(10)
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: word_count.py <input_path> <output_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
EOF

# Lancer en local
spark-submit \
    --master local[*] \
    --name "WordCount" \
    /tmp/word_count.py \
    /tmp/input_text/ \
    /tmp/output_wordcount/

# Lancer sur YARN en mode cluster
spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --name "WordCount Production" \
    --driver-memory 2g \
    --executor-memory 4g \
    --executor-cores 2 \
    --num-executors 5 \
    --conf spark.sql.shuffle.partitions=100 \
    /tmp/word_count.py \
    hdfs://namenode:9000/data/input/ \
    hdfs://namenode:9000/data/output/wordcount/
```

**Script avec plusieurs fichiers Python :**

```bash
# Structure du projet
# mon_projet/
# ├── main.py
# ├── utils/
# │   ├── __init__.py
# │   ├── transformations.py
# │   └── io_helpers.py
# └── config.yml

# Créer une archive des modules Python
cd mon_projet
zip -r utils.zip utils/

# Soumettre avec l'archive
spark-submit \
    --master local[*] \
    --py-files utils.zip \
    --files config.yml \
    main.py
```

**Script avec dépendances Maven (packages) :**

```bash
# Connexion S3 (hadoop-aws) + Delta Lake
spark-submit \
    --master local[*] \
    --packages \
        org.apache.hadoop:hadoop-aws:3.3.4,\
        io.delta:delta-spark_2.12:3.1.0 \
    --conf spark.sql.extensions=io.delta.sql.DeltaSparkSessionExtension \
    --conf spark.sql.catalog.spark_catalog=\
org.apache.spark.sql.delta.catalog.DeltaCatalog \
    mon_pipeline_delta.py

# Kafka Streaming
spark-submit \
    --master local[3] \
    --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 \
    mon_stream_kafka.py

# PostgreSQL via JDBC
spark-submit \
    --master local[*] \
    --jars /opt/drivers/postgresql-42.6.0.jar \
    pipeline_jdbc.py
```

**Job Scala (fichier JAR) :**

```bash
# Compiler le projet Scala avec sbt ou Maven
cd mon-projet-scala
sbt package
# Produit : target/scala-2.12/mon-app_2.12-1.0.jar

# Soumettre le JAR
spark-submit \
    --master spark://master:7077 \
    --deploy-mode cluster \
    --class com.monentreprise.MonApp \
    --driver-memory 2g \
    --executor-memory 4g \
    target/scala-2.12/mon-app_2.12-1.0.jar \
    arg1 arg2
```

### 7.4 Client vs Cluster deploy-mode

```
--deploy-mode client (défaut) :

  Machine locale ──── spark-submit ────► Cluster
        │                                    │
     DRIVER ◄──── résultats/logs ──── Executors
  (sur la machine locale)

  Avantages : logs visibles directement, débogage facile
  Inconvénients : connexion réseau maintenue pendant tout le job


--deploy-mode cluster :

  Machine locale ──── spark-submit ────► Cluster Manager
                                              │
                                         DRIVER (sur un nœud du cluster)
                                              │
                                         Executors

  Avantages : le terminal peut être fermé, pas de latence réseau Driver↔Executors
  Inconvénients : logs dans le cluster (à récupérer via spark-submit --status ou UI)
```

### 7.5 Superviser un job soumis

```bash
# Soumettre en arrière-plan
spark-submit --master yarn --deploy-mode cluster mon_job.py &

# Lister les applications YARN
yarn application -list

# Voir les logs d'une application YARN
yarn logs -applicationId application_1234567890_0001

# Annuler une application YARN
yarn application -kill application_1234567890_0001

# Voir l'état d'un job Standalone
spark-submit --status <driver-id> --master spark://master:7077

# Annuler un job Standalone
spark-submit --kill <driver-id> --master spark://master:7077
```

---

## 8. Gestion des dépendances

### 8.1 `--packages` : résolution Maven automatique

```bash
# Syntaxe : groupId:artifactId:version
spark-submit \
    --packages groupId:artifactId:version \
    mon_script.py

# Exemples courants
spark-submit \
    --packages \
        org.apache.hadoop:hadoop-aws:3.3.4,\
        com.amazonaws:aws-java-sdk-bundle:1.12.262 \
    mon_script_s3.py

# Spécifier un repository Maven alternatif
spark-submit \
    --repositories https://repo.company.com/maven2 \
    --packages com.company:ma-lib:1.0.0 \
    mon_script.py

# Les packages sont mis en cache dans ~/.ivy2/cache
# Pour forcer le retéléchargement :
rm -rf ~/.ivy2/cache
```

### 8.2 `--jars` : JARs locaux

```bash
# Un seul JAR
spark-submit --jars /opt/drivers/postgresql-42.6.0.jar mon_script.py

# Plusieurs JARs (séparés par des virgules)
spark-submit \
    --jars /opt/drivers/postgresql-42.6.0.jar,/opt/libs/ma-lib.jar \
    mon_script.py

# Utiliser un glob (Spark 3.x)
spark-submit --jars /opt/drivers/*.jar mon_script.py
```

### 8.3 `--py-files` : modules Python

```bash
# Fichiers Python individuels
spark-submit --py-files utils.py,helpers.py mon_script.py

# Archive ZIP d'un package Python
zip -r monpackage.zip monpackage/
spark-submit --py-files monpackage.zip mon_script.py

# Fichier egg (wheel Python)
spark-submit --py-files ma_bibliotheque-1.0.egg mon_script.py
```

### 8.4 `--files` : fichiers de données

Les fichiers spécifiés avec `--files` sont distribués sur **chaque Executor** et accessibles dans le répertoire de travail courant.

```bash
# Distribuer un fichier de configuration
spark-submit \
    --files /local/config.yml,/local/modele.pkl \
    mon_script.py

# Dans le script, accéder au fichier distribué :
# import os
# config_path = os.environ.get("SPARK_YARN_STAGING_DIR", ".") + "/config.yml"
# Ou plus simplement, SparkFiles.get() :
# from pyspark import SparkFiles
# config_path = SparkFiles.get("config.yml")
```

---

## 9. Configuration avancée

### 9.1 Fichier `spark-defaults.conf`

Ce fichier définit des valeurs par défaut pour toutes les applications Spark lancées sur la machine.

```bash
# Copier le template
cp $SPARK_HOME/conf/spark-defaults.conf.template \
   $SPARK_HOME/conf/spark-defaults.conf

# Éditer le fichier
cat > $SPARK_HOME/conf/spark-defaults.conf << 'EOF'
# ─── Ressources ───────────────────────────────────────────────────────────────
spark.driver.memory                  2g
spark.executor.memory                4g
spark.executor.cores                 2

# ─── SQL ─────────────────────────────────────────────────────────────────────
spark.sql.shuffle.partitions         50
spark.sql.adaptive.enabled           true
spark.sql.adaptive.coalescePartitions.enabled  true
spark.sql.adaptive.skewJoin.enabled  true

# ─── Serialisation ────────────────────────────────────────────────────────────
spark.serializer                     org.apache.spark.serializer.KryoSerializer

# ─── Logs ─────────────────────────────────────────────────────────────────────
spark.eventLog.enabled               true
spark.eventLog.dir                   /tmp/spark-events
spark.history.fs.logDirectory        /tmp/spark-events

# ─── UI ───────────────────────────────────────────────────────────────────────
spark.ui.enabled                     true
spark.ui.retainedJobs                200
spark.ui.retainedStages              200
EOF
```

### 9.2 Fichier de propriétés par application

Plutôt que de passer toutes les options en ligne de commande, on peut les regrouper dans un fichier :

```bash
# Créer un fichier de propriétés
cat > /tmp/prod_config.conf << 'EOF'
spark.master                         yarn
spark.deploy.mode                    cluster
spark.driver.memory                  4g
spark.executor.memory                8g
spark.executor.cores                 4
spark.executor.instances             20
spark.sql.shuffle.partitions         200
spark.sql.adaptive.enabled           true
spark.serializer                     org.apache.spark.serializer.KryoSerializer
spark.eventLog.enabled               true
spark.eventLog.dir                   hdfs:///spark-events
EOF

# Utiliser le fichier
spark-submit \
    --properties-file /tmp/prod_config.conf \
    mon_script.py
```

### 9.3 Priorité des configurations

Spark applique les configurations dans cet ordre (du moins au plus prioritaire) :

```
1. spark-defaults.conf         (le moins prioritaire)
2. --properties-file           (remplace spark-defaults.conf)
3. --conf clé=valeur           (remplace les fichiers)
4. SparkConf dans le code      (le plus prioritaire)
```

```bash
# Vérifier la configuration active depuis le shell
pyspark --master local[*]
# >>> sc.getConf().getAll()     # Affiche toutes les configurations
# >>> spark.conf.get("spark.sql.shuffle.partitions")
```

### 9.4 Mode dynamique des ressources (YARN)

```bash
spark-submit \
    --master yarn \
    --conf spark.dynamicAllocation.enabled=true \
    --conf spark.dynamicAllocation.minExecutors=2 \
    --conf spark.dynamicAllocation.maxExecutors=50 \
    --conf spark.dynamicAllocation.initialExecutors=5 \
    --conf spark.shuffle.service.enabled=true \
    mon_script.py
```

---

## 10. Monitoring et logs

### 10.1 La Spark UI

La Spark UI est une interface web disponible pendant l'exécution d'une application.

```bash
# URL d'accès (pendant l'exécution)
# Application locale :
http://localhost:4040

# Si plusieurs applications tournent simultanément :
http://localhost:4040    # 1ère application
http://localhost:4041    # 2ème application
http://localhost:4042    # 3ème application...

# Sur YARN :
http://<yarn-resourcemanager>:8088/proxy/<application-id>/

# Sur cluster Standalone :
http://<master-host>:8080       # Spark Master UI
http://<worker-host>:8081       # Worker UI
```

### 10.2 Spark History Server

Le Spark History Server permet de consulter les logs des applications **après leur fin**.

```bash
# Configurer le logging dans spark-defaults.conf
spark.eventLog.enabled   true
spark.eventLog.dir       /tmp/spark-events   # ou hdfs:///spark-events

# Créer le répertoire de logs
mkdir -p /tmp/spark-events

# Lancer le History Server
$SPARK_HOME/sbin/start-history-server.sh

# Accéder à l'interface
# http://localhost:18080

# Arrêter le History Server
$SPARK_HOME/sbin/stop-history-server.sh

# Lancer avec des options spécifiques
SPARK_HISTORY_OPTS="-Dspark.history.ui.port=18080 \
                    -Dspark.history.fs.logDirectory=/tmp/spark-events" \
$SPARK_HOME/sbin/start-history-server.sh
```

### 10.3 Contrôle du niveau de log

```bash
# En ligne de commande (appliqué à toute l'application)
pyspark --master local[*] --conf spark.logLevel=WARN
spark-submit --master local[*] --conf spark.logLevel=ERROR mon_script.py

# Dans le code Python
spark.sparkContext.setLogLevel("ERROR")  # ERROR, WARN, INFO, DEBUG
```

**Niveaux de log disponibles :**

| Niveau | Usage |
|---|---|
| `ERROR` | Uniquement les erreurs — sortie minimale |
| `WARN` | Erreurs + avertissements — **recommandé en développement** |
| `INFO` | Informations détaillées — verbeux mais utile pour le diagnostic |
| `DEBUG` | Très verbeux — pour déboguer Spark lui-même |

### 10.4 Personnaliser `log4j`

```bash
# Copier et éditer le template
cp $SPARK_HOME/conf/log4j2.properties.template \
   $SPARK_HOME/conf/log4j2.properties

# Réduire la verbosité globalement
cat >> $SPARK_HOME/conf/log4j2.properties << 'EOF'
rootLogger.level = WARN

# Garder INFO pour votre propre code
logger.myapp.name = com.monentreprise
logger.myapp.level = INFO

# Silencer les librairies très bavardes
logger.hadoop.name = org.apache.hadoop
logger.hadoop.level = ERROR
logger.hive.name = org.apache.hive
logger.hive.level = ERROR
EOF
```

### 10.5 Métriques et supervision

```bash
# Activer les métriques Prometheus
spark-submit \
    --conf spark.metrics.conf.*.sink.prometheus.class=\
org.apache.spark.metrics.sink.PrometheusServlet \
    --conf spark.metrics.conf.*.sink.prometheus.path=/metrics \
    mon_script.py

# Activer les métriques dans un fichier CSV (simple)
spark-submit \
    --conf spark.metrics.conf.*.sink.csv.class=\
org.apache.spark.metrics.sink.CsvSink \
    --conf spark.metrics.conf.*.sink.csv.directory=/tmp/spark-metrics \
    --conf spark.metrics.conf.*.sink.csv.period=10 \
    mon_script.py
```

---

## 11. Exemples complets de bout en bout

### 11.1 Pipeline ETL complet

```bash
# ── Créer les données d'entrée ─────────────────────────────────────────────────
mkdir -p /tmp/data/ventes
cat > /tmp/data/ventes/ventes.csv << 'EOF'
id,date,ville,produit,montant,quantite
1,2024-01-15,Paris,Laptop,999.99,2
2,2024-01-16,Lyon,Smartphone,599.99,3
3,2024-01-17,Paris,Tablette,299.99,1
4,2024-02-01,Nantes,Laptop,999.99,1
5,2024-02-02,Lyon,Casque,149.99,5
6,2024-02-15,Paris,Smartphone,599.99,2
7,2024-03-01,Bordeaux,Laptop,999.99,3
8,2024-03-10,Paris,Tablette,299.99,4
9,2024-03-15,Lyon,Laptop,999.99,1
10,2024-03-20,Nantes,Smartphone,599.99,2
EOF

# ── Créer le script ETL ────────────────────────────────────────────────────────
cat > /tmp/etl_ventes.py << 'PYTHON'
"""
Pipeline ETL : ventes CSV → rapport Parquet + stats console
Usage : spark-submit etl_ventes.py <input_path> <output_path>
"""
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

def main(input_path, output_path):
    spark = SparkSession.builder \
        .appName("ETL_Ventes") \
        .config("spark.sql.shuffle.partitions", "4") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print(f"\n{'='*50}")
    print("DÉMARRAGE DU PIPELINE ETL")
    print(f"{'='*50}")

    # 1. Lecture
    schema = StructType([
        StructField("id",       IntegerType(), False),
        StructField("date",     StringType(),  True),
        StructField("ville",    StringType(),  True),
        StructField("produit",  StringType(),  True),
        StructField("montant",  DoubleType(),  True),
        StructField("quantite", IntegerType(), True),
    ])
    df = spark.read \
        .option("header", "true") \
        .schema(schema) \
        .csv(input_path)
    print(f"\n[LECTURE] {df.count()} lignes chargées depuis {input_path}")

    # 2. Transformation
    df_enrichi = df \
        .withColumn("date",       F.to_date("date", "yyyy-MM-dd")) \
        .withColumn("mois",       F.date_format("date", "yyyy-MM")) \
        .withColumn("montant_ttc",F.round(F.col("montant") * F.col("quantite") * 1.2, 2)) \
        .withColumn("categorie",
            F.when(F.col("montant") > 500, "Premium")
             .otherwise("Standard")) \
        .filter(F.col("montant") > 0) \
        .dropDuplicates(["id"])

    # 3. Agrégation
    df_stats = df_enrichi \
        .groupBy("ville", "mois") \
        .agg(
            F.count("*").alias("nb_ventes"),
            F.round(F.sum("montant_ttc"), 2).alias("ca_ttc"),
            F.round(F.avg("montant"), 2).alias("prix_moy"),
        ) \
        .orderBy("ville", "mois")

    # 4. Affichage
    print("\n[RÉSULTATS] Chiffre d'affaires par ville et par mois :")
    df_stats.show(truncate=False)

    print("\n[RÉSULTATS] Top produits :")
    df_enrichi.groupBy("produit") \
        .agg(F.round(F.sum("montant_ttc"),2).alias("ca"),
             F.sum("quantite").alias("qte")) \
        .orderBy(F.col("ca").desc()) \
        .show()

    # 5. Écriture
    df_enrichi.write \
        .mode("overwrite") \
        .partitionBy("mois") \
        .parquet(f"{output_path}/detail/")

    df_stats.write \
        .mode("overwrite") \
        .option("header", "true") \
        .csv(f"{output_path}/stats/")

    print(f"\n[ÉCRITURE] Résultats sauvegardés dans {output_path}")
    print(f"{'='*50}\n")
    spark.stop()

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/data/ventes/",
         sys.argv[2] if len(sys.argv) > 2 else "/tmp/output/etl/")
PYTHON

# ── Exécuter le pipeline ───────────────────────────────────────────────────────
spark-submit \
    --master local[*] \
    --name "ETL Ventes" \
    --driver-memory 1g \
    --conf spark.sql.shuffle.partitions=4 \
    /tmp/etl_ventes.py \
    /tmp/data/ventes/ \
    /tmp/output/etl/

# ── Vérifier les sorties ───────────────────────────────────────────────────────
ls -lh /tmp/output/etl/detail/
ls -lh /tmp/output/etl/stats/
```

### 11.2 Job de streaming depuis des fichiers

```bash
cat > /tmp/streaming_job.py << 'PYTHON'
"""
Structured Streaming : surveiller un répertoire et agréger en temps réel.
Usage : spark-submit streaming_job.py
"""
import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

os.makedirs("/tmp/stream_input", exist_ok=True)
os.makedirs("/tmp/stream_checkpoint", exist_ok=True)

spark = SparkSession.builder \
    .appName("Streaming_Demo") \
    .master("local[3]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

schema = StructType([
    StructField("ville",   StringType(), True),
    StructField("montant", DoubleType(), True),
])

df_stream = spark.readStream \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .json("/tmp/stream_input/")

df_agg = df_stream \
    .groupBy("ville") \
    .agg(F.count("*").alias("n"),
         F.round(F.sum("montant"), 2).alias("ca"))

query = df_agg.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("checkpointLocation", "/tmp/stream_checkpoint/") \
    .trigger(once=True) \
    .start()

# Générer des données
import json, random, time
for i in range(3):
    villes = ["Paris","Lyon","Nantes"]
    data = [{"ville": random.choice(villes), "montant": round(random.uniform(10,500),2)}
            for _ in range(5)]
    with open(f"/tmp/stream_input/batch_{i}.json", "w") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")
    time.sleep(1)

query.awaitTermination(30)
spark.stop()
PYTHON

spark-submit \
    --master local[3] \
    --name "Streaming Demo" \
    /tmp/streaming_job.py
```

### 11.3 Benchmark de performances

```bash
cat > /tmp/benchmark.py << 'PYTHON'
"""
Benchmark : comparer les performances selon le nombre de partitions.
Usage : spark-submit benchmark.py
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time

spark = SparkSession.builder \
    .appName("Benchmark") \
    .master("local[*]") \
    .config("spark.sql.adaptive.enabled", "false") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

N = 5_000_000
df = spark.range(N) \
    .withColumn("valeur", (F.col("id") * 3.14).cast("double")) \
    .withColumn("groupe", (F.col("id") % 10).cast("string"))
df.cache()
df.count()

print(f"\nBenchmark sur {N:,} lignes — groupBy + sum")
print(f"{'Partitions':>12}  {'Temps (s)':>10}  {'Évaluation':>15}")
print("-" * 45)

for n_parts in [1, 2, spark.sparkContext.defaultParallelism,
                spark.sparkContext.defaultParallelism * 2, 50, 200]:
    spark.conf.set("spark.sql.shuffle.partitions", str(n_parts))
    t0 = time.time()
    df.groupBy("groupe").agg(F.sum("valeur"), F.count("*")).count()
    t = time.time() - t0
    eval_str = "✅ Optimal" if n_parts == spark.sparkContext.defaultParallelism * 2 \
               else ("❌ Trop" if n_parts > 100 else "")
    print(f"{n_parts:>12}  {t:>10.3f}  {eval_str:>15}")

spark.stop()
PYTHON

spark-submit --master local[*] --name "Benchmark" /tmp/benchmark.py
```

---

## 12. Référence rapide des options

### 12.1 Tableau de toutes les options `spark-submit`

```
Option                          Description
─────────────────────────────── ─────────────────────────────────────────────
--master URL                    URL du cluster (local[N], spark://, yarn, k8s)
--deploy-mode MODE              client (défaut) ou cluster
--name NOM                      Nom de l'application (Spark UI)
--class NOM_CLASSE              Classe principale (pour les JARs Scala/Java)
--jars JAR1,JAR2                JARs supplémentaires à distribuer
--packages gr:art:ver           Packages Maven (résolution automatique)
--exclude-packages gr:art       Exclure des packages transitifs
--repositories URL1,URL2        Repositories Maven additionnels
--py-files f1.py,f2.zip         Fichiers Python à distribuer aux Executors
--files f1,f2                   Fichiers à distribuer (accessibles via SparkFiles)
--archives arc1,arc2            Archives à extraire sur les Executors
--conf CLE=VALEUR               Configuration Spark (répétable)
--properties-file FICHIER       Fichier de propriétés Spark
--driver-memory MEM             Mémoire du Driver (ex: 2g, 512m)
--driver-cores N                Cœurs pour le Driver (cluster mode only)
--executor-memory MEM           Mémoire par Executor
--executor-cores N              Cœurs par Executor
--num-executors N               Nombre d'Executors (YARN/Standalone)
--total-executor-cores N        Total de cœurs (Standalone uniquement)
--driver-java-options "OPTS"    Options JVM pour le Driver
--driver-library-path "PATH"    Chemin des librairies natives du Driver
--driver-class-path "PATH"      Classpath du Driver
--executor-java-options "OPTS"  Options JVM pour les Executors
--queue NOM                     Queue YARN
--principal PRINCIPAL           Kerberos principal
--keytab FICHIER                Fichier Kerberos keytab
--verbose                       Sortie verbose
--version                       Afficher la version de Spark
--status ID                     Statut d'un job (Standalone)
--kill ID                       Annuler un job (Standalone)
--help                          Afficher l'aide
```

### 12.2 URLs de master

```
local             Mode local, 1 thread
local[N]          Mode local, N threads
local[*]          Mode local, tous les cœurs disponibles
local[N,M]        Mode local, N threads, M tentatives en cas d'échec
spark://HOST:PORT Cluster Standalone (port par défaut : 7077)
spark://H1:P,H2:P Multi-master Standalone (HA)
yarn              YARN (la config Hadoop doit être dans HADOOP_CONF_DIR)
k8s://HTTPS://URL Kubernetes
mesos://HOST:PORT Apache Mesos (déprécié)
```

### 12.3 Unités de mémoire

```
512m    512 mébioctets
2g      2 gibioctets
4g      4 gibioctets
8192m   8192 mébioctets (= 8 Go)
```

### 12.4 Commandes de gestion du cluster Standalone

```bash
# Démarrer le Master
$SPARK_HOME/sbin/start-master.sh
# → Accessible sur http://localhost:8080

# Démarrer un Worker (se connecte au Master)
$SPARK_HOME/sbin/start-worker.sh spark://localhost:7077

# Démarrer Master + Workers (tous définis dans conf/workers)
$SPARK_HOME/sbin/start-all.sh

# Arrêter tout
$SPARK_HOME/sbin/stop-all.sh

# Démarrer le History Server
$SPARK_HOME/sbin/start-history-server.sh

# Arrêter le History Server
$SPARK_HOME/sbin/stop-history-server.sh
```

### 12.5 Configurations Spark les plus utilisées

```bash
# ─── Performances ─────────────────────────────────────────────────────────────
spark.sql.shuffle.partitions=200          # Partitions après shuffle (défaut)
spark.sql.adaptive.enabled=true           # AQE — optimisation dynamique
spark.sql.adaptive.coalescePartitions.enabled=true
spark.sql.adaptive.skewJoin.enabled=true
spark.serializer=org.apache.spark.serializer.KryoSerializer

# ─── Mémoire ──────────────────────────────────────────────────────────────────
spark.driver.memory=2g
spark.executor.memory=4g
spark.memory.fraction=0.8               # % de la RAM utilisée pour le calcul
spark.memory.storageFraction=0.5        # % du précédent réservé au cache

# ─── Jointures ────────────────────────────────────────────────────────────────
spark.sql.autoBroadcastJoinThreshold=10mb   # Seuil broadcast automatique
spark.sql.broadcastTimeout=300              # Timeout du broadcast (secondes)

# ─── Parquet / IO ─────────────────────────────────────────────────────────────
spark.sql.parquet.compression.codec=snappy
spark.sql.parquet.filterPushdown=true
spark.sql.parquet.mergeSchema=false         # Désactiver pour les performances

# ─── UI et Monitoring ─────────────────────────────────────────────────────────
spark.ui.enabled=true
spark.ui.port=4040
spark.eventLog.enabled=true
spark.eventLog.dir=hdfs:///spark-events    # ou chemin local

# ─── Réseau ───────────────────────────────────────────────────────────────────
spark.network.timeout=120s
spark.executor.heartbeatInterval=10s
spark.rpc.message.maxSize=256              # Taille max des messages RPC (Mo)
```

---

> 💡 **Astuce finale** : les exemples de scripts officiels Spark sont une mine d'or.  
> Ils sont disponibles dans `$SPARK_HOME/examples/src/main/python/` et peuvent  
> être exécutés directement avec `spark-submit` pour valider une installation :
>
> ```bash
> spark-submit $SPARK_HOME/examples/src/main/python/pi.py 100
> # → Calcule π avec 100 itérations Monte-Carlo
> # → Pi is roughly 3.14...
>
> spark-submit $SPARK_HOME/examples/src/main/python/wordcount.py \
>     $SPARK_HOME/README.md
> # → Word count sur le README de Spark
> ```
