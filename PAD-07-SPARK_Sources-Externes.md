# Module 7 — Interaction avec des sources externes

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 6 — Partitionnement, shuffles et collecte

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Lire et écrire des fichiers locaux dans les formats courants (CSV, JSON, Parquet)
- Connecter Spark à Amazon S3 (et aux équivalents GCS, Azure Blob Storage)
- Interroger une base de données relationnelle via JDBC
- Comprendre les bases de Spark Structured Streaming avec Kafka
- Appréhender le rôle de Delta Lake comme couche transactionnelle

---

## 1. Vue d'ensemble des sources de données

Spark peut lire et écrire depuis une grande variété de sources, grâce à une **API unifiée** basée sur `spark.read` et `df.write`.

```
                        ┌──────────────────────┐
                        │      SPARK CLUSTER   │
  ┌──────────────┐      │                      │      ┌──────────────┐
  │ Fichiers     │◄────►│   spark.read.*()     │      │  Fichiers    │
  │ locaux/HDFS  │      │   df.write.*()       │◄────►│  de sortie   │
  └──────────────┘      │                      │      └──────────────┘
  ┌──────────────┐      │   DataFrame API      │      ┌──────────────┐
  │ Amazon S3    │◄────►│   Spark SQL          │◄────►│ Amazon S3    │
  │ GCS / Azure  │      │   Structured         │      │ GCS / Azure  │
  └──────────────┘      │   Streaming          │      └──────────────┘
  ┌──────────────┐      │                      │      ┌──────────────┐
  │ Bases SQL    │◄────►│   JDBC Connector     │◄────►│ Bases SQL    │
  │ (JDBC)       │      │                      │      │ (JDBC)       │
  └──────────────┘      │                      │      └──────────────┘
  ┌──────────────┐      │                      │
  │ Apache Kafka │◄────►│   Kafka Connector    │
  │ (streaming)  │      │                      │
  └──────────────┘      └──────────────────────┘
```

---

## 2. Lecture et écriture de fichiers locaux

### 2.1 L'API `spark.read` — options communes

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("SourcesExternes") \
    .master("local[*]") \
    .getOrCreate()

# Structure générale
df = spark.read \
    .format("csv") \           # Format : csv, json, parquet, orc, text...
    .option("clé", "valeur") \ # Options spécifiques au format
    .schema(mon_schema) \      # Schéma explicite (optionnel)
    .load("chemin/vers/data")  # Chemin local, HDFS, S3...

# Options communes à tous les formats
df = spark.read \
    .option("inferSchema", "true") \       # Déduction auto des types (coûteux !)
    .option("nullValue", "NULL") \         # Valeur à interpréter comme null
    .option("nanValue", "NaN") \           # Valeur à interpréter comme NaN
    .option("mode", "DROPMALFORMED") \     # Comportement si ligne invalide
    .load("data/")
    # mode : PERMISSIVE (défaut), DROPMALFORMED, FAILFAST
```

### 2.2 CSV

```python
# Lecture minimale
df = spark.read.csv("data/ventes.csv")

# Lecture complète avec options
df = spark.read \
    .option("header",       "true") \    # Première ligne = noms de colonnes
    .option("sep",          ";") \       # Séparateur (défaut : ",")
    .option("encoding",     "UTF-8") \   # Encodage
    .option("inferSchema",  "true") \    # Déduire les types automatiquement
    .option("quote",        '"') \       # Caractère de délimitation de chaîne
    .option("escape",       "\\") \      # Caractère d'échappement
    .option("multiLine",    "false") \   # Champs sur plusieurs lignes ?
    .option("dateFormat",   "dd/MM/yyyy") \  # Format des dates
    .option("timestampFormat", "dd/MM/yyyy HH:mm:ss") \
    .csv("data/ventes.csv")

# Lecture de plusieurs fichiers d'un coup
df = spark.read.option("header", "true").csv("data/ventes_*.csv")
df = spark.read.option("header", "true").csv("data/2024/", "data/2025/")

# ⚠️ Bonne pratique : toujours fournir un schéma explicite en production
#    inferSchema = true nécessite un second passage sur les données → lent
from pyspark.sql.types import *
schema = StructType([
    StructField("id",      IntegerType(), False),
    StructField("nom",     StringType(),  True),
    StructField("montant", DoubleType(),  True),
    StructField("date",    DateType(),    True),
])
df = spark.read.schema(schema).option("header", "true").csv("data/ventes.csv")

# Écriture CSV
df.write \
  .option("header", "true") \
  .option("sep", ";") \
  .option("encoding", "UTF-8") \
  .mode("overwrite") \
  .csv("output/ventes_export/")
```

### 2.3 JSON

```python
# Lecture JSON Lines (1 objet JSON par ligne — format standard Spark)
df = spark.read.json("data/utilisateurs.json")

# JSON multiligne (un seul objet JSON sur plusieurs lignes)
df = spark.read \
    .option("multiLine", "true") \
    .option("allowComments", "true") \       # Autoriser les commentaires //
    .option("allowUnquotedFieldNames", "true") \
    .json("data/config.json")

# JSON imbriqué : Spark crée automatiquement des colonnes StructType
df.printSchema()
# root
#  |-- id: long
#  |-- nom: string
#  |-- adresse: struct
#  |    |-- rue: string
#  |    |-- ville: string
#  |    |-- code_postal: string
#  |-- commandes: array
#  |    |-- element: struct
#  |    |    |-- produit: string
#  |    |    |-- montant: double

# Accéder aux champs imbriqués
df.select(
    "nom",
    F.col("adresse.ville").alias("ville"),
    F.col("adresse.code_postal").alias("cp"),
    F.explode("commandes").alias("commande")
).select("nom", "ville", "cp", "commande.produit", "commande.montant").show()

# Écriture JSON
df.write.mode("overwrite").json("output/utilisateurs_export/")
```

### 2.4 Parquet

```python
# Lecture Parquet (schéma embarqué → pas besoin d'inferSchema)
df = spark.read.parquet("data/transactions/")

# Lecture de plusieurs chemins
df = spark.read.parquet("data/2023/", "data/2024/")

# Lecture avec filtre poussé (partition pruning)
df = spark.read.parquet("data/ventes/") \
    .filter(F.col("pays") == "France") \
    .filter(F.col("annee") == 2024)
# → Spark ne lit que data/ventes/pays=France/annee=2024/

# Écriture Parquet avec compression et partitionnement
df.write \
  .option("compression", "snappy") \   # snappy, gzip, lz4, zstd, none
  .partitionBy("pays", "annee") \
  .mode("overwrite") \
  .parquet("output/ventes_parquet/")
```

### 2.5 Autres formats

```python
# ORC (Optimized Row Columnar — populaire dans l'écosystème Hive)
df = spark.read.orc("data/fichier.orc")
df.write.orc("output/")

# Texte brut (1 ligne = 1 Row avec colonne "value")
df = spark.read.text("data/roman.txt")
df.show()
# +------------------------------------------+
# |value                                     |
# +------------------------------------------+
# |Il était une fois dans un pays lointain...|

# Fichiers binaires (Spark 3.x)
df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load("data/images/")
# Colonnes : path, modificationTime, length, content (bytes)

# Format Avro (nécessite spark-avro)
df = spark.read.format("avro").load("data/fichier.avro")
```

---

## 3. Connexion à Amazon S3

### 3.1 Architecture S3 et Spark

Amazon S3 est un **stockage objet** (pas un système de fichiers) accessible via HTTPS. Spark y accède grâce au connecteur **hadoop-aws** qui implémente le protocole `s3a://`.

```
┌─────────────────────────────────────────────┐
│              SPARK CLUSTER                  │
│                                             │
│  Executor 1 ──────────────────────────────► │
│  Executor 2 ──────────── S3A Connector ───► │──► Amazon S3
│  Executor 3 ──────────────────────────────► │    (us-east-1)
│  (lecture parallèle depuis S3)              │
└─────────────────────────────────────────────┘

Protocoles S3 :
  s3://   → ancien protocole (Hadoop 2.x, à éviter)
  s3n://  → protocole intermédiaire (déprécié)
  s3a://  → protocole moderne, performant, recommandé
```

### 3.2 Configuration des credentials S3

**Méthode 1 — Variables d'environnement (recommandée en local) :**
```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
export AWS_DEFAULT_REGION=eu-west-1
```

**Méthode 2 — Configuration Spark (dans le code) :**
```python
spark = SparkSession.builder \
    .appName("S3Demo") \
    .master("local[*]") \
    .config("spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262") \
    .getOrCreate()

hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()

# Credentials (à ne jamais écrire en dur dans le code de production !)
hadoop_conf.set("fs.s3a.access.key",    "AKIAIOSFODNN7EXAMPLE")
hadoop_conf.set("fs.s3a.secret.key",    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY")
hadoop_conf.set("fs.s3a.endpoint",      "s3.amazonaws.com")
hadoop_conf.set("fs.s3a.region",        "eu-west-1")

# Implémentation du système de fichiers S3A
hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
```

**Méthode 3 — IAM Role (recommandée en production sur AWS) :**
```python
# Sur une instance EC2 ou EMR avec un IAM Role attaché,
# aucun credential n'est nécessaire : Spark utilise automatiquement le rôle IAM
hadoop_conf.set("fs.s3a.aws.credentials.provider",
                "com.amazonaws.auth.InstanceProfileCredentialsProvider")
```

**Méthode 4 — Fichier de configuration `spark-defaults.conf` :**
```properties
# $SPARK_HOME/conf/spark-defaults.conf
spark.hadoop.fs.s3a.access.key    AKIAIOSFODNN7EXAMPLE
spark.hadoop.fs.s3a.secret.key    wJalrXUtnFEMI/K7MDENG...
spark.hadoop.fs.s3a.endpoint      s3.eu-west-1.amazonaws.com
spark.hadoop.fs.s3a.region        eu-west-1
```

### 3.3 Lecture depuis S3

```python
# Lecture d'un fichier unique
df = spark.read.parquet("s3a://mon-bucket/data/transactions.parquet")

# Lecture d'un dossier complet
df = spark.read.parquet("s3a://mon-bucket/data/transactions/")

# Lecture avec wildcards
df = spark.read.csv("s3a://mon-bucket/logs/2024-*.csv")

# Lecture partitionnée (partition pruning automatique)
df = spark.read \
    .parquet("s3a://mon-bucket/ventes/") \
    .filter(F.col("annee") == 2024) \
    .filter(F.col("pays") == "France")
# Spark lit uniquement : s3a://mon-bucket/ventes/annee=2024/pays=France/

# Lecture JSON depuis S3
df_logs = spark.read \
    .option("multiLine", "false") \
    .json("s3a://mon-bucket/logs/app-*.json")
```

### 3.4 Écriture vers S3

```python
# Écriture Parquet partitionnée sur S3
df.write \
  .mode("overwrite") \
  .option("compression", "snappy") \
  .partitionBy("annee", "mois") \
  .parquet("s3a://mon-bucket/output/ventes/")

# Écriture CSV sur S3
df.coalesce(1) \
  .write \
  .option("header", "true") \
  .mode("overwrite") \
  .csv("s3a://mon-bucket/exports/rapport_mensuel/")
```

### 3.5 Optimisations spécifiques à S3

S3 n'est pas un système de fichiers traditionnel : il a une **latence de listage élevée** et des contraintes de cohérence qui peuvent affecter les performances.

```python
# ── Activer le committer S3A Magic (écriture atomique, Spark 3.x) ─────────────
hadoop_conf.set("fs.s3a.committer.magic.enabled", "true")
spark.conf.set("spark.sql.sources.commitProtocolClass",
    "org.apache.spark.internal.io.cloud.PathOutputCommitProtocol")
spark.conf.set("spark.sql.parquet.output.committer.class",
    "org.apache.spark.internal.io.cloud.BindingParquetOutputCommitter")

# ── Augmenter les threads de chargement S3 ────────────────────────────────────
hadoop_conf.set("fs.s3a.threads.max",        "64")
hadoop_conf.set("fs.s3a.connection.maximum", "64")
hadoop_conf.set("fs.s3a.block.size",         "134217728")  # 128 Mo

# ── Activer le cache de métadonnées S3 ───────────────────────────────────────
hadoop_conf.set("fs.s3a.metadatastore.impl",
    "org.apache.hadoop.fs.s3a.s3guard.DynamoDBMetadataStore")

# ── Utiliser S3 Select (pushdown de filtres côté S3) ─────────────────────────
spark.conf.set("spark.sql.parquet.filterPushdown",  "true")
spark.conf.set("spark.sql.parquet.recordFiltering", "true")
```

### 3.6 GCS (Google Cloud Storage) et Azure Blob Storage

La même logique s'applique aux autres clouds — seuls le protocole et les credentials changent.

**Google Cloud Storage :**
```python
# Protocole : gs://
spark = SparkSession.builder \
    .config("spark.jars.packages",
            "com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.15") \
    .getOrCreate()

hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
hadoop_conf.set("fs.gs.impl",
    "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
hadoop_conf.set("google.cloud.auth.service.account.enable", "true")
hadoop_conf.set("google.cloud.auth.service.account.json.keyfile",
    "/chemin/vers/service-account.json")

df = spark.read.parquet("gs://mon-bucket-gcs/data/")
```

**Azure Blob Storage / ADLS Gen2 :**
```python
# Protocole : wasbs:// (Blob) ou abfss:// (ADLS Gen2)
hadoop_conf.set(
    "fs.azure.account.key.moncompte.blob.core.windows.net",
    "ma_cle_de_compte_azure=="
)

df = spark.read.parquet(
    "wasbs://mon-conteneur@moncompte.blob.core.windows.net/data/"
)

# ADLS Gen2 avec OAuth
hadoop_conf.set("fs.azure.account.auth.type.moncompte.dfs.core.windows.net",
    "OAuth")
hadoop_conf.set("fs.azure.account.oauth.provider.type.moncompte.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider")
hadoop_conf.set("fs.azure.account.oauth2.client.id.moncompte.dfs.core.windows.net",
    "mon-client-id")

df = spark.read.parquet(
    "abfss://mon-conteneur@moncompte.dfs.core.windows.net/data/"
)
```

---

## 4. Connexion à des bases de données via JDBC

### 4.1 Principe du connecteur JDBC

JDBC (*Java Database Connectivity*) est l'interface standard Java pour communiquer avec des bases de données relationnelles. Spark peut **lire et écrire** dans n'importe quelle base disposant d'un driver JDBC.

```
SPARK CLUSTER                    BASE DE DONNÉES
┌────────────┐   JDBC / SQL     ┌────────────────────┐
│  Driver    │◄────────────────►│  PostgreSQL        │
│            │                  │  MySQL / MariaDB   │
│  Executors │◄────────────────►│  Oracle            │
│  (lecture  │   Partitionnée   │  SQL Server        │
│  parallèle)│                  │  Snowflake         │
└────────────┘                  │  Redshift          │
                                └────────────────────┘
```

### 4.2 Configuration et dépendances

```bash
# Ajouter le driver JDBC au classpath Spark
# Option 1 : via spark-submit
spark-submit --jars /chemin/vers/postgresql-42.6.0.jar mon_script.py

# Option 2 : via la variable d'environnement
export SPARK_CLASSPATH=/chemin/vers/postgresql-42.6.0.jar

# Option 3 : via les packages Maven (téléchargement automatique)
pyspark --packages org.postgresql:postgresql:42.6.0
```

```python
# Dans le code
spark = SparkSession.builder \
    .config("spark.jars.packages", "org.postgresql:postgresql:42.6.0") \
    .getOrCreate()
```

**Drivers JDBC courants :**

| Base de données | Package Maven | Classe driver |
|---|---|---|
| PostgreSQL | `org.postgresql:postgresql:42.6.0` | `org.postgresql.Driver` |
| MySQL | `com.mysql:mysql-connector-j:8.1.0` | `com.mysql.cj.jdbc.Driver` |
| SQL Server | `com.microsoft.sqlserver:mssql-jdbc:12.4.0.jre11` | `com.microsoft.sqlserver.jdbc.SQLServerDriver` |
| Oracle | `com.oracle.database.jdbc:ojdbc11:23.2.0.0` | `oracle.jdbc.OracleDriver` |
| Snowflake | `net.snowflake:spark-snowflake_2.12:2.12.0` | `net.snowflake.client.jdbc.SnowflakeDriver` |

### 4.3 Lecture depuis une base de données

```python
# Paramètres de connexion
jdbc_url  = "jdbc:postgresql://mon-serveur:5432/ma_base"
properties = {
    "user":     "mon_utilisateur",
    "password": "mon_mot_de_passe",
    "driver":   "org.postgresql.Driver"
}

# ── Lecture d'une table entière ───────────────────────────────────────────────
df = spark.read.jdbc(
    url=jdbc_url,
    table="ventes",
    properties=properties
)
# ⚠️ Lecture séquentielle en 1 seule partition → lent pour les grandes tables !

# ── Lecture parallèle (recommandée pour les grandes tables) ───────────────────
df = spark.read.jdbc(
    url=jdbc_url,
    table="ventes",
    column="id",           # Colonne de partitionnement (numérique ou date)
    lowerBound=1,          # Valeur minimale
    upperBound=10_000_000, # Valeur maximale
    numPartitions=20,      # Nombre de partitions (= requêtes parallèles)
    properties=properties
)
# → Spark exécute 20 requêtes SQL en parallèle :
#   SELECT * FROM ventes WHERE id >= 1        AND id < 500001
#   SELECT * FROM ventes WHERE id >= 500001   AND id < 1000001
#   ...

# ── Lecture avec une requête SQL personnalisée ────────────────────────────────
query = "(SELECT id, nom, montant FROM ventes WHERE annee = 2024) AS ventes_2024"
df = spark.read.jdbc(url=jdbc_url, table=query, properties=properties)

# ── Lecture partitionnée par une colonne de type date ────────────────────────
df = spark.read.jdbc(
    url=jdbc_url,
    table="transactions",
    column="date_transaction",
    lowerBound="2024-01-01",
    upperBound="2024-12-31",
    numPartitions=12,
    properties=properties
)
```

### 4.4 Écriture vers une base de données

```python
# ── Écriture simple ──────────────────────────────────────────────────────────
df.write.jdbc(
    url=jdbc_url,
    table="resultats_ml",
    mode="overwrite",       # overwrite, append, ignore, error
    properties=properties
)

# ── Écriture avec options avancées ───────────────────────────────────────────
df.write.jdbc(
    url=jdbc_url,
    table="resultats_ml",
    mode="append",
    properties={
        **properties,
        "batchsize":     "10000",    # Nombre de lignes par batch INSERT
        "isolationLevel":"READ_COMMITTED",
        "truncate":      "false",    # Truncate la table si mode=overwrite
    }
)

# ⚠️ Chaque Executor effectue ses propres INSERT → peut surcharger la base
# Conseils :
# → Réduire le nombre de partitions avant l'écriture : df.coalesce(4).write.jdbc(...)
# → Augmenter batchsize pour réduire le nombre de requêtes
# → Préférer écrire en Parquet sur S3 puis charger via COPY (Redshift, Snowflake)
```

### 4.5 Bonnes pratiques JDBC

```python
# ✅ Ne jamais stocker les credentials en dur dans le code
import os
properties = {
    "user":     os.environ["DB_USER"],
    "password": os.environ["DB_PASSWORD"],
    "driver":   "org.postgresql.Driver"
}

# ✅ Utiliser des secrets managers en production (AWS Secrets Manager, Vault...)
import boto3, json
client = boto3.client("secretsmanager", region_name="eu-west-1")
secret = json.loads(client.get_secret_value(SecretId="prod/spark/db")["SecretString"])
properties = {"user": secret["username"], "password": secret["password"], ...}

# ✅ Toujours filtrer côté base avant de charger en Spark
query = "(SELECT * FROM ventes WHERE annee = 2024 AND montant > 100) AS t"
df = spark.read.jdbc(url=jdbc_url, table=query, properties=properties)

# ✅ Indexer la colonne de partitionnement dans la base de données
# CREATE INDEX idx_ventes_id ON ventes(id);   → lectures parallèles efficaces
```

---

## 5. Introduction à Spark Structured Streaming

### 5.1 Principe du streaming avec Spark

Spark Structured Streaming permet de traiter des **flux de données en temps réel** en utilisant exactement la même API que les DataFrames batch. Un flux est modélisé comme une **table infinie** à laquelle de nouvelles lignes sont ajoutées en continu.

```
                    Flux de données (Kafka, fichiers, socket...)
                         │    │    │    │    │
                         ▼    ▼    ▼    ▼    ▼
                    ┌────────────────────────────┐
                    │   Table infinie (unbounded) │
                    │                            │
                    │  t=0 : [ligne1, ligne2]    │
                    │  t=1 : [ligne3]            │
                    │  t=2 : [ligne4, ligne5]    │
                    │  ...                       │
                    └────────────────────────────┘
                              │
                              ▼ Transformations (identiques au batch)
                              │ filter(), groupBy(), join()...
                              ▼
                    ┌────────────────────────────┐
                    │   Table de résultats       │
                    │   (mise à jour continue)   │
                    └────────────────────────────┘
                              │
                              ▼ Sink (sortie)
                    Console, Fichiers, Kafka, JDBC...
```

### 5.2 Modes de déclenchement (*trigger*)

| Mode | Description | Usage |
|---|---|---|
| `Trigger.Once()` | Traite toutes les données disponibles, puis s'arrête | Migration batch/stream |
| `Trigger.AvailableNow()` | Idem, en multi-batch (Spark 3.3+) | Pipelines micro-batch |
| `Trigger.ProcessingTime("1 minute")` | Déclenche toutes les N secondes/minutes | Temps quasi-réel |
| `Trigger.Continuous("1 second")` | Latence très faible (~ms) — expérimental | Temps réel strict |

### 5.3 Lecture d'un flux Kafka

```python
# Dépendance nécessaire
# spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0

df_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka-broker-1:9092,kafka-broker-2:9092") \
    .option("subscribe", "topic-ventes") \           # 1 topic
    .option("subscribePattern", "topic-.*") \        # Pattern regex de topics
    .option("startingOffsets", "earliest") \         # earliest, latest, ou JSON d'offsets
    .option("maxOffsetsPerTrigger", "10000") \       # Limite par batch
    .option("kafka.security.protocol", "SASL_SSL") \ # Sécurité (si nécessaire)
    .load()

# Structure d'un message Kafka dans Spark :
# +----+--------------------+-----+---------+------+--------------------+-------------+
# | key|               value|topic|partition|offset|           timestamp|timestampType|
# +----+--------------------+-----+---------+------+--------------------+-------------+
# Toutes les colonnes sont de type binaire (bytes)

# Désérialiser le contenu (supposé être du JSON)
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema_vente = StructType([
    StructField("id",      StringType(),  True),
    StructField("produit", StringType(),  True),
    StructField("montant", DoubleType(),  True),
    StructField("ville",   StringType(),  True),
])

df_ventes = df_stream \
    .select(F.col("value").cast("string").alias("json_str")) \
    .select(F.from_json(F.col("json_str"), schema_vente).alias("data")) \
    .select("data.*")
```

### 5.4 Transformations et agrégations sur un flux

```python
# Agrégation par fenêtre temporelle glissante
df_agg = df_ventes \
    .withColumn("timestamp", F.current_timestamp()) \
    .groupBy(
        F.window(F.col("timestamp"), "10 minutes", "5 minutes"),  # fenêtre de 10min, glissant de 5min
        F.col("ville")
    ) \
    .agg(
        F.sum("montant").alias("ca_fenetre"),
        F.count("*").alias("nb_transactions")
    )

# Agrégation simple (sans fenêtre)
df_total = df_ventes.groupBy("produit").sum("montant")
```

### 5.5 Modes de sortie (*output mode*)

| Mode | Description | Contraintes |
|---|---|---|
| `"append"` | Écrit uniquement les nouvelles lignes | Pas d'agrégation sur état global |
| `"complete"` | Réécrit la table entière à chaque batch | Nécessite une agrégation |
| `"update"` | Écrit uniquement les lignes modifiées | Certains sinks seulement |

### 5.6 Écriture du flux (sinks)

```python
# ── Sink Console (développement/debug) ───────────────────────────────────────
query = df_agg.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("truncate", "false") \
    .trigger(processingTime="30 seconds") \
    .start()

# ── Sink Fichiers Parquet (production) ───────────────────────────────────────
query = df_ventes.writeStream \
    .outputMode("append") \
    .format("parquet") \
    .option("path",         "s3a://mon-bucket/stream-output/") \
    .option("checkpointLocation", "s3a://mon-bucket/checkpoints/ventes/") \
    .trigger(processingTime="1 minute") \
    .start()

# ── Sink Kafka (republier vers un autre topic) ────────────────────────────────
df_ventes \
    .select(F.to_json(F.struct("*")).alias("value")) \
    .writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("topic", "topic-ventes-enrichies") \
    .option("checkpointLocation", "/tmp/checkpoints/") \
    .start()

# ── Sink JDBC (base de données) ───────────────────────────────────────────────
def ecrire_en_base(batch_df, batch_id):
    """foreachBatch : traite chaque micro-batch comme un DataFrame batch"""
    batch_df.write.jdbc(
        url=jdbc_url,
        table="stream_resultats",
        mode="append",
        properties=properties
    )

query = df_agg.writeStream \
    .outputMode("update") \
    .foreachBatch(ecrire_en_base) \
    .option("checkpointLocation", "/tmp/checkpoints/agg/") \
    .start()

# Attendre la fin (ou interruption)
query.awaitTermination()
```

### 5.7 Gestion de l'état et checkpointing

```python
# Le checkpointLocation est OBLIGATOIRE pour la reprise après échec
# Il stocke :
#   - Les offsets Kafka traités
#   - L'état des agrégations (pour les fenêtres temporelles)
#   - Les métadonnées de progression

query = df_stream.writeStream \
    .option("checkpointLocation", "s3a://mon-bucket/checkpoints/mon-job/") \
    .start()

# Gérer un stream actif
print(f"Status : {query.status}")
print(f"Dernière progression : {query.lastProgress}")
query.stop()   # Arrêter proprement
```

---

## 6. Delta Lake — couche transactionnelle sur fichiers

### 6.1 Pourquoi Delta Lake ?

Les formats de fichiers classiques (Parquet, ORC) ne supportent pas nativement :
- Les **transactions ACID** (une écriture partielle peut corrompre les données)
- Les opérations **UPDATE** et **DELETE** sur des fichiers existants
- Le **schéma évolutif** contrôlé
- Le **time travel** (lire les données à un instant passé)

**Delta Lake** résout ces problèmes en ajoutant un **journal de transactions** (`_delta_log/`) à côté des fichiers Parquet.

```
output/ventes_delta/
├── _delta_log/
│   ├── 00000000000000000000.json   ← transaction 0 : création initiale
│   ├── 00000000000000000001.json   ← transaction 1 : INSERT
│   ├── 00000000000000000002.json   ← transaction 2 : UPDATE
│   └── ...
├── part-00000-xxxx.parquet
├── part-00001-xxxx.parquet
└── ...
```

### 6.2 Utilisation de base

```bash
# Installation
pip install delta-spark==3.1.0
```

```python
spark = SparkSession.builder \
    .appName("DeltaLake") \
    .config("spark.jars.packages", "io.delta:delta-spark_2.12:3.1.0") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog",
            "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .getOrCreate()

# Écriture au format Delta
df.write.format("delta").mode("overwrite").save("output/ventes_delta/")

# Lecture au format Delta
df_delta = spark.read.format("delta").load("output/ventes_delta/")
```

### 6.3 Opérations ACID

```python
from delta.tables import DeltaTable

dt = DeltaTable.forPath(spark, "output/ventes_delta/")

# ── UPDATE ────────────────────────────────────────────────────────────────────
dt.update(
    condition=F.col("ville") == "paris",
    set={"ville": F.lit("Paris")}   # Normalisation de la casse
)

# ── DELETE ────────────────────────────────────────────────────────────────────
dt.delete(condition=F.col("montant") < 0)

# ── MERGE (Upsert) ────────────────────────────────────────────────────────────
df_mises_a_jour = spark.createDataFrame([
    (1, "Alice", "Paris",  1800.0),  # Mise à jour
    (7, "Hugo",  "Toulouse", 900.0), # Nouveau
], ["id", "nom", "ville", "montant"])

dt.alias("existant") \
  .merge(
      df_mises_a_jour.alias("maj"),
      "existant.id = maj.id"
  ) \
  .whenMatchedUpdateAll() \     # Si l'id existe → UPDATE
  .whenNotMatchedInsertAll() \  # Si l'id est nouveau → INSERT
  .execute()
```

### 6.4 Time travel

```python
# Lire une version précédente (par numéro de version)
df_v0 = spark.read.format("delta") \
    .option("versionAsOf", 0) \
    .load("output/ventes_delta/")

# Lire à un instant passé (par timestamp)
df_hier = spark.read.format("delta") \
    .option("timestampAsOf", "2024-01-15 10:00:00") \
    .load("output/ventes_delta/")

# Voir l'historique des transactions
dt.history().select("version", "timestamp", "operation", "operationParameters").show()

# Restaurer une version précédente
dt.restoreToVersion(2)
dt.restoreToTimestamp("2024-01-15 10:00:00")
```

---

## 7. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import os

spark = SparkSession.builder \
    .appName("SourcesExternes") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# ─── 1. Lecture / Écriture locale ────────────────────────────────────────────
print("=" * 60)
print("=== 1. Fichiers locaux ===")
print("=" * 60)

# Créer un jeu de données de test
data = [
    (1, "Alice",    "Paris",   1500.0, "2024-01"),
    (2, "Bob",      "Lyon",     800.0, "2024-01"),
    (3, "Claire",   "Paris",   2200.0, "2024-02"),
    (4, "David",    "Lyon",    1100.0, "2024-02"),
    (5, "Emma",     "Nantes",   950.0, "2024-03"),
    (6, "François", "Paris",   3000.0, "2024-03"),
]
schema = StructType([
    StructField("id",      IntegerType(), False),
    StructField("nom",     StringType(),  True),
    StructField("ville",   StringType(),  True),
    StructField("montant", DoubleType(),  True),
    StructField("mois",    StringType(),  True),
])
df = spark.createDataFrame(data, schema)

# Écriture dans différents formats
os.makedirs("/tmp/spark_formats", exist_ok=True)
df.write.mode("overwrite").option("header","true").csv(    "/tmp/spark_formats/csv/")
df.write.mode("overwrite").json(                           "/tmp/spark_formats/json/")
df.write.mode("overwrite").option("compression","snappy").parquet("/tmp/spark_formats/parquet/")

# Lecture et comparaison
df_csv     = spark.read.option("header","true").option("inferSchema","true").csv("/tmp/spark_formats/csv/")
df_json    = spark.read.json("/tmp/spark_formats/json/")
df_parquet = spark.read.parquet("/tmp/spark_formats/parquet/")

print(f"\nCSV     : {df_csv.count()} lignes, {len(df_csv.columns)} colonnes")
print(f"JSON    : {df_json.count()} lignes, {len(df_json.columns)} colonnes")
print(f"Parquet : {df_parquet.count()} lignes, {len(df_parquet.columns)} colonnes")
print("\nSchéma Parquet (types préservés) :")
df_parquet.printSchema()

# Écriture partitionnée
df.write \
  .mode("overwrite") \
  .partitionBy("ville") \
  .parquet("/tmp/spark_formats/parquet_partitionne/")

# Lecture avec filtre → partition pruning
df_paris = spark.read \
    .parquet("/tmp/spark_formats/parquet_partitionne/") \
    .filter(F.col("ville") == "Paris")
print(f"\nLignes Paris (partition pruning) : {df_paris.count()}")
df_paris.explain()   # Vérifier que seule la partition Paris est lue

# ─── 2. Streaming simulé (source fichier) ────────────────────────────────────
print("\n" + "=" * 60)
print("=== 2. Structured Streaming (source fichier) ===")
print("=" * 60)

# Simuler un stream depuis un répertoire (Spark surveille les nouveaux fichiers)
os.makedirs("/tmp/spark_stream_input",  exist_ok=True)
os.makedirs("/tmp/spark_stream_output", exist_ok=True)
os.makedirs("/tmp/spark_checkpoints",   exist_ok=True)

# Écrire quelques fichiers de "données en arrivée"
for i, (ville, montant) in enumerate([("Paris", 500), ("Lyon", 300), ("Paris", 800)]):
    spark.createDataFrame(
        [(i, ville, float(montant))], ["id", "ville", "montant"]
    ).write.mode("overwrite").json(f"/tmp/spark_stream_input/batch_{i}.json")

# Définir le schéma du stream
schema_stream = StructType([
    StructField("id",      IntegerType(), True),
    StructField("ville",   StringType(),  True),
    StructField("montant", DoubleType(),  True),
])

# Lire comme un stream
df_stream = spark.readStream \
    .schema(schema_stream) \
    .option("maxFilesPerTrigger", 1) \   # 1 fichier par batch
    .json("/tmp/spark_stream_input/")

# Transformation : CA par ville
df_agg_stream = df_stream \
    .groupBy("ville") \
    .agg(F.sum("montant").alias("ca_cumule"), F.count("*").alias("nb"))

# Écriture en mode complete vers la console (pour la démo)
query = df_agg_stream.writeStream \
    .outputMode("complete") \
    .format("console") \
    .option("checkpointLocation", "/tmp/spark_checkpoints/demo/") \
    .trigger(once=True) \   # Traiter une seule fois puis s'arrêter
    .start()

query.awaitTermination(timeout=30)
print("Stream terminé.")

# ─── 3. Simulation S3 avec MinIO local ───────────────────────────────────────
# Note : en environnement réel, remplacer "s3a://..." par votre bucket S3
# La configuration hadoop_conf serait ajoutée ici (cf. section 3.2)
print("\n" + "=" * 60)
print("=== 3. Configuration S3 (référence) ===")
print("=" * 60)
print("""
Configuration S3 à ajouter à la SparkSession :

  hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
  hadoop_conf.set("fs.s3a.access.key",    os.environ["AWS_ACCESS_KEY_ID"])
  hadoop_conf.set("fs.s3a.secret.key",    os.environ["AWS_SECRET_ACCESS_KEY"])
  hadoop_conf.set("fs.s3a.endpoint",      "s3.eu-west-1.amazonaws.com")

  # Lecture
  df = spark.read.parquet("s3a://mon-bucket/data/ventes/")

  # Écriture
  df.write.mode("overwrite").parquet("s3a://mon-bucket/output/resultats/")
""")

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **API unifiée** | `spark.read` / `df.write` fonctionnent identiquement pour tous les formats et sources |
| **Parquet** | Format recommandé : schéma embarqué, colonnaire, compressé, partition pruning |
| **`inferSchema`** | Pratique mais coûteux — toujours préférer un schéma explicite en production |
| **S3 `s3a://`** | Protocole moderne pour S3 — configurer `hadoop-aws` et les credentials |
| **Credentials** | Jamais en dur dans le code — variables d'environnement ou IAM Role |
| **JDBC parallèle** | Spécifier `column`, `lowerBound`, `upperBound`, `numPartitions` pour éviter le goulot séquentiel |
| **Structured Streaming** | Même API que le batch — source → transformations → sink |
| **Checkpoint** | Obligatoire pour la reprise après échec d'un stream |
| **Delta Lake** | ACID + UPDATE/DELETE + time travel sur des fichiers Parquet |

---

## Exercices

### Exercice 1 — Formats de fichiers (25 min)
> 1. Créer un DataFrame de 50 000 lignes avec des données réalistes
> 2. L'écrire en CSV, JSON et Parquet (avec compression `snappy` et `gzip`)
> 3. Comparer les tailles sur disque et les temps de lecture pour chaque format
> 4. Lire le Parquet avec un filtre et vérifier le partition pruning dans `explain()`

### Exercice 2 — Connexion JDBC (35 min)
> 1. Installer PostgreSQL en local (ou utiliser Docker : `docker run -e POSTGRES_PASSWORD=test -p 5432:5432 postgres`)
> 2. Créer une table `clients` et y insérer 100 000 lignes
> 3. Lire la table depuis Spark avec une seule partition, puis avec 8 partitions parallèles
> 4. Comparer les temps de lecture et les plans d'exécution
> 5. Écrire le résultat d'une agrégation Spark dans une nouvelle table PostgreSQL

### Exercice 3 — Structured Streaming (40 min)
> 1. Créer un répertoire surveillé par Spark comme source de stream
> 2. Écrire un script Python qui génère un nouveau fichier JSON toutes les 5 secondes (simuler un flux d'événements)
> 3. Configurer un stream Spark qui lit ce répertoire, calcule le CA par ville dans une fenêtre de 30 secondes, et écrit les résultats en console
> 4. Observer les micro-batches et les mises à jour en mode `complete`

### Exercice 4 — Delta Lake (30 min)
> 1. Écrire un DataFrame en format Delta
> 2. Effectuer un UPDATE sur une colonne
> 3. Effectuer un MERGE entre les données existantes et un DataFrame de mises à jour
> 4. Afficher l'historique des transactions
> 5. Lire les données à la version 0 (avant les modifications) et comparer avec la version courante

---

## Pour aller plus loin

- 📖 **Spark Data Sources** : [spark.apache.org/docs/latest/sql-data-sources.html](https://spark.apache.org/docs/latest/sql-data-sources.html)
- 📖 **S3A Guide** : [hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html](https://hadoop.apache.org/docs/stable/hadoop-aws/tools/hadoop-aws/index.html)
- 📖 **Structured Streaming** : [spark.apache.org/docs/latest/structured-streaming-programming-guide.html](https://spark.apache.org/docs/latest/structured-streaming-programming-guide.html)
- 📖 **Delta Lake** : [docs.delta.io](https://docs.delta.io)
- 🛠️ **MinIO** : équivalent S3 open source pour tester en local — [min.io](https://min.io)
- 🛠️ **Kafka avec Docker** : `docker-compose` avec Zookeeper + Kafka + UI pour les TPs streaming

---

*Module précédent → **Module 6 : Partitionnement, shuffles et collecte***  
*Module suivant → **Module 8 : Feature Engineering avec PySpark***
