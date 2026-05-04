# Module 3 — Les structures de données Spark

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 2 — Installation et prise en main

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Comprendre le concept de RDD et ses propriétés fondamentales
- Créer et manipuler des DataFrames et des RDDs
- Distinguer les cas d'usage de chaque structure de données
- Utiliser Spark SQL pour interroger des données structurées
- Lire et interpréter un plan d'exécution Spark (DAG)

---

## 1. Vue d'ensemble des structures de données

Spark propose trois niveaux d'abstraction pour représenter des données distribuées, du plus bas niveau au plus haut niveau :

```
Niveau d'abstraction
        ▲
  Haut  │  ┌──────────────────────────────────────────┐
        │  │  Dataset[T]  (API Scala/Java typée)       │
        │  ├──────────────────────────────────────────┤
        │  │  DataFrame   (= Dataset[Row], PySpark)   │
        │  ├──────────────────────────────────────────┤
  Bas   │  │  RDD[T]      (Resilient Distributed      │
        │  │               Dataset)                   │
        ▼  └──────────────────────────────────────────┘

Plus le niveau est élevé :
  → Plus l'API est expressive et concise
  → Plus l'optimiseur Catalyst peut agir
  → Meilleures performances en général
```

---

## 2. Le RDD — Resilient Distributed Dataset

### 2.1 Définition et propriétés fondamentales

Le **RDD** est la structure de données primitive de Spark, introduite dès la version 1.0. Toutes les structures de plus haut niveau (DataFrame, Dataset) reposent sur le RDD en interne.

Un RDD est défini par cinq propriétés :

| Propriété | Description |
|---|---|
| **Resilient** (tolérant aux pannes) | Si une partition est perdue, Spark peut la recalculer grâce au lignage (*lineage*) |
| **Distributed** (distribué) | Les données sont découpées en **partitions** réparties sur les nœuds du cluster |
| **Dataset** (jeu de données) | Collection d'éléments pouvant être de n'importe quel type Python |
| **Immuable** | Un RDD ne peut pas être modifié — on en crée un nouveau à chaque transformation |
| **Lazily evaluated** | Les transformations ne sont pas exécutées immédiatement |

### 2.2 Le lignage (*lineage*) et la tolérance aux pannes

Le lignage est le **graphe des transformations** qui permet de reconstruire un RDD perdu. Spark ne duplique pas les données comme HDFS : il mémorise comment les recalculer.

```
rdd1 = sc.textFile("data.txt")          ← source (lecture disque)
rdd2 = rdd1.filter(lambda x: len(x) > 0)   ← transformation
rdd3 = rdd2.map(lambda x: x.upper())       ← transformation

# Si une partition de rdd3 est perdue sur un nœud :
# Spark relit les données correspondantes depuis data.txt
# et réapplique filter() puis map() — uniquement pour cette partition
```

### 2.3 Création d'un RDD

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.master("local[*]").appName("RDD").getOrCreate()
sc = spark.sparkContext

# ── Méthode 1 : paralléliser une collection Python ──────────────────────────
rdd1 = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8])
rdd1_partitionne = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8], numSlices=4)

print(f"Nombre de partitions : {rdd1.getNumPartitions()}")

# ── Méthode 2 : lire un fichier texte ───────────────────────────────────────
rdd_texte = sc.textFile("data/roman.txt")          # 1 ligne = 1 élément
rdd_textes = sc.textFile("data/*.txt")             # Plusieurs fichiers

# ── Méthode 3 : lire des fichiers clé-valeur ────────────────────────────────
rdd_kv = sc.sequenceFile("data/fichier.seq")

# ── Méthode 4 : depuis un DataFrame (conversion) ────────────────────────────
df = spark.read.csv("data/ventes.csv", header=True)
rdd_depuis_df = df.rdd   # Chaque élément est un objet Row
```

### 2.4 Types de RDD

```python
# RDD non typé : éléments de type quelconque
rdd_entiers  = sc.parallelize([1, 2, 3, 4, 5])
rdd_chaines  = sc.parallelize(["bonjour", "monde", "spark"])
rdd_tuples   = sc.parallelize([("Alice", 30), ("Bob", 25)])

# RDD Pair (Paire clé-valeur) : élément = (clé, valeur)
# Donne accès à des transformations spéciales : reduceByKey, groupByKey...
rdd_pair = sc.parallelize([
    ("Paris",  1500.0),
    ("Lyon",    800.0),
    ("Paris",  2200.0),
    ("Lyon",   1100.0),
])
```

### 2.5 Opérations sur les RDD : transformations vs actions

Les opérations sur un RDD sont de deux types :

**Transformations** (lazy — retournent un nouveau RDD, rien n'est calculé) :
```python
rdd2 = rdd1.map(lambda x: x * 2)
rdd3 = rdd2.filter(lambda x: x > 4)
rdd4 = rdd3.flatMap(lambda x: [x, x+1])
```

**Actions** (eager — déclenchent l'exécution du DAG et retournent un résultat) :
```python
rdd3.count()                # Nombre d'éléments
rdd3.collect()              # Rapatrie tous les éléments au Driver
rdd3.take(5)                # Rapatrie les 5 premiers éléments
rdd3.first()                # Premier élément
rdd3.reduce(lambda a, b: a + b)   # Agrégation
rdd3.saveAsTextFile("output/")    # Écriture sur disque
```

> ⚠️ **Règle d'or** : ne jamais appeler `.collect()` sur un RDD volumineux — cela rapatrie toutes les données dans la mémoire du Driver.

---

## 3. Le DataFrame

### 3.1 Définition

Le **DataFrame** est la structure de données de haut niveau de Spark, introduite en Spark 1.3. C'est une collection distribuée de données organisées en **colonnes nommées et typées**, analogue à :

- Un tableau Pandas en Python
- Une table SQL en base de données relationnelle
- Un fichier Excel avec un schéma

Contrairement aux RDD, les DataFrames ont un **schéma** (noms et types des colonnes) connu à l'avance, ce qui permet à l'optimiseur Catalyst d'appliquer des optimisations automatiques.

```python
# Un DataFrame Spark ressemble à ceci :
# +--------+------+-----------------+
# | nom    | ville| chiffre_affaires|
# +--------+------+-----------------+
# | Alice  | Paris|          1500.0 |
# | Bob    | Lyon |           800.0 |
# | Claire | Paris|          2200.0 |
# +--------+------+-----------------+
```

### 3.2 Création d'un DataFrame

```python
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder.master("local[*]").appName("DataFrame").getOrCreate()

# ── Méthode 1 : depuis une liste Python ─────────────────────────────────────
data = [("Alice", "Paris", 1500.0), ("Bob", "Lyon", 800.0)]
df1 = spark.createDataFrame(data, schema=["nom", "ville", "chiffre_affaires"])

# ── Méthode 2 : avec un schéma explicite ────────────────────────────────────
schema = StructType([
    StructField("nom",               StringType(), nullable=False),
    StructField("ville",             StringType(), nullable=True),
    StructField("chiffre_affaires",  DoubleType(), nullable=True),
])
df2 = spark.createDataFrame(data, schema=schema)

# ── Méthode 3 : lecture de fichiers ─────────────────────────────────────────
# CSV
df_csv = spark.read \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("sep", ";") \
    .csv("data/ventes.csv")

# JSON
df_json = spark.read.json("data/utilisateurs.json")

# Parquet (format colonnaire — le plus courant en production)
df_parquet = spark.read.parquet("data/transactions/")

# ── Méthode 4 : depuis un RDD ───────────────────────────────────────────────
rdd = sc.parallelize([("Alice", 30), ("Bob", 25)])
df_depuis_rdd = rdd.toDF(["nom", "age"])

# ── Méthode 5 : depuis Pandas (développement/prototypage) ───────────────────
import pandas as pd
pdf = pd.DataFrame({"nom": ["Alice", "Bob"], "age": [30, 25]})
df_depuis_pandas = spark.createDataFrame(pdf)
```

### 3.3 Le schéma (*schema*)

```python
# Afficher le schéma
df.printSchema()
# root
#  |-- nom: string (nullable = true)
#  |-- ville: string (nullable = true)
#  |-- chiffre_affaires: double (nullable = true)

# Accéder au schéma programmatiquement
print(df.schema)
print(df.dtypes)       # Liste de (nom_colonne, type_string)
print(df.columns)      # Liste des noms de colonnes
```

### 3.4 Les types de données Spark

| Type Spark | Type Python | Description |
|---|---|---|
| `StringType` | `str` | Chaîne de caractères |
| `IntegerType` | `int` | Entier 32 bits |
| `LongType` | `int` | Entier 64 bits |
| `FloatType` | `float` | Flottant 32 bits |
| `DoubleType` | `float` | Flottant 64 bits |
| `BooleanType` | `bool` | Booléen |
| `DateType` | `datetime.date` | Date (sans heure) |
| `TimestampType` | `datetime.datetime` | Date + heure |
| `ArrayType` | `list` | Tableau d'éléments |
| `MapType` | `dict` | Dictionnaire clé-valeur |
| `StructType` | — | Structure imbriquée (objet) |

### 3.5 Exploration d'un DataFrame

```python
# Aperçu des données
df.show()               # 20 premières lignes (par défaut)
df.show(5)              # 5 premières lignes
df.show(5, truncate=False)    # Sans tronquer les longues valeurs

# Statistiques descriptives
df.describe().show()    # count, mean, stddev, min, max pour les colonnes numériques
df.summary().show()     # Idem + percentiles (25%, 50%, 75%)

# Dimensions
df.count()              # Nombre de lignes
len(df.columns)         # Nombre de colonnes

# Valeurs distinctes et null
df.select("ville").distinct().show()
df.filter(df["chiffre_affaires"].isNull()).count()
```

### 3.6 Sélection et accès aux colonnes

```python
from pyspark.sql import functions as F

# Sélectionner des colonnes
df.select("nom", "ville").show()
df.select(df["nom"], df["ville"]).show()
df.select(F.col("nom"), F.col("ville")).show()

# Les trois syntaxes sont équivalentes
# F.col() est la plus recommandée (pas d'ambiguïté avec les variables Python)

# Accéder à une colonne (retourne un objet Column, pas des données)
col_nom = df["nom"]
col_nom = F.col("nom")
col_nom = df.nom          # Syntaxe attribut (déconseillée si nom = mot réservé Python)
```

---

## 4. Le Dataset — notion complémentaire

### 4.1 Qu'est-ce qu'un Dataset ?

Le **Dataset** est une API fortement typée disponible en **Scala et Java uniquement**. En Python, les DataFrames jouent ce rôle (un DataFrame Spark est en réalité un `Dataset[Row]`).

```scala
// Scala uniquement — pas disponible en PySpark
case class Vente(nom: String, ville: String, montant: Double)
val ds: Dataset[Vente] = spark.read.parquet("ventes.parquet").as[Vente]
ds.filter(_.montant > 1000).show()
```

En PySpark, les DataFrames offrent des garanties similaires via le schéma et les types de données. La notion de Dataset reste utile à connaître pour :
- Lire la documentation officielle Spark (souvent en Scala)
- Comprendre les messages d'erreur qui mentionnent `Dataset[Row]`

---

## 5. Spark SQL et les vues temporaires

### 5.1 Principe

Spark SQL permet d'interroger des DataFrames avec du **SQL standard (ANSI SQL)**. C'est particulièrement utile pour les Data Analysts ou pour des transformations complexes plus naturellement exprimées en SQL.

### 5.2 Créer une vue temporaire

```python
# Enregistrer un DataFrame comme vue SQL temporaire
df.createOrReplaceTempView("ventes")

# La vue est disponible pour toute la durée de la SparkSession
# Elle ne persiste pas entre deux sessions
```

### 5.3 Requêtes SQL

```python
# Requête SQL simple
result = spark.sql("""
    SELECT ville, 
           SUM(chiffre_affaires) AS ca_total,
           COUNT(*) AS nb_vendeurs
    FROM ventes
    WHERE chiffre_affaires > 500
    GROUP BY ville
    ORDER BY ca_total DESC
""")
result.show()

# Jointure entre deux DataFrames via SQL
df_clients.createOrReplaceTempView("clients")
df_commandes.createOrReplaceTempView("commandes")

spark.sql("""
    SELECT c.nom, c.ville, SUM(o.montant) AS total
    FROM clients c
    JOIN commandes o ON c.id = o.client_id
    GROUP BY c.nom, c.ville
""").show()
```

### 5.4 Vues globales (inter-sessions)

```python
# Vue globale : accessible depuis toutes les SparkSessions
df.createOrReplaceGlobalTempView("ventes_global")

# Accès obligatoirement via le préfixe "global_temp"
spark.sql("SELECT * FROM global_temp.ventes_global").show()
```

### 5.5 Équivalences SQL / API DataFrame

| SQL | API DataFrame |
|---|---|
| `SELECT col1, col2` | `.select("col1", "col2")` |
| `WHERE col > 100` | `.filter(F.col("col") > 100)` |
| `GROUP BY col` | `.groupBy("col")` |
| `ORDER BY col DESC` | `.orderBy(F.col("col").desc())` |
| `JOIN ... ON` | `.join(df2, on="id", how="inner")` |
| `LIMIT 10` | `.limit(10)` |
| `DISTINCT` | `.distinct()` |
| `AS alias` | `.alias("alias")` |

```python
# Ces deux expressions sont strictement équivalentes :

# Version SQL
spark.sql("""
    SELECT ville, SUM(chiffre_affaires) as ca
    FROM ventes
    WHERE chiffre_affaires > 1000
    GROUP BY ville
    ORDER BY ca DESC
""").show()

# Version API DataFrame
df.filter(F.col("chiffre_affaires") > 1000) \
  .groupBy("ville") \
  .agg(F.sum("chiffre_affaires").alias("ca")) \
  .orderBy(F.col("ca").desc()) \
  .show()
```

---

## 6. Quand choisir RDD vs DataFrame ?

### 6.1 Tableau de décision

| Critère | RDD | DataFrame |
|---|---|---|
| **Performance** | Moins optimisé (pas de Catalyst) | Optimisé automatiquement |
| **API** | Fonctionnelle (map, filter, reduce) | SQL-like (select, filter, groupBy) |
| **Typage** | Flexible — n'importe quel objet Python | Schéma strict (colonnes typées) |
| **Lisibilité** | Moins lisible pour des transformations complexes | Plus lisible, plus déclaratif |
| **Débogage** | Erreurs souvent plus claires | Plans d'exécution parfois opaques |
| **ML** | Certains algorithmes anciens (MLlib RDD-based) | API ML moderne (MLlib DataFrame-based) |
| **Données semi-structurées** | Idéal (objets Python libres) | Nécessite un schéma |

### 6.2 Règles pratiques

**Utiliser un DataFrame quand :**
- Les données sont structurées ou semi-structurées (CSV, JSON, Parquet)
- On veut profiter des optimisations de Catalyst
- On travaille avec Spark SQL, MLlib moderne, ou Structured Streaming
- C'est le cas d'usage **par défaut** — 90% des pipelines PySpark

**Utiliser un RDD quand :**
- Les données sont non structurées (texte brut, objets Python arbitraires)
- On a besoin d'un contrôle fin sur le partitionnement
- On utilise des API de bas niveau pour des algorithmes personnalisés
- On intègre du code Python pur complexe difficile à vectoriser

```python
# Exemple : traitement de texte non structuré → RDD naturel
rdd_mots = sc.textFile("corpus.txt") \
             .flatMap(lambda ligne: ligne.split(" ")) \
             .filter(lambda mot: len(mot) > 3) \
             .map(lambda mot: (mot.lower(), 1)) \
             .reduceByKey(lambda a, b: a + b)

# Exemple : analyse de ventes structurées → DataFrame naturel
df_analyse = spark.read.parquet("ventes/") \
                  .filter(F.col("montant") > 100) \
                  .groupBy("region", "produit") \
                  .agg(F.sum("montant").alias("ca"))
```

---

## 7. Le DAG et le plan d'exécution

### 7.1 Comment Spark construit le DAG

Chaque transformation Spark ajoute un nœud au DAG. Le DAG est compilé et optimisé par le **Catalyst Optimizer** uniquement lorsqu'une action est déclenchée.

```python
# Ces 4 lignes ne calculent rien — elles construisent le DAG
df_filtre   = df.filter(F.col("montant") > 100)          # Nœud 1
df_enrichi  = df_filtre.withColumn("tva", F.col("montant") * 0.2)  # Nœud 2
df_groupe   = df_enrichi.groupBy("region").sum("montant")  # Nœud 3
df_trie     = df_groupe.orderBy("sum(montant)")            # Nœud 4

# Cette ligne déclenche l'exécution complète du DAG
df_trie.show()   # ← ACTION
```

### 7.2 Les optimisations de Catalyst

Le **Catalyst Optimizer** analyse le DAG logique et applique automatiquement des optimisations :

**Predicate pushdown** : les filtres sont appliqués le plus tôt possible (idéalement au niveau de la source).
```python
# Spark réordonne automatiquement pour filtrer AVANT le join
df1.join(df2, "id").filter(F.col("montant") > 100)
# → Catalyst transforme en : df1.filter(...).join(df2, "id")
```

**Projection pushdown** : seules les colonnes nécessaires sont lues depuis la source.
```python
# Spark ne lit que les colonnes "nom" et "montant" depuis le fichier Parquet
df.select("nom", "montant").filter(...)
```

**Fusion de filtres** : plusieurs filtres consécutifs sont fusionnés en un seul passage.
```python
df.filter(A).filter(B).filter(C)
# → Catalyst transforme en : df.filter(A AND B AND C)
```

### 7.3 Lire un plan d'exécution

```python
df_resultat = df.filter(F.col("montant") > 100) \
                .groupBy("region") \
                .agg(F.sum("montant").alias("total"))

# Plan simple
df_resultat.explain()

# Plan détaillé avec les 4 niveaux
df_resultat.explain(mode="extended")
# - Parsed Logical Plan    : plan tel que décrit par le code
# - Analyzed Logical Plan  : résolution des types et des colonnes
# - Optimized Logical Plan : après optimisations Catalyst
# - Physical Plan          : plan réellement exécuté

# Plan en format JSON (Spark 3.x)
df_resultat.explain(mode="formatted")
```

**Exemple de plan physique annoté :**
```
== Physical Plan ==
AdaptiveSparkPlan                                ← AQE (optimisation dynamique)
+- == Final Plan ==
   *(2) HashAggregate(keys=[region], functions=[sum(montant)])
   +- AQEShuffleRead                             ← lecture après shuffle
      +- ShuffleQueryStage                       ← échange réseau (shuffle)
         +- Exchange hashpartitioning(region, 4) ← redistribution par clé
            +- *(1) HashAggregate(keys=[region], functions=[partial_sum(montant)])
               +- *(1) Filter (montant > 100)    ← filtre poussé vers la source
                  +- *(1) FileScan parquet [region, montant]  ← lecture fichier
```

### 7.4 Jobs, Stages et Tasks

```
ACTION (.show(), .count()...)
  │
  └── JOB (1 par action)
        │
        ├── STAGE 1 (calculs sans shuffle)
        │     ├── Task 1 (partition 1)
        │     ├── Task 2 (partition 2)
        │     └── Task 3 (partition 3)
        │
        └── STAGE 2 (après le shuffle)
              ├── Task 1 (partition 1)
              └── Task 2 (partition 2)

Règle : un nouveau Stage est créé à chaque opération de shuffle
(groupBy, join, repartition, sortBy...)
```

---

## 8. Conversions entre structures

```python
# ── DataFrame → RDD ─────────────────────────────────────────────────────────
rdd = df.rdd                  # RDD de Row objects
rdd_tuples = df.rdd.map(tuple)  # RDD de tuples Python

# ── RDD → DataFrame ─────────────────────────────────────────────────────────
df = rdd.toDF(["col1", "col2"])
df = spark.createDataFrame(rdd, schema=schema)

# ── DataFrame → Pandas ──────────────────────────────────────────────────────
pdf = df.toPandas()           # ⚠️ Rapatrie TOUT en mémoire Driver

# ── Pandas → DataFrame ──────────────────────────────────────────────────────
df = spark.createDataFrame(pdf)

# ── DataFrame → Spark SQL ───────────────────────────────────────────────────
df.createOrReplaceTempView("ma_table")
result = spark.sql("SELECT * FROM ma_table")
```

---

## 9. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

spark = SparkSession.builder \
    .appName("StructuresDonnees") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
sc = spark.sparkContext

# ─── 1. RDD ──────────────────────────────────────────────────────────────────
print("=== Démonstration RDD ===")
rdd = sc.parallelize(range(1, 21))
print(f"Partitions : {rdd.getNumPartitions()}")
print(f"Somme des pairs > 5 : {rdd.filter(lambda x: x % 2 == 0 and x > 5).sum()}")

# Comptage de mots (word count) — cas d'usage classique des RDD
texte = sc.parallelize([
    "spark est un framework distribue",
    "spark est rapide et puissant",
    "pyspark est l interface python de spark"
])
word_count = texte \
    .flatMap(lambda ligne: ligne.split(" ")) \
    .map(lambda mot: (mot, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .sortBy(lambda x: -x[1])

print("\nTop 5 mots :")
for mot, count in word_count.take(5):
    print(f"  {mot:20s} : {count}")

# ─── 2. DataFrame avec schéma explicite ─────────────────────────────────────
print("\n=== Démonstration DataFrame ===")
schema = StructType([
    StructField("id",      IntegerType(), False),
    StructField("nom",     StringType(),  True),
    StructField("ville",   StringType(),  True),
    StructField("montant", DoubleType(),  True),
    StructField("annee",   IntegerType(), True),
])

data = [
    (1, "Alice",   "Paris",  1500.0, 2023),
    (2, "Bob",     "Lyon",    800.0, 2023),
    (3, "Claire",  "Paris",  2200.0, 2024),
    (4, "David",   "Lyon",   1100.0, 2024),
    (5, "Emma",    "Nantes",  950.0, 2023),
    (6, "François","Paris",  3000.0, 2024),
    (7, "Gaëlle",  "Nantes",  400.0, 2023),
]

df = spark.createDataFrame(data, schema=schema)

df.printSchema()
df.show()

# ─── 3. Transformations DataFrame ────────────────────────────────────────────
df_enrichi = df \
    .filter(F.col("montant") > 500) \
    .withColumn("montant_ht", F.round(F.col("montant") / 1.2, 2)) \
    .withColumn("categorie", F.when(F.col("montant") > 1500, "TOP")
                               .when(F.col("montant") > 800, "MID")
                               .otherwise("LOW"))

print("DataFrame enrichi :")
df_enrichi.show()

# ─── 4. Spark SQL ────────────────────────────────────────────────────────────
df.createOrReplaceTempView("ventes")

result_sql = spark.sql("""
    SELECT 
        ville,
        annee,
        ROUND(SUM(montant), 2) AS ca_total,
        COUNT(*) AS nb_vendeurs,
        ROUND(AVG(montant), 2) AS ca_moyen
    FROM ventes
    WHERE montant > 500
    GROUP BY ville, annee
    ORDER BY ca_total DESC
""")

print("=== Résultat SQL ===")
result_sql.show()

# ─── 5. Plan d'exécution ─────────────────────────────────────────────────────
print("=== Plan d'exécution ===")
result_sql.explain()

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **RDD** | Structure primitive, immuable, distribuée, tolérante aux pannes via le lignage |
| **Lazy evaluation** | Aucun calcul avant une action — Spark optimise d'abord, calcule ensuite |
| **DataFrame** | Collection de colonnes typées avec schéma — structure recommandée par défaut |
| **Schéma** | Connu à l'avance → permet l'optimisation Catalyst et la validation des types |
| **Spark SQL** | SQL standard sur des DataFrames via des vues temporaires |
| **Catalyst** | Optimiseur qui réécrit et améliore automatiquement le plan d'exécution |
| **DAG** | Graphe des transformations — découpé en Jobs → Stages → Tasks à l'exécution |
| **RDD vs DataFrame** | DataFrame par défaut ; RDD pour les données non structurées ou le bas niveau |

---

## Exercices

### Exercice 1 — Word count (30 min)
> Implémenter le comptage de mots classique (*Word Count*) sur un fichier texte de votre choix, en utilisant un RDD.  
> 1. Lire le fichier avec `sc.textFile()`
> 2. Découper les lignes en mots (`flatMap`)
> 3. Mettre en minuscule et supprimer la ponctuation
> 4. Compter les occurrences (`map` + `reduceByKey`)
> 5. Afficher les 20 mots les plus fréquents (`sortBy` + `take`)

### Exercice 2 — Schéma et types (20 min)
> Créer un DataFrame représentant un catalogue de films avec les colonnes suivantes :
> `titre` (String), `annee` (Integer), `genre` (String), `note` (Double), `duree_min` (Integer).  
> 1. Définir un schéma explicite avec `StructType`
> 2. Créer le DataFrame avec au moins 8 films
> 3. Afficher les films de plus de 120 minutes avec une note > 7.5
> 4. Calculer la note moyenne par genre

### Exercice 3 — SQL vs API DataFrame (25 min)
> À partir du DataFrame de films (exercice 2) :  
> 1. Écrire une requête SQL qui retourne le meilleur film (note max) par genre
> 2. Écrire l'équivalent exact en API DataFrame
> 3. Comparer les plans d'exécution des deux approches avec `.explain()`
> 4. Sont-ils identiques ? Pourquoi ?

### Exercice 4 — Exploration du DAG (25 min)
> Construire un pipeline de 5 transformations sur un DataFrame de votre choix :  
> `filter` → `withColumn` → `groupBy` → `agg` → `orderBy`  
> 1. Avant d'appeler `.show()`, combien de jobs pensez-vous que Spark va créer ?
> 2. Appeler `.explain()` et compter le nombre d'échanges réseau (mots-clés `Exchange`, `Shuffle`)
> 3. Ouvrir la Spark UI après `.show()` et vérifier le nombre de stages

---

## Pour aller plus loin

- 📖 **Documentation RDD** : [spark.apache.org/docs/latest/rdd-programming-guide.html](https://spark.apache.org/docs/latest/rdd-programming-guide.html)
- 📖 **Documentation DataFrame** : [spark.apache.org/docs/latest/sql-programming-guide.html](https://spark.apache.org/docs/latest/sql-programming-guide.html)
- 📄 **Article fondateur RDD** : *Resilient Distributed Datasets* — Zaharia et al., NSDI 2012
- 📄 **Article Catalyst** : *Spark SQL: Relational Data Processing in Spark* — Armbrust et al., SIGMOD 2015
- 🛠️ **Spark SQL Functions** : liste complète des fonctions natives — [spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)

---

*Module précédent → **Module 2 : Installation et prise en main***  
*Module suivant → **Module 4 : Transformations de type mapper***
