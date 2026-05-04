# Module 6 — Partitionnement, shuffles et collecte

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 5 — Transformations de type reducer

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Comprendre le rôle et l'impact du partitionnement sur les performances Spark
- Utiliser `repartition()` et `coalesce()` à bon escient
- Identifier et corriger un problème de déséquilibre de données (*data skew*)
- Maîtriser les niveaux de persistance (`cache()`, `persist()`)
- Choisir le bon format de fichier pour l'écriture distribuée
- Collecter les données vers le Driver de manière sûre et efficace

---

## 1. Le partitionnement : fondement du calcul distribué

### 1.1 Qu'est-ce qu'une partition ?

Une **partition** est l'unité fondamentale de parallélisme dans Spark. Chaque RDD ou DataFrame est découpé en **N partitions**, et chaque partition est traitée par **une et une seule Task** sur un Executor.

```
DataFrame de 12 Go, 6 partitions :

  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ Partition 1 │   │ Partition 2 │   │ Partition 3 │
  │   2 Go      │   │   2 Go      │   │   2 Go      │
  └─────┬───────┘   └──────┬──────┘   └──────┬──────┘
        │                  │                  │
  ┌─────▼───────┐   ┌──────▼──────┐   ┌──────▼──────┐
  │  Executor 1 │   │  Executor 2 │   │  Executor 3 │
  │  (Task 1)   │   │  (Task 2)   │   │  (Task 3)   │
  └─────────────┘   └─────────────┘   └─────────────┘

  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
  │ Partition 4 │   │ Partition 5 │   │ Partition 6 │
  │   2 Go      │   │   2 Go      │   │   2 Go      │
  └─────┬───────┘   └──────┬──────┘   └──────┬──────┘
        │                  │                  │
  ┌─────▼───────┐   ┌──────▼──────┐   ┌──────▼──────┐
  │  Executor 1 │   │  Executor 2 │   │  Executor 3 │
  │  (Task 4)   │   │  (Task 5)   │   │  (Task 6)   │
  └─────────────┘   └─────────────┘   └─────────────┘
```

Le nombre de partitions détermine directement le **degré de parallélisme** : trop peu de partitions → sous-utilisation du cluster ; trop de partitions → overhead de scheduling.

### 1.2 Nombre de partitions : règles pratiques

```python
# Connaître le nombre de partitions courant
print(df.rdd.getNumPartitions())       # DataFrame
print(rdd.getNumPartitions())          # RDD

# Règle générale : 2 à 4 partitions par cœur CPU disponible
# Exemple : cluster de 10 Executors × 4 cœurs = 40 cœurs → 80 à 160 partitions

# Paramètre par défaut après un shuffle
spark.conf.get("spark.sql.shuffle.partitions")   # 200 par défaut
# ⚠️ 200 est trop élevé pour un cluster local ou de petits jeux de données !
# En local avec 8 cœurs : préférer 8 à 32
spark.conf.set("spark.sql.shuffle.partitions", "16")

# Taille cible recommandée par partition : 128 Mo à 256 Mo
# Pour 10 Go de données : 10 000 Mo / 128 Mo ≈ 78 → arrondir à 80 ou 96
```

### 1.3 Comment les partitions sont-elles créées ?

```python
# Lecture de fichiers : 1 partition par bloc HDFS (128 Mo par défaut)
df = spark.read.parquet("data/")
# → le nombre de partitions dépend du nombre et de la taille des fichiers

# Parallélisation explicite
rdd = sc.parallelize(range(1000), numSlices=8)   # 8 partitions

# Après un shuffle (groupBy, join, repartition...)
# → contrôlé par spark.sql.shuffle.partitions

# Lecture depuis une source externe (JDBC, Kafka...)
df = spark.read.jdbc(url, table, numPartitions=10,
                     partitionColumn="id",
                     lowerBound=0, upperBound=1000000)
```

---

## 2. `repartition()` et `coalesce()`

### 2.1 `repartition()` — repartitionnement complet

`repartition(n)` redistribue les données en **N nouvelles partitions** via un shuffle complet. Les données sont mélangées et réparties de manière équilibrée.

```python
# Augmenter le nombre de partitions (avant une opération lourde)
df_plus = df.repartition(100)

# Réduire le nombre de partitions (avant une écriture)
df_moins = df.repartition(10)

# Repartitionner par une colonne (Hash Partitioning)
# → Toutes les lignes avec la même valeur de "ville" seront dans la même partition
df_par_ville = df.repartition(8, F.col("ville"))

# Repartitionner par plusieurs colonnes
df_par_kv = df.repartition(16, F.col("pays"), F.col("region"))

# Équivalent RDD
rdd_new = rdd.repartition(50)
```

**Quand utiliser `repartition()` :**

```python
# ✅ Avant un groupBy ou un join intensif pour équilibrer la charge
df.repartition(100, "client_id").join(df2, "client_id")

# ✅ Pour augmenter le parallélisme sur un petit nombre de fichiers
df = spark.read.csv("data/un_seul_gros_fichier.csv")  # 1 partition
df = df.repartition(32)   # Parallélisation forcée

# ✅ Repartitionner par clé avant une écriture partitionnée
df.repartition("pays").write.partitionBy("pays").parquet("output/")
```

### 2.2 `coalesce()` — réduction sans shuffle

`coalesce(n)` réduit le nombre de partitions **sans shuffle complet** : il fusionne des partitions existantes sur le même Executor. C'est donc bien plus efficace que `repartition()` pour **réduire** le nombre de partitions.

```python
# Réduire de 200 à 10 partitions (sans shuffle)
df_coalesce = df.coalesce(10)

# Cas typique : avant une écriture pour obtenir moins de fichiers
df.coalesce(1).write.csv("output/resultat_unique.csv")
# ⚠️ coalesce(1) : tout converge vers 1 Executor → goulot d'étranglement !
```

**Différence fondamentale :**

```
repartition(N)               coalesce(N)
─────────────────────────    ─────────────────────────────────
Shuffle complet              Fusion locale (pas de shuffle)
Équilibré (même taille)      Déséquilibré possible
Peut augmenter ou réduire    Ne peut que réduire
Coûteux (réseau)             Peu coûteux
```

```python
# ── Règle de décision ────────────────────────────────────────────────────────
# Augmenter le nombre de partitions  → repartition()
# Diminuer et données équilibrées    → coalesce()
# Diminuer et données déséquilibrées → repartition() (pour ré-équilibrer)
# Avant une écriture                 → coalesce() (moins de fichiers, sans shuffle)
```

### 2.3 `partitionBy()` sur les RDD

```python
from pyspark import HashPartitioner, RangePartitioner

# Hash partitioning : même clé → même partition
rdd_pair = sc.parallelize([("A", 1), ("B", 2), ("A", 3), ("C", 4)])
rdd_hash = rdd_pair.partitionBy(4, HashPartitioner(4))

# Vérifier dans quelle partition se trouve chaque élément
rdd_hash.glom().zipWithIndex().collect()
# glom() : regroupe les éléments de chaque partition en une liste

# Persister après un partitionBy pour éviter de recalculer le shuffle
rdd_partitionne = rdd_pair.partitionBy(4).cache()
```

---

## 3. Le shuffle : anatomie et optimisation

### 3.1 Anatomie d'un shuffle

Un shuffle se déroule en trois phases :

```
Phase 1 — MAP SIDE (côté source)
  Chaque Executor :
  1. Exécute la transformation locale (map, filter...)
  2. Calcule la partition destination de chaque enregistrement (hash de la clé)
  3. Écrit les données triées par partition destination sur disque local

Phase 2 — TRANSFERT RÉSEAU
  Les Executors destination :
  4. Téléchargent les fichiers de shuffle depuis tous les Executors source

Phase 3 — REDUCE SIDE (côté destination)
  Chaque Executor destination :
  5. Fusionne et trie les données reçues
  6. Applique l'agrégation (reduceByKey, groupBy...)
```

### 3.2 Mesurer le coût d'un shuffle

Dans la **Spark UI → onglet Stages**, on peut lire pour chaque stage :

| Métrique | Description |
|---|---|
| **Shuffle Write** | Volume écrit sur disque côté source |
| **Shuffle Read** | Volume lu depuis le réseau côté destination |
| **Shuffle Spill (Memory)** | Données qui ne tenaient pas en RAM |
| **Shuffle Spill (Disk)** | Données écrites sur disque (signe d'un problème mémoire) |

```python
# Mesurer programmatiquement
df.groupBy("ville").count().explain()
# Chercher "Exchange hashpartitioning" dans le plan → c'est le shuffle
```

### 3.3 Le *Data Skew* — déséquilibre des partitions

Le **data skew** survient quand certaines partitions sont **beaucoup plus grandes que les autres**, créant des *stragglers* (tâches lentes) qui bloquent l'avancement du stage.

```
Distribution équilibrée :         Distribution skewed :
  Part. 1 : 1 000 000 lignes       Part. 1 :       100 lignes   ←  rapide
  Part. 2 : 1 000 000 lignes       Part. 2 :       200 lignes   ←  rapide
  Part. 3 : 1 000 000 lignes       Part. 3 : 9 800 000 lignes   ←  LENT !!
  Part. 4 : 1 000 000 lignes       Part. 4 :       150 lignes   ←  rapide

  Durée : ~10 min                  Durée : ~50 min (bloquée par Part. 3)
```

**Détecter le skew :**
```python
# Distribution des tailles de partitions
df.rdd.mapPartitionsWithIndex(
    lambda i, it: [(i, sum(1 for _ in it))]
).toDF(["partition", "count"]).orderBy("count").show()

# Vérifier les valeurs dominantes d'une colonne de jointure
df.groupBy("client_id").count().orderBy(F.col("count").desc()).show(10)
```

**Corriger le skew — techniques :**

```python
# ── Technique 1 : Salting (ajout d'un sel aléatoire) ─────────────────────────
import random

# Problème : "Paris" représente 80% des données
# Solution : distribuer "Paris" sur 10 partitions en ajoutant un suffixe

N_SALT = 10

# Côté grande table : ajouter un sel aléatoire
df_large = df_large.withColumn(
    "ville_salt",
    F.concat(F.col("ville"), F.lit("_"), (F.rand() * N_SALT).cast("int").cast("string"))
)

# Côté petite table : répliquer chaque ligne N_SALT fois
df_ref_repliquee = df_ref.withColumn(
    "sel", F.explode(F.array([F.lit(i) for i in range(N_SALT)]))
).withColumn(
    "ville_salt",
    F.concat(F.col("ville"), F.lit("_"), F.col("sel").cast("string"))
).drop("sel")

# Jointure sur la clé saltée
df_joint = df_large.join(df_ref_repliquee, "ville_salt").drop("ville_salt")


# ── Technique 2 : AQE — Adaptive Query Execution (Spark 3.x) ─────────────────
# Spark 3.x détecte et corrige automatiquement le skew pendant l'exécution
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
# → Les partitions > 5× la taille médiane sont automatiquement découpées


# ── Technique 3 : Broadcast join (si la table skewed est petite) ─────────────
df_large.join(broadcast(df_ref_skewee), "ville")


# ── Technique 4 : Filtrer et traiter séparément ───────────────────────────────
# Séparer les clés "chaudes" (fréquentes) des clés "froides" (rares)
cles_chaudes = ["Paris", "Lyon"]
df_chaud  = df.filter( F.col("ville").isin(cles_chaudes))
df_froid  = df.filter(~F.col("ville").isin(cles_chaudes))

# Broadcast join pour les clés chaudes (fréquentes)
resultat_chaud = df_chaud.join(broadcast(df_ref), "ville")
# Join standard pour les clés froides
resultat_froid = df_froid.join(df_ref, "ville")

# Réunir les résultats
resultat_final = resultat_chaud.union(resultat_froid)
```

---

## 4. Persistance et cache

### 4.1 Pourquoi persister ?

Sans persistance, chaque action sur un DataFrame **recalcule tout le DAG depuis le début**. Si le même DataFrame intermédiaire est utilisé plusieurs fois, cela conduit à des recalculs coûteux.

```python
# ❌ Sans cache : df_prepare recalculé 3 fois
df_prepare = spark.read.parquet("data/") \
    .filter(...).withColumn(...).groupBy(...).agg(...)

df_prepare.show()         # Recalcul complet
df_prepare.count()        # Recalcul complet
df_prepare.write.parquet("output/")  # Recalcul complet

# ✅ Avec cache : df_prepare calculé 1 fois, puis réutilisé depuis la mémoire
df_prepare.cache()
df_prepare.count()        # Déclenche le calcul ET le stockage en cache
df_prepare.show()         # Depuis le cache → très rapide
df_prepare.write.parquet("output/")  # Depuis le cache → très rapide
```

### 4.2 `cache()` et `persist()`

`cache()` est un raccourci pour `persist(StorageLevel.MEMORY_AND_DISK)`.

```python
from pyspark import StorageLevel

# cache() : niveau par défaut MEMORY_AND_DISK
df.cache()
rdd.cache()

# persist() : niveau explicite
df.persist(StorageLevel.MEMORY_ONLY)
df.persist(StorageLevel.MEMORY_AND_DISK)
df.persist(StorageLevel.DISK_ONLY)
df.persist(StorageLevel.MEMORY_ONLY_2)   # Répliqué sur 2 nœuds
df.persist(StorageLevel.OFF_HEAP)        # Hors du heap Java (Tungsten)
```

### 4.3 Les niveaux de persistance

| Niveau | RAM | Disque | Sérialisé | Réplication | Usage |
|---|:---:|:---:|:---:|:---:|---|
| `MEMORY_ONLY` | ✅ | ❌ | ❌ | ×1 | RDD petits, RAM abondante |
| `MEMORY_AND_DISK` | ✅ | ✅ | ❌ | ×1 | **Défaut** — bonne tolérance |
| `MEMORY_ONLY_SER` | ✅ | ❌ | ✅ | ×1 | Économise la RAM (Python moins efficace) |
| `MEMORY_AND_DISK_SER` | ✅ | ✅ | ✅ | ×1 | Équilibre RAM/disque |
| `DISK_ONLY` | ❌ | ✅ | ✅ | ×1 | Très grands datasets, RAM limitée |
| `MEMORY_ONLY_2` | ✅ | ❌ | ❌ | ×2 | Haute disponibilité |
| `OFF_HEAP` | ✅ | ❌ | ✅ | ×1 | Évite le GC Java |

**Choisir le bon niveau :**

```python
# Données fréquemment accédées + RAM suffisante
df.persist(StorageLevel.MEMORY_ONLY)

# Données fréquemment accédées + RAM limitée (débordement sur disque)
df.persist(StorageLevel.MEMORY_AND_DISK)   # ou cache()

# Données accédées une seule fois → pas besoin de cache
# Données en fin de pipeline → pas besoin de cache

# Checkpoint (alternative pour casser les longs lignages)
sc.setCheckpointDir("/tmp/spark-checkpoint")
rdd_long.checkpoint()   # Matérialise et coupe le lignage
```

### 4.4 Libérer le cache

```python
# Libérer explicitement (bonne pratique pour les jobs longs)
df.unpersist()
rdd.unpersist()

# Vérifier ce qui est en cache (Spark UI → onglet Storage)
spark.catalog.clearCache()   # Vider tout le cache de la session
```

### 4.5 Bonnes pratiques de cache

```python
# ✅ Toujours déclencher une action après cache() pour matérialiser
df.cache()
df.count()   # Force le calcul et le stockage

# ✅ Libérer le cache dès qu'il n'est plus nécessaire
df_temp.cache()
# ... utilisation ...
df_temp.unpersist()

# ❌ Ne pas cacher systématiquement — le cache consomme de la RAM
# Ne cacher que les DataFrames :
#   - calculés par un DAG long/coûteux
#   - utilisés plusieurs fois dans le pipeline

# ✅ Préférer cache() sur DataFrame vs RDD (Catalyst peut optimiser la sérialisation)
```

---

## 5. Collecte des données vers le Driver

### 5.1 Les opérations de collecte

Les opérations de collecte **rapatrient des données du cluster vers le Driver**. Elles sont nécessaires pour afficher des résultats ou les transmettre à du code Python, mais doivent être utilisées avec précaution.

```python
# ── Actions de collecte ───────────────────────────────────────────────────────
df.collect()           # ⚠️ Rapatrie TOUTES les lignes → risque OOM
df.take(n)             # Rapatrie les N premières lignes (sans garantie d'ordre)
df.first()             # Première ligne seulement
df.head(n)             # Identique à take(n)
df.show(n)             # Affiche N lignes dans le terminal (ne retourne rien)
df.toPandas()          # ⚠️ Rapatrie TOUT en Pandas → risque OOM

# ── Collecte d'une seule colonne ─────────────────────────────────────────────
valeurs = df.select("ville").rdd.flatMap(lambda x: x).collect()
# ou plus proprement :
valeurs = [row["ville"] for row in df.select("ville").collect()]

# ── Collecte scalaire ─────────────────────────────────────────────────────────
total  = df.count()                                      # Long
somme  = df.agg(F.sum("montant")).collect()[0][0]       # Double
maxi   = df.agg(F.max("montant")).first()[0]            # Double
```

### 5.2 Utiliser `collect()` en sécurité

```python
# ❌ Dangereux sur un gros DataFrame
df_10_to.collect()   # → OutOfMemoryError sur le Driver

# ✅ Toujours limiter avant de collecter
df.filter(...).limit(10000).collect()
df.sample(fraction=0.01).collect()   # Échantillon aléatoire de 1%

# ✅ Vérifier la taille avant de collecter
n = df.count()
if n < 100_000:
    rows = df.collect()
else:
    print(f"Trop grand pour collect() : {n} lignes — utiliser write() à la place")

# ✅ Utiliser toPandas() uniquement sur des sous-ensembles
df.filter(...).limit(5000).toPandas()
```

### 5.3 Diffuser des données du Driver vers le cluster

Parfois on veut partager une **petite structure de données Python** (dictionnaire, liste) avec tous les Executors. Les **variables broadcast** permettent de le faire efficacement.

```python
# ❌ Mauvaise pratique : variable Python capturée par fermeture (closure)
# Chaque tâche reçoit une copie sérialisée → inefficace
code_postal = {"Paris": "75", "Lyon": "69", "Nantes": "44"}
df.filter(lambda row: code_postal.get(row.ville, "") == "75")

# ✅ Variable broadcast : envoyée une seule fois à chaque Executor, puis mise en cache
code_postal_bc = sc.broadcast({"Paris": "75", "Lyon": "69", "Nantes": "44"})

# Accès dans une UDF
@udf(StringType())
def get_code(ville):
    return code_postal_bc.value.get(ville, "??")

df.withColumn("code", get_code(F.col("ville"))).show()

# Libérer la variable broadcast
code_postal_bc.unpersist()
code_postal_bc.destroy()   # Suppression définitive
```

### 5.4 Les accumulateurs — compteurs distribués

Les **accumulateurs** permettent de **collecter des métriques** depuis les Executors vers le Driver pendant l'exécution d'une transformation.

```python
# Accumulateur entier
compteur_erreurs = sc.accumulator(0)
compteur_nulls   = sc.accumulator(0)

# Utilisation dans une transformation
def traiter_ligne(row):
    global compteur_erreurs, compteur_nulls
    if row["montant"] is None:
        compteur_nulls.add(1)
        return None
    if row["montant"] < 0:
        compteur_erreurs.add(1)
        return None
    return row

rdd_propre = df.rdd.filter(traiter_ligne)
rdd_propre.count()   # Action nécessaire pour déclencher l'exécution

print(f"Lignes avec montant null   : {compteur_nulls.value}")
print(f"Lignes avec montant négatif: {compteur_erreurs.value}")

# ⚠️ Les accumulateurs ne sont mis à jour que lors des actions
# ⚠️ En cas de re-exécution d'une tâche (tolérance aux pannes), l'accumulateur
#    peut être incrémenté plusieurs fois → valeurs approchées pour les transformations
```

---

## 6. Écriture distribuée et formats de fichiers

### 6.1 L'API d'écriture

```python
# Syntaxe générale
df.write \
  .format("parquet") \    # ou "csv", "json", "orc", "delta"...
  .mode("overwrite") \    # ou "append", "ignore", "error" (défaut)
  .option("clé", "valeur") \
  .partitionBy("pays", "annee") \
  .save("chemin/vers/output/")

# Raccourcis
df.write.parquet("output/")
df.write.csv("output/", header=True, sep=";")
df.write.json("output/")
df.write.orc("output/")
```

### 6.2 Modes d'écriture

| Mode | Comportement si la destination existe |
|---|---|
| `"error"` (défaut) | Lance une exception |
| `"overwrite"` | Écrase les données existantes |
| `"append"` | Ajoute aux données existantes |
| `"ignore"` | Ne fait rien (silencieux) |

### 6.3 Les formats de fichiers

**CSV** — simple mais inefficace :
```python
df.write \
  .option("header", "true") \
  .option("sep", ";") \
  .option("encoding", "UTF-8") \
  .csv("output/donnees.csv")
```

**JSON** — flexible mais verbeux :
```python
df.write.json("output/donnees.json")
# Produit du JSON Lines (1 objet JSON par ligne)
```

**Parquet** — format colonnaire compressé, recommandé en production :
```python
df.write \
  .option("compression", "snappy") \   # snappy, gzip, lz4, zstd, none
  .parquet("output/donnees.parquet")

# Avantages Parquet :
# → Lecture colonnaire (ne lit que les colonnes demandées)
# → Compression efficace (5 à 10× vs CSV)
# → Schéma embarqué
# → Partitionnement natif
# → Predicate pushdown (filtres appliqués à la lecture)
```

**ORC** — similaire à Parquet, optimisé pour Hive :
```python
df.write.orc("output/donnees.orc")
```

**Delta Lake** — format ACID sur Parquet (Spark 3.x avec delta-spark) :
```python
df.write \
  .format("delta") \
  .mode("overwrite") \
  .save("output/donnees_delta/")

# Supporte : transactions ACID, time travel, MERGE, UPDATE, DELETE
```

### 6.4 Partitionnement des fichiers de sortie

Le **partitionnement physique** organise les fichiers en sous-répertoires par valeur de colonne — comme les partitions d'une table Hive.

```python
# Écriture partitionnée par pays et par année
df.write \
  .partitionBy("pays", "annee") \
  .parquet("output/ventes/")

# Structure des fichiers générés :
# output/ventes/
#   pays=France/
#     annee=2023/
#       part-00000-xxx.parquet
#       part-00001-xxx.parquet
#     annee=2024/
#       part-00000-xxx.parquet
#   pays=Allemagne/
#     annee=2023/
#       ...

# Avantage : lors d'une lecture filtrée, Spark ne lit que les sous-dossiers pertinents
df_lu = spark.read.parquet("output/ventes/")
df_france_2024 = df_lu.filter(
    (F.col("pays") == "France") & (F.col("annee") == 2024)
)
# → Spark lit uniquement output/ventes/pays=France/annee=2024/
```

### 6.5 Contrôler le nombre de fichiers de sortie

```python
# Par défaut : 1 fichier par partition → peut produire des centaines de petits fichiers

# ── Solution 1 : coalesce avant écriture ─────────────────────────────────────
df.coalesce(10).write.parquet("output/")   # → 10 fichiers

# ── Solution 2 : repartition avant écriture (avec ré-équilibrage) ─────────────
df.repartition(10).write.parquet("output/")

# ── Solution 3 : repartition par colonne de partition ─────────────────────────
# 1 fichier par valeur de pays
df.repartition(F.col("pays")).write.partitionBy("pays").parquet("output/")

# ── Solution 4 : maxRecordsPerFile ────────────────────────────────────────────
df.write \
  .option("maxRecordsPerFile", 1_000_000) \  # Max 1M lignes par fichier
  .parquet("output/")
```

---

## 7. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark import StorageLevel
import time

spark = SparkSession.builder \
    .appName("PartitionnementEtCollecte") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .getOrCreate()
sc = spark.sparkContext

# ─── 1. Génération d'un jeu de données de taille réaliste ────────────────────
print("=" * 60)
print("=== Création du jeu de données ===")
print("=" * 60)

# Simuler 100 000 transactions
import random
villes   = ["Paris"] * 60 + ["Lyon"] * 20 + ["Nantes"] * 10 + \
           ["Bordeaux"] * 5 + ["Marseille"] * 5  # Skew intentionnel : Paris domine
produits = ["A", "B", "C", "D", "E"]

data = [(
    i,
    random.choice(villes),
    random.choice(produits),
    round(random.uniform(10.0, 5000.0), 2),
    f"2024-{random.randint(1,12):02d}"
) for i in range(100_000)]

df = spark.createDataFrame(data, ["id", "ville", "produit", "montant", "mois"])
print(f"Lignes créées : {df.count()}")
print(f"Partitions initiales : {df.rdd.getNumPartitions()}")

# ─── 2. Observation du skew ──────────────────────────────────────────────────
print("\n=== Distribution par ville (skew intentionnel) ===")
df.groupBy("ville").count().orderBy(F.col("count").desc()).show()

print("=== Taille des partitions avant repartitionnement ===")
df.rdd.mapPartitionsWithIndex(
    lambda i, it: [(i, sum(1 for _ in it))]
).toDF(["partition", "nb_lignes"]).orderBy("partition").show()

# ─── 3. Repartitionnement ────────────────────────────────────────────────────
print("\n=== Repartitionnement par ville (hash) ===")
df_repart = df.repartition(8, F.col("ville"))
print(f"Partitions après repartition : {df_repart.rdd.getNumPartitions()}")

df_repart.rdd.mapPartitionsWithIndex(
    lambda i, it: [(i, sum(1 for _ in it))]
).toDF(["partition", "nb_lignes"]).orderBy("partition").show()

# ─── 4. Cache et réutilisation ───────────────────────────────────────────────
print("\n=== Démonstration du cache ===")

# Pipeline coûteux
df_prepare = df \
    .filter(F.col("montant") > 100) \
    .withColumn("montant_ttc", F.round(F.col("montant") * 1.2, 2)) \
    .withColumn("categorie",
        F.when(F.col("montant") > 2000, "GOLD")
         .when(F.col("montant") > 500, "SILVER")
         .otherwise("BRONZE")
    )

# Sans cache
t0 = time.time()
df_prepare.count()
df_prepare.groupBy("ville").agg(F.sum("montant")).count()
t_sans_cache = time.time() - t0
print(f"Temps sans cache  : {t_sans_cache:.2f}s")

# Avec cache
df_prepare.cache()
df_prepare.count()  # Matérialisation

t0 = time.time()
df_prepare.count()
df_prepare.groupBy("ville").agg(F.sum("montant")).count()
t_avec_cache = time.time() - t0
print(f"Temps avec cache  : {t_avec_cache:.2f}s")
print(f"Gain              : {t_sans_cache / max(t_avec_cache, 0.001):.1f}x")

df_prepare.unpersist()

# ─── 5. Broadcast variable ───────────────────────────────────────────────────
print("\n=== Broadcast variable ===")
objectifs = {"Paris": 200000.0, "Lyon": 80000.0, "Nantes": 40000.0,
             "Bordeaux": 30000.0, "Marseille": 30000.0}
objectifs_bc = sc.broadcast(objectifs)

@F.udf("double")
def get_objectif(ville):
    return objectifs_bc.value.get(ville, 0.0)

df_perf = df.groupBy("ville").agg(F.round(F.sum("montant"), 2).alias("ca")) \
  .withColumn("objectif", get_objectif(F.col("ville"))) \
  .withColumn("taux_%", F.round(F.col("ca") / F.col("objectif") * 100, 1)) \
  .orderBy(F.col("taux_%").desc())
df_perf.show()

objectifs_bc.unpersist()

# ─── 6. Accumulateur ─────────────────────────────────────────────────────────
print("\n=== Accumulateur ===")
acc_negatifs = sc.accumulator(0)
acc_nulls    = sc.accumulator(0)

def verifier(row):
    if row["montant"] is None:
        acc_nulls.add(1)
    elif row["montant"] < 0:
        acc_negatifs.add(1)
    return True

df.rdd.filter(verifier).count()
print(f"Montants négatifs : {acc_negatifs.value}")
print(f"Montants nulls    : {acc_nulls.value}")

# ─── 7. Collecte sécurisée ───────────────────────────────────────────────────
print("\n=== Collecte sécurisée ===")
n = df.count()
print(f"Nombre total de lignes : {n}")

# Collecte conditionnelle
resultat_agg = df.groupBy("produit") \
    .agg(F.round(F.sum("montant"), 2).alias("ca")) \
    .orderBy(F.col("ca").desc())

if resultat_agg.count() < 10_000:
    rows = resultat_agg.collect()
    print("\nCA par produit :")
    for row in rows:
        print(f"  Produit {row['produit']} : {row['ca']:>12.2f} €")

# ─── 8. Écriture avec partitionnement ────────────────────────────────────────
print("\n=== Écriture partitionnée ===")
df.coalesce(4) \
  .write \
  .mode("overwrite") \
  .partitionBy("ville") \
  .parquet("/tmp/spark_output/ventes_parquet/")

df_relu = spark.read.parquet("/tmp/spark_output/ventes_parquet/")
print(f"Lignes relues depuis Parquet : {df_relu.count()}")
print(f"Partitions du DataFrame relu : {df_relu.rdd.getNumPartitions()}")

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Partition** | Unité de parallélisme — 1 partition = 1 Task = 1 cœur CPU |
| **Nombre de partitions** | 2-4× le nombre de cœurs ; `spark.sql.shuffle.partitions` = 200 par défaut (souvent trop élevé) |
| **`repartition()`** | Shuffle complet — peut augmenter ou réduire, équilibré, coûteux |
| **`coalesce()`** | Fusion locale — ne peut que réduire, sans shuffle, moins coûteux |
| **Data skew** | Partitions déséquilibrées → stragglers ; corriger avec salting, AQE ou broadcast |
| **`cache()`** | Raccourci pour `MEMORY_AND_DISK` — toujours suivre d'une action pour matérialiser |
| **`persist()`** | Contrôle fin du niveau de stockage (RAM, disque, sérialisé, répliqué) |
| **Broadcast variable** | Partager une petite structure Python vers tous les Executors efficacement |
| **Accumulateur** | Collecter des métriques depuis les Executors vers le Driver |
| **Parquet** | Format de référence en production — colonnaire, compressé, partitionnable |
| **`collect()`** | ⚠️ Rapatrie tout au Driver — toujours limiter (`limit()`, `take()`, `count()` d'abord) |

---

## Exercices

### Exercice 1 — Impact du partitionnement (30 min)
> 1. Créer un DataFrame de 500 000 lignes avec `spark.range(500_000)`
> 2. Afficher le nombre de partitions initial
> 3. Tester `repartition(2)`, `repartition(16)`, `repartition(200)` et mesurer le temps d'un `groupBy().count()` pour chacun
> 4. Conclure : quel est le nombre de partitions optimal pour votre machine ?

### Exercice 2 — Détection et correction du skew (35 min)
> 1. Créer un DataFrame de 1 million de lignes où 90% des valeurs d'une colonne `categorie` valent `"A"`
> 2. Observer la distribution des partitions après un `groupBy("categorie").count()`
> 3. Implémenter la technique du **salting** pour équilibrer la charge
> 4. Comparer les temps d'exécution avant et après correction

### Exercice 3 — Stratégie de cache (25 min)
> Construire un pipeline en 5 étapes (lecture → filtre → join → agrégation → tri).  
> 1. Mesurer le temps total sans cache
> 2. Identifier l'étape intermédiaire la plus coûteuse (via Spark UI)
> 3. Mettre en cache le DataFrame après cette étape et relancer le pipeline
> 4. Mesurer le gain de performance
> 5. Utiliser `StorageLevel.DISK_ONLY` et comparer avec `MEMORY_ONLY`

### Exercice 4 — Écriture et lecture optimisées (30 min)
> À partir d'un jeu de données de ventes par pays et par année :
> 1. Écrire en CSV, Parquet et ORC — comparer les tailles sur disque
> 2. Écrire en Parquet partitionné par `pays` et `annee`
> 3. Lire uniquement les données `pays=France AND annee=2024` et vérifier dans le plan d'exécution que Spark ne lit que les bons sous-répertoires (*partition pruning*)
> 4. Comparer le temps de lecture avec et sans partition pruning

---

## Pour aller plus loin

- 📖 **Tuning Spark** : [spark.apache.org/docs/latest/tuning.html](https://spark.apache.org/docs/latest/tuning.html)
- 📖 **AQE (Adaptive Query Execution)** : [spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution](https://spark.apache.org/docs/latest/sql-performance-tuning.html#adaptive-query-execution)
- 📄 **Data Skew in Apache Spark** : article Databricks sur les stratégies anti-skew
- 📄 **Dremel (Parquet)** : *Dremel: Interactive Analysis of Web-Scale Datasets* — Melnik et al., VLDB 2010
- 🛠️ **Delta Lake** : [delta.io](https://delta.io) — couche transactionnelle ACID sur Parquet

---

*Module précédent → **Module 5 : Transformations de type reducer***  
*Module suivant → **Module 7 : Interaction avec des sources externes***
