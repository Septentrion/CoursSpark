# Module 5 — Transformations de type *reducer*

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 4 — Transformations de type mapper

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Comprendre le rôle et le coût des opérations de *shuffle* réseau
- Appliquer les transformations `reduceByKey()`, `groupByKey()` et `aggregateByKey()` sur des RDD
- Réaliser des agrégations complexes sur des DataFrames avec `groupBy()` et `agg()`
- Utiliser les fonctions de fenêtre (*Window functions*) pour des calculs avancés
- Réaliser et optimiser des jointures entre DataFrames

---

## 1. Le *reducer* : principe et coût réseau

### 1.1 Qu'est-ce qu'un reducer ?

Un **reducer** est une opération qui **agrège plusieurs éléments en un résultat unique**, généralement après regroupement par clé. Contrairement aux mappers (qui traitent chaque élément indépendamment), les reducers nécessitent de **rassembler des données issues de différentes partitions** — ce qui implique un transfert réseau appelé **shuffle**.

```
MAPPER                          REDUCER
Partition 1 : [A:1, B:3, A:2]  →  Partition A : [A:1, A:2, A:4] → A:7
Partition 2 : [B:1, A:4, C:2]  →  Partition B : [B:3, B:1]      → B:4
Partition 3 : [C:1, B:2, C:3]  →  Partition C : [C:2, C:1, C:3] → C:6
                    ↑
              SHUFFLE : redistribution réseau par clé
```

### 1.2 Le shuffle : opération coûteuse

Le **shuffle** est l'opération la plus coûteuse dans Spark. Il implique :

1. **Écriture sur disque** : chaque Executor sérialise et écrit les données à transférer
2. **Transfert réseau** : les données sont copiées vers les Executors destinataires
3. **Lecture et désérialisation** : les Executors destinataires reconstituent les données

```
Opérations déclenchant un shuffle :
  groupBy()       reduceByKey()     join()
  distinct()      repartition()     sortBy()
  groupByKey()    cogroup()         intersection()
```

**Stratégies pour minimiser le coût des shuffles :**

```python
# ❌ Mauvaise pratique : filtrer APRÈS un groupBy (shuffle inutile sur les données filtrées)
df.groupBy("ville").count().filter(F.col("count") > 10)

# ✅ Bonne pratique : filtrer AVANT le groupBy (moins de données à shuffler)
df.filter(F.col("montant") > 0).groupBy("ville").count().filter(F.col("count") > 10)

# ✅ Utiliser le broadcast join pour les petites tables (évite le shuffle)
from pyspark.sql.functions import broadcast
df_large.join(broadcast(df_petite), "id")

# ✅ Persister un DataFrame souvent réutilisé pour éviter de recalculer le shuffle
df_groupe = df.groupBy("ville").agg(...).cache()
df_groupe.count()   # Matérialisation
```

---

## 2. Reducers sur les RDD

### 2.1 `reduce()` — agrégation globale

`reduce()` applique une fonction binaire associative à tous les éléments d'un RDD pour produire une **valeur scalaire unique**. C'est une action (pas une transformation).

```python
rdd = sc.parallelize([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Somme
total = rdd.reduce(lambda a, b: a + b)
print(f"Somme : {total}")   # 55

# Maximum
maximum = rdd.reduce(lambda a, b: a if a > b else b)
print(f"Max : {maximum}")   # 10

# ⚠️ La fonction doit être commutative ET associative
# (le résultat doit être indépendant de l'ordre de traitement)
# ✅ (a + b) = (b + a)       → commutative
# ✅ (a + b) + c = a + (b + c) → associative
# ❌ (a - b)                 → ni commutative ni associative → résultat imprévisible
```

### 2.2 `reduceByKey()` — agrégation par clé

`reduceByKey()` opère sur un **RDD de paires (clé, valeur)** et agrège les valeurs ayant la même clé.

```
RDD : [("A",1), ("B",3), ("A",2), ("B",1), ("A",4), ("C",2)]
                           ↓ reduceByKey(+)
RDD : [("A",7), ("B",4), ("C",2)]
```

```python
rdd_ventes = sc.parallelize([
    ("Paris",  1500.0), ("Lyon",   800.0), ("Paris", 2200.0),
    ("Lyon",  1100.0),  ("Nantes",  950.0), ("Paris", 3000.0),
])

# Somme par ville
ca_par_ville = rdd_ventes.reduceByKey(lambda a, b: a + b)
print(ca_par_ville.collect())
# [('Paris', 6700.0), ('Lyon', 1900.0), ('Nantes', 950.0)]

# Maximum par ville
max_par_ville = rdd_ventes.reduceByKey(lambda a, b: max(a, b))
print(max_par_ville.collect())
# [('Paris', 3000.0), ('Lyon', 1100.0), ('Nantes', 950.0)]

# Comptage (word count pattern)
rdd_mots = sc.parallelize(["spark spark python spark python java".split()[i]
                            for i in range(6)])
counts = rdd_mots.map(lambda mot: (mot, 1)).reduceByKey(lambda a, b: a + b)
print(counts.collect())
# [('spark', 3), ('python', 2), ('java', 1)]
```

**Pourquoi `reduceByKey()` est-il plus efficace que `groupByKey()` ?**

```
groupByKey()  : Envoie TOUTES les valeurs au réseau, puis réduit
  Partition 1 : ("A",1), ("A",2) → réseau → Reducer A : [1, 2, 4] → 7
  Partition 2 : ("A",4)          → réseau ↗

reduceByKey() : Réduit LOCALEMENT d'abord, puis envoie le sous-total
  Partition 1 : ("A",1), ("A",2) → réduction locale → ("A",3) → réseau → Reducer A : 3+4=7
  Partition 2 : ("A",4)          → réduction locale → ("A",4) → réseau ↗

  → reduceByKey transfère BEAUCOUP moins de données sur le réseau
```

### 2.3 `groupByKey()` — regroupement par clé

`groupByKey()` regroupe toutes les valeurs d'une même clé dans un **itérable**. À utiliser uniquement quand on ne peut pas exprimer l'agrégation avec `reduceByKey()`.

```python
rdd_notes = sc.parallelize([
    ("Alice", 15), ("Bob", 12), ("Alice", 18),
    ("Bob", 16),   ("Alice", 14), ("Claire", 19),
])

# groupByKey → itérable de valeurs par clé
groupes = rdd_notes.groupByKey()
for etudiant, notes in groupes.collect():
    liste = list(notes)
    print(f"{etudiant} : {liste} → moyenne : {sum(liste)/len(liste):.1f}")
# Alice  : [15, 18, 14] → moyenne : 15.7
# Bob    : [12, 16]     → moyenne : 14.0
# Claire : [19]         → moyenne : 19.0

# ⚠️ ATTENTION : groupByKey() charge TOUTES les valeurs en mémoire par clé
# Si une clé a des millions de valeurs → risque de OutOfMemoryError
# Préférer aggregateByKey() ou combineByKey() dans ce cas
```

### 2.4 `aggregateByKey()` — agrégation flexible par clé

`aggregateByKey()` est la version la plus générale et la plus puissante. Elle permet de calculer un agrégat d'un **type différent** de celui des valeurs d'entrée.

Elle prend trois arguments :
- **`zeroValue`** : valeur initiale de l'accumulateur
- **`seqOp`** : fonction de fusion d'un élément dans l'accumulateur (locale, par partition)
- **`combOp`** : fonction de fusion de deux accumulateurs (entre partitions)

```python
rdd_notes = sc.parallelize([
    ("Alice", 15.0), ("Bob", 12.0), ("Alice", 18.0),
    ("Bob",   16.0), ("Alice", 14.0), ("Claire", 19.0),
])

# Calculer (somme, nombre) par étudiant pour obtenir la moyenne
zero = (0.0, 0)   # accumulateur : (somme, compte)

def seq_op(accum, valeur):
    return (accum[0] + valeur, accum[1] + 1)

def comb_op(accum1, accum2):
    return (accum1[0] + accum2[0], accum1[1] + accum2[1])

somme_compte = rdd_notes.aggregateByKey(zero, seq_op, comb_op)

moyennes = somme_compte.mapValues(lambda sc: round(sc[0] / sc[1], 2))
print(moyennes.collect())
# [('Alice', 15.67), ('Bob', 14.0), ('Claire', 19.0)]
```

### 2.5 `combineByKey()` — agrégation la plus générale

`combineByKey()` est similaire à `aggregateByKey()` mais avec un contrôle total sur la création de l'accumulateur initial.

```python
# Calculer min, max, somme par ville en une seule passe
rdd = sc.parallelize([
    ("Paris", 1500.0), ("Lyon", 800.0), ("Paris", 2200.0),
    ("Lyon", 1100.0), ("Paris", 3000.0),
])

def creer_combineur(valeur):
    return (valeur, valeur, valeur, 1)   # (min, max, somme, count)

def fusionner_valeur(combineur, valeur):
    return (
        min(combineur[0], valeur),
        max(combineur[1], valeur),
        combineur[2] + valeur,
        combineur[3] + 1
    )

def fusionner_combineurs(c1, c2):
    return (min(c1[0], c2[0]), max(c1[1], c2[1]), c1[2]+c2[2], c1[3]+c2[3])

stats = rdd.combineByKey(creer_combineur, fusionner_valeur, fusionner_combineurs)
for ville, (mn, mx, s, cnt) in stats.collect():
    print(f"{ville}: min={mn}, max={mx}, moy={s/cnt:.1f}, n={cnt}")
# Paris: min=1500.0, max=3000.0, moy=2233.3, n=3
# Lyon:  min=800.0,  max=1100.0, moy=950.0,  n=2
```

### 2.6 `sortByKey()` et `sortBy()` — tri

```python
# Trier par clé
rdd_pair.sortByKey(ascending=True).collect()
rdd_pair.sortByKey(ascending=False).collect()

# Trier par valeur
word_count.sortBy(lambda x: x[1], ascending=False).take(10)

# Trier par une expression complexe
rdd.sortBy(lambda x: (-x[1], x[0])).collect()  # D'abord par valeur desc, puis clé asc
```

---

## 3. Reducers sur les DataFrames

### 3.1 `groupBy()` et `agg()` — agrégations

`groupBy()` suivi de `agg()` est la combinaison standard pour les agrégations sur DataFrame.

```python
from pyspark.sql import functions as F

# Agrégation simple
df.groupBy("ville").count().show()
df.groupBy("ville").sum("montant").show()

# Agrégations multiples en une seule passe (recommandé)
df.groupBy("ville").agg(
    F.count("*").alias("nb_vendeurs"),
    F.sum("montant").alias("ca_total"),
    F.avg("montant").alias("ca_moyen"),
    F.min("montant").alias("ca_min"),
    F.max("montant").alias("ca_max"),
    F.stddev("montant").alias("ca_ecart_type"),
    F.collect_list("nom").alias("liste_vendeurs"),
    F.countDistinct("ville").alias("nb_villes_distinctes"),
).show()

# Groupement multi-colonnes
df.groupBy("ville", "annee").agg(
    F.sum("montant").alias("ca"),
    F.count("*").alias("n")
).orderBy("ville", "annee").show()
```

### 3.2 Les fonctions d'agrégation natives

```python
# ── Agrégations numériques ───────────────────────────────────────────────────
F.count("*")              # Nombre de lignes (incluant les nulls)
F.count(F.col("nom"))     # Nombre de valeurs non-null
F.countDistinct("ville")  # Nombre de valeurs distinctes
F.sum("montant")          # Somme
F.avg("montant")          # Moyenne (ignore les nulls)
F.mean("montant")         # Synonyme de avg
F.min("montant")          # Minimum
F.max("montant")          # Maximum
F.stddev("montant")       # Écart-type (échantillon)
F.stddev_pop("montant")   # Écart-type (population)
F.variance("montant")     # Variance
F.skewness("montant")     # Asymétrie (skewness)
F.kurtosis("montant")     # Aplatissement (kurtosis)
F.corr("col1", "col2")    # Corrélation de Pearson
F.covar_samp("c1", "c2")  # Covariance (échantillon)

# ── Agrégations sur collections ──────────────────────────────────────────────
F.collect_list("nom")     # Liste de valeurs (avec doublons)
F.collect_set("ville")    # Ensemble de valeurs (sans doublons)
F.array_join(F.collect_list("nom"), ", ")  # Concaténation en chaîne

# ── Agrégations sur chaînes ──────────────────────────────────────────────────
F.concat_ws(", ", F.collect_list("nom"))  # "Alice, Bob, Claire"

# ── Premier / dernier ────────────────────────────────────────────────────────
F.first("nom", ignorenulls=True)   # Premier élément non-null
F.last("nom",  ignorenulls=True)   # Dernier élément non-null

# ── Percentiles ──────────────────────────────────────────────────────────────
F.percentile_approx("montant", 0.5)          # Médiane (approchée)
F.percentile_approx("montant", [0.25, 0.75]) # Q1 et Q3
```

### 3.3 `pivot()` — tableau croisé dynamique

`pivot()` transforme des valeurs de ligne en colonnes — l'équivalent d'un tableau croisé dynamique Excel.

```python
# Données : ventes par ville et par année
data = [
    ("Paris", 2022, 5000.0), ("Paris", 2023, 6200.0), ("Paris", 2024, 7100.0),
    ("Lyon",  2022, 3100.0), ("Lyon",  2023, 3400.0), ("Lyon",  2024, 3800.0),
    ("Nantes",2022, 1800.0), ("Nantes",2023, 2100.0), ("Nantes",2024, 2400.0),
]
df_ventes = spark.createDataFrame(data, ["ville", "annee", "ca"])

# Pivot : une colonne par année
df_pivot = df_ventes \
    .groupBy("ville") \
    .pivot("annee", [2022, 2023, 2024]) \
    .sum("ca")

df_pivot.show()
# +------+------+------+------+
# | ville|  2022|  2023|  2024|
# +------+------+------+------+
# | Paris|5000.0|6200.0|7100.0|
# |  Lyon|3100.0|3400.0|3800.0|
# |Nantes|1800.0|2100.0|2400.0|
# +------+------+------+------+

# Unpivot (inverse du pivot) — Spark 3.4+
df_pivot.unpivot("ville", ["2022","2023","2024"], "annee", "ca").show()
```

---

## 4. Les fonctions de fenêtre (*Window functions*)

### 4.1 Principe

Les **fonctions de fenêtre** calculent une valeur pour chaque ligne en tenant compte d'un **ensemble de lignes voisines** (la "fenêtre"), sans réduire le nombre de lignes. Elles sont fondamentales pour des calculs comme :

- Rang d'un élément dans son groupe
- Comparaison avec la ligne précédente / suivante
- Cumul progressif
- Moyenne glissante

```python
from pyspark.sql.window import Window

# Définir une fenêtre : partitionnée par ville, triée par montant décroissant
fenetre = Window.partitionBy("ville").orderBy(F.col("montant").desc())
```

### 4.2 Fonctions de rang

```python
fenetre = Window.partitionBy("ville").orderBy(F.col("montant").desc())

df_rang = df.select(
    "nom", "ville", "montant",
    F.rank()       .over(fenetre).alias("rang"),         # Avec ex-aequo + saut
    F.dense_rank() .over(fenetre).alias("rang_dense"),   # Avec ex-aequo sans saut
    F.row_number() .over(fenetre).alias("numero_ligne"),  # Sans ex-aequo, ordre strict
    F.percent_rank().over(fenetre).alias("rang_pct"),     # Rang en proportion [0,1]
    F.ntile(4)     .over(fenetre).alias("quartile"),      # Découpage en N groupes égaux
)
df_rang.show()
```

**Différence entre `rank()`, `dense_rank()` et `row_number()` :**

```
Montant : [3000, 2200, 2200, 1500, 800]

rank()        : [1, 2, 2, 4, 5]   ← ex-aequo en 2, pas de rang 3
dense_rank()  : [1, 2, 2, 3, 4]   ← ex-aequo en 2, rang 3 conservé
row_number()  : [1, 2, 3, 4, 5]   ← pas d'ex-aequo, ordre strict
```

### 4.3 Fonctions de décalage (`lag` / `lead`)

`lag()` et `lead()` accèdent respectivement à la ligne **précédente** et **suivante** dans la fenêtre.

```python
fenetre_temps = Window.partitionBy("region").orderBy("date")

df_evolution = df.withColumn(
    "ca_precedent", F.lag("ca", 1).over(fenetre_temps)
).withColumn(
    "ca_suivant",   F.lead("ca", 1).over(fenetre_temps)
).withColumn(
    "evolution",    F.round(
        (F.col("ca") - F.col("ca_precedent")) / F.col("ca_precedent") * 100, 2
    )
)

df_evolution.select("region", "date", "ca", "ca_precedent", "evolution").show()
# +------+----------+------+-------------+---------+
# |region|      date|    ca| ca_precedent|evolution|
# +------+----------+------+-------------+---------+
# | Paris|2024-01-01|5000.0|         null|     null|
# | Paris|2024-02-01|5500.0|       5000.0|     10.0|
# | Paris|2024-03-01|4800.0|       5500.0|    -12.7|
```

### 4.4 Fonctions de cumul et de glissement

```python
# Fenêtre sans partition (toute la table)
fenetre_globale = Window.orderBy("date")

# Fenêtre glissante : 3 lignes précédentes + ligne courante
fenetre_glissante = Window.partitionBy("region") \
    .orderBy("date") \
    .rowsBetween(-2, 0)   # 2 lignes avant, ligne courante

# Fenêtre cumulée depuis le début jusqu'à la ligne courante
fenetre_cumul = Window.partitionBy("region") \
    .orderBy("date") \
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df_stats = df.withColumn(
    "cumul_ca",      F.sum("ca").over(fenetre_cumul)
).withColumn(
    "moy_glissante", F.round(F.avg("ca").over(fenetre_glissante), 2)
).withColumn(
    "min_cumul",     F.min("ca").over(fenetre_cumul)
).withColumn(
    "max_cumul",     F.max("ca").over(fenetre_cumul)
)
```

**Résumé des types de fenêtres :**

```python
# Fenêtre partitionnée et triée (pour rank, lag, lead)
Window.partitionBy("col").orderBy("col2")

# Fenêtre délimitée en nombre de lignes
Window.partitionBy("col").orderBy("col2").rowsBetween(début, fin)

# Fenêtre délimitée en valeur (rangeBetween)
Window.partitionBy("col").orderBy("col2").rangeBetween(-7, 0)

# Constantes utiles
Window.unboundedPreceding   # Depuis le début de la partition
Window.unboundedFollowing   # Jusqu'à la fin de la partition
Window.currentRow           # Ligne courante
```

---

## 5. Les jointures (*joins*)

### 5.1 Types de jointures

```python
# Données d'exemple
df_clients = spark.createDataFrame([
    (1, "Alice", "Paris"), (2, "Bob", "Lyon"),
    (3, "Claire", "Nantes"), (4, "David", "Paris"),
], ["id", "nom", "ville"])

df_commandes = spark.createDataFrame([
    (1, 1, 500.0), (2, 1, 300.0), (3, 2, 800.0),
    (4, 5, 200.0),   # client_id=5 n'existe pas dans df_clients
], ["commande_id", "client_id", "montant"])

# ── INNER JOIN : seulement les lignes présentes dans les deux tables
df_clients.join(df_commandes, df_clients["id"] == df_commandes["client_id"], "inner").show()

# ── LEFT JOIN : toutes les lignes de gauche + correspondances de droite (null sinon)
df_clients.join(df_commandes, df_clients["id"] == df_commandes["client_id"], "left").show()

# ── RIGHT JOIN : toutes les lignes de droite + correspondances de gauche
df_clients.join(df_commandes, df_clients["id"] == df_commandes["client_id"], "right").show()

# ── FULL OUTER JOIN : toutes les lignes des deux côtés
df_clients.join(df_commandes, df_clients["id"] == df_commandes["client_id"], "outer").show()

# ── LEFT ANTI JOIN : lignes de gauche SANS correspondance à droite
df_clients.join(df_commandes, df_clients["id"] == df_commandes["client_id"], "left_anti").show()
# → Clients sans commande

# ── LEFT SEMI JOIN : lignes de gauche AVEC correspondance à droite (sans colonnes droite)
df_clients.join(df_commandes, df_clients["id"] == df_commandes["client_id"], "left_semi").show()
# → Clients ayant au moins une commande

# ── CROSS JOIN : produit cartésien (toutes les combinaisons)
df_clients.crossJoin(df_commandes).show()
```

### 5.2 Jointure sur une colonne de même nom

```python
# Si la colonne de jointure a le même nom dans les deux DataFrames
df1.join(df2, on="id", how="inner")

# La colonne "id" n'apparaît qu'une seule fois dans le résultat
# (contrairement à la jointure avec expression booléenne)
```

### 5.3 Le Broadcast Join — optimisation clé

Quand l'une des tables est **petite** (typiquement < 10 Mo), Spark peut la diffuser à tous les Executors en mémoire, **évitant complètement le shuffle** de la grande table.

```python
# Sans broadcast : shuffle des deux tables → coûteux
df_large.join(df_reference, "code_produit")

# Avec broadcast : df_reference copié en mémoire sur tous les Executors
df_large.join(broadcast(df_reference), "code_produit")
```

```
Sans broadcast :               Avec broadcast :
  df_large  ──shuffle──►        df_large  ──(pas de shuffle)──►
  df_ref    ──shuffle──►        df_ref    ──broadcast──► chaque Executor
                    Reducer                            calcul local
```

**Configuration automatique :**
```python
# Spark applique le broadcast automatiquement si la table < seuil
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "10mb")  # 10 Mo par défaut

# Désactiver le broadcast automatique
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")
```

### 5.4 Éviter les colonnes ambiguës après jointure

```python
# ❌ Problème : deux colonnes "id" après jointure avec expression booléenne
df_joint = df_clients.join(df_commandes,
    df_clients["id"] == df_commandes["client_id"])
# df_joint.select("id")  → AnalysisException: ambiguous column name

# ✅ Solution 1 : renommer avant la jointure
df_c = df_clients.withColumnRenamed("id", "client_id")
df_c.join(df_commandes, "client_id")

# ✅ Solution 2 : utiliser des alias de DataFrame
df_clients.alias("c").join(
    df_commandes.alias("o"),
    F.col("c.id") == F.col("o.client_id")
).select(F.col("c.id"), "nom", "montant")

# ✅ Solution 3 : drop après jointure
df_joint.drop(df_commandes["client_id"])
```

---

## 6. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder \
    .appName("TransformationsReducer") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
sc = spark.sparkContext

# ─── 1. Démonstration RDD Reducers ───────────────────────────────────────────
print("=" * 60)
print("=== RDD : reduce, reduceByKey, aggregateByKey ===")
print("=" * 60)

rdd_ventes = sc.parallelize([
    ("Paris",  ("Alice",   1500.0)),
    ("Lyon",   ("Bob",      800.0)),
    ("Paris",  ("Claire",  2200.0)),
    ("Lyon",   ("David",   1100.0)),
    ("Nantes", ("Emma",     950.0)),
    ("Paris",  ("François",3000.0)),
    ("Nantes", ("Gaëlle",   400.0)),
])

# reduceByKey : somme du CA par ville
ca_par_ville = rdd_ventes \
    .mapValues(lambda v: v[1]) \
    .reduceByKey(lambda a, b: a + b)
print("\nCA par ville (reduceByKey) :")
for ville, ca in sorted(ca_par_ville.collect(), key=lambda x: -x[1]):
    print(f"  {ville:8s} : {ca:8.1f} €")

# aggregateByKey : (min, max, somme, count) en une seule passe
zero = (float("inf"), float("-inf"), 0.0, 0)
stats = rdd_ventes.mapValues(lambda v: v[1]).aggregateByKey(
    zero,
    lambda acc, v: (min(acc[0], v), max(acc[1], v), acc[2]+v, acc[3]+1),
    lambda a, b: (min(a[0],b[0]), max(a[1],b[1]), a[2]+b[2], a[3]+b[3])
)
print("\nStatistiques par ville (aggregateByKey) :")
print(f"  {'Ville':8s} {'Min':>8} {'Max':>8} {'Moy':>8} {'N':>4}")
for ville, (mn, mx, s, n) in sorted(stats.collect()):
    print(f"  {ville:8s} {mn:8.1f} {mx:8.1f} {s/n:8.1f} {n:4d}")

# ─── 2. Démonstration DataFrame Reducers ─────────────────────────────────────
print("\n" + "=" * 60)
print("=== DataFrame : groupBy, agg, pivot, window ===")
print("=" * 60)

data = [
    (1, "Alice",    "Paris",  1500.0, "2024-01"),
    (2, "Bob",      "Lyon",    800.0, "2024-01"),
    (3, "Claire",   "Paris",  2200.0, "2024-02"),
    (4, "David",    "Lyon",   1100.0, "2024-02"),
    (5, "Emma",     "Nantes",  950.0, "2024-01"),
    (6, "François", "Paris",  3000.0, "2024-03"),
    (7, "Gaëlle",   "Nantes",  400.0, "2024-02"),
    (8, "Henri",    "Lyon",   1600.0, "2024-03"),
    (9, "Inès",     "Paris",  1800.0, "2024-02"),
]
df = spark.createDataFrame(data, ["id", "nom", "ville", "ca", "mois"])

# ── groupBy + agg ─────────────────────────────────────────────────────────────
print("\nStatistiques par ville :")
df.groupBy("ville").agg(
    F.count("*").alias("nb"),
    F.round(F.sum("ca"), 2).alias("ca_total"),
    F.round(F.avg("ca"), 2).alias("ca_moyen"),
    F.min("ca").alias("ca_min"),
    F.max("ca").alias("ca_max"),
).orderBy(F.col("ca_total").desc()).show()

# ── pivot ─────────────────────────────────────────────────────────────────────
print("CA par ville et par mois (pivot) :")
df.groupBy("ville") \
  .pivot("mois", ["2024-01", "2024-02", "2024-03"]) \
  .sum("ca") \
  .show()

# ── Window functions ──────────────────────────────────────────────────────────
print("Rang par ville :")
w_rang = Window.partitionBy("ville").orderBy(F.col("ca").desc())
w_cumul = Window.partitionBy("ville").orderBy("mois") \
               .rowsBetween(Window.unboundedPreceding, Window.currentRow)

df.select(
    "nom", "ville", "mois", "ca",
    F.rank()      .over(w_rang).alias("rang"),
    F.dense_rank().over(w_rang).alias("rang_dense"),
    F.round(F.sum("ca").over(w_cumul), 2).alias("ca_cumule"),
    F.round(F.avg("ca").over(
        Window.partitionBy("ville")
              .orderBy("mois")
              .rowsBetween(-1, 1)
    ), 2).alias("moy_glissante"),
).orderBy("ville", "rang").show()

# ── lag / lead ────────────────────────────────────────────────────────────────
print("Évolution mensuelle du CA par ville :")
w_temps = Window.partitionBy("ville").orderBy("mois")
df_evol = df.withColumn("ca_prec", F.lag("ca", 1).over(w_temps)) \
            .withColumn("evolution_%",
                F.round((F.col("ca") - F.col("ca_prec")) / F.col("ca_prec") * 100, 1)
            )
df_evol.select("nom", "ville", "mois", "ca", "ca_prec", "evolution_%") \
       .orderBy("ville", "mois").show()

# ─── 3. Jointure avec broadcast ───────────────────────────────────────────────
print("=" * 60)
print("=== Jointure avec broadcast ===")
print("=" * 60)

df_objectifs = spark.createDataFrame([
    ("Paris",  5000.0), ("Lyon", 3000.0), ("Nantes", 1500.0)
], ["ville", "objectif"])

df_perf = df.groupBy("ville").agg(F.sum("ca").alias("ca_total")) \
  .join(broadcast(df_objectifs), "ville") \
  .withColumn("taux_realisation_%",
      F.round(F.col("ca_total") / F.col("objectif") * 100, 1)
  ) \
  .orderBy(F.col("taux_realisation_%").desc())

df_perf.show()

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Shuffle** | Opération la plus coûteuse — redistribution réseau des données par clé |
| **`reduceByKey()`** | Réduction locale avant shuffle → plus efficace que `groupByKey()` |
| **`groupByKey()`** | Envoie toutes les valeurs au réseau → à éviter sur gros volumes |
| **`aggregateByKey()`** | Version la plus flexible — peut produire un type différent des valeurs |
| **`groupBy().agg()`** | Agrégations multiples en une seule passe sur DataFrame |
| **`pivot()`** | Transforme des valeurs de lignes en colonnes |
| **Window functions** | Calculs sur une fenêtre de lignes sans réduire le DataFrame |
| **`rank()` / `lag()`** | Rang dans un groupe, accès aux lignes précédentes/suivantes |
| **Broadcast join** | Évite le shuffle en diffusant la petite table à tous les Executors |

---

## Exercices

### Exercice 1 — RDD Reducers (30 min)
> À partir d'un RDD de logs web de la forme `(ip, url, durée_ms)` :
> 1. Calculer le nombre de requêtes par IP avec `reduceByKey()`
> 2. Calculer la durée moyenne par URL avec `aggregateByKey()`
> 3. Trouver l'IP avec le plus grand nombre de requêtes
> 4. Comparer les temps d'exécution de `groupByKey()` vs `reduceByKey()` sur le même calcul

### Exercice 2 — Agrégations DataFrame (25 min)
> À partir d'un jeu de données de ventes (généré ou réel) :
> 1. Calculer les statistiques complètes (min, max, moy, écart-type, médiane) par catégorie de produit
> 2. Créer un tableau croisé `pivot()` : catégorie en ligne × trimestre en colonne, valeur = CA total
> 3. Identifier les catégories dont le CA a augmenté chaque trimestre consécutivement

### Exercice 3 — Window functions (35 min)
> Sur un DataFrame de salariés avec colonnes `(id, nom, département, salaire, date_embauche)` :
> 1. Calculer le rang de chaque salarié dans son département par salaire décroissant
> 2. Calculer l'écart entre le salaire de chaque salarié et la moyenne de son département
> 3. Pour chaque salarié, afficher le salaire du collègue embauché juste avant lui (même département)
> 4. Calculer le salaire cumulé par département trié par date d'embauche

### Exercice 4 — Jointures et optimisation (30 min)
> Vous disposez de trois DataFrames : `clients`, `commandes` et `produits`.
> 1. Réaliser une jointure triple pour obtenir : nom client, produit commandé, montant
> 2. Trouver les clients n'ayant passé aucune commande (anti-join)
> 3. Utiliser `broadcast()` sur la table `produits` et comparer le plan d'exécution avec et sans broadcast (`.explain()`)
> 4. Mesurer l'impact sur le temps d'exécution

---

## Pour aller plus loin

- 📖 **Window Functions** : [spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html](https://spark.apache.org/docs/latest/sql-ref-syntax-qry-select-window.html)
- 📖 **Join strategies** : [spark.apache.org/docs/latest/sql-performance-tuning.html](https://spark.apache.org/docs/latest/sql-performance-tuning.html)
- 📄 **Optimisation des shuffles** : article Databricks *"A Tale of Three Apache Spark APIs"*
- 🛠️ **Spark UI — SQL tab** : examiner les plans physiques des jointures pour identifier les stratégies choisies (BroadcastHashJoin, SortMergeJoin, ShuffleHashJoin)

---

*Module précédent → **Module 4 : Transformations de type mapper***  
*Module suivant → **Module 6 : Partitionnement, shuffles et collecte***
