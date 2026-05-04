# Module 4 — Transformations de type *mapper*

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 3 — Les structures de données Spark

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Distinguer transformations *lazy* et actions *eager*, et comprendre leurs implications
- Appliquer les transformations `map()` et `flatMap()` sur des RDD
- Manipuler les colonnes d'un DataFrame avec `select()`, `withColumn()`, `filter()`
- Créer et utiliser des UDF (*User-Defined Functions*) simples et vectorisées
- Choisir la bonne transformation selon le contexte et les contraintes de performance

---

## 1. Transformations *lazy* vs actions *eager*

### 1.1 Rappel du principe

Comme vu au Module 3, toutes les opérations Spark sont soit des **transformations** (lazy), soit des **actions** (eager). Ce principe est central pour comprendre les *mappers*.

```
TRANSFORMATIONS (lazy)          ACTIONS (eager)
─────────────────────           ───────────────
map()       → nouveau RDD/DF    count()    → Long
filter()    → nouveau RDD/DF    collect()  → List
flatMap()   → nouveau RDD/DF    show()     → Unit (affichage)
select()    → nouveau DF        take(n)    → List
withColumn()→ nouveau DF        first()    → Row
groupBy()   → RelGroupedDF      write()    → fichier
join()      → nouveau DF        toPandas() → DataFrame Pandas

Aucun calcul                    Déclenche l'exécution
n'est effectué                  complète du DAG
```

### 1.2 Pourquoi la lazy evaluation est un avantage

```python
# Ce pipeline NE CALCULE RIEN à ce stade
df = spark.read.parquet("data/transactions/")   # ← lecture différée
df = df.filter(F.col("montant") > 100)           # ← filtrage différé
df = df.select("client_id", "montant")           # ← projection différée
df = df.withColumn("tva", F.col("montant") * 0.2)# ← calcul différé

# Catalyst peut maintenant :
# 1. Fusionner filter + select en un seul passage
# 2. Pousser la projection au niveau de la lecture Parquet
#    (ne lire que les colonnes nécessaires depuis le disque)
# 3. Appliquer le filtre avant de charger les données en mémoire

df.show()   # ← ici seulement, Spark lit, filtre, projette et calcule
```

Sans lazy evaluation, chaque transformation nécessiterait un passage complet sur toutes les données — ce qui serait catastrophique pour les performances.

### 1.3 Forcer l'évaluation sans action finale

Parfois on veut matérialiser un DataFrame intermédiaire (pour le réutiliser plusieurs fois sans le recalculer). On utilise alors `cache()` ou `persist()` combiné avec une action légère :

```python
df_prepare = df.filter(...).withColumn(...).cache()
df_prepare.count()   # Force la matérialisation et le stockage en cache

# Maintenant df_prepare est en mémoire — les prochaines actions seront rapides
df_prepare.show()
df_prepare.groupBy("ville").count().show()
```

---

## 2. Transformations *mapper* sur les RDD

### 2.1 `map()` — transformation élément par élément

`map()` applique une fonction à **chaque élément** du RDD et retourne un nouveau RDD de même taille. C'est la transformation la plus fondamentale.

```
RDD source  : [1,  2,  3,  4,  5]
              ↓   ↓   ↓   ↓   ↓    f(x) = x * 2
RDD résultat: [2,  4,  6,  8, 10]
```

```python
rdd = sc.parallelize([1, 2, 3, 4, 5])

# Avec une lambda
rdd_double = rdd.map(lambda x: x * 2)
print(rdd_double.collect())   # [2, 4, 6, 8, 10]

# Avec une fonction nommée (préférable pour la lisibilité)
def normaliser(valeur, minimum, maximum):
    return (valeur - minimum) / (maximum - minimum)

rdd_normalise = rdd.map(lambda x: normaliser(x, 1, 5))
print(rdd_normalise.collect())  # [0.0, 0.25, 0.5, 0.75, 1.0]

# Sur des tuples (paires clé-valeur)
rdd_tuples = sc.parallelize([("Alice", 1500.0), ("Bob", 800.0)])
rdd_avec_tva = rdd_tuples.map(lambda t: (t[0], t[1], t[1] * 0.2))
print(rdd_avec_tva.collect())
# [('Alice', 1500.0, 300.0), ('Bob', 800.0, 160.0)]
```

### 2.2 `flatMap()` — map + aplatissement

`flatMap()` applique une fonction à chaque élément, mais cette fonction **retourne une liste** (ou tout itérable). Le résultat est **aplati** en un seul RDD.

```
RDD source  : ["bonjour monde", "spark est puissant"]
              ↓                  ↓
              ["bonjour","monde"] ["spark","est","puissant"]
                              ↓ aplatissement
RDD résultat: ["bonjour", "monde", "spark", "est", "puissant"]
```

```python
rdd_lignes = sc.parallelize([
    "bonjour monde",
    "spark est puissant",
    "pyspark et python"
])

# map() → RDD de listes
rdd_map = rdd_lignes.map(lambda ligne: ligne.split(" "))
print(rdd_map.collect())
# [['bonjour', 'monde'], ['spark', 'est', 'puissant'], ['pyspark', 'et', 'python']]

# flatMap() → RDD de mots (aplati)
rdd_mots = rdd_lignes.flatMap(lambda ligne: ligne.split(" "))
print(rdd_mots.collect())
# ['bonjour', 'monde', 'spark', 'est', 'puissant', 'pyspark', 'et', 'python']

print(f"map    : {rdd_map.count()} éléments")     # 3 listes
print(f"flatMap: {rdd_mots.count()} éléments")    # 8 mots
```

**Cas d'usage typiques de `flatMap()` :**

```python
# Expansion de données (1 ligne → plusieurs lignes)
rdd_commandes = sc.parallelize([
    {"client": "Alice", "produits": ["A", "B", "C"]},
    {"client": "Bob",   "produits": ["B", "D"]},
])
rdd_lignes_produit = rdd_commandes.flatMap(
    lambda c: [(c["client"], p) for p in c["produits"]]
)
print(rdd_lignes_produit.collect())
# [('Alice', 'A'), ('Alice', 'B'), ('Alice', 'C'), ('Bob', 'B'), ('Bob', 'D')]

# Filtrage + transformation combinés (retourner [] pour "supprimer" un élément)
rdd_nombres = sc.parallelize(range(-5, 6))
rdd_racines = rdd_nombres.flatMap(
    lambda x: [x ** 0.5] if x >= 0 else []   # Ignorer les négatifs
)
print(rdd_racines.collect())
# [0.0, 1.0, 1.414..., 1.732..., 2.0, 2.236...]
```

### 2.3 `mapPartitions()` — transformation par partition

`mapPartitions()` applique une fonction à une **partition entière** plutôt qu'à chaque élément individuellement. C'est utile pour amortir le coût d'une initialisation coûteuse (connexion BDD, chargement d'un modèle ML...).

```python
# Avec map() : connexion BDD ouverte et fermée pour CHAQUE élément ❌
rdd.map(lambda x: connexion_bd.query(x))

# Avec mapPartitions() : 1 connexion par partition ✅
def traiter_partition(elements):
    connexion = ouvrir_connexion_bd()   # 1 fois par partition
    resultats = [connexion.query(e) for e in elements]
    connexion.fermer()
    return resultats

rdd_resultat = rdd.mapPartitions(traiter_partition)
```

```python
# Exemple concret : enrichissement avec un modèle ML
def scorer_partition(partition):
    import joblib
    modele = joblib.load("modele.pkl")   # Chargé 1 fois par partition
    for ligne in partition:
        score = modele.predict([ligne.features])[0]
        yield (ligne.id, score)

rdd_scores = rdd_donnees.mapPartitions(scorer_partition)
```

### 2.4 `mapPartitionsWithIndex()` — avec numéro de partition

```python
def afficher_partition(index, elements):
    for element in elements:
        yield f"Partition {index} : {element}"

rdd = sc.parallelize(range(10), numSlices=3)
rdd.mapPartitionsWithIndex(afficher_partition).collect()
# ['Partition 0 : 0', 'Partition 0 : 1', 'Partition 0 : 2',
#  'Partition 1 : 3', 'Partition 1 : 4', 'Partition 1 : 5',
#  'Partition 2 : 6', 'Partition 2 : 7', 'Partition 2 : 8', 'Partition 2 : 9']
```

### 2.5 `filter()` sur les RDD

`filter()` conserve uniquement les éléments pour lesquels la fonction retourne `True`.

```python
rdd = sc.parallelize(range(1, 21))

rdd_pairs   = rdd.filter(lambda x: x % 2 == 0)
rdd_grands  = rdd.filter(lambda x: x > 10)
rdd_premier = rdd.filter(lambda x: all(x % i != 0 for i in range(2, x)) and x > 1)

print(rdd_pairs.collect())    # [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
print(rdd_grands.collect())   # [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
```

---

## 3. Transformations *mapper* sur les DataFrames

### 3.1 `select()` — projection de colonnes

`select()` sélectionne et transforme des colonnes. C'est l'équivalent du `SELECT` en SQL.

```python
from pyspark.sql import functions as F

# Sélection simple
df.select("nom", "ville").show()

# Avec renommage
df.select(
    F.col("nom").alias("vendeur"),
    F.col("montant")
).show()

# Avec transformations inline
df.select(
    "nom",
    "montant",
    (F.col("montant") * 1.2).alias("montant_ttc"),
    F.upper(F.col("ville")).alias("ville_maj"),
    F.length(F.col("nom")).alias("len_nom")
).show()

# Sélectionner toutes les colonnes + en ajouter une
df.select("*", (F.col("montant") / 1000).alias("montant_k")).show()

# Exclure une colonne
df.select([c for c in df.columns if c != "id"]).show()
# ou, plus élégant depuis Spark 3.x :
df.drop("id").show()
```

### 3.2 `withColumn()` — ajout ou modification d'une colonne

`withColumn()` ajoute une nouvelle colonne ou remplace une colonne existante.

```python
# Ajouter une colonne calculée
df = df.withColumn("montant_ttc", F.col("montant") * 1.2)

# Modifier une colonne existante (même nom)
df = df.withColumn("ville", F.upper(F.col("ville")))

# Arrondir
df = df.withColumn("montant_arrondi", F.round(F.col("montant"), 2))

# Condition (équivalent de IF/CASE WHEN)
df = df.withColumn("categorie",
    F.when(F.col("montant") > 2000, "GOLD")
     .when(F.col("montant") > 1000, "SILVER")
     .otherwise("BRONZE")
)

# Concaténation de chaînes
df = df.withColumn("nom_ville",
    F.concat(F.col("nom"), F.lit(" ("), F.col("ville"), F.lit(")"))
)

# Conversion de type (cast)
df = df.withColumn("annee_str", F.col("annee").cast("string"))
df = df.withColumn("montant_int", F.col("montant").cast(IntegerType()))
```

### 3.3 `withColumnRenamed()` — renommage de colonne

```python
# Renommer une colonne
df = df.withColumnRenamed("chiffre_affaires", "ca")

# Renommer plusieurs colonnes
for ancien, nouveau in [("nom", "vendeur"), ("montant", "ca")]:
    df = df.withColumnRenamed(ancien, nouveau)
```

### 3.4 `filter()` / `where()` — filtrage de lignes

`filter()` et `where()` sont **strictement synonymes** en Spark. On peut les utiliser indifféremment.

```python
# Syntaxe avec F.col()
df.filter(F.col("montant") > 1000).show()

# Syntaxe SQL string
df.filter("montant > 1000").show()
df.where("ville = 'Paris' AND montant > 500").show()

# Conditions multiples
df.filter(
    (F.col("montant") > 500) &
    (F.col("ville").isin(["Paris", "Lyon"])) &
    F.col("nom").isNotNull()
).show()

# Négation
df.filter(~F.col("ville").isin(["Nantes", "Bordeaux"])).show()

# Filtres sur chaînes
df.filter(F.col("nom").startswith("A")).show()
df.filter(F.col("ville").contains("aris")).show()
df.filter(F.col("nom").rlike("^[A-E].*")).show()   # Regex

# Filtres sur dates
df.filter(F.col("date") >= F.lit("2024-01-01")).show()
df.filter(F.year(F.col("date")) == 2024).show()
```

### 3.5 Les fonctions natives (`pyspark.sql.functions`)

Le module `pyspark.sql.functions` fournit plusieurs centaines de fonctions optimisées. En voici les principales catégories :

```python
from pyspark.sql import functions as F

# ── Mathématiques ────────────────────────────────────────────────────────────
F.abs(col)          # Valeur absolue
F.round(col, n)     # Arrondi à n décimales
F.ceil(col)         # Arrondi supérieur
F.floor(col)        # Arrondi inférieur
F.sqrt(col)         # Racine carrée
F.pow(col, n)       # Puissance
F.log(col)          # Logarithme naturel
F.exp(col)          # Exponentielle

# ── Chaînes de caractères ─────────────────────────────────────────────────────
F.upper(col)        # Majuscules
F.lower(col)        # Minuscules
F.trim(col)         # Supprimer espaces en début/fin
F.length(col)       # Longueur
F.substring(col, pos, len)  # Sous-chaîne
F.split(col, pattern)       # Découper (retourne ArrayType)
F.regexp_replace(col, pattern, replacement)  # Remplacement regex
F.regexp_extract(col, pattern, idx)          # Extraction regex
F.concat(col1, col2, ...)   # Concaténation
F.concat_ws(sep, col1, ...) # Concaténation avec séparateur
F.format_string("%s : %.2f", col1, col2)    # Formatage

# ── Dates et temps ────────────────────────────────────────────────────────────
F.current_date()            # Date du jour
F.current_timestamp()       # Timestamp actuel
F.year(col)                 # Extraire l'année
F.month(col)                # Extraire le mois
F.dayofweek(col)            # Jour de la semaine (1=Dim, 7=Sam)
F.datediff(col1, col2)      # Différence en jours
F.date_add(col, n)          # Ajouter n jours
F.to_date(col, "yyyy-MM-dd")  # Convertir en date
F.date_format(col, "dd/MM/yyyy")  # Formater une date

# ── Structures (Array, Map) ───────────────────────────────────────────────────
F.array(col1, col2)         # Créer un Array
F.array_contains(col, val)  # Test d'appartenance
F.array_distinct(col)       # Valeurs uniques
F.explode(col)              # Dé-imbrication (1 ligne → N lignes)
F.size(col)                 # Taille d'un Array ou Map

# ── Valeurs nulles ────────────────────────────────────────────────────────────
F.isnull(col)               # Test null
F.isnotnull(col)            # Test non null
F.coalesce(col1, col2, ...) # Premier non-null
F.fillna(valeur, subset=[]) # Remplacer les nulls

# ── Divers ───────────────────────────────────────────────────────────────────
F.lit(valeur)               # Valeur littérale constante
F.col("nom")                # Référence à une colonne
F.monotonically_increasing_id()  # ID unique croissant
F.hash(col)                 # Valeur de hachage
F.md5(col)                  # Hash MD5
```

---

## 4. Les UDF — *User-Defined Functions*

### 4.1 Pourquoi les UDF ?

Les fonctions natives de `pyspark.sql.functions` couvrent la grande majorité des besoins. Mais parfois, on doit appliquer une logique métier complexe qui n'existe pas nativement : c'est le rôle des **UDF**.

```python
# Exemple : normalisation d'un nom selon une règle métier spécifique
import re

def normaliser_nom(nom):
    if nom is None:
        return None
    nom = re.sub(r"[^a-zA-ZÀ-ÿ\s-]", "", nom)  # Supprimer les caractères spéciaux
    return " ".join(w.capitalize() for w in nom.strip().split())
```

### 4.2 Création et enregistrement d'une UDF

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DoubleType

# ── Méthode 1 : décorateur ───────────────────────────────────────────────────
@udf(returnType=StringType())
def normaliser_nom(nom):
    if nom is None:
        return None
    import re
    nom = re.sub(r"[^a-zA-ZÀ-ÿ\s-]", "", nom)
    return " ".join(w.capitalize() for w in nom.strip().split())

# ── Méthode 2 : enregistrement explicite ─────────────────────────────────────
def calculer_remise(montant, taux):
    if montant is None or taux is None:
        return None
    return round(montant * (1 - taux / 100), 2)

calculer_remise_udf = udf(calculer_remise, DoubleType())

# ── Méthode 3 : enregistrement SQL (utilisation dans spark.sql()) ─────────────
spark.udf.register("normaliser_nom_sql", normaliser_nom, StringType())
```

### 4.3 Utilisation des UDF

```python
# Dans l'API DataFrame
df = df.withColumn("nom_normalise", normaliser_nom(F.col("nom")))
df = df.withColumn("prix_remise", calculer_remise_udf(F.col("montant"), F.lit(10)))

# Dans Spark SQL (après enregistrement)
spark.sql("SELECT normaliser_nom_sql(nom) AS nom_ok FROM ventes").show()
```

### 4.4 Limitations et pièges des UDF

Les UDF Python ont un coût significatif par rapport aux fonctions natives :

```
Fonction native Spark
  → Exécutée directement dans la JVM
  → Vectorisée et optimisable par Catalyst
  → Très rapide

UDF Python classique
  → Sérialisation Row → Python (via Pickle)
  → Exécution dans l'interpréteur Python
  → Désérialisation Python → JVM
  → Catalyst ne peut pas optimiser l'intérieur de l'UDF
  → Potentiellement 10x plus lent qu'une fonction native
```

**Bonnes pratiques :**

```python
# ❌ UDF inutile — F.upper() existe nativement
@udf(StringType())
def mettre_majuscule(s):
    return s.upper() if s else None

# ✅ Toujours préférer la fonction native
df.withColumn("ville_maj", F.upper(F.col("ville")))

# ❌ UDF avec import dans la fonction → import répété à chaque appel
@udf(StringType())
def traitement(s):
    import re          # Import répété pour chaque élément !
    return re.sub(...)

# ✅ Import au niveau du module ou dans mapPartitions
import re
@udf(StringType())
def traitement(s):
    return re.sub(...) if s else None
```

---

## 5. Les Pandas UDF (UDF vectorisées)

### 5.1 Principe

Les **Pandas UDF** (aussi appelées *vectorized UDF*) résolvent le problème de performance des UDF classiques en traitant les données **par batch (lot)** plutôt qu'élément par élément.

Au lieu de sérialiser chaque `Row` individuellement, Spark transmet des **colonnes entières sous forme de `pandas.Series`** à la fonction Python — ce qui est beaucoup plus efficace grâce à Apache Arrow.

```
UDF classique :    [row1] → Python → JVM → [row2] → Python → JVM → ...
                   Sérialisation à chaque élément

Pandas UDF :       [col1_batch, col2_batch] → Python (pandas) → JVM
                   Sérialisation par batch via Apache Arrow → beaucoup plus rapide
```

### 5.2 Pandas UDF de type `SCALAR`

La fonction reçoit une ou plusieurs `pandas.Series` et retourne une `pandas.Series`.

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np

# ── Exemple 1 : normalisation min-max ────────────────────────────────────────
@pandas_udf(DoubleType())
def normaliser_minmax(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series([0.0] * len(series))
    return (series - min_val) / (max_val - min_val)

df = df.withColumn("montant_norm", normaliser_minmax(F.col("montant")))

# ── Exemple 2 : plusieurs colonnes en entrée ──────────────────────────────────
@pandas_udf(DoubleType())
def score_combine(montant: pd.Series, anciennete: pd.Series) -> pd.Series:
    return (montant * 0.7 + anciennete * 0.3).round(2)

df = df.withColumn("score", score_combine(F.col("montant"), F.col("anciennete")))

# ── Exemple 3 : traitement NLP avec une bibliothèque Python ──────────────────
@pandas_udf(StringType())
def detecter_langue(textes: pd.Series) -> pd.Series:
    from langdetect import detect
    return textes.apply(lambda t: detect(t) if t else None)

df = df.withColumn("langue", detecter_langue(F.col("description")))
```

### 5.3 Pandas UDF de type `GROUPED_MAP`

La fonction reçoit un `pandas.DataFrame` par groupe et retourne un `pandas.DataFrame`. Utile pour des opérations complexes par groupe.

```python
from pyspark.sql.functions import PandasUDFType

schema_sortie = StructType([
    StructField("ville",     StringType(),  True),
    StructField("nom",       StringType(),  True),
    StructField("montant",   DoubleType(),  True),
    StructField("rang",      IntegerType(), True),
])

@pandas_udf(schema_sortie, PandasUDFType.GROUPED_MAP)
def ranger_par_groupe(pdf: pd.DataFrame) -> pd.DataFrame:
    pdf["rang"] = pdf["montant"].rank(ascending=False).astype(int)
    return pdf

df_range = df.groupBy("ville").apply(ranger_par_groupe)
df_range.show()
```

### 5.4 Comparaison des performances

| Type | Vitesse relative | Cas d'usage |
|---|---|---|
| Fonction native `F.*` | ⚡⚡⚡⚡ (référence) | Toujours préférable si disponible |
| Pandas UDF (Arrow) | ⚡⚡⚡ (~10-100x vs UDF) | Logique custom sur grandes colonnes |
| UDF Python classique | ⚡ | Logique simple, petit volume |
| `mapPartitions()` | ⚡⚡⚡ | Chargement d'un modèle ML par partition |

---

## 6. Transformations sur les structures imbriquées

### 6.1 Colonnes de type Array

```python
from pyspark.sql.types import ArrayType

# Créer un DataFrame avec une colonne Array
data = [
    ("Alice", ["Python", "Spark", "SQL"]),
    ("Bob",   ["Java", "Scala"]),
    ("Claire",["Python", "R", "Spark", "ML"]),
]
df_skills = spark.createDataFrame(data, ["nom", "competences"])

# Taille du tableau
df_skills.withColumn("nb_competences", F.size(F.col("competences"))).show()

# Contient une valeur ?
df_skills.filter(F.array_contains(F.col("competences"), "Spark")).show()

# Exploser : 1 ligne par élément du tableau
df_skills.withColumn("competence", F.explode(F.col("competences"))) \
         .select("nom", "competence").show()

# Résultat :
# +------+----------+
# | nom  |competence|
# +------+----------+
# | Alice|    Python|
# | Alice|     Spark|
# | Alice|       SQL|
# | Bob  |      Java|
# ...
```

### 6.2 Colonnes de type Map (dictionnaire)

```python
# Créer un DataFrame avec une colonne Map
data = [
    ("Alice", {"maths": 18.0, "info": 16.5, "anglais": 14.0}),
    ("Bob",   {"maths": 12.0, "info": 19.0, "anglais": 17.5}),
]
df_notes = spark.createDataFrame(data, ["nom", "notes"])

# Accéder à une clé
df_notes.withColumn("note_maths", F.col("notes")["maths"]).show()

# Exploser le dictionnaire
df_notes.withColumn("kv", F.explode(F.col("notes"))) \
        .select("nom", "kv.key", "kv.value").show()

# Clés et valeurs
df_notes.withColumn("matieres", F.map_keys(F.col("notes"))).show()
df_notes.withColumn("toutes_notes", F.map_values(F.col("notes"))).show()
```

---

## 7. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, ArrayType
)
from pyspark.sql.functions import udf, pandas_udf
import pandas as pd
import re

spark = SparkSession.builder \
    .appName("TransformationsMapper") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
sc = spark.sparkContext

# ─── 1. Démonstration map / flatMap sur RDD ───────────────────────────────────
print("=" * 55)
print("=== RDD : map, flatMap, filter ===")
print("=" * 55)

corpus = sc.parallelize([
    "Apache Spark est un framework de calcul distribué",
    "PySpark permet d'utiliser Spark avec Python",
    "Les DataFrames Spark sont similaires à Pandas",
    "Spark est rapide grâce au calcul en mémoire",
])

# Word count avec map + flatMap
word_count = corpus \
    .flatMap(lambda ligne: re.sub(r"[^\w\s]", "", ligne.lower()).split()) \
    .map(lambda mot: (mot, 1)) \
    .reduceByKey(lambda a, b: a + b) \
    .filter(lambda x: x[1] > 1) \
    .sortBy(lambda x: -x[1])

print("\nMots apparaissant plusieurs fois :")
for mot, count in word_count.collect():
    print(f"  {mot:20s} : {count}")

# ─── 2. DataFrame : select, withColumn, filter ───────────────────────────────
print("\n" + "=" * 55)
print("=== DataFrame : transformations mapper ===")
print("=" * 55)

schema = StructType([
    StructField("id",         IntegerType(), False),
    StructField("nom",        StringType(),  True),
    StructField("ville",      StringType(),  True),
    StructField("montant",    DoubleType(),  True),
    StructField("competences",ArrayType(StringType()), True),
])

data = [
    (1, "alice martin",  "paris",  1500.0, ["Python", "Spark", "SQL"]),
    (2, "BOB DUPONT",    "LYON",    800.0, ["Java", "Scala"]),
    (3, "Claire Petit",  "Paris",  2200.0, ["Python", "ML", "Spark"]),
    (4, "david-leroy",   "nantes",  950.0, ["R", "Python"]),
    (5, "Emma GARNIER",  "Lyon",   3000.0, ["Spark", "Kafka", "SQL"]),
    (6, None,            "Paris",   400.0, ["Python"]),
]

df = spark.createDataFrame(data, schema=schema)

print("\nDataFrame brut :")
df.show(truncate=False)

# ─── 3. UDF de normalisation ──────────────────────────────────────────────────
@udf(StringType())
def normaliser_nom(nom):
    if nom is None:
        return "INCONNU"
    nom = re.sub(r"[^a-zA-ZÀ-ÿ\s-]", " ", nom)
    return " ".join(w.capitalize() for w in nom.strip().split())

# ─── 4. Pipeline de transformations ──────────────────────────────────────────
df_propre = df \
    .filter(F.col("montant").isNotNull()) \
    .withColumn("nom",           normaliser_nom(F.col("nom"))) \
    .withColumn("ville",         F.initcap(F.col("ville"))) \
    .withColumn("montant_ttc",   F.round(F.col("montant") * 1.2, 2)) \
    .withColumn("categorie",
        F.when(F.col("montant") >= 2000, "GOLD")
         .when(F.col("montant") >= 1000, "SILVER")
         .otherwise("BRONZE")
    ) \
    .withColumn("nb_competences", F.size(F.col("competences"))) \
    .withColumn("a_spark",        F.array_contains(F.col("competences"), "Spark"))

print("\nDataFrame après transformations mapper :")
df_propre.select(
    "nom", "ville", "montant", "montant_ttc",
    "categorie", "nb_competences", "a_spark"
).show(truncate=False)

# ─── 5. Pandas UDF ───────────────────────────────────────────────────────────
@pandas_udf(DoubleType())
def normaliser_minmax(series: pd.Series) -> pd.Series:
    return (series - series.min()) / (series.max() - series.min())

df_propre = df_propre.withColumn(
    "montant_norm", F.round(normaliser_minmax(F.col("montant")), 4)
)

print("\nAvec normalisation min-max (Pandas UDF) :")
df_propre.select("nom", "montant", "montant_norm").show()

# ─── 6. Explosion des compétences ────────────────────────────────────────────
df_competences = df_propre \
    .withColumn("competence", F.explode(F.col("competences"))) \
    .groupBy("competence") \
    .agg(
        F.count("*").alias("nb_personnes"),
        F.round(F.avg("montant"), 2).alias("ca_moyen_vendeurs")
    ) \
    .orderBy(F.col("nb_personnes").desc())

print("\nCompétences les plus répandues :")
df_competences.show()

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Lazy evaluation** | Les transformations construisent un plan — rien n'est calculé avant une action |
| **`map()`** | 1 élément → 1 élément, taille du RDD inchangée |
| **`flatMap()`** | 1 élément → N éléments, aplatissement du résultat |
| **`mapPartitions()`** | Fonction appliquée à la partition entière — idéal pour amortir des initialisations |
| **`select()`** | Projection de colonnes avec transformations inline |
| **`withColumn()`** | Ajout ou modification d'une colonne |
| **`filter()`** | Sélection de lignes selon une condition |
| **Fonctions natives** | Toujours préférer `F.*` aux UDF — optimisées par Catalyst |
| **UDF Python** | Logique custom — coût de sérialisation Python ↔ JVM |
| **Pandas UDF** | UDF vectorisées via Apache Arrow — bien plus rapides que les UDF classiques |
| **`explode()`** | Transforme une colonne Array/Map en plusieurs lignes |

---

## Exercices

### Exercice 1 — RDD et traitement de texte (30 min)
> Télécharger un texte long en français (par exemple un roman du Projet Gutenberg).  
> 1. Le charger avec `sc.textFile()`
> 2. Nettoyer le texte avec `flatMap()` (ponctuation, minuscules, mots vides)
> 3. Calculer la fréquence de chaque mot avec `map()` + `reduceByKey()`
> 4. Afficher les 20 mots les plus fréquents
> 5. Calculer la longueur moyenne des mots avec `map()` + `reduce()`

### Exercice 2 — Pipeline DataFrame (30 min)
> À partir d'un fichier CSV de clients (réels ou générés), construire un pipeline qui :  
> 1. Filtre les clients avec un champ `email` valide (regex)
> 2. Normalise les noms (majuscule initiale, pas de caractères spéciaux)
> 3. Ajoute une colonne `age_groupe` : `"18-25"`, `"26-40"`, `"41+"` selon la colonne `age`
> 4. Ajoute une colonne `anciennete_jours` : différence entre aujourd'hui et la colonne `date_inscription`
> 5. Sélectionne et renomme uniquement les colonnes utiles

### Exercice 3 — UDF vs fonction native (20 min)
> 1. Écrire une UDF Python qui met en majuscule la première lettre de chaque mot (`title case`)
> 2. Écrire l'équivalent avec `F.initcap()` (fonction native)
> 3. Mesurer le temps d'exécution des deux approches sur 100 000 lignes avec `time.time()`
> 4. Comparer les plans d'exécution avec `.explain()`

### Exercice 4 — Pandas UDF et ML (40 min)
> Créer un DataFrame de 10 000 lignes avec des features numériques aléatoires.  
> 1. Écrire une Pandas UDF qui applique une **standardisation z-score** à une colonne : `(x - mean) / std`
> 2. Écrire une Pandas UDF qui applique un modèle scikit-learn pré-entraîné (`joblib.load()`) pour scorer chaque ligne
> 3. Comparer les temps d'exécution avec une UDF classique équivalente

---

## Pour aller plus loin

- 📖 **API RDD** : [spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.RDD.html)
- 📖 **Fonctions DataFrame** : [spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html](https://spark.apache.org/docs/latest/api/python/reference/pyspark.sql/functions.html)
- 📖 **Pandas UDF** : [spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html](https://spark.apache.org/docs/latest/api/python/user_guide/sql/arrow_pandas.html)
- 📄 **Apache Arrow** : le format colonnaire qui rend les Pandas UDF efficaces — [arrow.apache.org](https://arrow.apache.org)
- 🛠️ **spark-testing-base** : framework de tests pour les transformations Spark — [github.com/holdenk/spark-testing-base](https://github.com/holdenk/spark-testing-base)

---

*Module précédent → **Module 3 : Les structures de données Spark***  
*Module suivant → **Module 5 : Transformations de type reducer***
