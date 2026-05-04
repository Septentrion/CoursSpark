# Module 6 (bis) — Feature Engineering avec PySpark

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 5 — Transformations de type reducer

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Enrichir un DataFrame avec de nouvelles colonnes calculées
- Créer et appliquer des UDF simples et vectorisées
- Construire des pipelines de traitement reproductibles avec `Pipeline`
- Appliquer les transformations classiques du feature engineering : binarisation, bucketing, normalisation, logarithme
- Tokeniser et vectoriser du texte avec TF-IDF et `FeatureHasher`
- Assembler des features hétérogènes en un vecteur unique avec `VectorAssembler`

---

## 1. Vue d'ensemble du feature engineering

### 1.1 Qu'est-ce que le feature engineering ?

Le **feature engineering** (ingénierie des features) est le processus de **transformation des données brutes en variables exploitables** par un algorithme de machine learning. C'est souvent l'étape la plus déterminante pour la qualité d'un modèle.

```
Données brutes                Feature engineering              Modèle ML
──────────────                ───────────────────              ─────────
"Alice, 32 ans,        →      age_norm      = 0.42     →      Random
 salaire=45000€"              salaire_log   = 10.71            Forest /
 ville="Paris"                ville_Paris   = 1                GBT /
 nb_achats=12                 achats_bucket = "MEDIUM"         ...
 texte="bon produit"          tfidf_vec     = [0.2, ...]
```

### 1.2 Les outils Spark MLlib pour le feature engineering

```
pyspark.ml.feature
├── Manipulation de colonnes
│   ├── VectorAssembler      → Fusion de colonnes en vecteur unique
│   ├── FeatureHasher        → Hachage de features hétérogènes
│   └── Interaction          → Produits d'interaction entre features
│
├── Transformations numériques
│   ├── Binarizer            → Seuillage binaire
│   ├── Bucketizer           → Découpage en intervalles
│   ├── QuantileDiscretizer  → Découpage en quantiles équilibrés
│   └── (SQL) log(), sqrt()  → Transformations mathématiques
│
├── Normalisation / Standardisation
│   ├── StandardScaler       → Centrage-réduction (z-score)
│   ├── MinMaxScaler         → Normalisation [0,1]
│   ├── MaxAbsScaler         → Normalisation [-1,1]
│   └── Normalizer           → Normalisation par norme L1/L2/Linf
│
├── Features catégorielles
│   ├── StringIndexer        → Encodage ordinal (chaîne → entier)
│   ├── OneHotEncoder        → Encodage one-hot
│   └── IndexToString        → Décodage (entier → chaîne)
│
└── Features textuelles
    ├── Tokenizer            → Découpage en mots
    ├── RegexTokenizer       → Découpage avec regex
    ├── StopWordsRemover     → Suppression mots vides
    ├── NGram                → Bigrammes, trigrammes
    ├── HashingTF            → TF par hachage
    ├── IDF                  → IDF sur corpus
    ├── Word2Vec             → Plongements lexicaux
    └── CountVectorizer      → TF avec dictionnaire explicite
```

---

## 2. Ajout de colonnes

### 2.1 `withColumn()` — rappel et usages avancés

`withColumn()` est la méthode de base pour créer ou modifier une colonne. Revisitée ici dans un contexte de feature engineering.

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder \
    .appName("FeatureEngineering") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Jeu de données de référence pour tout le module
data = [
    (1,  "Alice",   32, 45000.0, "Paris",    12, "Excellent produit, livraison rapide"),
    (2,  "Bob",     28, 28000.0, "Lyon",      3, "Produit correct mais lent"),
    (3,  "Claire",  45, 72000.0, "Paris",    28, "Très bon rapport qualité prix"),
    (4,  "David",   19, 15000.0, "Nantes",    1, "Déçu par la qualité"),
    (5,  "Emma",    55, 95000.0, "Paris",    50, "Parfait, je recommande vivement"),
    (6,  "François",38, 53000.0, "Lyon",      8, "Bien mais peut mieux faire"),
    (7,  "Gaëlle",  22, 22000.0, "Bordeaux",  2, "Produit moyen sans plus"),
    (8,  "Henri",   61, 110000.0,"Paris",    75, "Incroyable, au-delà de mes attentes"),
    (9,  "Inès",    34, 41000.0, "Nantes",    6, "Satisfaite de mon achat"),
    (10, "Jules",   29, 31000.0, "Bordeaux",  4, "Livraison rapide, produit ok"),
]
schema = StructType([
    StructField("id",         IntegerType(), False),
    StructField("nom",        StringType(),  True),
    StructField("age",        IntegerType(), True),
    StructField("salaire",    DoubleType(),  True),
    StructField("ville",      StringType(),  True),
    StructField("nb_achats",  IntegerType(), True),
    StructField("avis",       StringType(),  True),
])
df = spark.createDataFrame(data, schema)

print("=== Données brutes ===")
df.show(truncate=False)
```

### 2.2 Colonnes calculées — cas courants en feature engineering

```python
# ── Features numériques dérivées ─────────────────────────────────────────────
df = df \
    .withColumn("salaire_mensuel",
        F.round(F.col("salaire") / 12, 2)) \
    .withColumn("age_au_carre",
        F.pow(F.col("age"), 2)) \
    .withColumn("ratio_achat_age",
        F.round(F.col("nb_achats") / F.col("age"), 4)) \
    .withColumn("longueur_avis",
        F.length(F.col("avis"))) \
    .withColumn("nb_mots_avis",
        F.size(F.split(F.col("avis"), r"\s+")))

# ── Features conditionnelles (CASE WHEN) ─────────────────────────────────────
df = df \
    .withColumn("est_senior",
        (F.col("age") >= 50).cast(IntegerType())) \
    .withColumn("est_parisien",
        (F.col("ville") == "Paris").cast(IntegerType())) \
    .withColumn("client_fidele",
        F.when(F.col("nb_achats") >= 10, 1).otherwise(0))

# ── Features d'interaction ────────────────────────────────────────────────────
df = df \
    .withColumn("salaire_x_achats",
        F.col("salaire") * F.col("nb_achats")) \
    .withColumn("score_client",
        F.round(
            F.col("nb_achats") * 0.6 +
            (F.col("salaire") / 10000) * 0.4,
            2
        ))

print("=== DataFrame enrichi ===")
df.select("nom", "age", "salaire", "ratio_achat_age",
          "est_senior", "client_fidele", "score_client").show()
```

### 2.3 Ajout de colonnes multiples avec `select()`

Quand on ajoute beaucoup de colonnes, chaîner des `withColumn()` peut être lent (chaque appel crée un nouveau plan). Une alternative plus efficace est d'utiliser `select()` avec toutes les colonnes en une seule opération.

```python
# ❌ Moins efficace pour de nombreuses colonnes (N niveaux d'imbrication)
df = df.withColumn("a", ...) \
       .withColumn("b", ...) \
       .withColumn("c", ...) \
       # ... 20 withColumn() de plus

# ✅ Plus efficace : un seul select() pour toutes les nouvelles colonnes
df_features = df.select(
    "*",   # Garder toutes les colonnes existantes
    F.round(F.col("salaire") / 12, 2).alias("salaire_mensuel"),
    F.pow(F.col("age"), 2).cast(DoubleType()).alias("age_carre"),
    F.round(F.col("nb_achats") / F.col("age"), 4).alias("ratio_achat_age"),
    F.length(F.col("avis")).alias("longueur_avis"),
    (F.col("age") >= 50).cast(IntegerType()).alias("est_senior"),
)
```

---

## 3. User-Defined Functions (UDF)

### 3.1 UDF Python classiques

Les UDF permettent d'appliquer une logique Python arbitraire à chaque ligne du DataFrame.

```python
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, DoubleType, IntegerType
import re

# ── UDF de nettoyage de texte ─────────────────────────────────────────────────
@udf(StringType())
def nettoyer_texte(texte):
    """Normalise un texte : minuscules, suppression de la ponctuation."""
    if texte is None:
        return None
    texte = texte.lower()
    texte = re.sub(r"[^\w\sàâäéèêëîïôùûüç]", " ", texte)
    texte = re.sub(r"\s+", " ", texte).strip()
    return texte

# ── UDF de scoring de sentiment (règles simples) ──────────────────────────────
mots_positifs = {"excellent","bon","parfait","rapide","recommande",
                 "bien","satisfait","top","incroyable","super"}
mots_negatifs = {"déçu","lent","moyen","mauvais","décevant","nul",
                 "médiocre","négatif","problème","mauvaise"}

@udf(DoubleType())
def score_sentiment(texte):
    """Score de sentiment entre -1 (négatif) et +1 (positif)."""
    if texte is None:
        return 0.0
    mots = re.findall(r"\b\w+\b", texte.lower())
    pos = sum(1 for m in mots if m in mots_positifs)
    neg = sum(1 for m in mots if m in mots_negatifs)
    total = pos + neg
    return float((pos - neg) / total) if total > 0 else 0.0

# ── UDF de catégorisation de ville ───────────────────────────────────────────
grandes_villes = {"Paris", "Lyon", "Marseille", "Toulouse", "Nice"}

@udf(StringType())
def categoriser_ville(ville):
    if ville is None:
        return "inconnue"
    return "grande_ville" if ville in grandes_villes else "autre"

# ── Application des UDF ───────────────────────────────────────────────────────
df = df \
    .withColumn("avis_propre",      nettoyer_texte(F.col("avis"))) \
    .withColumn("sentiment",        score_sentiment(F.col("avis"))) \
    .withColumn("categorie_ville",  categoriser_ville(F.col("ville")))

df.select("nom", "avis", "avis_propre", "sentiment", "categorie_ville").show(truncate=False)
```

### 3.2 Enregistrement SQL des UDF

```python
# Enregistrement pour utilisation dans spark.sql()
spark.udf.register("score_sentiment_sql", score_sentiment, DoubleType())
spark.udf.register("nettoyer_texte_sql",  nettoyer_texte,  StringType())

df.createOrReplaceTempView("clients")
spark.sql("""
    SELECT nom, avis,
           score_sentiment_sql(avis) AS sentiment,
           nettoyer_texte_sql(avis)  AS avis_propre
    FROM clients
    ORDER BY sentiment DESC
""").show(truncate=False)
```

### 3.3 Pandas UDF (vectorisées)

Les Pandas UDF sont bien plus rapides que les UDF classiques car elles traitent les données par **batch** via Apache Arrow.

```python
from pyspark.sql.functions import pandas_udf
import pandas as pd
import numpy as np

# ── Pandas UDF scalaire ───────────────────────────────────────────────────────
@pandas_udf(DoubleType())
def normaliser_zscore(series: pd.Series) -> pd.Series:
    """Standardisation z-score sur une colonne."""
    return (series - series.mean()) / series.std()

@pandas_udf(DoubleType())
def winsoriser(series: pd.Series, pct_bas: float = 0.05, pct_haut: float = 0.95) -> pd.Series:
    """Écrêtage des valeurs extrêmes aux percentiles spécifiés."""
    bas  = series.quantile(pct_bas)
    haut = series.quantile(pct_haut)
    return series.clip(bas, haut)

@pandas_udf(StringType())
def extraire_initiales(series: pd.Series) -> pd.Series:
    """Extrait les initiales d'un nom complet."""
    def initiales(nom):
        if not nom:
            return ""
        return ".".join(m[0].upper() for m in nom.split() if m) + "."
    return series.apply(initiales)

# Application
df = df \
    .withColumn("salaire_zscore", normaliser_zscore(F.col("salaire"))) \
    .withColumn("salaire_winsorizé", winsoriser(F.col("salaire"))) \
    .withColumn("initiales", extraire_initiales(F.col("nom")))

df.select("nom", "initiales", "salaire",
          F.round("salaire_zscore", 4).alias("zscore"),
          F.round("salaire_winsorizé", 2).alias("winsorisé")).show()
```

---

## 4. Pipelines de traitement

### 4.1 Principe des Pipelines MLlib

Un **Pipeline** MLlib enchaîne des étapes de transformation (`Transformer`) et d'entraînement (`Estimator`) en un objet unique, reproductible et sérialisable.

```
Pipeline
  │
  ├── Stage 1 : StringIndexer    (Estimator → apprend le mapping)
  ├── Stage 2 : OneHotEncoder    (Transformer → applique le mapping)
  ├── Stage 3 : VectorAssembler  (Transformer → assemble les features)
  └── Stage 4 : StandardScaler   (Estimator → apprend μ et σ)

pipeline.fit(df_train)    → PipelineModel (tous les stages ajustés)
model.transform(df_test)  → DataFrame avec les features finales
```

### 4.2 Différence Estimator / Transformer

| Type | Exemples | Comportement |
|---|---|---|
| **Transformer** | `Tokenizer`, `Binarizer`, `VectorAssembler` | `transform(df)` → nouveau DataFrame |
| **Estimator** | `IDF`, `StandardScaler`, `StringIndexer` | `fit(df)` → Transformer ajusté, puis `transform()` |

### 4.3 Créer un pipeline de feature engineering

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler,
    StandardScaler, RegexTokenizer, StopWordsRemover,
    HashingTF, IDF, Binarizer, Bucketizer
)

# ── Stage 1 : Encoder la variable catégorielle "ville" ────────────────────────
ville_indexer = StringIndexer(
    inputCol="ville", outputCol="ville_idx",
    handleInvalid="keep"   # Valeurs inconnues → index spécial
)
ville_encoder = OneHotEncoder(
    inputCols=["ville_idx"], outputCols=["ville_ohe"]
)

# ── Stage 2 : Encoder "categorie_ville" ──────────────────────────────────────
cat_ville_indexer = StringIndexer(
    inputCol="categorie_ville", outputCol="cat_ville_idx",
    handleInvalid="keep"
)

# ── Stage 3 : Assembler les features numériques ───────────────────────────────
assembler = VectorAssembler(
    inputCols=["age", "salaire", "nb_achats", "longueur_avis",
               "sentiment", "ville_ohe", "cat_ville_idx"],
    outputCol="features_raw",
    handleInvalid="skip"
)

# ── Stage 4 : Standardiser les features ──────────────────────────────────────
scaler = StandardScaler(
    inputCol="features_raw", outputCol="features",
    withMean=True, withStd=True
)

# ── Assemblage du pipeline ────────────────────────────────────────────────────
pipeline = Pipeline(stages=[
    ville_indexer,
    ville_encoder,
    cat_ville_indexer,
    assembler,
    scaler
])

# ── Entraînement et transformation ───────────────────────────────────────────
pipeline_model = pipeline.fit(df)
df_transformed = pipeline_model.transform(df)

df_transformed.select("nom", "features").show(truncate=False)

# ── Sauvegarde et rechargement ────────────────────────────────────────────────
pipeline_model.save("/tmp/pipeline_fe_model")

from pyspark.ml import PipelineModel
pipeline_reloaded = PipelineModel.load("/tmp/pipeline_fe_model")
df_reloaded = pipeline_reloaded.transform(df)
print("Pipeline rechargé et appliqué avec succès !")
```

---

## 5. Binarisation

### 5.1 `Binarizer` — seuillage numérique

La **binarisation** transforme une colonne numérique en valeur binaire (0 ou 1) selon un seuil.

```python
from pyspark.ml.feature import Binarizer

# Binariser le nombre d'achats : 1 si > 5, 0 sinon
binarizer = Binarizer(
    threshold=5.0,
    inputCol="nb_achats",
    outputCol="acheteur_actif"
)

df_bin = binarizer.transform(df)
df_bin.select("nom", "nb_achats", "acheteur_actif").show()
```

### 5.2 Binarisation multiple avec `withColumn()`

Pour binariser plusieurs colonnes simultanément, `withColumn()` est plus souple :

```python
# Binarisation de plusieurs features en une seule passe
seuils = {
    "age_senior":        ("age",       50),
    "haut_salaire":      ("salaire",   60000),
    "acheteur_frequent": ("nb_achats", 10),
    "avis_long":         ("longueur_avis", 100),
}

df_multi_bin = df
for nouvelle_col, (source_col, seuil) in seuils.items():
    df_multi_bin = df_multi_bin.withColumn(
        nouvelle_col,
        (F.col(source_col) > seuil).cast(IntegerType())
    )

df_multi_bin.select("nom", "age_senior", "haut_salaire",
                    "acheteur_frequent", "avis_long").show()
```

---

## 6. Tokenisation

### 6.1 `Tokenizer` — découpage simple

Le `Tokenizer` découpe un texte sur les espaces et met tout en minuscules.

```python
from pyspark.ml.feature import Tokenizer, RegexTokenizer, NGram, StopWordsRemover

# Tokenizer basique (espace comme séparateur)
tokenizer = Tokenizer(inputCol="avis", outputCol="tokens_bruts")
df_tok = tokenizer.transform(df)
df_tok.select("avis", "tokens_bruts").show(3, truncate=False)
```

### 6.2 `RegexTokenizer` — découpage par expression régulière

```python
# RegexTokenizer : plus flexible et robuste
regex_tokenizer = RegexTokenizer(
    inputCol   ="avis",
    outputCol  ="tokens",
    pattern    =r"[\W_]+",    # Découper sur tout caractère non alphanumérique
    minTokenLength=2,          # Ignorer les tokens de moins de 2 caractères
    toLowercase=True
)
df_tok = regex_tokenizer.transform(df)
df_tok.select("avis", "tokens").show(truncate=False)
```

### 6.3 `StopWordsRemover` — suppression des mots vides

```python
# Mots vides français personnalisés
stop_words_fr = [
    "le","la","les","de","du","des","un","une","et","est","en",
    "pour","par","sur","avec","dans","au","aux","il","elle","ils",
    "elles","je","tu","nous","vous","ce","qui","que","mais","ou",
    "donc","or","ni","car","mon","ton","son","ma","ta","sa","mes",
    "tes","ses","notre","votre","leur","leurs","à","y","si","plus",
    "très","bien","mal","pas","ne","se","me","te","lui","leur"
]

remover = StopWordsRemover(
    inputCol ="tokens",
    outputCol="tokens_filtres",
    stopWords=stop_words_fr
)
df_tok = remover.transform(df_tok)
df_tok.select("tokens", "tokens_filtres").show(3, truncate=False)
```

### 6.4 `NGram` — bigrammes et trigrammes

```python
# N-grammes : capturer les relations entre mots consécutifs
bigram  = NGram(n=2, inputCol="tokens_filtres", outputCol="bigrams")
trigram = NGram(n=3, inputCol="tokens_filtres", outputCol="trigrams")

df_ngram = bigram.transform(df_tok)
df_ngram = trigram.transform(df_ngram)

df_ngram.select("tokens_filtres", "bigrams", "trigrams").show(3, truncate=False)
# Exemple :
# tokens_filtres : ["excellent", "produit", "livraison", "rapide"]
# bigrams        : ["excellent produit", "produit livraison", "livraison rapide"]
# trigrams       : ["excellent produit livraison", "produit livraison rapide"]
```

---

## 7. Standardisation et Normalisation

### 7.1 Différence entre standardisation et normalisation

```
Standardisation (z-score)          Normalisation (min-max)
──────────────────────────         ───────────────────────
x' = (x - μ) / σ                  x' = (x - min) / (max - min)

Résultat : μ'=0, σ'=1             Résultat : x' ∈ [0, 1]
Préserve les outliers              Sensible aux outliers
Utile pour : SVM, PCA, régression  Utile pour : réseaux de neurones,
linéaire, k-means                  KNN, visualisation
```

### 7.2 `StandardScaler`

```python
from pyspark.ml.feature import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer

# StandardScaler nécessite un vecteur en entrée
assembler_temp = VectorAssembler(
    inputCols=["age", "salaire", "nb_achats"],
    outputCol="features_brutes"
)
df_vec = assembler_temp.transform(df)

# Standardisation z-score
standard_scaler = StandardScaler(
    inputCol ="features_brutes",
    outputCol="features_std",
    withMean =True,   # Centrer (soustraire la moyenne)
    withStd  =True    # Réduire (diviser par l'écart-type)
)
scaler_model = standard_scaler.fit(df_vec)
df_scaled = scaler_model.transform(df_vec)

print("Paramètres appris :")
print(f"  Moyennes  : {scaler_model.mean}")
print(f"  Écarts-type : {scaler_model.std}")
df_scaled.select("nom", "features_brutes", "features_std").show(3, truncate=False)
```

### 7.3 `MinMaxScaler`

```python
# Normalisation min-max → [0, 1]
minmax_scaler = MinMaxScaler(
    inputCol ="features_brutes",
    outputCol="features_minmax",
    min=0.0,   # Borne inférieure souhaitée
    max=1.0    # Borne supérieure souhaitée
)
minmax_model = minmax_scaler.fit(df_vec)
df_minmax    = minmax_model.transform(df_vec)

print("Min appris :", minmax_model.originalMin)
print("Max appris :", minmax_model.originalMax)
df_minmax.select("nom", "features_brutes", "features_minmax").show(3, truncate=False)
```

### 7.4 `MaxAbsScaler`

```python
# MaxAbsScaler → normalise chaque feature par sa valeur absolue maximale
# Résultat ∈ [-1, 1] — préserve les valeurs nulles (utile pour données sparse)
maxabs_scaler = MaxAbsScaler(
    inputCol="features_brutes", outputCol="features_maxabs"
)
maxabs_model = maxabs_scaler.fit(df_vec)
df_maxabs    = maxabs_model.transform(df_vec)
df_maxabs.select("nom", "features_maxabs").show(3, truncate=False)
```

### 7.5 `Normalizer` — normalisation par norme

```python
# Normalizer : normalise chaque LIGNE (vecteur) par sa norme L1, L2 ou Linf
# Utile pour les données textuelles et les similarités cosinus
normalizer_l2 = Normalizer(
    inputCol="features_brutes",
    outputCol="features_l2",
    p=2.0   # p=1 → norme L1, p=2 → norme L2, p=inf → norme Linf
)
df_norm = normalizer_l2.transform(df_vec)
df_norm.select("nom", "features_l2").show(3, truncate=False)
```

### 7.6 Choisir le bon scaler

| Scaler | Formule | Quand l'utiliser |
|---|---|---|
| `StandardScaler` | (x - μ) / σ | SVM, PCA, régression, k-means |
| `MinMaxScaler` | (x - min) / (max - min) | Réseaux de neurones, KNN |
| `MaxAbsScaler` | x / max(\|x\|) | Données sparse (texte, recommandation) |
| `Normalizer` L2 | x / \|\|x\|\|₂ | Similarité cosinus, NLP |

---

## 8. VectorAssembler

### 8.1 Principe

`VectorAssembler` est l'outil incontournable du feature engineering Spark : il **fusionne plusieurs colonnes** (numériques, booléennes, vectorielles) en **un seul vecteur de features**, format attendu par tous les algorithmes MLlib.

```
DataFrame :
  age   salaire   nb_achats   ville_ohe       sentiment
  32    45000     12          [1.0, 0.0, 0.0]  0.8
                    ↓ VectorAssembler
  features
  [32.0, 45000.0, 12.0, 1.0, 0.0, 0.0, 0.8]
```

### 8.2 Usage de base

```python
from pyspark.ml.feature import VectorAssembler

# Colonnes à assembler (numériques et vectorielles)
feature_cols = [
    "age",
    "salaire",
    "nb_achats",
    "longueur_avis",
    "nb_mots_avis",
    "sentiment",
    "est_senior",
    "client_fidele",
    "est_parisien",
]

assembler = VectorAssembler(
    inputCols  =feature_cols,
    outputCol  ="features",
    handleInvalid="skip"    # "skip" : ignorer les lignes avec null
                            # "keep" : remplacer null par 0
                            # "error": lever une exception (défaut)
)

df_assembled = assembler.transform(df)
df_assembled.select("nom", "features").show(truncate=False)

# Vérifier la dimension du vecteur
from pyspark.ml.linalg import Vectors
n_features = len(df_assembled.first()["features"])
print(f"Dimension du vecteur de features : {n_features}")
```

### 8.3 Combiner des vecteurs existants

```python
# Assembler des vecteurs déjà existants (ex : TF-IDF + features numériques)
# Note : VectorAssembler accepte des colonnes de type Vector en entrée

assembler_combine = VectorAssembler(
    inputCols=["features_std",   # Vecteur StandardScaler
               "tfidf_vec",      # Vecteur TF-IDF
               "sentiment"],     # Scalaire numérique
    outputCol="features_combined"
)
```

### 8.4 Récupérer les noms des features après assemblage

```python
# Utile pour interpréter les importances de features après un modèle
def get_feature_names(pipeline_model, assembler_stage_idx):
    """Récupère les noms de features d'un VectorAssembler dans un Pipeline."""
    assembler = pipeline_model.stages[assembler_stage_idx]
    return assembler.getInputCols()

# Ou directement si on a l'objet assembler
print("Features dans le vecteur :")
for i, col in enumerate(assembler.getInputCols()):
    print(f"  [{i:2d}] {col}")
```

---

## 9. Bucketing (discrétisation)

### 9.1 `Bucketizer` — intervalles fixes

Le **bucketing** transforme une variable continue en variable catégorielle discrète en la découpant en intervalles (buckets) définis manuellement.

```python
from pyspark.ml.feature import Bucketizer, QuantileDiscretizer

# Bucketing de l'âge en tranches définies manuellement
age_bucketizer = Bucketizer(
    splits   =[-float("inf"), 25, 35, 50, float("inf")],
    #           │              │   │   │   └── Senior (50+)
    #           │              │   │   └────── Adulte (35-50)
    #           │              │   └────────── Jeune adulte (25-35)
    #           │              └────────────── Jeune (< 25)
    #           └───────────────────────────── Borne inférieure
    inputCol ="age",
    outputCol="tranche_age",
    handleInvalid="keep"   # NaN → bucket spécial
)

df_bucket = age_bucketizer.transform(df)

# Ajouter un label lisible
df_bucket = df_bucket.withColumn("tranche_age_label",
    F.when(F.col("tranche_age") == 0, "Jeune (<25)")
     .when(F.col("tranche_age") == 1, "Jeune adulte (25-35)")
     .when(F.col("tranche_age") == 2, "Adulte (35-50)")
     .otherwise("Senior (50+)")
)

df_bucket.select("nom", "age", "tranche_age", "tranche_age_label").show()

# Bucketing du salaire
salaire_bucketizer = Bucketizer(
    splits   =[-float("inf"), 25000, 45000, 75000, float("inf")],
    inputCol ="salaire",
    outputCol="tranche_salaire"
)
df_bucket = salaire_bucketizer.transform(df_bucket)
```

### 9.2 `QuantileDiscretizer` — intervalles par quantiles

`QuantileDiscretizer` calcule automatiquement les bornes pour obtenir des **buckets de taille approximativement égale** (équilibrés).

```python
# Discrétisation en 4 quartiles équilibrés
quantile_disc = QuantileDiscretizer(
    numBuckets      =4,
    inputCol        ="salaire",
    outputCol       ="quartile_salaire",
    relativeError   =0.01,    # Précision des quantiles (0 = exact mais lent)
    handleInvalid   ="keep"
)

# QuantileDiscretizer est un Estimator : il doit être ajusté sur les données
qd_model = quantile_disc.fit(df)
df_quantile = qd_model.transform(df)

print("Bornes des quartiles :", qd_model.getSplits())
df_quantile.select("nom", "salaire", "quartile_salaire") \
    .orderBy("salaire").show()

# Bucketing multiple (plusieurs colonnes en même temps) — Spark 3.x
multi_bucketizer = Bucketizer(
    splitsArray=[
        [-float("inf"), 25, 35, 50, float("inf")],      # Age
        [-float("inf"), 25000, 50000, float("inf")],     # Salaire
    ],
    inputCols=["age", "salaire"],
    outputCols=["age_bucket", "salaire_bucket"]
)
df_multi_bucket = multi_bucketizer.transform(df)
df_multi_bucket.select("nom", "age", "age_bucket",
                        "salaire", "salaire_bucket").show()
```

---

## 10. Transformation logarithmique

### 10.1 Pourquoi transformer en logarithme ?

La transformation logarithmique est utilisée pour :
- **Réduire l'asymétrie** (*skewness*) des distributions à longue queue droite
- **Stabiliser la variance** quand elle croît avec la moyenne
- **Linéariser** des relations exponentielles
- Rendre des distributions proches de la **loi normale** (attendue par certains algorithmes)

```
Distribution originale du salaire :      Distribution après log
──────────────────────────────────       ──────────────────────
  │ ██                                     │    ████
  │ ████                                   │  ████████
  │ ██████                                 │████████████
  │ ████████                               │  ████████
  │ ██████████████                         │    ████
  └─────────────────→                      └──────────────────→
  Asymétrie à droite (longue queue)        Distribution plus symétrique
```

### 10.2 Implémentation avec Spark

```python
# ── Transformation log naturel ────────────────────────────────────────────────
df = df.withColumn("log_salaire",    F.log(F.col("salaire")))
df = df.withColumn("log1p_achats",   F.log1p(F.col("nb_achats")))  # log(1+x) pour gérer x=0
df = df.withColumn("log2_longueur",  F.log2(F.col("longueur_avis")))
df = df.withColumn("log10_salaire",  F.log10(F.col("salaire")))

# ── Transformation racine carrée (alternative douce) ─────────────────────────
df = df.withColumn("sqrt_achats",    F.sqrt(F.col("nb_achats")))

# ── Transformation Box-Cox (généralisation) ───────────────────────────────────
# x^λ si λ ≠ 0, log(x) si λ = 0
lambda_bc = 0.5   # Racine carrée

@pandas_udf(DoubleType())
def boxcox(series: pd.Series) -> pd.Series:
    from scipy.stats import boxcox as scipy_boxcox
    result, _ = scipy_boxcox(series.clip(lower=0.001))
    return pd.Series(result)

df = df.withColumn("salaire_boxcox", boxcox(F.col("salaire")))

# ── Comparer les distributions ────────────────────────────────────────────────
print("=== Statistiques avant/après transformation log ===")
df.select(
    F.round(F.mean("salaire"),      2).alias("salaire_moy"),
    F.round(F.stddev("salaire"),    2).alias("salaire_std"),
    F.round(F.skewness("salaire"),  4).alias("salaire_skew"),
    F.round(F.mean("log_salaire"),  4).alias("log_salaire_moy"),
    F.round(F.stddev("log_salaire"),4).alias("log_salaire_std"),
    F.round(F.skewness("log_salaire"),4).alias("log_salaire_skew"),
).show()

# ── Attention aux valeurs nulles et négatives ─────────────────────────────────
# log(0) = -∞ → erreur !
# ✅ Toujours vérifier ou utiliser log1p(x) = log(1+x) pour x ≥ 0
df = df.withColumn("log_safe_achats",
    F.when(F.col("nb_achats") > 0, F.log(F.col("nb_achats")))
     .otherwise(0.0)
)
# Ou simplement :
df = df.withColumn("log1p_achats_safe", F.log1p(F.col("nb_achats")))

df.select("nom", "nb_achats", "log_safe_achats", "log1p_achats_safe").show()
```

---

## 11. TF-IDF

### 11.1 Rappel du principe

**TF-IDF** a été introduit dans le Module 9. Dans ce module, nous l'abordons comme une étape de **feature engineering** à intégrer dans un pipeline MLlib complet.

```
TF(t, d)  = fréquence du terme t dans le document d
IDF(t, D) = log(N / df(t))   → rareté du terme dans le corpus
TF-IDF    = TF × IDF          → importance du terme pour caractériser d
```

### 11.2 Pipeline TF-IDF complet

```python
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover,
    HashingTF, IDF, CountVectorizer
)

# ── Méthode 1 : HashingTF + IDF (rapide, sans dictionnaire) ──────────────────
pipeline_tfidf_hash = Pipeline(stages=[
    RegexTokenizer(
        inputCol="avis_propre", outputCol="tokens_tfidf",
        pattern=r"\W+", minTokenLength=3
    ),
    StopWordsRemover(
        inputCol="tokens_tfidf", outputCol="tokens_propres",
        stopWords=stop_words_fr
    ),
    HashingTF(
        inputCol="tokens_propres", outputCol="tf_hash",
        numFeatures=200,      # Taille de l'espace de hachage
        binary=False          # True = présence/absence, False = fréquence
    ),
    IDF(
        inputCol="tf_hash", outputCol="tfidf_hash",
        minDocFreq=1          # Ignorer les termes dans < N documents
    ),
])

model_tfidf_hash = pipeline_tfidf_hash.fit(df)
df_tfidf = model_tfidf_hash.transform(df)
df_tfidf.select("nom", "tokens_propres", "tfidf_hash").show(3, truncate=False)

# ── Méthode 2 : CountVectorizer + IDF (avec dictionnaire explicite) ───────────
pipeline_tfidf_count = Pipeline(stages=[
    RegexTokenizer(
        inputCol="avis", outputCol="tokens_cv",
        pattern=r"\W+", minTokenLength=3, toLowercase=True
    ),
    StopWordsRemover(
        inputCol="tokens_cv", outputCol="tokens_cv_filtres",
        stopWords=stop_words_fr
    ),
    CountVectorizer(
        inputCol="tokens_cv_filtres", outputCol="tf_count",
        vocabSize=500,        # Taille du vocabulaire (N mots les plus fréquents)
        minDF=1.0,            # Fréquence minimale dans le corpus
        minTF=1.0             # Fréquence minimale dans le document
    ),
    IDF(
        inputCol="tf_count", outputCol="tfidf_count"
    ),
])

model_tfidf_count = pipeline_tfidf_count.fit(df)
df_tfidf_count = model_tfidf_count.transform(df)

# Accéder au vocabulaire
cv_model = model_tfidf_count.stages[2]  # CountVectorizerModel
print(f"\nTaille du vocabulaire : {len(cv_model.vocabulary)}")
print(f"10 premiers mots : {cv_model.vocabulary[:10]}")

# ── Quelle méthode choisir ? ─────────────────────────────────────────────────
print("""
HashingTF   : rapide, pas de dictionnaire, risque de collision de hash
              → à préférer pour les très grands corpus
CountVectorizer : dictionnaire explicite, interprétable
              → à préférer pour les corpus de taille raisonnable
""")
```

### 11.3 Intégration dans un pipeline ML complet

```python
from pyspark.ml.classification import LogisticRegression

# Pipeline complet : texte → TF-IDF → Modèle
# (Ici, prédire si le sentiment est positif ou négatif)
df_label = df.withColumn("label",
    F.when(F.col("sentiment") > 0, 1.0).otherwise(0.0)
)

pipeline_ml = Pipeline(stages=[
    RegexTokenizer(inputCol="avis", outputCol="tokens_ml",
                   pattern=r"\W+", minTokenLength=3, toLowercase=True),
    StopWordsRemover(inputCol="tokens_ml", outputCol="tokens_ml_filtres",
                     stopWords=stop_words_fr),
    HashingTF(inputCol="tokens_ml_filtres", outputCol="tf_ml", numFeatures=100),
    IDF(inputCol="tf_ml", outputCol="tfidf_ml"),
    LogisticRegression(featuresCol="tfidf_ml", labelCol="label", maxIter=10),
])

train_ml, test_ml = df_label.randomSplit([0.8, 0.2], seed=42)
model_ml = pipeline_ml.fit(train_ml)
predictions = model_ml.transform(test_ml)
predictions.select("nom", "avis", "label", "prediction", "probability").show(truncate=False)
```

---

## 12. FeatureHasher

### 12.1 Principe

`FeatureHasher` convertit un ensemble de features **hétérogènes** (numériques, booléennes, catégorielles) en un **vecteur sparse de taille fixe** via une fonction de hachage. Contrairement à `VectorAssembler` + `OneHotEncoder`, il ne nécessite **pas d'étape d'ajustement** (pas de `fit()`) : c'est un `Transformer` pur.

```
Features hétérogènes           FeatureHasher                Vecteur sparse
────────────────────           ─────────────                ─────────────
age       = 32         →       hash("age=32")       →       [0, 0, 1, 0, ...]
ville     = "Paris"            hash("ville=Paris")          [1, 0, 0, 0, ...]
client    = True               hash("client=true")          [0, 1, 0, 0, ...]
nb_achats = 12                 hash("nb_achats") × 12       [0, 0, 0, 2, ...]
                                                            (vecteur de taille N fixe)
```

### 12.2 Usage

```python
from pyspark.ml.feature import FeatureHasher

# FeatureHasher accepte les colonnes numériques, booléennes et string
hasher = FeatureHasher(
    inputCols =["age", "salaire", "nb_achats", "ville", "client_fidele"],
    outputCol ="features_hashed",
    numFeatures=256,       # Taille du vecteur de sortie (puissance de 2 recommandée)
    categoricalCols=["ville"]  # Colonnes à traiter comme catégorielles
)

# FeatureHasher est un Transformer (pas d'étape fit() nécessaire)
df_hashed = hasher.transform(df)
df_hashed.select("nom", "ville", "age", "salaire", "features_hashed").show(3, truncate=False)
```

### 12.3 Avantages et limites

```python
print("""
AVANTAGES du FeatureHasher :
  ✅ Pas d'étape fit() → peut être appliqué à de nouvelles données sans réentraînement
  ✅ Gère les features catégorielles sans pré-indexation
  ✅ Très rapide et scalable
  ✅ Gère les nouveaux niveaux catégoriels inconnus à l'entraînement
  ✅ Utile pour le NLP en ligne (features inconnues à l'avance)

LIMITES :
  ⚠️ Collisions de hash : deux features différentes peuvent donner le même index
     → Augmenter numFeatures réduit ce risque (256, 512, 1024...)
  ⚠️ Non interprétable : impossible de récupérer quel feature correspond à quel index
  ⚠️ Les colonnes numériques sont traitées différemment des catégorielles
     → Spécifier categoricalCols pour les colonnes à encoder comme catégories
""")

# Comparaison FeatureHasher vs VectorAssembler
print("=== Comparaison ===")
print(f"VectorAssembler : vecteur dense de taille {len(feature_cols)}")
print(f"FeatureHasher   : vecteur sparse de taille 256 (configurable)")
print("FeatureHasher adapté pour : features nombreuses, catégorielles, online learning")
print("VectorAssembler adapté pour : features connues et stables, interprétabilité")
```

---

## 13. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    RegexTokenizer, StopWordsRemover, HashingTF, IDF,
    Binarizer, Bucketizer, QuantileDiscretizer,
    VectorAssembler, StandardScaler, MinMaxScaler,
    StringIndexer, OneHotEncoder, FeatureHasher
)
from pyspark.sql.functions import udf, pandas_udf
import pandas as pd
import numpy as np
import re

spark = SparkSession.builder \
    .appName("FeatureEngineeringComplet") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ─── Données brutes ───────────────────────────────────────────────────────────
data = [
    (1,"Alice",  32,45000.0,"Paris",  12,"Excellent produit livraison rapide",True),
    (2,"Bob",    28,28000.0,"Lyon",    3,"Produit correct mais lent",          False),
    (3,"Claire", 45,72000.0,"Paris",  28,"Très bon rapport qualité prix",      True),
    (4,"David",  19,15000.0,"Nantes",  1,"Déçu par la qualité",                False),
    (5,"Emma",   55,95000.0,"Paris",  50,"Parfait je recommande vivement",      True),
    (6,"François",38,53000.0,"Lyon",   8,"Bien mais peut mieux faire",          True),
    (7,"Gaëlle", 22,22000.0,"Bordeaux",2,"Produit moyen sans plus",             False),
    (8,"Henri",  61,110000.0,"Paris", 75,"Incroyable au-delà de mes attentes",  True),
    (9,"Inès",   34,41000.0,"Nantes",  6,"Satisfaite de mon achat",             True),
    (10,"Jules", 29,31000.0,"Bordeaux",4,"Livraison rapide produit ok",          True),
]
schema = StructType([
    StructField("id",          IntegerType(), False),
    StructField("nom",         StringType(),  True),
    StructField("age",         IntegerType(), True),
    StructField("salaire",     DoubleType(),  True),
    StructField("ville",       StringType(),  True),
    StructField("nb_achats",   IntegerType(), True),
    StructField("avis",        StringType(),  True),
    StructField("premium",     BooleanType(), True),
])
df = spark.createDataFrame(data, schema)

print("=" * 65)
print("=== 1. Features numériques dérivées ===")
print("=" * 65)
df = df \
    .withColumn("log1p_salaire",  F.log1p(F.col("salaire"))) \
    .withColumn("log1p_achats",   F.log1p(F.col("nb_achats"))) \
    .withColumn("sqrt_achats",    F.sqrt(F.col("nb_achats").cast(DoubleType()))) \
    .withColumn("ratio_achat_age",F.round(F.col("nb_achats") / F.col("age"), 4)) \
    .withColumn("nb_mots_avis",   F.size(F.split(F.col("avis"), r"\s+")))
df.select("nom","salaire","log1p_salaire","nb_achats","log1p_achats","sqrt_achats").show()

print("=" * 65)
print("=== 2. UDF : sentiment et nettoyage ===")
print("=" * 65)
mots_pos = {"excellent","bon","parfait","rapide","recommande","bien","satisfait","incroyable"}
mots_neg = {"déçu","lent","moyen","mauvais","décevant"}

@udf(DoubleType())
def sentiment(texte):
    if not texte: return 0.0
    mots = re.findall(r"\b\w+\b", texte.lower())
    p = sum(1 for m in mots if m in mots_pos)
    n = sum(1 for m in mots if m in mots_neg)
    return float((p-n)/(p+n)) if p+n > 0 else 0.0

@pandas_udf(DoubleType())
def zscore_pandas(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()

df = df \
    .withColumn("sentiment", sentiment(F.col("avis"))) \
    .withColumn("salaire_zscore", zscore_pandas(F.col("salaire")))
df.select("nom","avis","sentiment", F.round("salaire_zscore",4).alias("z_salaire")).show(truncate=False)

print("=" * 65)
print("=== 3. Binarisation et Bucketing ===")
print("=" * 65)
binarizer = Binarizer(threshold=5.0, inputCol="nb_achats", outputCol="acheteur_actif")
age_bucket = Bucketizer(
    splits=[-float("inf"), 25, 35, 50, float("inf")],
    inputCol="age", outputCol="tranche_age"
)
sal_quantile = QuantileDiscretizer(numBuckets=3, inputCol="salaire", outputCol="quartile_sal")

df = binarizer.transform(df)
df = age_bucket.transform(df)
sal_model = sal_quantile.fit(df)
df = sal_model.transform(df)
print(f"Bornes quantiles salaire : {sal_model.getSplits()}")
df.select("nom","age","tranche_age","nb_achats","acheteur_actif",
          "salaire","quartile_sal").show()

print("=" * 65)
print("=== 4. TF-IDF sur les avis ===")
print("=" * 65)
stop_fr = ["le","la","les","de","du","des","un","une","et","est","en",
           "pour","par","sur","avec","mais","très","bien","par","je"]

tfidf_pipe = Pipeline(stages=[
    RegexTokenizer(inputCol="avis", outputCol="tokens", pattern=r"\W+",
                   minTokenLength=3, toLowercase=True),
    StopWordsRemover(inputCol="tokens", outputCol="tokens_f", stopWords=stop_fr),
    HashingTF(inputCol="tokens_f", outputCol="tf", numFeatures=100),
    IDF(inputCol="tf", outputCol="tfidf"),
])
tfidf_model = tfidf_pipe.fit(df)
df = tfidf_model.transform(df)
df.select("nom","tokens_f","tfidf").show(3, truncate=False)

print("=" * 65)
print("=== 5. VectorAssembler + StandardScaler ===")
print("=" * 65)
feat_cols = ["age","salaire","nb_achats","log1p_salaire","log1p_achats",
             "sqrt_achats","ratio_achat_age","nb_mots_avis","sentiment",
             "acheteur_actif","tranche_age","quartile_sal"]

assembler = VectorAssembler(inputCols=feat_cols, outputCol="features_raw",
                             handleInvalid="skip")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features",
                            withMean=True, withStd=True)

df_vec = assembler.transform(df)
scaler_model = scaler.fit(df_vec)
df_final = scaler_model.transform(df_vec)
df_final.select("nom","features").show(3, truncate=False)
print(f"\nDimension finale du vecteur : {len(df_final.first()['features'])}")

print("=" * 65)
print("=== 6. FeatureHasher ===")
print("=" * 65)
hasher = FeatureHasher(
    inputCols=["age","salaire","nb_achats","ville","premium"],
    outputCol="features_hashed",
    numFeatures=128,
    categoricalCols=["ville"]
)
df_hashed = hasher.transform(df)
df_hashed.select("nom","ville","premium","features_hashed").show(3, truncate=False)

print("=" * 65)
print("=== 7. Pipeline complet sauvegardé ===")
print("=" * 65)
pipeline_complet = Pipeline(stages=[
    StringIndexer(inputCol="ville", outputCol="ville_idx", handleInvalid="keep"),
    OneHotEncoder(inputCols=["ville_idx"], outputCols=["ville_ohe"]),
    VectorAssembler(
        inputCols=["age","log1p_salaire","log1p_achats","sentiment","acheteur_actif","ville_ohe"],
        outputCol="features_pipeline", handleInvalid="skip"
    ),
    MinMaxScaler(inputCol="features_pipeline", outputCol="features_norm"),
])

pipeline_complet_model = pipeline_complet.fit(df)
df_pipeline = pipeline_complet_model.transform(df)
df_pipeline.select("nom","features_norm").show(3, truncate=False)

pipeline_complet_model.save("/tmp/pipeline_complet_fe")
print("Pipeline sauvegardé dans /tmp/pipeline_complet_fe")

spark.stop()
```

---

## Résumé du module

| Technique | Outil Spark | Cas d'usage |
|---|---|---|
| **Colonnes dérivées** | `withColumn()`, `select()` | Ratios, interactions, indicateurs binaires |
| **UDF Python** | `@udf`, `spark.udf.register()` | Logique métier custom, règles complexes |
| **Pandas UDF** | `@pandas_udf` | UDF performantes par batch via Arrow |
| **Pipeline** | `Pipeline`, `PipelineModel` | Enchaîner et sauvegarder les étapes de FE |
| **Binarizer** | `Binarizer` | Seuillage numérique → 0/1 |
| **Tokenisation** | `Tokenizer`, `RegexTokenizer`, `NGram` | Découper le texte en tokens |
| **Stop Words** | `StopWordsRemover` | Supprimer les mots peu informatifs |
| **Standardisation** | `StandardScaler` | Centrage-réduction (μ=0, σ=1) |
| **Normalisation** | `MinMaxScaler`, `MaxAbsScaler`, `Normalizer` | Mise à l'échelle dans [0,1] ou par norme |
| **VectorAssembler** | `VectorAssembler` | Fusionner toutes les features en 1 vecteur |
| **Bucketing** | `Bucketizer`, `QuantileDiscretizer` | Discrétiser les variables continues |
| **Log transform** | `F.log()`, `F.log1p()`, `F.sqrt()` | Réduire le skew, stabiliser la variance |
| **TF-IDF** | `HashingTF` + `IDF`, `CountVectorizer` + `IDF` | Vectoriser le texte |
| **FeatureHasher** | `FeatureHasher` | Hachage de features hétérogènes sans fit() |

---

## Exercices

### Exercice 1 — Pipeline de feature engineering (35 min)
> À partir d'un jeu de données de logements (prix, surface, nombre de pièces, ville, description textuelle) :
> 1. Ajouter des features dérivées : `prix_m2`, `log_prix`, `surface_bucket` (petit/moyen/grand)
> 2. Tokeniser et vectoriser les descriptions avec TF-IDF
> 3. Encoder la ville avec `StringIndexer` + `OneHotEncoder`
> 4. Assembler toutes les features avec `VectorAssembler`
> 5. Standardiser avec `StandardScaler`
> 6. Encapsuler toutes ces étapes dans un `Pipeline` et le sauvegarder

### Exercice 2 — UDF avancées (30 min)
> 1. Écrire une UDF Python qui détecte la langue d'un texte (utiliser `langdetect`)
> 2. Écrire une Pandas UDF qui applique une transformation Box-Cox (via `scipy.stats`)
> 3. Comparer les temps d'exécution sur 10 000 lignes : UDF Python vs Pandas UDF vs fonction native `F.log()`
> 4. Expliquer les différences observées

### Exercice 3 — Comparaison des scalers (25 min)
> Sur un DataFrame avec des colonnes de distributions très différentes (age, revenu, population d'une ville) :
> 1. Appliquer `StandardScaler`, `MinMaxScaler` et `MaxAbsScaler`
> 2. Calculer les statistiques (min, max, moy, std) des vecteurs résultants
> 3. Introduire un outlier extrême et observer l'impact sur chaque scaler
> 4. Conclure sur le choix du scaler selon le contexte

### Exercice 4 — FeatureHasher vs VectorAssembler (30 min)
> Sur un jeu de données e-commerce avec ~50 features (catégorielles, numériques, textuelles) :
> 1. Construire un pipeline `StringIndexer` + `OneHotEncoder` + `VectorAssembler`
> 2. Construire un pipeline `FeatureHasher` équivalent
> 3. Comparer : taille des vecteurs, temps de traitement, précision d'un GBT entraîné sur chacun
> 4. Dans quels cas préférez-vous l'un à l'autre ?

---

## Pour aller plus loin

- 📖 **Spark MLlib Feature** : [spark.apache.org/docs/latest/ml-features.html](https://spark.apache.org/docs/latest/ml-features.html)
- 📖 **Spark MLlib Pipelines** : [spark.apache.org/docs/latest/ml-pipeline.html](https://spark.apache.org/docs/latest/ml-pipeline.html)
- 📄 **Feature Engineering for ML** : *Feature Engineering for Machine Learning* — Rachel Zheng & Amanda Casari, O'Reilly 2018
- 🛠️ **Feature-engine** : bibliothèque Python de feature engineering compatible Pandas — [feature-engine.readthedocs.io](https://feature-engine.readthedocs.io)
- 🛠️ **Featuretools** : feature engineering automatique — [featuretools.alteryx.com](https://featuretools.alteryx.com)

---

*Module précédent → **Module 7 : Interaction avec des sources externes***  
*Module suivant → **Module 9 : Algorithmes de graphe***
