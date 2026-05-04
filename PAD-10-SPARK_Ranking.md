# Module 9 — Algorithmes de ranking

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 8 — Algorithmes de graphe

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Comprendre les enjeux et les métriques d'évaluation d'un système de ranking
- Implémenter PageRank comme algorithme de ranking sur graphe
- Utiliser ALS (Alternating Least Squares) pour le ranking collaboratif
- Construire un pipeline TF-IDF pour le ranking de documents textuels
- Appréhender les principes du *Learning to Rank* (LTR)
- Évaluer un système de ranking avec NDCG, MAP et MRR

---

## 1. Qu'est-ce que le ranking ?

### 1.1 Définition et enjeux

Le **ranking** (ou ordonnancement) est la tâche qui consiste à **trier un ensemble d'items par ordre de pertinence** par rapport à une requête ou un contexte donné. C'est l'un des problèmes centraux de l'IA appliquée.

```
Requête / Contexte          Items non ordonnés         Résultat rankéé
──────────────────          ──────────────────         ───────────────
"machine learning"    →     [Doc A, Doc B,       →     1. Doc C  ★★★★★
(moteur de recherche)       Doc C, Doc D,               2. Doc A  ★★★★
                            Doc E]                      3. Doc E  ★★★
                                                        4. Doc B  ★★
                                                        5. Doc D  ★

user_id = 42          →     [Film 1, Film 2,     →     1. Film 3  ★★★★★
(recommandation)            Film 3, Film 4,             2. Film 1  ★★★★
                            Film 5]                     3. Film 5  ★★★
```

### 1.2 Taxonomie des approches de ranking

```
                    ALGORITHMES DE RANKING
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
   Basé sur graphe    Collaboratif      Basé sur contenu
   ──────────────     ────────────      ────────────────
   PageRank           ALS (Matrix       TF-IDF
   HITS               Factorization)    BM25
   TextRank           SVD               Word2Vec
                      KNN               BERT embeddings
                           │
                           ▼
                    Learning to Rank
                    ────────────────
                    Pointwise  (régression/classification)
                    Pairwise   (RankSVM, RankBoost)
                    Listwise   (LambdaMART, LambdaRank)
```

### 1.3 Cas d'usage industriels

| Domaine | Requête | Items rankés | Algorithme typique |
|---|---|---|---|
| **Moteur de recherche** | Requête texte | Pages web | BM25 + PageRank + LTR |
| **Recommandation** | Utilisateur | Films, produits, articles | ALS, collaborative filtering |
| **Résumé automatique** | Document | Phrases | TextRank |
| **Publicité** | Contexte + utilisateur | Annonces | LTR (LambdaMART) |
| **Tri de résultats e-commerce** | Recherche produit | Articles du catalogue | BM25 + scoring ML |
| **Feed de réseaux sociaux** | Utilisateur + temps | Posts | Modèles prédictifs d'engagement |
| **Médecine** | Symptômes | Diagnostics | Scoring probabiliste |

---

## 2. Métriques d'évaluation

Avant d'implémenter un algorithme de ranking, il faut savoir comment l'évaluer. Les métriques présentées ici supposent l'existence d'une **vérité terrain** (*ground truth*) indiquant quels items sont pertinents pour chaque requête.

### 2.1 Precision@K et Recall@K

```
Precision@K = (items pertinents dans les K premiers) / K

Recall@K    = (items pertinents dans les K premiers) / (total items pertinents)
```

```python
def precision_at_k(recommended, relevant, k):
    """
    recommended : liste ordonnée des items recommandés
    relevant    : ensemble des items pertinents (vérité terrain)
    k           : nombre de résultats considérés
    """
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / k

def recall_at_k(recommended, relevant, k):
    top_k = recommended[:k]
    hits  = sum(1 for item in top_k if item in relevant)
    return hits / len(relevant) if relevant else 0.0

# Exemple
recommended = ["Film3", "Film1", "Film5", "Film2", "Film4"]
relevant    = {"Film1", "Film3", "Film4"}   # Films que l'utilisateur aurait aimé

print(f"Precision@3 : {precision_at_k(recommended, relevant, 3):.3f}")  # 2/3 = 0.667
print(f"Recall@3    : {recall_at_k(recommended, relevant, 3):.3f}")     # 2/3 = 0.667
print(f"Precision@5 : {precision_at_k(recommended, relevant, 5):.3f}")  # 3/5 = 0.600
print(f"Recall@5    : {recall_at_k(recommended, relevant, 5):.3f}")     # 3/3 = 1.000
```

### 2.2 MAP — Mean Average Precision

La **MAP** (*Mean Average Precision*) tient compte de la **position** des items pertinents dans la liste, pas seulement de leur présence dans le top-K.

```
Average Precision (AP) pour une requête =
  Σ [Precision@k × is_relevant(k)] / nb_items_pertinents
  k=1 à N

MAP = moyenne des AP sur toutes les requêtes
```

```python
def average_precision(recommended, relevant):
    """AP pour une seule requête."""
    hits, sum_precisions = 0, 0.0
    for k, item in enumerate(recommended, start=1):
        if item in relevant:
            hits += 1
            sum_precisions += hits / k
    return sum_precisions / len(relevant) if relevant else 0.0

def mean_average_precision(all_recommended, all_relevant):
    """MAP sur plusieurs requêtes."""
    aps = [average_precision(rec, rel)
           for rec, rel in zip(all_recommended, all_relevant)]
    return sum(aps) / len(aps)

# Exemple multi-requêtes
all_rec = [
    ["Film3", "Film1", "Film5", "Film2", "Film4"],
    ["Doc_A", "Doc_C", "Doc_B", "Doc_D", "Doc_E"],
]
all_rel = [
    {"Film1", "Film3", "Film4"},
    {"Doc_B", "Doc_D"},
]

map_score = mean_average_precision(all_rec, all_rel)
print(f"MAP : {map_score:.4f}")
```

### 2.3 MRR — Mean Reciprocal Rank

La **MRR** (*Mean Reciprocal Rank*) mesure où se trouve le **premier** item pertinent dans la liste classée.

```
Reciprocal Rank = 1 / rang_du_premier_item_pertinent

MRR = moyenne des Reciprocal Ranks sur toutes les requêtes
```

```python
def reciprocal_rank(recommended, relevant):
    for rank, item in enumerate(recommended, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0   # Aucun item pertinent trouvé

def mean_reciprocal_rank(all_recommended, all_relevant):
    rrs = [reciprocal_rank(rec, rel)
           for rec, rel in zip(all_recommended, all_relevant)]
    return sum(rrs) / len(rrs)

mrr = mean_reciprocal_rank(all_rec, all_rel)
print(f"MRR : {mrr:.4f}")
```

### 2.4 NDCG — Normalized Discounted Cumulative Gain

La **NDCG** (*Normalized Discounted Cumulative Gain*) est la métrique la plus complète : elle tient compte à la fois de la **position** et de la **pertinence graduée** (un item peut être "très pertinent", "pertinent", "peu pertinent").

```
DCG@K = Σ (2^rel_i - 1) / log2(i + 1)
        i=1 à K

IDCG@K = DCG@K du classement idéal (meilleur score possible)

NDCG@K = DCG@K / IDCG@K    (normalisé entre 0 et 1)
```

```python
import math

def dcg_at_k(relevances, k):
    """
    relevances : liste de scores de pertinence dans l'ordre du ranking
    k          : nombre de résultats considérés
    """
    return sum(
        (2 ** rel - 1) / math.log2(i + 2)   # i+2 car i commence à 0
        for i, rel in enumerate(relevances[:k])
    )

def ndcg_at_k(recommended_relevances, k):
    """
    recommended_relevances : pertinences dans l'ordre proposé par le système
    """
    # Classement idéal : trier par pertinence décroissante
    ideal_relevances = sorted(recommended_relevances, reverse=True)
    dcg  = dcg_at_k(recommended_relevances, k)
    idcg = dcg_at_k(ideal_relevances, k)
    return dcg / idcg if idcg > 0 else 0.0

# Exemple : pertinences des items recommandés (0=non pertinent, 1=peu, 2=pertinent, 3=très pertinent)
recommended_relevances = [3, 0, 2, 1, 0]   # Ordre proposé par le système
ideal_relevances       = [3, 2, 1, 0, 0]   # Ordre idéal (trié par pertinence)

print(f"DCG@5  : {dcg_at_k(recommended_relevances, 5):.4f}")
print(f"IDCG@5 : {dcg_at_k(ideal_relevances, 5):.4f}")
print(f"NDCG@5 : {ndcg_at_k(recommended_relevances, 5):.4f}")

# NDCG avec Spark (pour grande échelle)
from pyspark.sql import functions as F
from pyspark.sql.window import Window

def ndcg_spark(df_scores, df_ground_truth, k=10):
    """
    df_scores       : DataFrame (user_id, item_id, score_prédit)
    df_ground_truth : DataFrame (user_id, item_id, relevance)
    """
    w = Window.partitionBy("user_id").orderBy(F.col("score_predit").desc())

    df_ranked = df_scores \
        .withColumn("rank", F.rank().over(w)) \
        .filter(F.col("rank") <= k) \
        .join(df_ground_truth, ["user_id", "item_id"], "left") \
        .fillna(0, subset=["relevance"])

    df_dcg = df_ranked.withColumn(
        "gain",
        (F.pow(F.lit(2), F.col("relevance")) - 1) /
        F.log2(F.col("rank") + F.lit(1))
    ).groupBy("user_id").agg(F.sum("gain").alias("dcg"))

    # IDCG : classement idéal
    w_ideal = Window.partitionBy("user_id").orderBy(F.col("relevance").desc())
    df_idcg = df_ground_truth \
        .withColumn("rank_ideal", F.rank().over(w_ideal)) \
        .filter(F.col("rank_ideal") <= k) \
        .withColumn(
            "gain_ideal",
            (F.pow(F.lit(2), F.col("relevance")) - 1) /
            F.log2(F.col("rank_ideal") + F.lit(1))
        ).groupBy("user_id").agg(F.sum("gain_ideal").alias("idcg"))

    ndcg_df = df_dcg.join(df_idcg, "user_id") \
        .withColumn("ndcg", F.col("dcg") / F.col("idcg"))

    return ndcg_df.agg(F.avg("ndcg").alias(f"NDCG@{k}")).first()[0]
```

### 2.5 Récapitulatif des métriques

| Métrique | Pertinence binaire | Pertinence graduée | Sensible à la position | Usage typique |
|---|:---:|:---:|:---:|---|
| Precision@K | ✅ | ❌ | Partiel | Recherche d'information |
| Recall@K | ✅ | ❌ | Partiel | Recommandation exhaustive |
| MAP | ✅ | ❌ | ✅ | Moteurs de recherche |
| MRR | ✅ | ❌ | ✅ | QA, résultats uniques |
| NDCG | ✅ | ✅ | ✅ | **Standard industriel** |

---

## 3. PageRank comme algorithme de ranking

PageRank a été traité en détail dans le Module 8. Voici son positionnement dans le contexte du ranking.

### 3.1 PageRank pour le ranking de pages web

```python
from graphframes import GraphFrame
from pyspark.sql import functions as F

# Graphe de pages web : sommets = pages, arêtes = liens hypertexte
vertices = spark.createDataFrame([
    ("p1", "Accueil"),
    ("p2", "Machine Learning"),
    ("p3", "Deep Learning"),
    ("p4", "NLP"),
    ("p5", "Reinforcement Learning"),
    ("p6", "Glossaire"),
], ["id", "titre"])

edges = spark.createDataFrame([
    ("p1","p2"), ("p1","p3"), ("p1","p4"),
    ("p2","p3"), ("p2","p6"), ("p3","p4"),
    ("p3","p6"), ("p4","p2"), ("p4","p5"),
    ("p5","p2"), ("p5","p3"), ("p6","p1"),
], ["src", "dst"])

g = GraphFrame(vertices, edges)

# PageRank → score de ranking pour les pages
pr = g.pageRank(resetProbability=0.15, tol=0.001)

# Intégrer avec un score de pertinence textuelle pour un ranking hybride
requete_pertinence = spark.createDataFrame([
    ("p2", 0.9), ("p3", 0.8), ("p4", 0.6),
    ("p5", 0.3), ("p6", 0.1), ("p1", 0.05),
], ["id", "pertinence_texte"])

# Score hybride = α × pertinence_texte + β × PageRank (normalisé)
pr_norm = pr.vertices.withColumn(
    "pr_norm",
    F.col("pagerank") / pr.vertices.agg(F.max("pagerank")).first()[0]
)

ranking_hybride = pr_norm.join(requete_pertinence, "id") \
    .withColumn("score_final",
        F.lit(0.7) * F.col("pertinence_texte") +
        F.lit(0.3) * F.col("pr_norm")
    ) \
    .join(vertices, "id") \
    .select("titre", "pertinence_texte",
            F.round("pagerank", 4).alias("pagerank"),
            F.round("score_final", 4).alias("score_final")) \
    .orderBy(F.col("score_final").desc())

print("=== Ranking hybride (texte + PageRank) ===")
ranking_hybride.show()
```

---

## 4. ALS — Ranking collaboratif

### 4.1 Principe du filtrage collaboratif

Le **filtrage collaboratif** (*Collaborative Filtering*) prédit les préférences d'un utilisateur en se basant sur les préférences d'utilisateurs **similaires**, sans analyser le contenu des items.

```
Matrice utilisateurs × items (notes de 1 à 5, ? = inconnu) :

              Film1  Film2  Film3  Film4  Film5
Alice           5      3      ?      1      4
Bob             ?      4      2      ?      ?
Claire          3      ?      4      2      ?
David           ?      2      ?      5      3

Objectif : prédire les "?" et recommander les films avec les plus hautes prédictions
```

### 4.2 ALS — Alternating Least Squares

ALS est une technique de **factorisation de matrice** qui décompose la matrice utilisateurs × items en deux matrices de faible rang :

```
       R (m × n)            ≈       U (m × k)      ×     V^T (k × n)
(utilisateurs × items)          (utilisateurs           (items
                                × facteurs latents)      × facteurs latents)^T

Où k << min(m, n) est le nombre de facteurs latents (hyperparamètre)

Chaque facteur latent capture une "dimension" abstraite :
  Genre, époque, style, complexité... (non interprétables directement)
```

**L'algorithme ALS alterne entre deux étapes :**
1. Fixer **V**, optimiser **U** (résolution de systèmes linéaires)
2. Fixer **U**, optimiser **V** (résolution de systèmes linéaires)
3. Répéter jusqu'à convergence

### 4.3 Implémentation avec Spark MLlib

```python
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import functions as F

# ── 1. Préparer les données ───────────────────────────────────────────────────
# ALS nécessite : user_id (Int), item_id (Int), rating (Float)
ratings_data = [
    (1, 101, 5.0), (1, 102, 3.0), (1, 104, 1.0), (1, 105, 4.0),
    (2, 102, 4.0), (2, 103, 2.0),
    (3, 101, 3.0), (3, 103, 4.0), (3, 104, 2.0),
    (4, 102, 2.0), (4, 104, 5.0), (4, 105, 3.0),
    (5, 101, 4.0), (5, 102, 1.0), (5, 103, 5.0),
    (6, 101, 2.0), (6, 104, 4.0), (6, 105, 5.0),
    (7, 102, 5.0), (7, 103, 3.0), (7, 105, 2.0),
    (8, 101, 1.0), (8, 103, 4.0), (8, 104, 3.0),
]
df_ratings = spark.createDataFrame(
    ratings_data, ["user_id", "item_id", "rating"]
)

# ── 2. Split train/test ───────────────────────────────────────────────────────
train, test = df_ratings.randomSplit([0.8, 0.2], seed=42)
print(f"Train : {train.count()} notes, Test : {test.count()} notes")

# ── 3. Entraîner le modèle ALS ────────────────────────────────────────────────
als = ALS(
    userCol      = "user_id",
    itemCol      = "item_id",
    ratingCol    = "rating",
    rank         = 10,            # k : nombre de facteurs latents
    maxIter      = 20,            # Nombre d'itérations ALS
    regParam     = 0.1,           # Régularisation (évite le surapprentissage)
    implicitPrefs= False,         # True pour les données implicites (clics, vues...)
    coldStartStrategy = "drop",   # Gérer les utilisateurs/items inconnus
    seed         = 42
)

model = als.fit(train)

# ── 4. Évaluation ─────────────────────────────────────────────────────────────
predictions = model.transform(test)
predictions.select("user_id", "item_id", "rating", "prediction").show(10)

evaluator = RegressionEvaluator(
    metricName  ="rmse",
    labelCol    ="rating",
    predictionCol="prediction"
)
rmse = evaluator.evaluate(predictions)
print(f"RMSE sur le test : {rmse:.4f}")

# ── 5. Générer des recommandations ────────────────────────────────────────────
# Top 5 items recommandés pour chaque utilisateur
recs_users = model.recommendForAllUsers(5)
recs_users.show(truncate=False)

# Top 5 utilisateurs pour chaque item (ciblage)
recs_items = model.recommendForAllItems(5)
recs_items.show(truncate=False)

# Recommandations pour un sous-ensemble d'utilisateurs
users_subset = df_ratings.select("user_id").distinct().limit(3)
recs_subset  = model.recommendForUserSubset(users_subset, 5)
recs_subset.show(truncate=False)
```

### 4.4 ALS avec données implicites

Dans de nombreux cas réels, on n'a pas de notes explicites (1 à 5) mais des **signaux implicites** (vues, clics, temps de lecture...).

```python
# Données implicites : nombre d'écoutes d'un morceau
ecoutes_data = [
    (1, 201, 12), (1, 202, 3),  (1, 205, 8),
    (2, 201, 1),  (2, 203, 15), (2, 204, 2),
    (3, 202, 7),  (3, 203, 4),  (3, 205, 20),
    (4, 201, 5),  (4, 204, 10), (4, 205, 1),
]
df_ecoutes = spark.createDataFrame(ecoutes_data, ["user_id", "track_id", "nb_ecoutes"])

# ALS implicite : le "rating" est une mesure de confiance, pas une préférence explicite
als_implicit = ALS(
    userCol      = "user_id",
    itemCol      = "track_id",
    ratingCol    = "nb_ecoutes",
    rank         = 8,
    maxIter      = 15,
    regParam     = 0.01,
    implicitPrefs= True,    # ← Mode implicite
    alpha        = 40.0,    # Contrôle la confiance : confidence = 1 + alpha × rating
    coldStartStrategy = "drop"
)

model_implicit = als_implicit.fit(df_ecoutes)
model_implicit.recommendForAllUsers(5).show(truncate=False)
```

### 4.5 Optimisation des hyperparamètres ALS

```python
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

# Grille de recherche
param_grid = ParamGridBuilder() \
    .addGrid(als.rank,     [5, 10, 20]) \
    .addGrid(als.maxIter,  [10, 20]) \
    .addGrid(als.regParam, [0.01, 0.1, 1.0]) \
    .build()

# Validation croisée
cv = CrossValidator(
    estimator   = als,
    estimatorParamMaps = param_grid,
    evaluator   = RegressionEvaluator(
        metricName="rmse", labelCol="rating", predictionCol="prediction"
    ),
    numFolds    = 3,
    seed        = 42
)

cv_model = cv.fit(train)
best_model = cv_model.bestModel
print(f"Meilleur rank     : {best_model.rank}")
print(f"Meilleur regParam : {best_model._java_obj.parent().getRegParam()}")
```

---

## 5. TF-IDF — Ranking de documents textuels

### 5.1 Principe

**TF-IDF** (*Term Frequency - Inverse Document Frequency*) est une mesure de l'importance d'un mot dans un document au sein d'une collection.

```
TF(t, d)  = nombre d'occurrences du terme t dans le document d
             ────────────────────────────────────────────────────
             nombre total de termes dans d

IDF(t, D) = log(N / df(t))
             où N = nombre total de documents
             et df(t) = nombre de documents contenant t

TF-IDF(t, d, D) = TF(t, d) × IDF(t, D)

Intuition :
  - Un mot fréquent dans un document mais rare dans le corpus → score élevé
    (mot caractéristique du document)
  - Un mot très fréquent dans tous les documents ("le", "de"...) → score faible
    (mot peu discriminant)
```

### 5.2 Pipeline TF-IDF avec Spark MLlib

```python
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, HashingTF, IDF,
    RegexTokenizer, CountVectorizer, NGram
)
from pyspark.ml import Pipeline
from pyspark.sql.types import StringType

# ── Corpus de documents ───────────────────────────────────────────────────────
corpus = spark.createDataFrame([
    (1, "Apache Spark est un framework de calcul distribué pour le big data"),
    (2, "Le machine learning avec Spark MLlib permet de traiter de grands volumes"),
    (3, "Les DataFrames Spark sont similaires aux DataFrames Pandas"),
    (4, "Le deep learning nécessite des GPU pour entraîner des réseaux de neurones"),
    (5, "Spark Streaming permet de traiter des flux de données en temps réel"),
    (6, "Le traitement du langage naturel utilise des modèles de deep learning"),
    (7, "La recommandation collaborative avec ALS dans Spark MLlib"),
    (8, "PageRank est un algorithme de graphe utilisé par les moteurs de recherche"),
], ["id", "texte"])

# ── Pipeline de vectorisation TF-IDF ─────────────────────────────────────────
# Étape 1 : Tokenisation
tokenizer = RegexTokenizer(
    inputCol="texte", outputCol="mots",
    pattern=r"\W+",       # Découper sur les non-alphanumériques
    minTokenLength=2      # Ignorer les tokens de moins de 2 caractères
)

# Étape 2 : Suppression des mots vides
stop_words_fr = ["le", "la", "les", "de", "du", "des", "un", "une",
                 "et", "est", "en", "pour", "par", "sur", "avec",
                 "dans", "au", "aux", "il", "elle", "ils", "elles"]
remover = StopWordsRemover(
    inputCol="mots", outputCol="mots_filtres",
    stopWords=stop_words_fr
)

# Étape 3 : TF (Term Frequency) avec hachage
hashing_tf = HashingTF(
    inputCol="mots_filtres", outputCol="tf",
    numFeatures=1000     # Taille du dictionnaire haché
)

# Étape 4 : IDF (Inverse Document Frequency)
idf = IDF(
    inputCol="tf", outputCol="tfidf",
    minDocFreq=1      # Ignorer les termes présents dans < N documents
)

# Assemblage du pipeline
pipeline = Pipeline(stages=[tokenizer, remover, hashing_tf, idf])
model_tfidf = pipeline.fit(corpus)
df_vectorise = model_tfidf.transform(corpus)

df_vectorise.select("id", "mots_filtres", "tfidf").show(truncate=False)
```

### 5.3 Ranking de documents par similarité cosinus

```python
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import udf
from pyspark.ml.linalg import VectorUDT
import numpy as np

# ── Fonction de similarité cosinus ───────────────────────────────────────────
@F.udf("double")
def cosine_similarity(v1, v2):
    """Similarité cosinus entre deux vecteurs sparse."""
    arr1 = v1.toArray()
    arr2 = v2.toArray()
    norm1 = float(np.linalg.norm(arr1))
    norm2 = float(np.linalg.norm(arr2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(arr1, arr2) / (norm1 * norm2))

# ── Vectoriser la requête ─────────────────────────────────────────────────────
def vectoriser_requete(requete_texte):
    df_requete = spark.createDataFrame([(99, requete_texte)], ["id", "texte"])
    return model_tfidf.transform(df_requete).select("tfidf").first()["tfidf"]

# ── Ranker les documents par rapport à une requête ────────────────────────────
def ranker_documents(requete, top_k=5):
    vecteur_requete = vectoriser_requete(requete)

    # Broadcast du vecteur requête pour comparaison distribuée
    vecteur_bc = spark.sparkContext.broadcast(vecteur_requete)

    @F.udf("double")
    def sim_avec_requete(doc_vecteur):
        req = vecteur_bc.value
        d   = doc_vecteur
        arr_r = req.toArray()
        arr_d = d.toArray()
        n1 = float(np.linalg.norm(arr_r))
        n2 = float(np.linalg.norm(arr_d))
        if n1 == 0 or n2 == 0:
            return 0.0
        return float(np.dot(arr_r, arr_d) / (n1 * n2))

    return df_vectorise \
        .withColumn("score", sim_avec_requete(F.col("tfidf"))) \
        .select("id", "texte", F.round("score", 4).alias("similarite")) \
        .orderBy(F.col("similarite").desc()) \
        .limit(top_k)

# Test du moteur de ranking
print("=== Ranking pour 'machine learning deep learning' ===")
ranker_documents("machine learning deep learning", top_k=5).show(truncate=False)

print("=== Ranking pour 'Spark traitement données' ===")
ranker_documents("Spark traitement données", top_k=5).show(truncate=False)
```

### 5.4 BM25 — amélioration de TF-IDF

**BM25** (*Best Match 25*) est le successeur standard de TF-IDF, utilisé par la plupart des moteurs de recherche modernes. Il corrige deux limitations de TF-IDF :
- La **saturation de la fréquence** : au-delà d'un certain seuil, ajouter plus d'occurrences d'un terme n'augmente plus le score
- La **normalisation par longueur** : les documents longs sont pénalisés pour éviter l'avantage des textes verbeux

```
             TF(t,d) × (k1 + 1)
BM25(t,d) = ──────────────────────────────────────── × IDF(t)
             TF(t,d) + k1 × (1 - b + b × |d| / avgdl)

Où :
  k1    = paramètre de saturation (typiquement 1.2 à 2.0)
  b     = paramètre de normalisation par longueur (typiquement 0.75)
  |d|   = longueur du document
  avgdl = longueur moyenne des documents du corpus
```

```python
def bm25_score(tf, df_count, n_docs, doc_len, avg_doc_len, k1=1.5, b=0.75):
    """Calcul du score BM25 pour un terme dans un document."""
    idf = math.log((n_docs - df_count + 0.5) / (df_count + 0.5) + 1)
    tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
    return idf * tf_norm

# Implémentation BM25 sur Spark
def bm25_pipeline(corpus_df, text_col="texte", id_col="id", k1=1.5, b=0.75):
    """Pipeline BM25 complet sur un corpus Spark."""

    # Tokenisation
    tok = RegexTokenizer(inputCol=text_col, outputCol="tokens", pattern=r"\W+")
    df_tok = tok.transform(corpus_df)
    df_tok = df_tok.withColumn("doc_len", F.size("tokens"))

    # Longueur moyenne des documents
    avg_dl = df_tok.agg(F.avg("doc_len")).first()[0]

    # TF : compter les occurrences de chaque mot dans chaque document
    df_tf = df_tok.withColumn("mot", F.explode("tokens")) \
        .groupBy(id_col, "mot") \
        .agg(F.count("*").alias("tf")) \
        .join(df_tok.select(id_col, "doc_len"), id_col)

    # DF : compter dans combien de documents apparaît chaque mot
    n_docs = corpus_df.count()
    df_df = df_tf.groupBy("mot").agg(F.countDistinct(id_col).alias("df_count"))

    # Score BM25
    bm25_udf = F.udf(
        lambda tf, df_count, doc_len:
            bm25_score(tf, df_count, n_docs, doc_len, avg_dl, k1, b),
        "double"
    )

    df_bm25 = df_tf.join(df_df, "mot") \
        .withColumn("bm25", bm25_udf(
            F.col("tf").cast("double"),
            F.col("df_count").cast("double"),
            F.col("doc_len").cast("double")
        ))

    return df_bm25

df_bm25 = bm25_pipeline(corpus)

# Ranking BM25 pour une requête
def ranker_bm25(requete, df_bm25, top_k=5):
    mots_requete = requete.lower().split()
    return df_bm25 \
        .filter(F.col("mot").isin(mots_requete)) \
        .groupBy("id") \
        .agg(F.sum("bm25").alias("score_bm25")) \
        .join(corpus, "id") \
        .select("id", "texte", F.round("score_bm25", 4).alias("score")) \
        .orderBy(F.col("score").desc()) \
        .limit(top_k)

print("=== BM25 pour 'Spark MLlib machine learning' ===")
ranker_bm25("Spark MLlib machine learning", df_bm25).show(truncate=False)
```

---

## 6. Learning to Rank (LTR)

### 6.1 Principe

Le **Learning to Rank** entraîne un **modèle de machine learning** à prédire l'ordre optimal des résultats, en apprenant à partir d'exemples étiquetés (requête → liste ordonnée).

```
Données d'entraînement :
  Requête q1 : [(item_A, rel=3), (item_B, rel=1), (item_C, rel=0)]
  Requête q2 : [(item_D, rel=2), (item_E, rel=2), (item_F, rel=1)]

Features de chaque (requête, item) :
  - Score TF-IDF
  - PageRank du document
  - Fraîcheur du contenu
  - Historique de clics
  - Longueur du document
  - ...

Objectif : apprendre une fonction f(requête, item) → score
           telle que l'ordre induit par f maximise le NDCG
```

### 6.2 Les trois paradigmes LTR

```
POINTWISE              PAIRWISE               LISTWISE
──────────────         ──────────────────     ──────────────────────────
Prédit la              Prédit si A doit       Optimise directement
pertinence de          être ranké avant B     une métrique de ranking
chaque item            pour une requête       (NDCG, MAP...)
séparément

Modèles :              Modèles :              Modèles :
Régression             RankSVM                LambdaMART
linéaire               RankBoost              LambdaRank
Random Forest          RankNet                ListNet

Simple à               Capture les            Meilleure performance
implémenter            préférences            mais plus complexe
                       relatives
```

### 6.3 Implémentation LTR Pointwise avec Spark MLlib

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import RegressionEvaluator

# ── 1. Créer le jeu de données d'entraînement ─────────────────────────────────
# Chaque ligne = (requête, document, features, pertinence)
ltr_data = spark.createDataFrame([
    # (query_id, doc_id, tfidf_score, pagerank, fraicheur_jours, longueur, clics, pertinence)
    (1, "d1", 0.82, 0.45, 10,  850, 120, 3.0),
    (1, "d2", 0.71, 0.12, 30, 1200,  45, 1.0),
    (1, "d3", 0.65, 0.38,  5,  600,  80, 2.0),
    (1, "d4", 0.30, 0.05, 90,  300,  10, 0.0),
    (1, "d5", 0.55, 0.22, 15,  950,  60, 2.0),
    (2, "d6", 0.90, 0.60,  3, 1100, 200, 3.0),
    (2, "d7", 0.45, 0.08, 60,  400,  20, 0.0),
    (2, "d8", 0.78, 0.35, 20,  750, 110, 2.0),
    (2, "d9", 0.62, 0.18, 45,  500,  35, 1.0),
    (3, "d10",0.88, 0.55,  7,  900, 150, 3.0),
    (3, "d11",0.40, 0.10, 80,  350,  15, 0.0),
    (3, "d12",0.75, 0.42, 12,  800,  90, 2.0),
    (3, "d13",0.58, 0.25, 35,  650,  50, 1.0),
], ["query_id", "doc_id", "tfidf", "pagerank", "fraicheur",
    "longueur", "clics", "pertinence"])

# ── 2. Assembler les features ─────────────────────────────────────────────────
feature_cols = ["tfidf", "pagerank", "fraicheur", "longueur", "clics"]

assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features"
)
df_features = assembler.transform(ltr_data)

# ── 3. Split train / test (par query_id pour éviter la fuite de données) ──────
train_queries = [1, 2]
test_queries  = [3]

df_train = df_features.filter(F.col("query_id").isin(train_queries))
df_test  = df_features.filter(F.col("query_id").isin(test_queries))

# ── 4. Entraîner un GBT Regressor (approche Pointwise) ────────────────────────
gbt = GBTRegressor(
    featuresCol ="features",
    labelCol    ="pertinence",
    maxIter     =50,
    maxDepth    =4,
    stepSize    =0.1,
    seed        =42
)
model_ltr = gbt.fit(df_train)

# ── 5. Prédire et ranker ───────────────────────────────────────────────────────
df_pred = model_ltr.transform(df_test)

w_rank = Window.partitionBy("query_id").orderBy(F.col("prediction").desc())
df_ranked = df_pred \
    .withColumn("rang_predit",    F.rank().over(w_rank)) \
    .withColumn("rang_ideal",
        F.rank().over(Window.partitionBy("query_id")
                            .orderBy(F.col("pertinence").desc()))
    )

df_ranked.select(
    "query_id", "doc_id", "pertinence",
    F.round("prediction", 3).alias("score_predit"),
    "rang_predit", "rang_ideal"
).orderBy("query_id", "rang_predit").show()

# ── 6. Importance des features ─────────────────────────────────────────────────
print("\n=== Importance des features ===")
for feat, imp in sorted(
    zip(feature_cols, model_ltr.featureImportances),
    key=lambda x: -x[1]
):
    print(f"  {feat:12s} : {imp:.4f} {'█' * int(imp * 40)}")
```

### 6.4 LambdaMART — LTR listwise (concept)

LambdaMART est l'algorithme LTR état de l'art, utilisé en production par Bing, Yahoo, et la plupart des grands moteurs de recherche.

```
Principe :
  1. Calculer les λ_ij (gradients pondérés par le gain NDCG) pour chaque paire (i, j)
  2. Entraîner un arbre de décision boosté (MART) pour prédire ces gradients
  3. Itérer : chaque arbre corrige les erreurs du précédent

Avantages :
  → Optimise directement NDCG (métrique de ranking)
  → Robuste au bruit dans les annotations
  → Scalable sur de grands corpus

Implémentation :
  → LightGBM : lightgbm.train(..., objective="lambdarank")
  → XGBoost  : xgb.train(..., objective="rank:ndcg")
```

```python
# LambdaMART avec LightGBM (hors Spark MLlib — appel local)
# En production : utiliser les Pandas UDF pour distribuer l'inférence

import lightgbm as lgb
import numpy as np

# Convertir en pandas pour l'entraînement LightGBM
train_pd = df_train.toPandas()
X_train  = train_pd[feature_cols].values
y_train  = train_pd["pertinence"].values
# group : nombre de documents par requête (requis par LambdaMART)
groups   = train_pd.groupby("query_id").size().values

dtrain = lgb.Dataset(X_train, label=y_train, group=groups)

params = {
    "objective"    : "lambdarank",
    "metric"       : "ndcg",
    "ndcg_eval_at" : [3, 5, 10],
    "num_leaves"   : 31,
    "learning_rate": 0.05,
    "min_data_in_leaf": 1,
}

model_lgbm = lgb.train(params, dtrain, num_boost_round=100)

# Inférence distribuée avec Pandas UDF
feature_cols_bc = spark.sparkContext.broadcast(feature_cols)
model_bc        = spark.sparkContext.broadcast(model_lgbm)

@F.pandas_udf("double")
def predict_lgbm(tfidf, pagerank, fraicheur, longueur, clics):
    import pandas as pd
    X = pd.DataFrame({
        "tfidf":    tfidf,
        "pagerank": pagerank,
        "fraicheur":fraicheur,
        "longueur": longueur,
        "clics":    clics
    })
    return pd.Series(model_bc.value.predict(X[feature_cols_bc.value]))

df_scored = df_test.withColumn(
    "score_lambdamart",
    predict_lgbm("tfidf", "pagerank", "fraicheur", "longueur", "clics")
)
df_scored.select("doc_id", "pertinence", F.round("score_lambdamart", 4).alias("score")) \
         .orderBy(F.col("score").desc()).show()
```

---

## 7. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml import Pipeline
import math, numpy as np

spark = SparkSession.builder \
    .appName("RankingAlgorithms") \
    .master("local[*]") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# ─── 1. ALS — Recommandation de films ────────────────────────────────────────
print("=" * 60)
print("=== ALS : Recommandation de films ===")
print("=" * 60)

ratings = spark.createDataFrame([
    (1,1,5.0),(1,2,3.0),(1,4,1.0),(1,5,4.0),(1,6,2.0),
    (2,1,4.0),(2,2,4.0),(2,3,2.0),(2,5,3.0),
    (3,1,3.0),(3,3,4.0),(3,4,2.0),(3,6,5.0),
    (4,2,2.0),(4,4,5.0),(4,5,3.0),(4,6,4.0),
    (5,1,4.0),(5,2,1.0),(5,3,5.0),(5,4,3.0),
    (6,1,2.0),(6,4,4.0),(6,5,5.0),(6,6,3.0),
    (7,2,5.0),(7,3,3.0),(7,5,2.0),(7,6,4.0),
    (8,1,1.0),(8,3,4.0),(8,4,3.0),(8,5,2.0),
], ["user_id","film_id","note"])

films = spark.createDataFrame([
    (1,"Inception"),(2,"Interstellar"),(3,"The Matrix"),
    (4,"Pulp Fiction"),(5,"Fight Club"),(6,"Parasite"),
], ["film_id","titre"])

train, test = ratings.randomSplit([0.8, 0.2], seed=42)

als = ALS(userCol="user_id", itemCol="film_id", ratingCol="note",
          rank=5, maxIter=15, regParam=0.1, coldStartStrategy="drop", seed=42)
model = als.fit(train)

print("\nTop 3 recommandations par utilisateur :")
recs = model.recommendForAllUsers(3)
recs.select("user_id",
    F.explode("recommendations").alias("rec")
).select(
    "user_id",
    F.col("rec.film_id").alias("film_id"),
    F.round("rec.rating", 3).alias("score_predit")
).join(films, "film_id") \
 .select("user_id", "titre", "score_predit") \
 .orderBy("user_id", F.col("score_predit").desc()) \
 .show(15, truncate=False)

# ─── 2. TF-IDF — Moteur de recherche ─────────────────────────────────────────
print("=" * 60)
print("=== TF-IDF : Moteur de recherche de cours ===")
print("=" * 60)

cours = spark.createDataFrame([
    (1, "Introduction au machine learning supervisé et non supervisé"),
    (2, "Deep learning et réseaux de neurones convolutifs pour les images"),
    (3, "Traitement du langage naturel avec les Transformers et BERT"),
    (4, "Apprentissage par renforcement et algorithmes Q-learning"),
    (5, "Spark et le calcul distribué pour le big data"),
    (6, "Recommandation collaborative et filtrage basé contenu"),
    (7, "Détection d'anomalies et apprentissage non supervisé"),
    (8, "Séries temporelles et prévision avec LSTM"),
], ["id", "titre"])

stop_words = ["et","le","la","les","de","du","des","un","une","pour",
              "par","sur","avec","au","aux","en","les","à"]

pipeline_tfidf = Pipeline(stages=[
    RegexTokenizer(inputCol="titre", outputCol="tokens", pattern=r"\W+"),
    StopWordsRemover(inputCol="tokens", outputCol="tokens_filtres",
                     stopWords=stop_words),
    HashingTF(inputCol="tokens_filtres", outputCol="tf", numFeatures=500),
    IDF(inputCol="tf", outputCol="tfidf"),
])
model_tfidf = pipeline_tfidf.fit(cours)
df_vect = model_tfidf.transform(cours)

def rechercher(requete, top_k=5):
    df_req   = spark.createDataFrame([(99, requete)], ["id", "titre"])
    vect_req = model_tfidf.transform(df_req).first()["tfidf"]
    vect_bc  = spark.sparkContext.broadcast(vect_req)

    @F.udf("double")
    def sim(v):
        a = np.array(vect_bc.value.toArray())
        b = np.array(v.toArray())
        n1, n2 = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a,b)/(n1*n2)) if n1>0 and n2>0 else 0.0

    return df_vect \
        .withColumn("score", sim(F.col("tfidf"))) \
        .select("id", "titre", F.round("score", 4).alias("score")) \
        .orderBy(F.col("score").desc()) \
        .limit(top_k)

print("\nRecherche : 'deep learning images'")
rechercher("deep learning images").show(truncate=False)

print("Recherche : 'apprentissage non supervisé'")
rechercher("apprentissage non supervisé").show(truncate=False)

# ─── 3. Évaluation NDCG ──────────────────────────────────────────────────────
print("=" * 60)
print("=== Évaluation NDCG@5 ===")
print("=" * 60)

# Vérité terrain simulée pour la requête "deep learning images"
ground_truth = {2: 3, 3: 2, 8: 1, 1: 0, 4: 0, 5: 0, 6: 0, 7: 0}

df_ranked = rechercher("deep learning images", top_k=8) \
    .withColumn("rang", F.monotonically_increasing_id() + 1)

results = [(row["id"], row["titre"], row["score"],
            ground_truth.get(row["id"], 0))
           for row in df_ranked.collect()]

print(f"\n{'Rang':>4} {'Titre':45s} {'Score':>7} {'Pertinence':>10}")
print("-" * 75)
for i, (doc_id, titre, score, rel) in enumerate(results, 1):
    barre = "★" * rel
    print(f"{i:>4} {titre:45s} {score:>7.4f} {rel:>5} {barre}")

# Calcul NDCG
relevances_ordonnees = [r for _, _, _, r in results]
ndcg5  = ndcg_at_k(relevances_ordonnees, 5)
ndcg8  = ndcg_at_k(relevances_ordonnees, 8)
print(f"\nNDCG@5 : {ndcg5:.4f}")
print(f"NDCG@8 : {ndcg8:.4f}")

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Ranking** | Ordonner des items par pertinence pour une requête ou un contexte |
| **Precision@K / Recall@K** | Métriques simples — ne tiennent pas compte de la position exacte |
| **MAP** | Moyenne des précisions aux positions des items pertinents — sensible à l'ordre |
| **MRR** | Position du premier item pertinent — utile pour les systèmes à réponse unique |
| **NDCG** | Standard industriel — tient compte de la position ET de la pertinence graduée |
| **PageRank** | Ranking par autorité dans un graphe — combinable avec un score textuel |
| **ALS** | Factorisation de matrice pour le ranking collaboratif — `pyspark.ml.recommendation` |
| **ALS implicite** | Pour les signaux implicites (clics, vues) — paramètre `alpha` pour la confiance |
| **TF-IDF** | Ranking documentaire classique — implémenté dans `pyspark.ml.feature` |
| **BM25** | Amélioration de TF-IDF avec saturation et normalisation par longueur |
| **LTR Pointwise** | Modèle ML (GBT) entraîné à prédire la pertinence item par item |
| **LambdaMART** | LTR listwise état de l'art — optimise directement NDCG |

---

## Exercices

### Exercice 1 — Métriques d'évaluation (30 min)
> Implémenter en PySpark (et non en Python pur) les fonctions de calcul de :
> 1. Precision@K et Recall@K
> 2. Average Precision et MAP
> 3. NDCG@K avec pertinence graduée (0, 1, 2, 3)
>
> Les tester sur un jeu de données de recommandation synthétique de 100 utilisateurs, 50 items et 500 interactions annotées.

### Exercice 2 — ALS et recommandation (40 min)
> À partir du jeu de données MovieLens (disponible sur [grouplens.org](https://grouplens.org/datasets/movielens/)) :
> 1. Charger les ratings et les métadonnées des films
> 2. Entraîner un modèle ALS avec optimisation des hyperparamètres (grid search)
> 3. Générer le top 10 de recommandations pour 5 utilisateurs de votre choix
> 4. Calculer le NDCG@10 sur un ensemble de test
> 5. Comparer ALS explicite vs implicite (utiliser le nombre de ratings comme signal implicite)

### Exercice 3 — Moteur de recherche TF-IDF (35 min)
> Construire un mini-moteur de recherche sur un corpus de 100 articles Wikipedia :
> 1. Télécharger et charger les articles avec Spark
> 2. Construire le pipeline TF-IDF (tokenisation → stop words → HashingTF → IDF)
> 3. Implémenter la recherche par similarité cosinus
> 4. Améliorer avec des bigrammes (NGram) et comparer le NDCG
> 5. Comparer TF-IDF vs BM25 sur 10 requêtes de référence

### Exercice 4 — Pipeline LTR complet (50 min)
> Construire un système de ranking hybride :
> 1. Features : score TF-IDF, PageRank du document, fraîcheur, longueur
> 2. Collecter ou simuler des annotations de pertinence (0-3) pour 20 requêtes × 10 documents
> 3. Entraîner un GBT Regressor (approche Pointwise) et un modèle LambdaMART (LightGBM)
> 4. Évaluer les deux approches avec NDCG@5 et MAP
> 5. Analyser l'importance des features dans chaque modèle
> 6. Déployer l'inférence en distribué avec une Pandas UDF Spark

---

## Pour aller plus loin

- 📖 **ALS dans Spark MLlib** : [spark.apache.org/docs/latest/ml-collaborative-filtering.html](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html)
- 📖 **Spark MLlib Feature** : [spark.apache.org/docs/latest/ml-features.html](https://spark.apache.org/docs/latest/ml-features.html)
- 📄 **ALS original** : *Collaborative Filtering for Implicit Feedback Datasets* — Hu, Koren & Volinsky, ICDM 2008
- 📄 **BM25** : *The Probabilistic Relevance Framework: BM25 and Beyond* — Robertson & Zaragoza, 2009
- 📄 **LambdaMART** : *From RankNet to LambdaRank to LambdaMART* — Burges, Microsoft Research, 2010
- 📄 **NDCG** : *Cumulated Gain-Based Evaluation of IR Techniques* — Järvelin & Kekäläinen, ACM TOIS 2002
- 🛠️ **LightGBM** : implémentation LambdaMART très efficace — [lightgbm.readthedocs.io](https://lightgbm.readthedocs.io)
- 🛠️ **MovieLens** : jeu de données standard pour évaluer les systèmes de recommandation — [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/)
- 🛠️ **LETOR** : benchmark officiel pour le Learning to Rank — [microsoft.com/en-us/research/project/letor](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/)

---

*Module précédent → **Module 9 : Algorithmes de graphe***

---

# 🎓 Félicitations — Cours complet !

Vous avez couvert l'intégralité du programme Spark / PySpark pour Master 1 IA :

| Module | Thème |
|:---:|---|
| 1 | Introduction à Spark et à l'écosystème Big Data |
| 2 | Installation et prise en main |
| 3 | Les structures de données Spark |
| 4 | Transformations de type *mapper* |
| 5 | Transformations de type *reducer* |
| 6 | Partitionnement, shuffles et collecte |
| 7 | Interaction avec des sources externes |
| 8 | Feature Engineering avec PySpark |
| 9 | Algorithmes de graphe |
| 10 | Algorithmes de ranking |
