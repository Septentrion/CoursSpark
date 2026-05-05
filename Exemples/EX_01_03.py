# ── Étape 1 : Installation (dans le terminal) ─────────────────────────────────
# pip install pyspark jupyter

# ── Étape 2 : Imports et création de la SparkSession ─────────────────────────
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.appName("PremierProgramme_Module1")
    .master("local[*]")
    .config("spark.sql.shuffle.partitions", "4")
    .getOrCreate()
)

# Vérification
print(f"Spark version  : {spark.version}")
print(f"Python version : {spark.sparkContext.pythonVer}")
print(f"Master         : {spark.sparkContext.master}")
print(f"Parallélisme   : {spark.sparkContext.defaultParallelism}")

# ── Étape 3 : Création d'un DataFrame depuis une liste Python ─────────────────
data = [
    ("Alice", "Data Science", 28, 3200.0),
    ("Bob", "Développement", 35, 2800.0),
    ("Claire", "Data Science", 22, 2500.0),
    ("David", "DevOps", 41, 3500.0),
    ("Emma", "Développement", 29, 2900.0),
    ("François", "Data Science", 33, 3100.0),
    ("Gaëlle", "DevOps", 26, 2700.0),
    ("Henri", "Développement", 38, 3300.0),
]
colonnes = ["nom", "departement", "age", "salaire"]
df = spark.createDataFrame(data, schema=colonnes)

# ── Étape 4 : Exploration du DataFrame ───────────────────────────────────────
print("\n=== Schéma ===")
df.printSchema()

print("\n=== Aperçu complet ===")
df.show()

print(f"\nNombre de lignes      : {df.count()}")
print(f"Nombre de colonnes    : {len(df.columns)}")
print(f"Nombre de partitions  : {df.rdd.getNumPartitions()}")

# ── Étape 5 : Filtrage et transformations ─────────────────────────────────────
print("\n=== Data Scientists de moins de 30 ans ===")
df_filtre = df.filter((F.col("departement") == "Data Science") & (F.col("age") < 30))
df_filtre.show()
print(f"Résultat : {df_filtre.count()} personne(s)")

# ── Étape 6 : Agrégation par département ─────────────────────────────────────
print("\n=== Statistiques par département ===")
df.groupBy("departement").agg(
    F.count("*").alias("nb_employes"),
    F.round(F.avg("salaire"), 2).alias("salaire_moyen"),
    F.min("age").alias("age_min"),
    F.max("age").alias("age_max"),
).orderBy("departement").show()

# ── Étape 7 : Observer le plan d'exécution (= le DAG) ────────────────────────
print("\n=== Plan d'exécution (DAG) ===")
df.filter(F.col("salaire") > 3000).groupBy("departement").count().explain()

# ── Notes sur la Spark UI ─────────────────────────────────────────────────────
print("""
=== Observer le DAG dans la Spark UI ===

1. Ouvrir un navigateur et aller sur : http://localhost:4040

2. Onglet "Jobs" :
   → Chaque appel à .count(), .show(), .collect() crée un Job
   → Observer les Jobs créés par ce programme

3. Onglet "Stages" :
   → Cliquer sur un Job pour voir ses Stages
   → Un Stage = ensemble de Tasks sans shuffle entre elles

4. Onglet "SQL / DataFrame" :
   → Visualisation graphique du DAG
   → Voir les opérations : Scan, Filter, HashAggregate, Exchange (shuffle)

5. Onglet "Executors" :
   → En mode local[*], un seul Executor (le Driver)
   → Observer la mémoire utilisée et les Tasks complétées
""")

# ── Fermeture propre ──────────────────────────────────────────────────────────
input("Appuyez sur Entrée pour arrêter Spark...")
spark.stop()
print("SparkSession fermée.")
