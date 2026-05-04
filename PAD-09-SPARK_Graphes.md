# Module 8 — Algorithmes de graphe avec GraphX / GraphFrames

> **Public cible** : Master 1 Intelligence Artificielle  
> **Durée estimée** : 3 heures  
> **Prérequis** : Module 7 — Interaction avec des sources externes

---

## Objectifs pédagogiques

À l'issue de ce module, l'étudiant sera capable de :

- Modéliser un problème réel sous forme de graphe (sommets, arêtes, propriétés)
- Créer et manipuler un graphe avec GraphFrames en PySpark
- Appliquer les algorithmes fondamentaux : BFS, composantes connexes, triangle counting
- Implémenter et interpréter PageRank sur un graphe distribué
- Identifier les cas d'usage concrets des graphes en IA et en data science

---

## 1. Introduction à la théorie des graphes

### 1.1 Définitions fondamentales

Un **graphe** G = (V, E) est une structure mathématique composée de :
- **V** (*Vertices*) : un ensemble de **sommets** (nœuds)
- **E** (*Edges*) : un ensemble d'**arêtes** (liens) entre sommets

```
Graphe non orienté :          Graphe orienté (digraphe) :
    Alice ──── Bob                Alice ──► Bob
      │    ╲                        │
    Claire   David               Claire ──► David
      │                             ▲
    Emma                          Emma

Chaque arête est               Chaque arête a un
bidirectionnelle               sens (source → destination)
```

**Propriétés importantes :**

| Propriété | Description | Exemple |
|---|---|---|
| **Orienté / Non orienté** | Les arêtes ont-elles un sens ? | Twitter (orienté) vs Facebook (non orienté) |
| **Pondéré** | Les arêtes ont-elles un poids ? | Distance routière, force d'une relation |
| **Connexe** | Peut-on atteindre tout sommet depuis n'importe quel autre ? | Réseau sans îlots isolés |
| **Cyclique** | Existe-t-il des chemins qui reviennent au point de départ ? | Graphes DAG vs graphes généraux |
| **Biparti** | Les sommets se divisent en 2 groupes, arêtes seulement entre groupes | Utilisateurs ↔ Produits |

### 1.2 Vocabulaire courant

```
Degré d'un sommet    : nombre d'arêtes connectées à ce sommet
Degré entrant        : arêtes arrivant sur le sommet (graphe orienté)
Degré sortant        : arêtes partant du sommet (graphe orienté)
Chemin               : suite de sommets reliés par des arêtes
Plus court chemin    : chemin minimisant le nombre d'arêtes (ou leur poids)
Diamètre             : longueur du plus long des plus courts chemins
Voisin               : sommet directement relié par une arête
Triangle             : 3 sommets tous reliés entre eux deux à deux
Composante connexe   : sous-graphe connexe maximal
Communauté           : groupe de sommets très interconnectés
```

### 1.3 Cas d'usage en IA et data science

| Domaine | Sommets | Arêtes | Algorithmes |
|---|---|---|---|
| **Réseaux sociaux** | Utilisateurs | Amitiés, follows | PageRank, communautés, influence |
| **Knowledge Graph** | Entités | Relations | Recherche de chemins, inférence |
| **Détection de fraude** | Comptes, transactions | Virements | Composantes connexes, cycles |
| **Moteurs de recherche** | Pages web | Liens hypertexte | PageRank |
| **Recommandation** | Utilisateurs, produits | Achats, notes | Propagation de labels |
| **Bioinformatique** | Gènes, protéines | Interactions | Clustering, chemins |
| **Logistique** | Entrepôts, villes | Routes | Plus court chemin, flot max |
| **NLP** | Mots, documents | Co-occurrences | TextRank (résumé auto) |

---

## 2. GraphX vs GraphFrames

### 2.1 GraphX (API Scala)

**GraphX** est la bibliothèque de graphes native de Spark, disponible en **Scala et Java uniquement**. Elle est très performante mais non accessible directement depuis PySpark.

```scala
// GraphX — Scala uniquement
import org.apache.spark.graphx._
val vertices: RDD[(VertexId, String)] = sc.parallelize(Seq((1L, "Alice"), (2L, "Bob")))
val edges: RDD[Edge[String]] = sc.parallelize(Seq(Edge(1L, 2L, "ami")))
val graph = Graph(vertices, edges)
graph.pageRank(0.001).vertices.collect()
```

### 2.2 GraphFrames (API Python/Scala)

**GraphFrames** est une bibliothèque open source construite sur les DataFrames Spark, qui expose une **API Python complète** et donne accès à la plupart des algorithmes de GraphX.

```
GraphX (bas niveau)          GraphFrames (haut niveau)
────────────────────         ─────────────────────────
RDD-based                    DataFrame-based
Scala/Java seulement         Python, Scala, Java, R
API verbeuse                 API concise et expressive
Très performant              Légèrement moins rapide
Natif dans Spark             Package externe (graphframes)
```

### 2.3 Installation de GraphFrames

```bash
# Via spark-submit
spark-submit --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12 mon_script.py

# Via pyspark en ligne de commande
pyspark --packages graphframes:graphframes:0.8.3-spark3.5-s_2.12

# Dans le code Python
spark = SparkSession.builder \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .getOrCreate()

# Import
from graphframes import GraphFrame
```

---

## 3. Modélisation et création d'un graphe

### 3.1 Structure d'un GraphFrame

Un `GraphFrame` est composé de **deux DataFrames** :
- **`vertices`** : les sommets, avec une colonne obligatoire `id`
- **`edges`** : les arêtes, avec des colonnes obligatoires `src` et `dst`

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

spark = SparkSession.builder \
    .appName("GraphFrames") \
    .master("local[*]") \
    .config("spark.jars.packages", "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .getOrCreate()

# ── DataFrame des sommets ─────────────────────────────────────────────────────
# Colonne obligatoire : "id"
# Colonnes libres : attributs des sommets
vertices = spark.createDataFrame([
    ("alice",   "Alice Martin",   "Ingénieure",  35),
    ("bob",     "Bob Dupont",     "Data Analyst", 28),
    ("claire",  "Claire Petit",   "Chercheuse",   42),
    ("david",   "David Leroy",    "Ingénieur",    31),
    ("emma",    "Emma Garnier",   "Étudiante",    24),
    ("francois","François Blanc", "Manager",      47),
    ("gaelle",  "Gaëlle Morel",   "Designeuse",   33),
], ["id", "nom", "metier", "age"])

# ── DataFrame des arêtes ──────────────────────────────────────────────────────
# Colonnes obligatoires : "src" (source), "dst" (destination)
# Colonnes libres : attributs des arêtes
edges = spark.createDataFrame([
    ("alice",    "bob",      "collègue",  5),
    ("alice",    "claire",   "amie",      8),
    ("bob",      "david",    "collègue",  3),
    ("bob",      "emma",     "ami",       7),
    ("claire",   "francois", "collègue",  6),
    ("david",    "alice",    "collègue",  4),
    ("david",    "gaelle",   "ami",       9),
    ("emma",     "alice",    "étudiante", 2),
    ("francois", "alice",    "manager",   5),
    ("francois", "claire",   "collègue",  7),
    ("gaelle",   "bob",      "amie",      6),
    ("claire",   "david",    "collègue",  4),
], ["src", "dst", "relation", "force"])

# ── Création du GraphFrame ────────────────────────────────────────────────────
g = GraphFrame(vertices, edges)

# Inspecter le graphe
print(f"Sommets : {g.vertices.count()}")
print(f"Arêtes  : {g.edges.count()}")

g.vertices.show()
g.edges.show()
```

### 3.2 Propriétés de base du graphe

```python
# ── Degrés des sommets ────────────────────────────────────────────────────────
print("=== Degrés (total) ===")
g.degrees.orderBy(F.col("degree").desc()).show()

print("=== Degrés entrants ===")
g.inDegrees.orderBy(F.col("inDegree").desc()).show()

print("=== Degrés sortants ===")
g.outDegrees.orderBy(F.col("outDegree").desc()).show()

# ── Requêtes sur sommets et arêtes ───────────────────────────────────────────
# Trouver tous les collègues d'Alice
g.edges.filter(
    (F.col("src") == "alice") & (F.col("relation") == "collègue")
).show()

# Qui a une relation de force > 7 ?
g.edges.filter(F.col("force") > 7).show()

# Sommets de plus de 30 ans
g.vertices.filter(F.col("age") > 30).show()
```

---

## 4. Recherche de motifs (*Motif Finding*)

### 4.1 Principe

La recherche de motifs permet de trouver des **structures récurrentes** dans un graphe en utilisant une syntaxe déclarative inspirée de Cypher (Neo4j).

```python
# Syntaxe : (sommet)-[arête]->(sommet)
# Les noms entre parenthèses/crochets sont des variables de capture

# ── Tous les chemins A → B → C ────────────────────────────────────────────────
chemins = g.find("(a)-[e1]->(b); (b)-[e2]->(c)")
chemins.select("a.id", "b.id", "c.id", "e1.relation", "e2.relation").show()

# ── Relations réciproques (A → B ET B → A) ───────────────────────────────────
reciproques = g.find("(a)-[e1]->(b); (b)-[e2]->(a)")
reciproques.select("a.id", "b.id").show()

# ── Triangles (A → B → C → A) ────────────────────────────────────────────────
triangles = g.find("(a)-[e1]->(b); (b)-[e2]->(c); (c)-[e3]->(a)")
triangles.select("a.id", "b.id", "c.id").show()

# ── Amis d'amis (avec filtre) ─────────────────────────────────────────────────
amis_amis = g.find("(a)-[e1]->(b); (b)-[e2]->(c)") \
    .filter("e1.relation = 'ami' AND e2.relation = 'ami'") \
    .filter("a.id != c.id") \    # Exclure A lui-même
    .select("a.id", "c.id").distinct()
amis_amis.show()
```

---

## 5. BFS — Breadth-First Search

### 5.1 Principe

La **recherche en largeur (BFS)** explore un graphe en partant d'un sommet source et en visitant tous les voisins de niveau en niveau. Elle permet de trouver le **plus court chemin** (en nombre d'arêtes) entre deux sommets.

```
Graphe :  alice → bob → emma
                ↘
          alice → claire → francois

BFS depuis "alice" vers "francois" :
  Niveau 0 : {alice}
  Niveau 1 : {bob, claire}          ← voisins d'alice
  Niveau 2 : {emma, francois} ✓    ← voisins de bob et claire
  Chemin trouvé : alice → claire → francois (distance = 2)
```

### 5.2 BFS avec GraphFrames

```python
# BFS : chemin le plus court d'Alice vers François
resultat_bfs = g.bfs(
    fromExpr="id = 'alice'",          # Condition de départ
    toExpr  ="id = 'francois'",       # Condition d'arrivée
    edgeFilter="relation != 'ennemi'", # Filtrer certaines arêtes (optionnel)
    maxPathLength=5                    # Profondeur maximale de recherche
)
resultat_bfs.show(truncate=False)

# Le résultat contient les colonnes :
# from, e0, v1, e1, v2, ... to
# où v_i sont les sommets intermédiaires et e_i les arêtes empruntées
```

### 5.3 BFS personnalisé avec Pregel

Pour des BFS plus complexes, on peut utiliser l'API **Pregel** qui permet d'implémenter des algorithmes itératifs sur graphe (paradigme *think like a vertex*).

```python
# Pregel : chaque sommet envoie des messages à ses voisins
# à chaque itération, jusqu'à convergence

from graphframes.lib import AggregateMessages as AM

# Calculer la distance minimale depuis un sommet source
source_id = "alice"

# Initialiser : distance 0 pour la source, infini pour les autres
vertices_init = g.vertices.withColumn(
    "dist",
    F.when(F.col("id") == source_id, 0).otherwise(float("inf"))
)
g_init = GraphFrame(vertices_init, g.edges)

# Fonction d'agrégation des messages
agg_msgs = F.min(AM.msg)

# Fonction d'envoi de messages
send_msgs = AM.src["dist"] + F.lit(1)

# Condition d'arrêt : plus aucune mise à jour
def update_vertices(old_dist, new_dist):
    return F.when(new_dist < old_dist, new_dist).otherwise(old_dist)

# Note : en PySpark, Pregel s'utilise via l'API Java sous-jacente
# Pour simplifier, on utilise des itérations manuelles :

def bfs_distances(g, source_id, max_iter=10):
    """Calcule les distances BFS depuis source_id."""
    dist_df = g.vertices.withColumn(
        "distance",
        F.when(F.col("id") == source_id, 0).otherwise(float("inf"))
    )
    for _ in range(max_iter):
        # Propager : pour chaque arête (src, dst), proposer dist(src) + 1
        updates = g.edges.join(
            dist_df.select("id", "distance"),
            g.edges["src"] == dist_df["id"]
        ).select(
            F.col("dst").alias("id"),
            (F.col("distance") + 1).alias("new_dist")
        ).groupBy("id").agg(F.min("new_dist").alias("new_dist"))

        # Mettre à jour : conserver la distance minimale
        dist_df = dist_df.join(updates, "id", "left") \
            .withColumn("distance",
                F.when(
                    F.col("new_dist") < F.col("distance"),
                    F.col("new_dist")
                ).otherwise(F.col("distance"))
            ).drop("new_dist")
    return dist_df

distances = bfs_distances(g, "alice")
distances.orderBy("distance").show()
```

---

## 6. Composantes connexes

### 6.1 Composantes connexes (*Connected Components*)

Une **composante connexe** est un sous-ensemble de sommets tel que tout sommet est accessible depuis tout autre par un chemin, et qu'aucun chemin ne permet de rejoindre un sommet hors de ce sous-ensemble.

```
Graphe avec 3 composantes connexes :

  Composante 1 :    Composante 2 :    Composante 3 :
  alice ── bob      claire            david ── emma
    |                                    |
  gaelle                              francois

Algorithme : assigne un même identifiant à tous les sommets
             d'une même composante
```

```python
# Composantes connexes
spark.sparkContext.setCheckpointDir("/tmp/graphframes-checkpoints")

cc = g.connectedComponents()
cc.select("id", "component").orderBy("component").show()

# Résultat :
# +--------+-----------+
# |      id|  component|
# +--------+-----------+
# |   alice|          0|
# |     bob|          0|
# |  claire|          0|
# |   david|          0|
# |    emma|          0|
# |francois|          0|
# |  gaelle|          0|
# +--------+-----------+
# (Tout le monde est dans la même composante ici)

# Compter les composantes et leur taille
cc.groupBy("component") \
  .agg(F.count("*").alias("taille"),
       F.collect_list("id").alias("membres")) \
  .orderBy(F.col("taille").desc()) \
  .show(truncate=False)
```

### 6.2 Composantes fortement connexes (*Strongly Connected Components*)

Dans un graphe **orienté**, une composante fortement connexe (SCC) est un ensemble de sommets où chaque sommet est accessible depuis chaque autre en suivant le sens des arêtes.

```python
# Strongly Connected Components
scc = g.stronglyConnectedComponents(maxIter=10)
scc.select("id", "component").orderBy("component").show()

# Cas d'usage : détection de cycles dans un graphe de dépendances,
# identification de groupes d'influence dans un réseau social orienté
```

---

## 7. Comptage de triangles

### 7.1 Principe et importance

Un **triangle** est un ensemble de 3 sommets A, B, C tels que les arêtes A-B, B-C et A-C existent toutes. Le comptage de triangles est fondamental pour calculer le **coefficient de clustering**, qui mesure à quel point les voisins d'un sommet sont eux-mêmes connectés entre eux.

```
Coefficient de clustering local de v =
    (nombre de triangles passant par v)
    ─────────────────────────────────────────────
    (nombre de paires de voisins de v possibles)
    = triangles(v) / (deg(v) × (deg(v) - 1) / 2)
```

```python
# Comptage de triangles par sommet
triangles_count = g.triangleCount()
triangles_count.select("id", "count") \
    .orderBy(F.col("count").desc()) \
    .show()

# Calcul du coefficient de clustering
deg_df = g.degrees
tc_df  = g.triangleCount()

clustering = tc_df.join(deg_df, "id") \
    .withColumn("coeff_clustering",
        F.when(F.col("degree") <= 1, 0.0)
         .otherwise(
             F.round(
                 F.col("count") * 2 /
                 (F.col("degree") * (F.col("degree") - 1)),
                 4
             )
         )
    ) \
    .select("id", "count", "degree", "coeff_clustering") \
    .orderBy(F.col("coeff_clustering").desc())

clustering.show()
```

---

## 8. PageRank

### 8.1 Histoire et intuition

**PageRank** a été inventé par Larry Page et Sergey Brin à Stanford en 1998, et est devenu le fondement de Google Search. L'intuition est simple :

> *Une page web est importante si de nombreuses pages importantes pointent vers elle.*

C'est une définition récursive qui se résout par itération.

### 8.2 Formule mathématique

Pour un graphe de N sommets, le PageRank PR(v) d'un sommet v est défini par :

```
                    PR(u)
PR(v) = (1 - d) + d × Σ  ────────
                   u→v  deg_out(u)

où :
  d         = facteur d'amortissement (damping factor), typiquement 0.85
  u → v     = tous les sommets u ayant une arête vers v
  deg_out(u) = degré sortant de u (nombre d'arêtes partant de u)
  (1 - d)   = probabilité de "téléportation" aléatoire vers n'importe quelle page
```

### 8.3 Interprétation probabiliste

PageRank modélise le comportement d'un **surfeur aléatoire** sur le graphe :
- Avec probabilité **d (=0.85)** : il suit un lien de la page courante
- Avec probabilité **(1-d) (=0.15)** : il "téléporte" vers une page aléatoire

Le PageRank de v est la probabilité que ce surfeur se trouve sur v à un instant donné.

### 8.4 Algorithme itératif

```
Initialisation : PR(v) = 1/N pour tout v

Itération t+1 :
  Pour chaque sommet v :
               PR_t(u)
  PR_{t+1}(v) = (1-d) + d × Σ  ────────
                        u→v  deg_out(u)

Répéter jusqu'à convergence (|PR_{t+1} - PR_t| < ε)
```

### 8.5 Implémentation avec GraphFrames

```python
# ── PageRank avec GraphFrames ─────────────────────────────────────────────────

# Méthode 1 : convergence automatique (recommandée)
pr_converge = g.pageRank(resetProbability=0.15, tol=0.001)
# resetProbability = 1 - d = probabilité de téléportation

pr_converge.vertices \
    .select("id", "nom", "pagerank") \
    .orderBy(F.col("pagerank").desc()) \
    .show()

pr_converge.edges \
    .select("src", "dst", "weight") \
    .orderBy(F.col("weight").desc()) \
    .show()

# Méthode 2 : nombre fixe d'itérations
pr_iter = g.pageRank(resetProbability=0.15, maxIter=20)
pr_iter.vertices.orderBy(F.col("pagerank").desc()).show()
```

### 8.6 Implémentation manuelle (pédagogique)

```python
def pagerank_manuel(g, d=0.85, n_iter=20, verbose=False):
    """
    Implémentation manuelle de PageRank pour illustrer l'algorithme.
    d : facteur d'amortissement (damping factor)
    """
    n = g.vertices.count()

    # Initialisation : PR = 1/N pour tous les sommets
    pr = g.vertices.select("id").withColumn("pagerank", F.lit(1.0 / n))

    # Calculer le degré sortant de chaque sommet
    out_deg = g.outDegrees  # DataFrame : id, outDegree

    for i in range(n_iter):
        # Contribution de chaque sommet u vers ses voisins :
        # contribution(u → v) = PR(u) / deg_out(u)
        contributions = g.edges \
            .join(pr.withColumnRenamed("id", "src")
                    .withColumnRenamed("pagerank", "pr_src"), "src") \
            .join(out_deg.withColumnRenamed("id", "src")
                         .withColumnRenamed("outDegree", "deg_src"), "src") \
            .withColumn("contrib", F.col("pr_src") / F.col("deg_src")) \
            .groupBy("dst") \
            .agg(F.sum("contrib").alias("sum_contrib")) \
            .withColumnRenamed("dst", "id")

        # Mise à jour : PR_new(v) = (1-d)/N + d × sum_contrib(v)
        pr_new = g.vertices.select("id") \
            .join(contributions, "id", "left") \
            .withColumn("sum_contrib", F.coalesce(F.col("sum_contrib"), F.lit(0.0))) \
            .withColumn("pagerank",
                F.lit((1.0 - d) / n) + F.lit(d) * F.col("sum_contrib")
            ) \
            .select("id", "pagerank")

        if verbose:
            delta = pr.join(pr_new.withColumnRenamed("pagerank", "pr_new"), "id") \
                .withColumn("delta", F.abs(F.col("pagerank") - F.col("pr_new"))) \
                .agg(F.max("delta")).first()[0]
            print(f"Itération {i+1:2d} — delta max : {delta:.8f}")

        pr = pr_new

    return pr

# Appel
pr_result = pagerank_manuel(g, d=0.85, n_iter=15, verbose=True)

# Joindre avec les attributs des sommets
pr_final = pr_result.join(g.vertices, "id") \
    .select("id", "nom", "metier", "pagerank") \
    .orderBy(F.col("pagerank").desc())

print("\n=== PageRank final ===")
pr_final.show()
```

### 8.7 Variantes de PageRank

```python
# ── PageRank personnalisé (Personalized PageRank) ─────────────────────────────
# La téléportation se fait vers un sous-ensemble de sommets (pas tous)
# → Mesure la proximité d'un sommet par rapport à un sommet de référence
# Non disponible directement dans GraphFrames — nécessite une implémentation manuelle

# ── PageRank pondéré ───────────────────────────────────────────────────────────
# Si les arêtes ont des poids, on pondère les contributions
# contribution(u → v) = PR(u) × poids(u→v) / Σ poids(arêtes sortantes de u)

edges_ponderees = g.edges  # Colonne "force" disponible

contributions_pond = edges_ponderees \
    .groupBy("src") \
    .agg(F.sum("force").alias("poids_total_sortant")) \
    .join(edges_ponderees, "src") \
    .withColumn("poids_normalise", F.col("force") / F.col("poids_total_sortant"))
# Puis utiliser poids_normalise comme contribution à la place de 1/deg_out
```

---

## 9. Label Propagation — détection de communautés

### 9.1 Principe

L'algorithme de **propagation de labels** (*Label Propagation Algorithm*, LPA) détecte des communautés (clusters) dans un graphe sans avoir à spécifier leur nombre à l'avance. Chaque sommet adopte le label le plus fréquent parmi ses voisins, jusqu'à stabilisation.

```
Initialisation : chaque sommet a son propre label (son id)

Itération 1 :
  alice   (voisins : bob, claire, david) → label majoritaire parmi {bob, claire, david}
  bob     (voisins : alice, gaelle)      → label majoritaire parmi {alice, gaelle}
  ...

Répéter jusqu'à stabilisation → les groupes de même label forment des communautés
```

```python
# Label Propagation avec GraphFrames
lpa = g.labelPropagation(maxIter=10)
lpa.select("id", "nom", "label").orderBy("label").show()

# Visualiser les communautés
lpa.groupBy("label") \
   .agg(
       F.count("*").alias("taille"),
       F.collect_list("nom").alias("membres")
   ) \
   .orderBy(F.col("taille").desc()) \
   .show(truncate=False)
```

---

## 10. Plus courts chemins (*Shortest Paths*)

```python
# Plus court chemin depuis un ensemble de sommets "landmarks"
# Retourne, pour chaque sommet, sa distance à chaque landmark

landmarks = ["alice", "francois"]
sp = g.shortestPaths(landmarks=landmarks)
sp.select("id", "distances").show(truncate=False)

# Résultat :
# +--------+---------------------------+
# |      id|                  distances|
# +--------+---------------------------+
# |   alice|       {alice -> 0, ...}   |
# |     bob|       {alice -> 1, ...}   |
# |  claire|       {alice -> 1, ...}   |
# ...
```

---

## 11. Programme complet illustratif

```python
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from graphframes import GraphFrame

spark = SparkSession.builder \
    .appName("GraphFramesDemo") \
    .master("local[*]") \
    .config("spark.jars.packages",
            "graphframes:graphframes:0.8.3-spark3.5-s_2.12") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/gf-checkpoints")
spark.sparkContext.setLogLevel("WARN")

# ─── 1. Construction du graphe ───────────────────────────────────────────────
print("=" * 60)
print("=== Graphe : réseau de citations académiques ===")
print("=" * 60)

# Sommets : articles scientifiques
vertices = spark.createDataFrame([
    ("p1",  "Deep Learning Survey",           2021, 850),
    ("p2",  "Transformer Architecture",        2017, 2100),
    ("p3",  "BERT: Pre-training",             2018, 3200),
    ("p4",  "GPT-3: Language Models",          2020, 1800),
    ("p5",  "Attention Is All You Need",       2017, 4500),
    ("p6",  "ResNet: Deep Residual Learning",  2016, 3800),
    ("p7",  "GAN: Generative Networks",        2014, 2900),
    ("p8",  "Word2Vec Representations",        2013, 3100),
    ("p9",  "Adam Optimizer",                  2015, 2400),
    ("p10", "Dropout Regularization",          2014, 2200),
], ["id", "titre", "annee", "nb_citations"])

# Arêtes : citations (src cite dst)
edges = spark.createDataFrame([
    ("p1", "p2",  "cite", 1), ("p1", "p3",  "cite", 1),
    ("p1", "p5",  "cite", 1), ("p1", "p6",  "cite", 1),
    ("p1", "p9",  "cite", 1), ("p2", "p5",  "cite", 1),
    ("p2", "p8",  "cite", 1), ("p3", "p2",  "cite", 1),
    ("p3", "p5",  "cite", 1), ("p3", "p8",  "cite", 1),
    ("p3", "p9",  "cite", 1), ("p4", "p2",  "cite", 1),
    ("p4", "p3",  "cite", 1), ("p4", "p5",  "cite", 1),
    ("p4", "p9",  "cite", 1), ("p5", "p8",  "cite", 1),
    ("p6", "p9",  "cite", 1), ("p6", "p10", "cite", 1),
    ("p7", "p9",  "cite", 1), ("p7", "p10", "cite", 1),
    ("p10","p9",  "cite", 1),
], ["src", "dst", "type", "poids"])

g = GraphFrame(vertices, edges)
print(f"\nArticles  : {g.vertices.count()}")
print(f"Citations : {g.edges.count()}")

# ─── 2. Analyse des degrés ───────────────────────────────────────────────────
print("\n=== Articles les plus cités (degré entrant) ===")
g.inDegrees \
  .join(g.vertices, "id") \
  .select("titre", "annee", "nb_citations", "inDegree") \
  .orderBy(F.col("inDegree").desc()) \
  .show(truncate=False)

print("=== Articles qui citent le plus (degré sortant) ===")
g.outDegrees \
  .join(g.vertices, "id") \
  .select("titre", "annee", "outDegree") \
  .orderBy(F.col("outDegree").desc()) \
  .show(truncate=False)

# ─── 3. BFS ─────────────────────────────────────────────────────────────────
print("\n=== BFS : chemin de p1 vers p8 ===")
bfs_result = g.bfs(
    fromExpr="id = 'p1'",
    toExpr  ="id = 'p8'",
    maxPathLength=4
)
bfs_result.show(truncate=False)

# ─── 4. Composantes connexes ─────────────────────────────────────────────────
print("\n=== Composantes connexes ===")
cc = g.connectedComponents()
cc.join(g.vertices.select("id", "titre"), "id") \
  .select("titre", "component") \
  .orderBy("component") \
  .show(truncate=False)

# ─── 5. Triangles ────────────────────────────────────────────────────────────
print("\n=== Comptage de triangles ===")
tc = g.triangleCount()
tc.join(g.vertices, "id") \
  .select("titre", "count") \
  .orderBy(F.col("count").desc()) \
  .show(truncate=False)

# ─── 6. PageRank ─────────────────────────────────────────────────────────────
print("\n=== PageRank (impact scientifique) ===")
pr = g.pageRank(resetProbability=0.15, tol=0.001)
pr.vertices \
  .join(g.vertices.select("id", "nb_citations"), "id") \
  .select("id", "titre", "nb_citations",
          F.round("pagerank", 4).alias("pagerank")) \
  .orderBy(F.col("pagerank").desc()) \
  .show(truncate=False)

# Corrélation PageRank vs nombre de citations brut ?
combined = pr.vertices \
    .join(g.vertices.select("id", "nb_citations"), "id") \
    .select("pagerank", "nb_citations")
corr = combined.corr("pagerank", "nb_citations")
print(f"\nCorrélation PageRank ↔ Citations brutes : {corr:.4f}")

# ─── 7. Label Propagation ────────────────────────────────────────────────────
print("\n=== Communautés (Label Propagation) ===")
lpa = g.labelPropagation(maxIter=10)
lpa.join(g.vertices.select("id", "titre"), "id") \
   .groupBy("label") \
   .agg(
       F.count("*").alias("taille"),
       F.collect_list("titre").alias("articles")
   ) \
   .orderBy(F.col("taille").desc()) \
   .show(truncate=False)

# ─── 8. Shortest Paths ───────────────────────────────────────────────────────
print("\n=== Plus courts chemins depuis p5 et p7 ===")
sp = g.shortestPaths(landmarks=["p5", "p7"])
sp.join(g.vertices.select("id", "titre"), "id") \
  .select("titre", "distances") \
  .show(truncate=False)

spark.stop()
```

---

## Résumé du module

| Concept | Points clés à retenir |
|---|---|
| **Graphe** | G = (V, E) — sommets + arêtes, orienté ou non, pondéré ou non |
| **GraphFrames** | API Python sur DataFrames — `vertices` (col. `id`) + `edges` (col. `src`, `dst`) |
| **Motif Finding** | Recherche déclarative de structures avec la syntaxe `(a)-[e]->(b)` |
| **BFS** | Plus court chemin en nombre d'arêtes — `g.bfs(fromExpr, toExpr)` |
| **Composantes connexes** | Sous-graphes isolés — `g.connectedComponents()` |
| **Triangle Count** | Mesure de transitivité locale — base du coefficient de clustering |
| **PageRank** | Score d'importance basé sur les citations récursives — `g.pageRank(resetProbability, tol)` |
| **Label Propagation** | Détection de communautés non supervisée — `g.labelPropagation(maxIter)` |
| **Shortest Paths** | Distances depuis des sommets "landmark" — `g.shortestPaths(landmarks)` |

---

## Exercices

### Exercice 1 — Modélisation (20 min)
> Modéliser un réseau de transport ferroviaire de votre région :
> - Sommets : gares (avec attributs : nom, ville, nb_quais)
> - Arêtes : liaisons directes (avec attributs : durée_min, distance_km, compagnie)
>
> 1. Créer le GraphFrame correspondant
> 2. Trouver la gare avec le plus grand nombre de connexions directes
> 3. Lister toutes les gares accessibles depuis Paris en moins de 2 étapes (BFS)

### Exercice 2 — Réseau social (35 min)
> Générer un réseau social aléatoire de 50 utilisateurs et 200 relations :
> 1. Calculer les degrés entrant/sortant et identifier les 5 utilisateurs les plus influents
> 2. Détecter les composantes connexes — y a-t-il des utilisateurs isolés ?
> 3. Calculer le coefficient de clustering moyen du réseau
> 4. Appliquer Label Propagation pour détecter les communautés

### Exercice 3 — PageRank (40 min)
> Sur un graphe de pages web simplifiées (10 sommets, 25 arêtes) :
> 1. Implémenter PageRank manuellement (sans GraphFrames) en suivant l'algorithme itératif de la section 8.6
> 2. Comparer avec le résultat de `g.pageRank()`
> 3. Faire varier le facteur d'amortissement `d` entre 0.5 et 0.95 et observer l'impact sur les scores
> 4. Que se passe-t-il avec des "dangling nodes" (sommets sans arête sortante) ?

### Exercice 4 — Détection de fraude (45 min)
> Modéliser un réseau de transactions bancaires :
> - Sommets : comptes bancaires (type : particulier, entreprise, offshore)
> - Arêtes : virements (montant, date, devise)
>
> 1. Détecter les comptes "offshore" à moins de 2 sauts d'un compte particulier (BFS)
> 2. Trouver des cycles de longueur 3 (virements circulaires suspects) via la recherche de motifs
> 3. Calculer le PageRank et identifier les comptes les plus "centraux" du réseau
> 4. Détecter les composantes fortement connexes — que représentent-elles ici ?

---

## Pour aller plus loin

- 📖 **GraphFrames** : [graphframes.github.io](https://graphframes.github.io/graphframes/docs/_site/user-guide.html)
- 📖 **GraphX** (Scala) : [spark.apache.org/docs/latest/graphx-programming-guide.html](https://spark.apache.org/docs/latest/graphx-programming-guide.html)
- 📄 **PageRank original** : *The PageRank Citation Ranking: Bringing Order to the Web* — Page, Brin et al., Stanford 1998
- 📄 **Label Propagation** : *Near Linear Time Algorithm to Detect Community Structures in Large-Scale Networks* — Raghavan et al., 2007
- 🛠️ **NetworkX** : bibliothèque Python de graphes pour prototyper avant de passer à GraphFrames sur de gros volumes — [networkx.org](https://networkx.org)
- 🛠️ **Neo4j** : base de données graphe avec le langage de requête Cypher, pour visualiser et explorer des graphes — [neo4j.com](https://neo4j.com)
- 🎓 **Stanford Network Analysis Project (SNAP)** : jeux de données graphes réels pour les TPs — [snap.stanford.edu](https://snap.stanford.edu/data/)

---

*Module précédent → **Module 8 : Feature Engineering avec PySpark***  
*Module suivant → **Module 10 : Algorithmes de ranking***
