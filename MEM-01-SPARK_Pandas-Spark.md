# Tableau comparatif — Pandas vs PySpark

> Ce document recense les équivalences entre les opérations courantes de **Pandas** et **PySpark**.  
> Il est destiné aux data scientists maîtrisant Pandas et souhaitant transposer leurs réflexes vers PySpark.

---

## Légende

| Symbole | Signification |
|:---:|---|
| ✅ | Fonctionnalité disponible et directe |
| ⚠️ | Disponible mais avec des nuances importantes |
| ❌ | Non disponible nativement (contournement nécessaire) |
| 🐢 | Opération coûteuse — à utiliser avec précaution |

---

## 1. Création et chargement de données

| Opération | Pandas | PySpark |
|---|---|---|
| Depuis une liste de dicts | `pd.DataFrame([{"a":1},{"a":2}])` | `spark.createDataFrame([{"a":1},{"a":2}])` |
| Depuis une liste de tuples | `pd.DataFrame([(1,"a"),(2,"b")], columns=["id","val"])` | `spark.createDataFrame([(1,"a"),(2,"b")], ["id","val"])` |
| Depuis un CSV | `pd.read_csv("f.csv", sep=";", header=0)` | `spark.read.option("sep",";").option("header","true").csv("f.csv")` |
| Depuis un JSON | `pd.read_json("f.json")` | `spark.read.json("f.json")` |
| Depuis un Parquet | `pd.read_parquet("f.parquet")` | `spark.read.parquet("f.parquet")` |
| Depuis un Excel | `pd.read_excel("f.xlsx")` | ❌ Passer par Pandas puis `spark.createDataFrame(pdf)` |
| Depuis une base SQL | `pd.read_sql(query, conn)` | `spark.read.jdbc(url, table, properties=props)` |
| Depuis Pandas | — | `spark.createDataFrame(pdf)` |
| Vers Pandas | — | `df.toPandas()` 🐢 (rapatrie tout en mémoire Driver) |
| Taille en mémoire | `df.memory_usage(deep=True).sum()` | ❌ Non disponible directement |
| Dimensions | `df.shape` → `(n_lignes, n_cols)` | `(df.count(), len(df.columns))` 🐢 |

---

## 2. Inspection et exploration

| Opération | Pandas | PySpark |
|---|---|---|
| Aperçu des N premières lignes | `df.head(5)` / `df[:5]` | `df.show(5)` ou `df.head(5)` (retourne des Row) |
| Aperçu des N dernières lignes | `df.tail(5)` | ❌ `df.orderBy(...).tail(5)` (coûteux) |
| Schéma / types | `df.dtypes` | `df.printSchema()` / `df.dtypes` |
| Liste des colonnes | `df.columns` | `df.columns` |
| Noms + types | `df.dtypes` (Series) | `df.dtypes` (liste de tuples) |
| Statistiques descriptives | `df.describe()` | `df.describe()` / `df.summary()` (+ percentiles) |
| Nombre de lignes | `len(df)` | `df.count()` 🐢 |
| Nombre de colonnes | `len(df.columns)` | `len(df.columns)` |
| Valeurs uniques d'une colonne | `df["col"].unique()` | `df.select("col").distinct().collect()` 🐢 |
| Nombre de valeurs uniques | `df["col"].nunique()` | `df.select("col").distinct().count()` 🐢 |
| Compter les nulls | `df.isnull().sum()` | `df.select([F.count(F.when(F.col(c).isNull(),c)).alias(c) for c in df.columns])` |
| Distribution des valeurs | `df["col"].value_counts()` | `df.groupBy("col").count().orderBy(F.col("count").desc())` |
| Corrélation entre colonnes | `df.corr()` | `df.stat.corr("col1", "col2")` (par paire seulement) |
| Afficher sans troncature | `pd.set_option("display.max_colwidth", None)` | `df.show(truncate=False)` |
| Afficher en HTML (notebook) | `display(df)` | `df.limit(N).toPandas()` 🐢 |

---

## 3. Sélection de colonnes

| Opération | Pandas | PySpark |
|---|---|---|
| Une colonne (Series/Column) | `df["col"]` ou `df.col` | `df["col"]` ou `F.col("col")` ou `df.col` |
| Plusieurs colonnes | `df[["a","b","c"]]` | `df.select("a","b","c")` |
| Toutes les colonnes sauf une | `df.drop("col", axis=1)` | `df.drop("col")` |
| Toutes les colonnes sauf plusieurs | `df.drop(["a","b"], axis=1)` | `df.drop("a","b")` |
| Colonnes par position | `df.iloc[:, 0:3]` | `df.select(df.columns[:3])` |
| Colonnes par condition de nom | `df.filter(regex="^prix")` | `df.select([c for c in df.columns if c.startswith("prix")])` |
| Renommer une colonne | `df.rename(columns={"old":"new"})` | `df.withColumnRenamed("old","new")` |
| Renommer plusieurs colonnes | `df.rename(columns={"a":"x","b":"y"})` | Chaîner les `withColumnRenamed()` ou `toDF(new_names)` |
| Réordonner les colonnes | `df[["b","a","c"]]` | `df.select("b","a","c")` |
| Colonnes avec expression | `df.assign(ttc=df.prix*1.2)` | `df.withColumn("ttc", F.col("prix")*1.2)` |

---

## 4. Sélection de lignes (filtrage)

| Opération | Pandas | PySpark |
|---|---|---|
| Par condition simple | `df[df["age"] > 30]` | `df.filter(F.col("age") > 30)` ou `df.where("age > 30")` |
| Par conditions multiples (ET) | `df[(df["a"]>1) & (df["b"]<5)]` | `df.filter((F.col("a")>1) & (F.col("b")<5))` |
| Par conditions multiples (OU) | `df[(df["a"]>1) \| (df["b"]<5)]` | `df.filter((F.col("a")>1) \| (F.col("b")<5))` |
| Négation | `df[~df["col"].isin([1,2])]` | `df.filter(~F.col("col").isin([1,2]))` |
| Valeur dans une liste | `df[df["ville"].isin(["Paris","Lyon"])]` | `df.filter(F.col("ville").isin(["Paris","Lyon"]))` |
| Valeur nulle | `df[df["col"].isnull()]` | `df.filter(F.col("col").isNull())` |
| Valeur non nulle | `df[df["col"].notna()]` | `df.filter(F.col("col").isNotNull())` |
| Par index (position) | `df.iloc[10:20]` | ❌ Pas d'index — utiliser `limit()` ou `monotonically_increasing_id()` |
| Par label d'index | `df.loc["label"]` | ❌ Pas d'index nommé |
| N premières lignes | `df.head(10)` / `df[:10]` | `df.limit(10)` |
| Échantillon aléatoire | `df.sample(n=100)` | `df.sample(fraction=0.01, seed=42)` |
| Dupliquer les lignes | `df[df["col"].duplicated()]` | `df.groupBy(df.columns).count().filter(F.col("count")>1)` |
| Supprimer les doublons | `df.drop_duplicates()` | `df.distinct()` ou `df.dropDuplicates(["col1","col2"])` |

---

## 5. Masques (équivalent des masques booléens Pandas)

| Opération | Pandas | PySpark |
|---|---|---|
| Créer un masque | `mask = df["age"] > 30` (Series bool) | `cond = F.col("age") > 30` (Column) |
| Appliquer un masque | `df[mask]` | `df.filter(cond)` |
| Inverser un masque | `df[~mask]` | `df.filter(~cond)` |
| Combiner masques (ET) | `df[mask1 & mask2]` | `df.filter(cond1 & cond2)` |
| Combiner masques (OU) | `df[mask1 \| mask2]` | `df.filter(cond1 \| cond2)` |
| Valeur conditionnelle | `np.where(mask, val_vrai, val_faux)` | `F.when(cond, val_vrai).otherwise(val_faux)` |
| CASE WHEN multiple | `np.select([c1,c2,c3],[v1,v2,v3], default)` | `F.when(c1,v1).when(c2,v2).when(c3,v3).otherwise(default)` |
| Remplacer selon masque | `df.loc[mask, "col"] = valeur` | `df.withColumn("col", F.when(cond, valeur).otherwise(F.col("col")))` |
| Masque sur chaîne | `df["col"].str.contains("motif")` | `F.col("col").contains("motif")` ou `.rlike("regex")` |
| Masque sur valeurs NaN | `df["col"].isna()` | `F.col("col").isNull()` |
| Masque sur valeurs infinies | `np.isinf(df["col"])` | `F.col("col") == float("inf")` |
| Comptage du masque | `mask.sum()` | `df.filter(cond).count()` 🐢 |

---

## 6. Modifications et remplacements

| Opération | Pandas | PySpark |
|---|---|---|
| Ajouter une colonne | `df["new"] = valeur` ou `df.assign(new=...)` | `df.withColumn("new", expr)` |
| Modifier une colonne | `df["col"] = df["col"] * 2` | `df.withColumn("col", F.col("col") * 2)` |
| Remplacer des valeurs | `df["col"].replace({1: "un", 2: "deux"})` | `df.withColumn("col", F.when(F.col("col")==1,"un").when(F.col("col")==2,"deux").otherwise(F.col("col")))` |
| Remplacer les NaN | `df["col"].fillna(0)` | `df.fillna(0, subset=["col"])` |
| Remplacer tous les NaN | `df.fillna({"a":0, "b":"??"})` | `df.fillna({"a":0, "b":"??"})` |
| Supprimer les lignes NaN | `df.dropna()` | `df.dropna()` ou `df.na.drop()` |
| Supprimer si N NaN | `df.dropna(thresh=N)` | `df.dropna(thresh=N)` |
| Modifier par condition | `df.loc[df["a"]>1, "b"] = 99` | `df.withColumn("b", F.when(F.col("a")>1, 99).otherwise(F.col("b")))` |
| Conversion de type | `df["col"].astype(int)` | `df.withColumn("col", F.col("col").cast("integer"))` |
| Arrondir | `df["col"].round(2)` | `F.round(F.col("col"), 2)` |
| Valeur absolue | `df["col"].abs()` | `F.abs(F.col("col"))` |
| Clip (écrêtage) | `df["col"].clip(lower=0, upper=100)` | `F.greatest(F.lit(0), F.least(F.lit(100), F.col("col")))` |
| Appliquer une fonction | `df["col"].apply(func)` | `df.withColumn("col", udf_func(F.col("col")))` ⚠️ UDF coûteux |
| Appliquer ligne par ligne | `df.apply(func, axis=1)` | ❌ Pas d'équivalent direct — utiliser plusieurs `withColumn()` |

---

## 7. Broadcasting

| Concept | Pandas | PySpark |
|---|---|---|
| **Principe** | Broadcast automatique lors d'opérations entre Series et scalaires ou entre DataFrames de tailles compatibles | Deux types distincts : broadcast de variables (Executors) et broadcast join (tables) |
| Opération scalaire | `df["col"] * 2` (broadcast automatique) | `F.col("col") * 2` (broadcast automatique de la constante) |
| Constante littérale | `df["col"] + 10` | `F.col("col") + F.lit(10)` — `F.lit()` crée une colonne constante |
| Opération entre deux colonnes | `df["a"] + df["b"]` | `F.col("a") + F.col("b")` |
| Broadcast d'un scalaire calculé | `mean = df["col"].mean()` / `df["col"] - mean` | Calculer la moyenne puis utiliser `F.lit(mean_val)` ou une Pandas UDF |
| Broadcast d'une liste | `df["col"].isin([1,2,3])` | `F.col("col").isin([1,2,3])` — broadcast implicite de la liste |
| **Broadcast d'une variable Python** | Non nécessaire (même processus) | `bc = sc.broadcast(dict_or_list)` puis `bc.value` dans une UDF |
| **Broadcast join (table)** | ❌ Non applicable (mono-machine) | `df_large.join(F.broadcast(df_petite), "id")` — évite le shuffle |
| Broadcast d'un modèle ML | Non applicable | `model_bc = sc.broadcast(modele)` → dans Pandas UDF : `model_bc.value.predict(X)` |
| Libérer un broadcast | Non applicable | `bc.unpersist()` / `bc.destroy()` |
| Seuil broadcast auto | Non applicable | `spark.conf.set("spark.sql.autoBroadcastJoinThreshold","10mb")` |

---

## 8. Opérations sur les chaînes de caractères

| Opération | Pandas (`str.`) | PySpark (`F.`) |
|---|---|---|
| Minuscules | `df["col"].str.lower()` | `F.lower(F.col("col"))` |
| Majuscules | `df["col"].str.upper()` | `F.upper(F.col("col"))` |
| Initiale majuscule | `df["col"].str.title()` | `F.initcap(F.col("col"))` |
| Longueur | `df["col"].str.len()` | `F.length(F.col("col"))` |
| Supprimer espaces | `df["col"].str.strip()` | `F.trim(F.col("col"))` |
| Contient | `df["col"].str.contains("motif")` | `F.col("col").contains("motif")` |
| Commence par | `df["col"].str.startswith("pref")` | `F.col("col").startswith("pref")` |
| Finit par | `df["col"].str.endswith("suf")` | `F.col("col").endswith("suf")` |
| Remplacer | `df["col"].str.replace("old","new")` | `F.regexp_replace(F.col("col"),"old","new")` |
| Extraire regex | `df["col"].str.extract(r"(\d+)")` | `F.regexp_extract(F.col("col"), r"(\d+)", 1)` |
| Découper | `df["col"].str.split(" ")` | `F.split(F.col("col"), " ")` → ArrayType |
| Concaténer | `df["a"].str.cat(df["b"], sep="-")` | `F.concat(F.col("a"), F.lit("-"), F.col("b"))` |
| Sous-chaîne | `df["col"].str[2:5]` | `F.substring(F.col("col"), 3, 3)` ⚠️ index 1-based |
| Compter occurrences | `df["col"].str.count("motif")` | ❌ via `F.size(F.split(...))` - 1 |
| Format string | `"Hello " + df["col"]` | `F.concat(F.lit("Hello "), F.col("col"))` |
| Regex match | `df["col"].str.match(r"^\d+$")` | `F.col("col").rlike(r"^\d+$")` |
| Padding | `df["col"].str.pad(10, side="left")` | `F.lpad(F.col("col"), 10, " ")` |

---

## 9. Opérations sur les dates

| Opération | Pandas | PySpark |
|---|---|---|
| Convertir en date | `pd.to_datetime(df["col"])` | `F.to_date(F.col("col"), "yyyy-MM-dd")` |
| Date du jour | `pd.Timestamp.today()` | `F.current_date()` |
| Timestamp actuel | `pd.Timestamp.now()` | `F.current_timestamp()` |
| Extraire l'année | `df["col"].dt.year` | `F.year(F.col("col"))` |
| Extraire le mois | `df["col"].dt.month` | `F.month(F.col("col"))` |
| Extraire le jour | `df["col"].dt.day` | `F.dayofmonth(F.col("col"))` |
| Jour de la semaine | `df["col"].dt.dayofweek` (0=Lundi) | `F.dayofweek(F.col("col"))` (1=Dimanche) ⚠️ |
| Différence en jours | `(df["a"] - df["b"]).dt.days` | `F.datediff(F.col("a"), F.col("b"))` |
| Ajouter N jours | `df["col"] + pd.Timedelta(days=7)` | `F.date_add(F.col("col"), 7)` |
| Formater en chaîne | `df["col"].dt.strftime("%d/%m/%Y")` | `F.date_format(F.col("col"), "dd/MM/yyyy")` |
| Début du mois | `df["col"].dt.to_period("M").dt.start_time` | `F.trunc(F.col("col"), "month")` |
| Début de l'année | `df["col"].dt.to_period("Y").dt.start_time` | `F.trunc(F.col("col"), "year")` |
| Timestamp → Date | `df["col"].dt.date` | `F.to_date(F.col("col"))` |

---

## 10. Agrégations

| Opération | Pandas | PySpark |
|---|---|---|
| Somme | `df["col"].sum()` | `df.agg(F.sum("col"))` |
| Moyenne | `df["col"].mean()` | `df.agg(F.avg("col"))` |
| Min / Max | `df["col"].min()` / `df["col"].max()` | `df.agg(F.min("col"))` / `df.agg(F.max("col"))` |
| Écart-type | `df["col"].std()` | `df.agg(F.stddev("col"))` |
| Médiane | `df["col"].median()` | `df.agg(F.percentile_approx("col", 0.5))` ⚠️ approchée |
| Percentile | `df["col"].quantile(0.75)` | `df.agg(F.percentile_approx("col", 0.75))` |
| Compter | `df["col"].count()` / `len(df)` | `df.count()` / `df.agg(F.count("col"))` |
| Compter distincts | `df["col"].nunique()` | `df.agg(F.countDistinct("col"))` |
| Grouper + agréger | `df.groupby("col").agg({"a":"sum","b":"mean"})` | `df.groupBy("col").agg(F.sum("a"), F.avg("b"))` |
| Grouper + fonction | `df.groupby("col").agg(lambda x: x.max()-x.min())` | Via `pandas_udf` de type `GROUPED_MAP` |
| Pivot | `df.pivot_table(index="a", columns="b", values="c", aggfunc="sum")` | `df.groupBy("a").pivot("b").sum("c")` |
| Cumsum | `df["col"].cumsum()` | `F.sum("col").over(Window.orderBy(...).rowsBetween(Window.unboundedPreceding, 0))` |
| Rolling | `df["col"].rolling(3).mean()` | `F.avg("col").over(Window.orderBy(...).rowsBetween(-2, 0))` |
| Rang | `df["col"].rank()` | `F.rank().over(Window.orderBy("col"))` |
| Shift / Lag | `df["col"].shift(1)` | `F.lag("col", 1).over(Window.orderBy(...))` |
| Diff | `df["col"].diff(1)` | `F.col("col") - F.lag("col",1).over(Window.orderBy(...))` |
| Transformer sans réduire | `df.groupby("g")["col"].transform("mean")` | `F.avg("col").over(Window.partitionBy("g"))` |

---

## 11. Jointures

| Opération | Pandas | PySpark |
|---|---|---|
| Inner join | `df1.merge(df2, on="id", how="inner")` | `df1.join(df2, "id", "inner")` |
| Left join | `df1.merge(df2, on="id", how="left")` | `df1.join(df2, "id", "left")` |
| Right join | `df1.merge(df2, on="id", how="right")` | `df1.join(df2, "id", "right")` |
| Full outer join | `df1.merge(df2, on="id", how="outer")` | `df1.join(df2, "id", "outer")` |
| Semi-join | `df1[df1["id"].isin(df2["id"])]` | `df1.join(df2, "id", "left_semi")` |
| Anti-join | `df1[~df1["id"].isin(df2["id"])]` | `df1.join(df2, "id", "left_anti")` |
| Cross join | `df1.merge(df2, how="cross")` | `df1.crossJoin(df2)` |
| Clés différentes | `df1.merge(df2, left_on="a", right_on="b")` | `df1.join(df2, df1["a"]==df2["b"])` |
| Plusieurs clés | `df1.merge(df2, on=["a","b"])` | `df1.join(df2, ["a","b"])` |
| Broadcast join | ❌ Non applicable | `df1.join(F.broadcast(df2), "id")` |
| Éviter colonnes dupliquées | Suffixes : `merge(..., suffixes=("_l","_r"))` | `df1.alias("l").join(df2.alias("r"), ...)` puis `F.col("l.col")` |

---

## 12. Tri

| Opération | Pandas | PySpark |
|---|---|---|
| Tri ascendant | `df.sort_values("col")` | `df.orderBy("col")` ou `df.sort("col")` |
| Tri descendant | `df.sort_values("col", ascending=False)` | `df.orderBy(F.col("col").desc())` |
| Tri multi-colonnes | `df.sort_values(["a","b"], ascending=[True,False])` | `df.orderBy(F.col("a"), F.col("b").desc())` |
| NaN en premier | `df.sort_values("col", na_position="first")` | `df.orderBy(F.col("col").asc_nulls_first())` |
| NaN en dernier | `df.sort_values("col", na_position="last")` | `df.orderBy(F.col("col").asc_nulls_last())` |
| Rang dans un groupe | `df.groupby("g")["col"].rank()` | `F.rank().over(Window.partitionBy("g").orderBy("col"))` |

---

## 13. Restructuration des données

| Opération | Pandas | PySpark |
|---|---|---|
| Empiler verticalement | `pd.concat([df1,df2], axis=0)` | `df1.union(df2)` / `df1.unionByName(df2)` |
| Empiler horizontalement | `pd.concat([df1,df2], axis=1)` | ❌ Utiliser un join sur index artificiel |
| Exploser une liste | `df.explode("col")` | `df.withColumn("col", F.explode(F.col("col")))` |
| Melt (wide → long) | `df.melt(id_vars=["id"], value_vars=["a","b"])` | `df.unpivot(["id"], ["a","b"], "variable", "value")` (Spark 3.4+) |
| Pivot (long → wide) | `df.pivot_table(index="a", columns="b", values="c")` | `df.groupBy("a").pivot("b").agg(F.first("c"))` |
| Stack (MultiIndex) | `df.stack()` | ❌ Non applicable (pas de MultiIndex) |

---

## 14. Valeurs manquantes

| Opération | Pandas | PySpark |
|---|---|---|
| Détecter les nulls | `df.isnull()` / `df.isna()` | `F.col("col").isNull()` |
| Compter les nulls | `df.isnull().sum()` | `df.select([F.count(F.when(F.col(c).isNull(),c)).alias(c) for c in df.columns])` |
| Remplir les nulls (valeur fixe) | `df.fillna(0)` | `df.fillna(0)` |
| Remplir les nulls (dict) | `df.fillna({"a":0, "b":"N/A"})` | `df.fillna({"a":0, "b":"N/A"})` |
| Remplir par la moyenne | `df["col"].fillna(df["col"].mean())` | `mean = df.agg(F.avg("col")).first()[0]` / `df.fillna(mean, subset=["col"])` |
| Forward fill | `df["col"].fillna(method="ffill")` | `F.last("col", ignorenulls=True).over(Window.orderBy(...).rowsBetween(Window.unboundedPreceding,0))` |
| Back fill | `df["col"].fillna(method="bfill")` | `F.first("col", ignorenulls=True).over(Window.orderBy(...).rowsBetween(0, Window.unboundedFollowing))` |
| Premier non-null | `df[["a","b"]].bfill(axis=1)` | `F.coalesce(F.col("a"), F.col("b"))` |
| Supprimer lignes NaN | `df.dropna()` | `df.dropna()` |
| Supprimer colonnes NaN | `df.dropna(axis=1)` | ❌ Calculer manuellement puis `df.drop(cols_nulles)` |
| Remplacer NaN par interpolation | `df["col"].interpolate()` | ❌ Non natif — via Pandas UDF ou Window functions |

---

## 15. Performance et mémoire

| Aspect | Pandas | PySpark |
|---|---|---|
| **Modèle d'exécution** | Eager (immédiat) | Lazy (différé jusqu'à une action) |
| **Parallélisme** | Mono-thread (sauf via `multiprocessing`) | Multi-nœuds, multi-cœurs natif |
| **Scalabilité** | Limité à la RAM d'une machine | Pétaoctets (cluster) |
| **Copie des données** | Par valeur (modifier une copie ne modifie pas l'original) | Immuable — chaque transformation crée un nouveau DF |
| **Cache en mémoire** | Automatique (tout est en RAM) | Explicite : `df.cache()` / `df.persist()` |
| **Optimiseur** | Aucun | Catalyst Optimizer (réécriture automatique du plan) |
| **Conversion Pandas → Spark** | — | `spark.createDataFrame(pdf)` (coûteux pour gros volumes) |
| **Conversion Spark → Pandas** | — | `df.toPandas()` 🐢 rapatrie tout au Driver |
| **API Pandas sur Spark** | — | `import pyspark.pandas as ps` (API quasi-identique) |
| **Inférence de schéma** | Automatique | `inferSchema=True` (coûteux) → préférer schéma explicite |
| **Index** | ✅ Natif (Int64Index, DatetimeIndex...) | ❌ Pas d'index — ordre non garanti |
| **Ordre des lignes** | Garanti (par défaut) | ⚠️ Non garanti sans `orderBy()` |

---

## 16. Opérations sur les structures imbriquées (spécifique PySpark)

Ces opérations n'ont pas d'équivalent direct dans Pandas standard.

| Opération | PySpark | Remarque |
|---|---|---|
| Créer un Array | `F.array(F.col("a"), F.col("b"))` | Type `ArrayType` |
| Taille d'un Array | `F.size(F.col("arr"))` | Équivalent à `len()` |
| Accéder à un élément | `F.col("arr")[0]` | Index 0-based |
| Contient une valeur | `F.array_contains(F.col("arr"), "val")` | Booléen |
| Exploser un Array | `F.explode(F.col("arr"))` | 1 ligne par élément |
| Exploser (garder nulls) | `F.explode_outer(F.col("arr"))` | Conserve les lignes vides |
| Flatten (Array imbriqués) | `F.flatten(F.col("arr_of_arr"))` | Aplatit 1 niveau |
| Dédupliquer un Array | `F.array_distinct(F.col("arr"))` | — |
| Trier un Array | `F.array_sort(F.col("arr"))` | — |
| Zipper deux Arrays | `F.arrays_zip(F.col("a"), F.col("b"))` | → Array de structs |
| Créer un Map | `F.create_map(F.lit("k"), F.col("v"))` | Type `MapType` |
| Accéder à une clé | `F.col("map")["cle"]` | — |
| Clés d'un Map | `F.map_keys(F.col("map"))` | → Array |
| Valeurs d'un Map | `F.map_values(F.col("map"))` | → Array |
| Exploser un Map | `F.explode(F.col("map"))` | → colonnes `key`, `value` |
| Accès à un champ struct | `F.col("struct_col.champ")` | Notation pointée |
| Convertir JSON → struct | `F.from_json(F.col("col"), schema)` | — |
| Convertir struct → JSON | `F.to_json(F.col("col"))` | → StringType |

---

## Récapitulatif des pièges courants

| Piège | Description | Solution |
|---|---|---|
| **Ordre non garanti** | `df.show()` peut afficher dans n'importe quel ordre | Ajouter `orderBy()` avant `show()` |
| **Pas d'index** | Impossible de faire `df.iloc[5]` ou `df.loc["label"]` | Utiliser `filter()` ou créer une colonne `id` avec `monotonically_increasing_id()` |
| **`collect()` dangereux** | Rapatrie TOUT au Driver → `MemoryError` | Toujours `limit()` avant `collect()` ou `toPandas()` |
| **Modification sur place** | `df["col"] = ...` ne fonctionne pas | Toujours `df = df.withColumn(...)` (immuabilité) |
| **UDF coûteuses** | Les UDF Python brisent l'optimisation Catalyst | Préférer les fonctions natives `F.*`, puis Pandas UDF |
| **`inferSchema` lent** | Spark lit les données deux fois | Toujours fournir un schéma explicite en production |
| **200 shuffle partitions** | Valeur par défaut trop élevée en local | `spark.conf.set("spark.sql.shuffle.partitions","8")` |
| **`toPandas()` sur gros volume** | Sature la RAM du Driver | `df.limit(1000).toPandas()` ou écrire en fichier |
| **Index 1-based en Spark** | `substring(col, 1, 3)` commence à 1 (pas 0) | Contrairement à Python, l'index est 1-based dans Spark SQL |
| **NaN vs null** | Pandas distingue `NaN` (float) et `None`. Spark utilise `null` pour tout | `F.isnan()` pour NaN, `F.isnull()` pour null |
