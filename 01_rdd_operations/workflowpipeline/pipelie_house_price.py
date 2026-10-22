
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer, StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# =========================================================
# STEP 1: CREATE SPARK SESSION
# =========================================================
spark = (
    SparkSession.builder
    .appName("HousePriceETLMLPipeline")
    .getOrCreate()
)

# =========================================================
# STEP 2: DATA EXTRACTION AND LOADING
# =========================================================
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("house_prices.csv")
)

print("Extracted house-price data:")
df.show(5)
df.printSchema()

# Remove duplicate records.
df = df.dropDuplicates()

# Remove unnecessary identifier.
if "property_id" in df.columns:
    df = df.drop("property_id")

# Data validation: prices and areas must be positive.
# Missing predictor values are allowed for later imputation.
df = df.filter(
    (F.col("price") > 0)
    & (F.col("area").isNull() | (F.col("area") > 0))
    & (
        F.col("bedrooms").isNull()
        | (F.col("bedrooms") >= 0)
    )
    & (
        F.col("bathrooms").isNull()
        | (F.col("bathrooms") >= 0)
    )
    & (F.col("age").isNull() | (F.col("age") >= 0))
)

numeric_cols = ["area", "bedrooms", "bathrooms", "age"]
categorical_cols = ["neighborhood"]

# Split data before estimating preprocessing parameters.
train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)

# =========================================================
# STEP 3: OUTLIER DETECTION AND TREATMENT
# Learn percentile bounds from the training set only.
# =========================================================
def learn_outlier_bounds(data, columns):
    bounds = {}

    for col_name in columns:
        q = data.approxQuantile(
            col_name, [0.01, 0.99], 0.01
        )

        if len(q) == 2:
            bounds[col_name] = (q[0], q[1])

    return bounds


def apply_outlier_bounds(data, bounds):
    for col_name, (lower, upper) in bounds.items():
        data = data.withColumn(
            col_name,
            F.when(
                F.col(col_name) < lower, F.lit(lower)
            ).when(
                F.col(col_name) > upper, F.lit(upper)
            ).otherwise(F.col(col_name))
        )

    return data


bounds = learn_outlier_bounds(train_raw, numeric_cols)

train = apply_outlier_bounds(train_raw, bounds)
test = apply_outlier_bounds(test_raw, bounds)

# =========================================================
# STEP 4: HANDLE MISSING CATEGORICAL VALUES
# =========================================================
for col_name in categorical_cols:
    train = train.fillna({col_name: "Unknown"})
    test = test.fillna({col_name: "Unknown"})

# =========================================================
# STEP 5: CORRELATION ANALYSIS
# Remove one predictor from each highly correlated pair.
# =========================================================
correlation_matrix = train.stat.corr
drop_cols = set()

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col_a = numeric_cols[i]
        col_b = numeric_cols[j]

        corr = correlation_matrix(col_a, col_b)

        if corr is not None and abs(corr) > 0.90:
            drop_cols.add(col_b)

print("Highly correlated features to remove:", drop_cols)

numeric_cols = [
    c for c in numeric_cols if c not in drop_cols
]

# =========================================================
# STEP 6: FEATURE PREPROCESSING
# Median imputation, categorical encoding, vector assembly.
# =========================================================
imputed_cols = [c + "_imputed" for c in numeric_cols]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=imputed_cols,
    strategy="median"
)

category_indexers = [
    StringIndexer(
        inputCol=c,
        outputCol=c + "_index",
        handleInvalid="keep"
    )
    for c in categorical_cols
]

encoder = OneHotEncoder(
    inputCols=[c + "_index" for c in categorical_cols],
    outputCols=[c + "_encoded" for c in categorical_cols],
    handleInvalid="keep"
)

assembler = VectorAssembler(
    inputCols=imputed_cols + [
        c + "_encoded" for c in categorical_cols
    ],
    outputCol="features",
    handleInvalid="keep"
)

# =========================================================
# STEP 7: RANDOM FOREST REGRESSION MODEL
# =========================================================
regressor = RandomForestRegressor(
    featuresCol="features",
    labelCol="price",
    predictionCol="predicted_price",
    numTrees=100,
    maxDepth=10,
    seed=42
)

pipeline = Pipeline(stages=[
    imputer,
    *category_indexers,
    encoder,
    assembler,
    regressor
])

# =========================================================
# STEP 8: MODEL TRAINING
# =========================================================
model = pipeline.fit(train)

print("Random Forest Regression model trained successfully.")

# =========================================================
# STEP 9: PREDICTIONS
# =========================================================
predictions = model.transform(test)

predictions.select(
    "price", "predicted_price"
).show(20, truncate=False)

# =========================================================
# STEP 10: EVALUATION METRICS
# =========================================================
def evaluate_regression(data, metric_name):
    evaluator = RegressionEvaluator(
        labelCol="price",
        predictionCol="predicted_price",
        metricName=metric_name
    )
    return evaluator.evaluate(data)


rmse = evaluate_regression(predictions, "rmse")
mae = evaluate_regression(predictions, "mae")
r2 = evaluate_regression(predictions, "r2")

print("RMSE:", rmse)
print("MAE :", mae)
print("R2  :", r2)

# =========================================================
# STEP 11: SAVE PREDICTIONS
# =========================================================
(
    predictions.select("price", "predicted_price")
    .write.mode("overwrite")
    .option("header", True)
    .csv("output/house_price_predictions")
)

print("Predictions saved to output/house_price_predictions")

spark.stop()
