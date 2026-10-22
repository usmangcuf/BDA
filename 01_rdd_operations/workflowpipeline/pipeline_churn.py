
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer, StringIndexer, OneHotEncoder, VectorAssembler
)
from pyspark.ml.functions import vector_to_array
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import (
    MulticlassClassificationEvaluator,
    BinaryClassificationEvaluator
)

# =========================================================
# STEP 1: Create Spark Session
# =========================================================
spark = (
    SparkSession.builder
    .appName("CustomerChurnETLMLPipeline")
    .getOrCreate()
)

# =========================================================
# STEP 2: DATA EXTRACTION AND LOADING
# =========================================================
df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv("synthetic_customer_churn_100k.csv")
)

print("Extracted data:")
df.show(5)
df.printSchema()

# Remove duplicate records and unnecessary identifier.
df = df.dropDuplicates()

if "CustomerId" in df.columns:
    df = df.drop("CustomerId")

# Validate the target and required fields.
df = df.filter(
    F.col("churn").isin("Yes", "No")
    & (F.col("Age").isNull() | (F.col("Age") > 0))
    & (
        F.col("MonthlyCharges").isNull()
        | (F.col("MonthlyCharges") >= 0)
    )
)

# Split BEFORE learning preprocessing parameters.
train_raw, test_raw = df.randomSplit([0.8, 0.2], seed=42)

numeric_cols = ["Age", "Tenure", "MonthlyCharges"]
categorical_cols = ["Contract"]

# =========================================================
# STEP 3: NOISE AND OUTLIER HANDLING
# Learn 1st and 99th percentile bounds from training data.
# Then apply the same bounds to both datasets.
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

# Categorical missing values are handled consistently.
for col_name in categorical_cols:
    train = train.fillna({col_name: "Unknown"})
    test = test.fillna({col_name: "Unknown"})

# =========================================================
# STEP 4: CORRELATION ANALYSIS
# Remove one of a pair of highly correlated numeric features.
# Threshold: absolute Pearson correlation > 0.90.
# =========================================================
correlation_matrix = train.stat.corr
drop_cols = set()

for i in range(len(numeric_cols)):
    for j in range(i + 1, len(numeric_cols)):
        col_a = numeric_cols[i]
        col_b = numeric_cols[j]

        corr = correlation_matrix(col_a, col_b)

        if corr is not None and abs(corr) > 0.90:
            # Keep the earlier feature; remove the later one.
            drop_cols.add(col_b)

print("Highly correlated features to remove:", drop_cols)

numeric_cols = [
    c for c in numeric_cols if c not in drop_cols
]

# =========================================================
# STEP 5: FEATURE PREPROCESSING AND ML PIPELINE
# Impute missing numeric values using training medians.
# Encode categories and assemble the feature vector.
# =========================================================
imputed_cols = [c + "_imputed" for c in numeric_cols]

imputer = Imputer(
    inputCols=numeric_cols,
    outputCols=imputed_cols,
    strategy="median"
)

label_indexer = StringIndexer(
    inputCol="Churn",
    outputCol="label",
    handleInvalid="skip"
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

classifier = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=50,
    regParam=0.01
)

pipeline = Pipeline(stages=[
    imputer,
    label_indexer,
    *category_indexers,
    encoder,
    assembler,
    classifier
])

# =========================================================
# STEP 6: MODEL TRAINING
# =========================================================
model = pipeline.fit(train)

print("Logistic Regression model trained successfully.")

# =========================================================
# STEP 7: PREDICTIONS
# =========================================================
predictions = model.transform(test)

predictions.select(
    "churn", "label", "prediction", "probability"
).show(20, truncate=False)

# =========================================================
# STEP 8: EVALUATION METRICS
# =========================================================
accuracy_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="accuracy"
)

precision_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedPrecision"
)

recall_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="weightedRecall"
)

f1_eval = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction",
    metricName="f1"
)

auc_eval = BinaryClassificationEvaluator(
    labelCol="label",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

print("Accuracy :", accuracy_eval.evaluate(predictions))
print("Precision:", precision_eval.evaluate(predictions))
print("Recall   :", recall_eval.evaluate(predictions))
print("F1 Score :", f1_eval.evaluate(predictions))
print("ROC-AUC  :", auc_eval.evaluate(predictions))

# =========================================================
# STEP 9: SAVE PREDICTIONS
# =========================================================
output_df = predictions.select(
    "churn",
    "prediction",
    F.col("probability").alias("probability_vector")
).withColumn(
    "probability",
    F.to_json(vector_to_array("probability_vector"))
).drop("probability_vector")

# Save predictions as CSV.
output_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("output/churn_predictions")

print("Predictions saved successfully!")



spark.stop()
