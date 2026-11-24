from pyspark.sql import SparkSession

# Create Spark Session
spark = SparkSession.builder \
    .appName("PySpark Example") \
    .getOrCreate()

# Sample data
data = [
    (1, "Ali", 21),
    (2, "Ahmed", 23),
    (3, "Sara", 20)
]

# Create DataFrame
df = spark.createDataFrame(data, ["ID", "Name", "Age"])

# Show DataFrame
df.show()

spark.stop()
