from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("SparkConnectionTest")
    .master("local[*]").getOrCreate()
)

print("Spark Version:", spark.version)

data = [
    (1, "Usman"),
    (2, "Ali"),
    (3, "Ahmed")
]

df = spark.createDataFrame(data, ["ID", "Name"])

df.show()

spark.stop()