from pyspark.sql import SparkSession
from pyspark.sql.functions import avg

spark = SparkSession.builder.appName("Group By").getOrCreate()

data = [
    ("CS", 80),
    ("CS", 90),
    ("IT", 70),
    ("IT", 75)
]

df = spark.createDataFrame(data, ["Department", "Marks"])

df.groupBy("Department").agg(avg("Marks").alias("Average")).show()

spark.stop()

