from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.appName("Add Column").getOrCreate()

data = [
    ("Ali", 50),
    ("Sara", 70)
]

df = spark.createDataFrame(data, ["Name", "Marks"])

# Add 10 bonus marks
df = df.withColumn("FinalMarks", col("Marks") + 10)

df.show()
input("Press Enter to stop Spark...")

spark.stop()
