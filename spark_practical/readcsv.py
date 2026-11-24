from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Read CSV").getOrCreate()

df = spark.read.csv("weather.csv", header=True, inferSchema=True)

df.show()
df.printSchema()

spark.stop()
