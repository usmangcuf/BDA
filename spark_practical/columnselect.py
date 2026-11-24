from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Select Columns").getOrCreate()

data = [
    (1, "Ali", 21),
    (2, "Ahmed", 23)
]

df = spark.createDataFrame(data, ["ID", "Name", "Age"])

df.select("Name", "Age").show()

spark.stop()

