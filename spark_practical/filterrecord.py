from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("Filter Data").getOrCreate()

data = [
    (1, "Ali", 21),
    (2, "Ahmed", 18),
    (3, "Sara", 25)
]

df = spark.createDataFrame(data, ["ID", "Name", "Age"])

filtered_df = df.filter(df.Age > 20)

filtered_df.show()

spark.stop()
