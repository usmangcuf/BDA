from pyspark import SparkContext

sc = SparkContext.getOrCreate()

print("Spark version:", sc.version)
print("PySpark is working!")

sc.stop()