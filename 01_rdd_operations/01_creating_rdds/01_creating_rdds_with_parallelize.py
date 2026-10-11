from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("RDDExample")
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("ERROR")

numbers = [10, 20, 30, 40, 50]

rdd = sc.parallelize(numbers)

print("Original RDD:", rdd.collect())
print("Doubled:", rdd.map(lambda x: x * 2).collect())
print("Sum:", rdd.sum())

sc.stop()