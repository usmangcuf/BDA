from pyspark import SparkContext

sc = SparkContext("local", "TextFileExample")

# Read a text file using TextInputFormat
rdd = sc.hadoopFile("/Users/mac/Downloads/BDA/01_rdd_operations/01_creating_rdds/data/movies.csv",
                    "org.apache.hadoop.mapred.TextInputFormat",
                    "org.apache.hadoop.io.LongWritable",
                    "org.apache.hadoop.io.Text")

# Convert RDD to display as tuples
result = rdd.map(lambda x: (x[0],x[1]))

print(result.collect())
