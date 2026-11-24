try:
    from pyspark.sql import SparkSession  # type: ignore[import]
except ImportError:
    SparkSession = None

spark = None
if SparkSession is not None:
    spark = SparkSession.builder.appName("Sorting").getOrCreate()

data = [
    ("Ali", 50),
    ("Sara", 90),
    ("Ahmed", 70)
]

if spark is not None:
    df = spark.createDataFrame(data, ["Name", "Marks"])
    df.orderBy("Marks", ascending=False).show()
    # spark.stop()
else:
    print("pyspark is not available; cannot create SparkSession.")
