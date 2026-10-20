import sys
import boto3
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from awsglue.dynamicframe import DynamicFrame
from datetime import datetime

args = getResolvedOptions(sys.argv, ['JOB_NAME'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Initialize DynamoDB client
dynamodb = boto3.client('dynamodb')
dynamo_table = 'cdc_tracking'

# Common S3 base path
s3_base_path = "s3://ecom-tg-project-layers/bronze/"

# Common DB connection config
db_config = {
    "url": "jdbc:mysql://ecom-db.ct2888064y9g.us-east-1.rds.amazonaws.com:3306/inventory",
    "user": "admin",
    "password": "admin_123",
    "className": "com.mysql.cj.jdbc.Driver"
}

# Tables and hash keys
tables = {
    "orders": "order_id",
    "customers": "customer_id",
    "products": "product_id"
}

# Function to fetch lastmodified from DynamoDB
def get_lastmodified_value(record_id):
    try:
        response = dynamodb.get_item(
            TableName=dynamo_table,
            Key={'record_id': {'S': record_id}}
        )
        if 'Item' in response:
            timestamp = int(response['Item']['lastmodified']['N'])
            return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"Error fetching data from DynamoDB: {e}")
        return "1970-01-01 00:00:00"

# Function to update lastmodified in DynamoDB
def update_lastmodified_value(record_id, lastmodified_str):
    try:
        timestamp = int(datetime.strptime(lastmodified_str, "%Y-%m-%d %H:%M:%S").timestamp())
        dynamodb.put_item(
            TableName=dynamo_table,
            Item={
                'record_id': {'S': record_id},
                'lastmodified': {'N': str(timestamp)}
            }
        )
        print(f"Updated lastmodified for {record_id}: {lastmodified_str}")
    except Exception as e:
        print(f"Error updating DynamoDB: {e}")

# Loop through each table
for table, hash_key in tables.items():
    print(f"\nProcessing table: {table}")
    lastmodified = get_lastmodified_value(table)
    print(f"Last modified timestamp for {table}: {lastmodified}")

    sample_query = f"SELECT * FROM {table} WHERE lastmodified > '{lastmodified}' AND"

    try:
        df_dynamic = glueContext.create_dynamic_frame.from_options(
            connection_type="mysql",
            connection_options={
                **db_config,
                "dbtable": table,
                "sampleQuery": sample_query,
                "hashfield": hash_key,
                "enablePartitioningForSampleQuery": True,
                "sampleSize": "20"
            },
            transformation_ctx=f"{table}_node"
        )

        if df_dynamic.count() > 0:
            df = df_dynamic.toDF().repartition(1)
            df_dynamic_single = DynamicFrame.fromDF(df, glueContext, f"{table}_single")

            glueContext.write_dynamic_frame.from_options(
                frame=df_dynamic_single,
                connection_type="s3",
                format="csv",
                connection_options={
                    "path": f"{s3_base_path}{table}_delta/",
                    "partitionKeys": []
                },
                transformation_ctx=f"s3_write_{table}"
            )

            max_lastmodified = df.agg({"lastmodified": "max"}).collect()[0][0]
            update_lastmodified_value(table, str(max_lastmodified))
        else:
            print(f"No new changes for table: {table}")

    except Exception as e:
        print(f"Error processing table {table}: {e}")

job.commit()
