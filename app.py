import sys
import boto3
from pyspark.sql import SparkSession

# Configure Spark session for Iceberg
spark = SparkSession.builder \
    .config("spark.sql.extensions", "org.apache.iceberg.spark.extensions.IcebergSparkSessionExtensions") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.iceberg.spark.SparkCatalog") \
    .config("spark.sql.catalog.spark_catalog.catalog-impl", "org.apache.iceberg.aws.glue.GlueCatalog") \
    .config("spark.sql.catalog.spark_catalog.warehouse", "s3://omopdatatesting/synthea10/") \
    .config("spark.sql.catalog.spark_catalog.io-impl", "org.apache.iceberg.aws.s3.S3FileIO") \
    .getOrCreate()

# S3 Bucket and SQL file path
s3_bucket = "omopdatatesting"
sql_file_key = "files/result_statement.txt"

# Initialize S3 Client
s3_client = boto3.client("s3")

# Read the SQL file from S3
try:
    response = s3_client.get_object(Bucket=s3_bucket, Key=sql_file_key)
    sql_commands = response["Body"].read().decode("utf-8")
    sql_statements = [stmt.strip() for stmt in sql_commands.split(";") if stmt.strip()]
except Exception as e:
    print(f"Error reading SQL file from S3: {e}")
    sys.exit(1)

    
# Execute each SQL statement
for sql in sql_statements:
    try:
        print(f"Executing SQL: {sql}")
        spark.sql(sql)
        print("SQL executed successfully")
    except Exception as e:
        print(f"Error executing SQL: {e}")

print("🎉 All SQL commands executed successfully!")
