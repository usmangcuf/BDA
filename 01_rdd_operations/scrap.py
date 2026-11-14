
# ==========================================================
# APACHE SPARK WEB SCRAPING
# BS DATA SCIENCE PRACTICAL
# ==========================================================

import requests
import time

from bs4 import BeautifulSoup

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    DoubleType,
    IntegerType
)

# ----------------------------------------------------------
# STEP 1: CREATE SPARK SESSION
# ----------------------------------------------------------

spark = SparkSession.builder \
    .appName("BSDS Web Scraping Practical") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Apache Spark Started Successfully!")

# ----------------------------------------------------------
# STEP 2: DEFINE WEBSITE PAGE URLS
# ----------------------------------------------------------

base_url = "https://books.toscrape.com/"

page_urls = [
    base_url,
    base_url + "catalogue/page-2.html",
    base_url + "catalogue/page-3.html"
]

# ----------------------------------------------------------
# STEP 3: DEFINE WEB SCRAPING FUNCTION
# ----------------------------------------------------------

def scrape_page(url):

    try:

        headers = {
            "User-Agent": "BSDS-Educational-Web-Scraper/1.0"
        }

        response = requests.get(
            url,
            headers=headers,
            timeout=20
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        books = []

        # Find all book containers
        book_items = soup.find_all(
            "article",
            class_="product_pod"
        )

        for book in book_items:

            # Extract book title
            title_tag = book.find("h3").find("a")

            title = title_tag.get("title", "").strip()

            # Extract book price
            price_tag = book.find(
                "p",
                class_="price_color"
            )

            price_text = price_tag.text.strip()

            # Convert price into numeric value
            price = float(
                price_text.replace("£", "").replace("Â", "")
            )

            # Extract star rating
            rating_tag = book.find(
                "p",
                class_="star-rating"
            )

            rating_classes = rating_tag.get("class", [])

            rating_words = {
                "One": 1,
                "Two": 2,
                "Three": 3,
                "Four": 4,
                "Five": 5
            }

            rating = 0

            for word, value in rating_words.items():

                if word in rating_classes:
                    rating = value

            # Extract availability
            availability_tag = book.find(
                "p",
                class_="availability"
            )

            availability = availability_tag.text.strip()

            # Extract product URL
            relative_url = title_tag.get("href")

            product_url = requests.compat.urljoin(
                url,
                relative_url
            )

            # Store extracted information
            books.append((
                title,
                price,
                rating,
                availability,
                product_url
            ))

        return books

    except Exception as e:

        print("Error scraping:", url, e)

        return []


# ----------------------------------------------------------
# STEP 4: DISTRIBUTE WEB SCRAPING TASKS USING SPARK
# ----------------------------------------------------------

page_rdd = spark.sparkContext.parallelize(
    page_urls,
    len(page_urls)
)

# Each partition processes its assigned page
book_rdd = page_rdd.flatMap(scrape_page)

# Count extracted records
print("Total Books Scraped:", book_rdd.count())


# ----------------------------------------------------------
# STEP 5: DEFINE SPARK DATAFRAME SCHEMA
# ----------------------------------------------------------

schema = StructType([

    StructField("title", StringType(), True),

    StructField("price", DoubleType(), True),

    StructField("rating", IntegerType(), True),

    StructField("availability", StringType(), True),

    StructField("product_url", StringType(), True)

])


# ----------------------------------------------------------
# STEP 6: CONVERT RDD INTO SPARK DATAFRAME
# ----------------------------------------------------------

books_df = spark.createDataFrame(
    book_rdd,
    schema
)

print("\nSCRAPED BOOK DATA")

books_df.show(20, truncate=False)


# ----------------------------------------------------------
# STEP 7: PERFORM DATA ANALYSIS USING SPARK
# ----------------------------------------------------------

from pyspark.sql.functions import (
    avg,
    min,
    max,
    count,
    col
)

print("\n1. BOOKS WITH PRICE GREATER THAN 30")

books_df.filter(
    col("price") > 30
).show(truncate=False)


print("\n2. AVERAGE BOOK PRICE")

books_df.select(
    avg("price").alias("Average Price")
).show()


print("\n3. MINIMUM AND MAXIMUM BOOK PRICE")

books_df.select(
    min("price").alias("Minimum Price"),
    max("price").alias("Maximum Price")
).show()


print("\n4. NUMBER OF BOOKS BY RATING")

books_df.groupBy(
    "rating"
).agg(
    count("*").alias("Total Books")
).orderBy(
    "rating"
).show()


print("\n5. BOOKS SORTED BY PRICE")

books_df.orderBy(
    col("price").desc()
).show(10, truncate=False)


# ----------------------------------------------------------
# STEP 8: SAVE SCRAPED DATA
# ----------------------------------------------------------

books_df.write \
    .mode("overwrite") \
    .option("header", True) \
    .csv("output/books_csv")

books_df.write \
    .mode("overwrite") \
    .json("output/books_json")

print("\nData successfully saved!")

# ----------------------------------------------------------
# STEP 9: STOP SPARK SESSION
# ----------------------------------------------------------

spark.stop()

print("Apache Spark Stopped Successfully!")
