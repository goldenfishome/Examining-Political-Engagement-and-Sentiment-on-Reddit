# Databricks notebook source
# MAGIC %md
# MAGIC # Preparing reddit data using Spark for Machine Learning
# MAGIC 
# MAGIC # Project Group #27
# MAGIC #### Clara Richter, Elise Rust, Yujia Jin
# MAGIC ##### ANLY 502
# MAGIC ##### Project Deliverable #3
# MAGIC #####Dec 5, 2022
# MAGIC 
# MAGIC The original dataset for this notebook is described in [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf) paper.

# COMMAND ----------

## Load necessary packages
#import findspark
#findspark.init()
import pandas as pd
import numpy as np
import json
import pyspark.sql.functions as f
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType,BooleanType,DateType
from pyspark.sql import SparkSession
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler, VectorIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, Model

# COMMAND ----------

spark = SparkSession.builder.appName("reddit").getOrCreate()

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load in Data

# COMMAND ----------

## Read in sentment dataframes again
## 1) Comments
comments_sentiment = spark.read.parquet("dbfs:/tmp/out/com_kpis_sent.parquet")

## 2) Submissions
titles_sentiment = spark.read.parquet("dbfs:/tmp/out/sub_kpis_sent.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View data and ensure cleanliness

# COMMAND ----------

comments_sentiment.show(5)

# COMMAND ----------

titles_sentiment.show(5)

# COMMAND ----------

# Print schema and get datatypes
print(comments_sentiment.printSchema())
print(titles_sentiment.printSchema())

# COMMAND ----------

# MAGIC %md
# MAGIC #### Subset relevant ML columns
# MAGIC 
# MAGIC Our analysis leads us to use the comments dataframe to try to do 'subreddit' classification and the titles dataframe to try to do 'CPI' and 'DOW' regression. Given the results from the NLP analysis, there are more distinct differences across text and other variables for subreddits amongst the comments; thus, subreddit classification via comments makes more analytical sense. Additionally, the titles dataframe consists mostly of headlines which may offer insight into economic trends such as inflation and stock market crashes in regression analysis.

# COMMAND ----------

# select columns for ML regression for KPIs from posts
cols = ['DOW', 'CPI', 'Unemp_Rate', 'title']

titles_KPI = titles_sentiment.select(cols).na.drop() # Drop NAs
titles_KPI.show(5)

# COMMAND ----------

# select columns for ML predicting subreddit from comments
cols = ['subreddit', 'body', ]

com_subred = comments_sentiment.select(cols).na.drop() # Drop NAs
com_subred.show(5)

# COMMAND ----------

com_subred.groupBy("subreddit").count().show()

# COMMAND ----------

rep_group = com_subred.filter(com_subred.subreddit=="Republican")
pol_group = com_subred.filter(com_subred.subreddit=="politics").limit(40710)
dem_group = com_subred.filter(com_subred.subreddit=="democrats").limit(40710)

# COMMAND ----------

# join all
com_subred = rep_group.union(pol_group)
com_subred = com_subred.union(dem_group)
com_subred.groupBy("subreddit").count().show()

# COMMAND ----------

com_subred.show(5)

# COMMAND ----------

com_subred_pd = com_subred.toPandas()

# COMMAND ----------

# select columns for ML predicting subreddit from titles
cols = ['subreddit', 'title']

title_subred = titles_sentiment.select(cols).na.drop() # Drop NAs
title_subred.show(5)

# COMMAND ----------

# check the number of each subreddit
title_subred.groupBy("subreddit").count().show()

# COMMAND ----------

rep_title = title_subred.filter(title_subred.subreddit=="Republican").limit(13561)
pol_title = title_subred.filter(title_subred.subreddit=="politics").limit(13561)
dem_title = title_subred.filter(title_subred.subreddit=="democrats")

# COMMAND ----------

# join all
title_subred = rep_title.union(pol_title)
title_subred = title_subred.union(dem_title)
title_subred.groupBy("subreddit").count().show()

# COMMAND ----------

title_subred_pd = title_subred.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Preparation for Modeling
# MAGIC 
# MAGIC * Remaining feature transformations and additional textual processing steps
# MAGIC 
# MAGIC ##### Contained in Pipeline in ML Notebook
# MAGIC * ML transformations (string indexer and vectorizer)
# MAGIC * Split data into training and testing data

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Remaining Textual Processing: Tokenize text

# COMMAND ----------

!pip install nltk
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
st = SnowballStemmer('english')

# COMMAND ----------

### Define function data_clean(text) to automate this 
def data_clean(text):
    # change to lower and remove spaces on either side
    cleaned_text = text.apply(lambda x: x.lower().strip())

    # remove extra spaces in between
    cleaned_text = cleaned_text.apply(lambda x: re.sub(' +', ' ', x))

    # remove punctuation
    cleaned_text = cleaned_text.apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))

    # remove stopwords and get the stem
    cleaned_text = cleaned_text.apply(lambda x: ' '.join(st.stem(text) for text in x.split() if text not in stop_words))

    return cleaned_text
    

# Clean comments
com_subred_pd['body'] = data_clean(com_subred_pd['body'])
title_subred_pd['title'] = data_clean(title_subred_pd['title'])

# COMMAND ----------

title_subred_pd

# COMMAND ----------

com_subred_pd

# COMMAND ----------

# https://datascience-enthusiast.com/Python/PySpark_ML_with_Text_part1.html
from pyspark.ml.feature import Tokenizer

tokenizer = Tokenizer(inputCol="title", outputCol="words")
words_kpi = tokenizer.transform(titles_KPI)
words_kpi.show(5)

# COMMAND ----------

from pyspark.ml.feature import CountVectorizer

count = CountVectorizer(inputCol="words", outputCol="rawFeatures")
model = count.fit(words_kpi)
featurizedData_kpi = model.transform(words_kpi)
featurizedData_kpi.show(5)

# COMMAND ----------

# Apply term frequency–inverse document frequency (TF-IDF)
# (down-weighs features which appear frequently)
from pyspark.ml.feature import  IDF

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData_kpi)
title_data = idfModel.transform(featurizedData_kpi)

title_data = title_data.select("DOW", "CPI", "Unemp_Rate", "features")
title_data.show(5)

# COMMAND ----------

# prep text data from predicting subreddits from comments
com_subred = spark.createDataFrame(com_subred_pd)

# Tokenize text
tokenizer = Tokenizer(inputCol="body", outputCol="words")
words_subred = tokenizer.transform(com_subred)

# Vectorize
count = CountVectorizer(inputCol="words", outputCol="rawFeatures")
model = count.fit(words_subred)
featurizedData_subred = model.transform(words_subred)

# apply TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData_subred)

com_data = idfModel.transform(featurizedData_subred)
com_data = com_data.select("subreddit", "features")
com_data.show(5)

# COMMAND ----------

# prep text data from predicting subreddits from title
title_subred = spark.createDataFrame(title_subred_pd)

# Tokenize text
tokenizer = Tokenizer(inputCol="title", outputCol="words")
words_subred = tokenizer.transform(title_subred)

# Vectorize
count = CountVectorizer(inputCol="words", outputCol="rawFeatures")
model = count.fit(words_subred)
featurizedData_subred = model.transform(words_subred)

# apply TF-IDF
idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData_subred)

title_subred_data = idfModel.transform(featurizedData_subred)
title_subred_data = title_subred_data.select("subreddit", "features")
title_subred_data.show(5)

# COMMAND ----------

# export dataset for predicting KPIs from posts
title_data.write.parquet("/tmp/out/title_KPI.parquet")

# COMMAND ----------

# export dataset for predicting subreddits from commments
com_data.write.parquet("/tmp/out/com_subred1.parquet")

# COMMAND ----------

# export dataset for predicting subreddits from titles
title_subred_data.write.parquet("/tmp/out/title_subred_data.parquet")

# COMMAND ----------


