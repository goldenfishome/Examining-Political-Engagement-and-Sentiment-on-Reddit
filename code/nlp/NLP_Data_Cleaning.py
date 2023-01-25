# Databricks notebook source
# MAGIC %md
# MAGIC # Cleaning and Preparing reddit data using Spark for Natural language Processing
# MAGIC 
# MAGIC # Project Group #27
# MAGIC #### Clara Richter, Elise Rust, Yujia Jin
# MAGIC ##### ANLY 502
# MAGIC ##### Project Deliverable #2
# MAGIC #####Nov 22, 2022
# MAGIC 
# MAGIC The original dataset for this notebook is described in [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf) paper.

# COMMAND ----------

# MAGIC %md
# MAGIC Spark and Colab setup --> adapted from Lab 9 Setup

# COMMAND ----------

# Install Spark, PySpark and Spark NLP
! pip install -q pyspark==3.1.2 spark-nlp

# COMMAND ----------

## Load in required packages
import os
import pandas as pd
import numpy as np
import json
import pyspark
from datetime import datetime
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline

# COMMAND ----------

# MAGIC %md
# MAGIC Start Spark Session

# COMMAND ----------

spark = SparkSession.builder \
        .appName("SparkNLP") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.2.1") \
    .master('yarn') \
    .getOrCreate()

# COMMAND ----------

spark

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Loading and Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Read in Data (Reddit and External)
# MAGIC 
# MAGIC Previously generated and cleaned in Project Deliverable 1

# COMMAND ----------

######## ONLY RUN ONCE!!!!!!!!!
os.chdir("../..") # Move current working directory back up to root
print(os.getcwd())

# COMMAND ----------

## Read in submissions data
sub_all=spark.read.parquet("/tmp/output/sub_all1.parquet")

# COMMAND ----------

## Read in comments data
com_all=spark.read.parquet("/tmp/output/com_all1.parquet")

# COMMAND ----------

import pandas as pd

sub_all_pd = sub_all.toPandas()
sub_all_pd

# COMMAND ----------

com_all_pd = com_all.toPandas()
com_all_pd

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Generation
# MAGIC 
# MAGIC 2 more dummy variables (using regex)

# COMMAND ----------

#### Create dummy variables about topics using regex commands
## We already have 4-5 created from previous assignment but 2 additional topic dummy variables are generated below
import re

## 1) Education
sub_all_pd["dummy_education"] = sub_all_pd.title.str.contains(r'(?=.*school)|(?=.*education)|(?=.*learn)|(?=.*class)|(?=.*teacher)')
com_all_pd["dummy_education"] = com_all_pd.body.str.contains(r'(?=.*school)|(?=.*education)|(?=.*learn)|(?=.*class)|(?=.*teacher)')

## 2) Trump
sub_all_pd["dummy_trump"] = sub_all_pd.title.str.contains(r'(?=.*trump)|(?=.*Trump)')
com_all_pd["dummy_trump"] = com_all_pd.body.str.contains(r'(?=.*trump)|(?=.*Trump)')

# COMMAND ----------

## View new dummy variables
print(sub_all_pd[["title", "dummy_trump", "dummy_education"]])
print(com_all_pd[["body", "dummy_trump", "dummy_education"]])

# COMMAND ----------

# MAGIC %md
# MAGIC #### External Dataset
# MAGIC 
# MAGIC We are combining various Key Economic Performance Indicators (e.g. inflation rate, unemployment rate, DOW) by month between 2020 and 2021 from The Bureau of Labor Statistics (https://data.bls.gov/timeseries/). Many of our NLP questions are related to political sentiment and polarization along party lines, topic lines, etc. and there is much research that indicates that economic conditions are highly correlated to political sentiment.
# MAGIC 
# MAGIC **Articles discussing this correlation between economic growth and politics:**
# MAGIC 1. https://link.springer.com/chapter/10.1007/978-1-349-26284-7_9
# MAGIC 2. https://www.imf.org/en/Publications/fandd/issues/2020/06/political-economy-of-economic-policy-jeff-frieden
# MAGIC 3. https://www.investopedia.com/ask/answers/031615/what-impact-does-economics-have-government-policy.asp

# COMMAND ----------

##### Read in 3 separate small datasets:
## 1) Consumer Price Index by month between 2020-2021 (https://data.bls.gov/pdq/SurveyOutputServlet)
cpi = pd.read_csv("data/csv/nlp/bls_data/cpi.csv")

## 2) U.S. Unemployment Rate by month between 2020-2021 (https://data.bls.gov/pdq/SurveyOutputServlet)
unemployment = pd.read_csv("data/csv/nlp/bls_data/unemployment.csv")

## 3) DOW by month between 2020-2021 (https://www.statista.com/statistics/261690/monthly-performance-of-djia-index/)
dow = pd.read_csv("data/csv/nlp/bls_data/dow.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Clean external datasets

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1) Consumer Price Index (CPI)

# COMMAND ----------

## Select only last 3 rows (where CPI data is stored) and reset headers
cpi = cpi.tail(3)
cpi.columns = cpi.iloc[0]
cpi = cpi[1:]

print(cpi)

# COMMAND ----------

# Remove unnecessary columns
cpi = cpi.drop(columns=["HALF1", "HALF2"])

# Rename columns
cpi.columns = ["Year", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# COMMAND ----------

## Gather data into correct date-value rows
cpi_clean = pd.melt(cpi, id_vars=["Year"], value_vars=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], var_name="Month", value_name="CPI")
print(cpi_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2) Unemployment Data --> same source, repeat steps as above

# COMMAND ----------

## Select only last 3 rows (where unemployment data is stored) and reset headers
unemployment = unemployment.tail(3)
unemployment.columns = unemployment.iloc[0]
unemployment = unemployment[1:]

print(unemployment)

# COMMAND ----------

# Rename columns
unemployment.columns = ["Year", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# COMMAND ----------

## Gather data into correct date-value rows
unemployment_clean = pd.melt(unemployment, id_vars=["Year"], value_vars=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], var_name="Month", value_name="Unemp_Rate")
print(unemployment_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 3) DOW

# COMMAND ----------

## Remove unnecessary rows (title/header rows non-compliant with .csv format)
dow_clean = dow.iloc[2:]
dow_clean.columns = ["Date", "DOW"] # Rename columns

print(dow_clean)

# COMMAND ----------

# Split "Date" into "Month/Year" to match other Economic KPIs
new = dow_clean["Date"].str.split(" ") # Split on space
dow_clean["Month"] = new.str[0] # Create month column
dow_clean["Year"] = new.str[1] # Create year column
dow_clean = dow_clean.drop(columns=["Date"]) # Drop old date column

# COMMAND ----------

## Adjust datetype to match
dow_clean["Month"] = pd.to_datetime(dow_clean['Month'], format='%b').dt.strftime('%B')
dow_clean["Year"] = pd.to_datetime(dow_clean['Year'], format='%y').dt.strftime('%Y')

# re-order columns
dow_clean = dow_clean[["Year", "Month", "DOW"]]
print(dow_clean)

# COMMAND ----------

## Filter for only years 2020-2021
dow_clean = dow_clean[(dow_clean["Year"]=='2020') | (dow_clean["Year"]=='2021')]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Join into economic KPI dataset

# COMMAND ----------

kpis = pd.merge(cpi_clean, unemployment_clean, on=["Year","Month"])
kpis = pd.merge(kpis, dow_clean, on=["Year","Month"])

## Write to .csv to Git
kpis.to_csv("data/csv/nlp/bls_data/external_data.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Conduct EDA for external dataset

# COMMAND ----------

## Read in KPIs dataset again
kpis = pd.read_csv("data/csv/nlp/bls_data/external_data.csv")
kpis.head()

# COMMAND ----------

# Drop "Unnamed: 0"
kpis = kpis.drop(columns=["Unnamed: 0"])

# Check datatypes
for col in kpis.columns:
    print(col, kpis[col].dtype)

# COMMAND ----------

### Do datatype conversion

# Remove commas from DOW
kpis = kpis.replace(",", "", regex=True)

# Month to object --> the format of Reddit data
## i.e. Convert month to '01' format instead of 'January'
kpis["Month"] = pd.to_datetime(kpis['Month'], format='%B').dt.strftime('%m').astype("int")

# CPI, Unemp_Rate, DOW to float
kpis["CPI"] = kpis["CPI"].astype("float")
kpis["Unemp_Rate"] = kpis["Unemp_Rate"].astype("float")
kpis["DOW"] = kpis["DOW"].astype("float")

print(kpis.dtypes)

# COMMAND ----------

# Check for null/missing/incorrect values
print("Missing values")
print(kpis.isnull().sum())

# COMMAND ----------

# Get shape of external data set
print("Shape of training: ", len(kpis), len(kpis.columns)) # Get number of rows in the joined dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary Tables + Charts for External Dataset

# COMMAND ----------

#### Chart 1) Summary Table by Year
summary_table1 = kpis.groupby(kpis["Year"]).describe()
summary_table1.to_csv("data/csv/nlp/summary_table1.csv")
summary_table1

# COMMAND ----------

#### Chart 2) Summary Table by Month
summary_table2 = kpis.groupby(kpis["Month"]).describe()
summary_table2.to_csv("summary_table2.csv")
summary_table2

# COMMAND ----------

## For plotting, convert "Year", "Month" into one datetime variable
kpis["date"] = pd.to_datetime(kpis.Year.astype(str) + '/' + kpis.Month.astype(str)+'/01')
kpis_plotting = pd.melt(kpis, id_vars=["date"], value_vars=["CPI", "Unemp_Rate"])
print(kpis_plotting)

# COMMAND ----------

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

## Graph #1: Plot CPI and Unemployment over time --> on a similar scale
sns.lineplot(x="date", y="value", hue="variable", data = kpis_plotting).set(title="Key Performance Indicators: Consumer Price Index and Unemployment Rate")
#plt.savefig("data/plots/KPIs_fig1.png")

# COMMAND ----------

## Graph #1: Plot DOW over time --> on a totally separate scale
sns.lineplot(x="date", y="DOW", data = kpis).set(title="Key Performance Indicators: DOW")
#plt.savefig("data/plots/KPIs_DOW.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) Join External Data to Reddit Data

# COMMAND ----------

#### We're joining on the 'Month' and 'Year' columns --> let's make sure they're in the same format
print(sub_all_pd[['month','year']].dtypes)
print(com_all_pd[['month','year']].dtypes)
print(kpis[['Month','Year']].dtypes)

# COMMAND ----------

sub_kpis = pd.merge(sub_all_pd, kpis, left_on=["year","month"], right_on=["Year","Month"])
com_kpis = pd.merge(com_all_pd, kpis, left_on=["year","month"], right_on=["Year","Month"])
print(sub_kpis)

# COMMAND ----------

# Drop duplicate columns
sub_kpis = sub_kpis.drop(columns=["Month", "Year"])
com_kpis = com_kpis.drop(columns=["Month","Year"])

# Check datatypes one more time to ensure they stayed correct during join
print(sub_kpis.dtypes)
print(com_kpis.dtypes)

# COMMAND ----------

### Save newly joined full dataset to DBFS
sub_spark = spark.createDataFrame(sub_kpis)
sub_spark.write.parquet('/tmp/out/sub_kpis.parquet')

com_spark = spark.createDataFrame(com_kpis)
com_spark.write.parquet('/tmp/out/com_kpis.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ## NLP Text Exploration

# COMMAND ----------

pip install nltk

# COMMAND ----------

## Read in full joined datasets again
sub = spark.read.parquet('/tmp/out/sub_kpis.parquet')
print(sub)

# COMMAND ----------

com = spark.read.parquet('/tmp/out/com_kpis.parquet')
print(com)

# COMMAND ----------

# Read in necessary  packages
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
from nltk.stem.snowball import SnowballStemmer
st = SnowballStemmer('english')
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# COMMAND ----------

##### Examine data again
sub.printSchema()

# COMMAND ----------

com.printSchema()

# COMMAND ----------

# Print size of data
print("Size of submissions dataframe", sub.count())
print("Size of comments dataframe", com.count())

# COMMAND ----------

# Print number of partitions
print("Number of partitions in submissions dataframe: ", sub.rdd.getNumPartitions())
print("Number of partitions in comments dataframe: ", com.rdd.getNumPartitions())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Basic data text checks/analysis
# MAGIC 
# MAGIC Identify most common words overall (+ over time), model distribution of text lengths

# COMMAND ----------

# Show text from submissions and comments dataframes
sub.select('title').show()

# COMMAND ----------

com.select('body').show()

# COMMAND ----------

sub_small = sub.sample(0.01,123)

# COMMAND ----------

com_small = com.sample(0.01, 123)

# COMMAND ----------

sub_titles = sub.toPandas()[['subreddit','title']]
com_body = com.toPandas()[['subreddit', 'body']]

# Print submissions output
sub_titles

# COMMAND ----------

# Print comments output
com_body

# COMMAND ----------

# Initial text cleaning (basic space removal and punctuation removal)

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
    

# Clean titles and comments
sub_titles['title'] = data_clean(sub_titles['title'])
com_body['body'] = data_clean(com_body['body'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Plot word counts and identify most common words

# COMMAND ----------

# For submissions
sub_word_cnt = pd.DataFrame(sub_titles[['title']].apply(lambda x: x.str.split(expand=True).stack()).stack().value_counts())
sub_word_cnt = sub_word_cnt.reset_index()
sub_word_cnt.columns = ['word','count']
top_words = sub_word_cnt[:50]

# COMMAND ----------

import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
plt.xticks(rotation=90)
FS = 18
plt.xlabel('Words', fontsize=FS)
plt.ylabel('Frequency', fontsize=FS)
plt.title("Top 50 Frequent Words in Titles of Posts")
plt.bar(top_words['word'], top_words['count'])
plt.savefig('data/plots/top_50_words.png')# Save to data/plots folder

# COMMAND ----------

# MAGIC %md
# MAGIC ###### Wordclouds of top words for 1) titles and 2) comments

# COMMAND ----------

!pip install wordcloud

# COMMAND ----------

### Define function to automate wordcloud generation
def generate_word_cloud(my_text, title):
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud, STOPWORDS
    # Define a function to plot word cloud
    def plot_cloud(wordcloud):
        # Set figure size
        plt.figure(figsize=(10, 10))
        # Display image
        plt.imshow(wordcloud) 
        # No axis details
        plt.axis("off");

    # Generate word cloud
    wordcloud = WordCloud(
        width = 1500,
        height = 1000, 
        random_state=1, 
        #background_color='salmon', 
        colormap='Pastel1', 
        collocations=False,
        stopwords = STOPWORDS).generate(my_text)
    plot_cloud(wordcloud)
    plt.show()
    plt.savefig('data/plots/'+title+'.png')

# COMMAND ----------

# Submissions titles
politics_titles = sub_titles[sub_titles['subreddit'] == 'politics']
politics_text = ' '.join(i for i in politics_titles['title'])

republican_titles = sub_titles[sub_titles['subreddit'] == 'Republican']
republican_text = ' '.join(i for i in republican_titles['title'])

democrats_titles = sub_titles[sub_titles['subreddit'] == 'democrats']
democrats_text = ' '.join(i for i in democrats_titles['title'])

# COMMAND ----------

### Submissions titles wordclouds
generate_word_cloud(politics_text, "politics_titles_wordcloud")
generate_word_cloud(republican_text, "republican_titles_wordcloud")
generate_word_cloud(democrats_text, "democrats_titles_wordcloud")

# COMMAND ----------

# Comments
politics_titles = com_body[com_body['subreddit'] == 'politics']
politics_text = ' '.join(i for i in politics_titles['body'])

republican_titles = com_body[com_body['subreddit'] == 'Republican']
republican_text = ' '.join(i for i in republican_titles['body'])

democrats_titles = com_body[com_body['subreddit'] == 'democrats']
democrats_text = ' '.join(i for i in democrats_titles['body'])

### Comments titles wordclouds
generate_word_cloud(politics_text, "politics_comments_wordcloud")
generate_word_cloud(republican_text, "republican_comments_wordcloud")
generate_word_cloud(democrats_text, "democrats_comments_wordcloud")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Build bag of words dictionary

# COMMAND ----------

### For the titles
sub_titles

# COMMAND ----------

bow_sub_dict= (sub_titles[['title']].apply(lambda x: x.str.split(expand=True).stack()).stack().value_counts()).to_dict()
bow_sub_dict

# COMMAND ----------

# Export bag of words
import json
json.dump({'bag_of_words' : bow_sub_dict}, 
          fp = open('data/csv/bag_of_words.json','w'))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3) Identify the most important words (TFIDF vectorized)

# COMMAND ----------

from sklearn.feature_extraction.text import TfidfVectorizer

# TFIDF Vectorize text and titles
tfidf_vec = TfidfVectorizer() # Initialize TFIDFVectorizer

sub_tfidf = tfidf_vec.fit_transform(sub_titles['title'])
SubColumnNames = tfidf_vec.get_feature_names()

com_tfidf = tfidf_vec.fit_transform(com_body['body'])
ComColumnNames = tfidf_vec.get_feature_names()

# COMMAND ----------

# View output and extract most important words
sub_tfidf = pd.DataFrame(sub_tfidf.toarray(),columns=SubColumnNames) # Convert to array, then dataframe
#com_tfidf = pd.DataFrame(com_tfidf.toarray()) # Convert to array, then dataframe

# COMMAND ----------

## Sum each column (summing frequency)
imp_words_sorted = sub_tfidf.sum(axis=0).sort_values(ascending=False)
top_words = imp_words_sorted.head(50)
print(top_words)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4) Plot distribution of text lengths

# COMMAND ----------

from pyspark.sql.functions import length

# COMMAND ----------

sub_lengths = sub.withColumn('title_length', length("title"))
sub_lengths = sub_lengths.select('title', 'title_length', 'month', 'year')
sub_lengths.show()

# COMMAND ----------

### Plot distribution of title lengths (https://stackoverflow.com/questions/39154325/pyspark-show-histogram-of-a-data-frame-column)

sub_length_hist = sub_lengths.select('title_length').rdd.flatMap(lambda x: x).histogram(11)

# Loading the Computed Histogram into a Pandas Dataframe for plotting
hist_df = pd.DataFrame(
    list(zip(*sub_length_hist)), 
    columns=['bin', 'frequency'])
hist_df["bin"] = np.floor(hist_df.bin).astype(int)
hist_df.set_index('bin').plot(kind='bar').get_figure().savefig('data/plots/titles_length_dist.png')

# COMMAND ----------

com_lengths = com.withColumn('comment_length', length("body"))
com_lengths = com_lengths.select('body', 'comment_length', 'month', 'year')
com_lengths.show()

# COMMAND ----------

### Plot distribution of comment lengths (https://stackoverflow.com/questions/39154325/pyspark-show-histogram-of-a-data-frame-column)

com_length_hist = com_lengths.select('comment_length').rdd.flatMap(lambda x: x).histogram(11)

# Loading the Computed Histogram into a Pandas Dataframe for plotting
hist_df = pd.DataFrame(
    list(zip(*sub_length_hist)), 
    columns=['bin', 'frequency'])
hist_df["bin"] = np.floor(hist_df.bin).astype(int)
hist_df.set_index('bin').plot(kind='bar').get_figure().savefig('data/plots/comments_length_dist.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ## NLP Text Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Clean data → stemming, punctuation, stopwords, etc. (5 procedures)

# COMMAND ----------

## Load in required packages
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
from pyspark.sql.functions import *

# COMMAND ----------

title_df = sub.select('title')
comments_df = com.select('body')

# COMMAND ----------

## 1) Stem words
stemmer = Stemmer() \
    .setInputCols(["cleanTokens"]) \
    .setOutputCol("stem")

## 2) Remove stopwords
stopwords_cleaner = StopWordsCleaner()\
    .setInputCols("token")\
    .setOutputCol("cleanTokens")\
    .setCaseSensitive(False)\
    #.setStopWords(["no", "without"]) (e.g. read a list of words from a txt)

# 3) Lowercase all words
# 4) Remove non-alphanumeric characters
normalizer = Normalizer() \
    .setInputCols(["stem"]) \
    .setOutputCol("normalized")\
    .setLowercase(True)\
    .setCleanupPatterns(["[^\w\d\s]"]) # remove punctuations (keep alphanumeric chars)
    # if we don't set CleanupPatterns, it will only keep alphabet letters ([^A-Za-z])
    
documentAssembler = DocumentAssembler()\
    .setInputCol("title")\
    .setOutputCol("document")

# 5) Tokenize
tokenizer = Tokenizer() \
    .setInputCols(["document"]) \
    .setOutputCol("token")

tokenassembler = TokenAssembler()\
    .setInputCols(["document", "normalized"]) \
    .setOutputCol("clean_text")

nlpPipeline = Pipeline(stages=[documentAssembler, 
                               tokenizer,
                               stopwords_cleaner,
                               stemmer,
                              normalizer,
                              tokenassembler])

# COMMAND ----------

## Submissions first
sub_result = nlpPipeline.fit(title_df).transform(title_df)

sub_result = sub_result.select('clean_text.result', explode(sub_result.clean_text.result).alias('clean_text'))
sub_result = sub_result.select('clean_text')
sub_result.show()

# COMMAND ----------

## Comments
## Rename 'body' to 'title' so it works with our NLP Pipeline
comments_df = comments_df.withColumnRenamed("body", "title")

com_result = nlpPipeline.fit(comments_df).transform(comments_df)

com_result = com_result.select('clean_text.result', explode(com_result.clean_text.result).alias('clean_text'))
com_result = com_result.select('clean_text')
com_result.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save data to DBFS to be used for models
# MAGIC 
# MAGIC Separate notebooks for models

# COMMAND ----------

### Save newly joined full dataset to DBFS
sub_result.write.parquet('/tmp/out/sub_result.parquet')

com_result.write.parquet('/tmp/out/com_result.parquet')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Close Spark Session

# COMMAND ----------

spark.stop()
