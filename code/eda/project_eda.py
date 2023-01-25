# Databricks notebook source
# MAGIC %md
# MAGIC # EDA for Project Group #27
# MAGIC #### Clara Richter, Elise Rust, Yujia Jin
# MAGIC ##### ANLY 502
# MAGIC ##### Project Deliverable #1
# MAGIC #####Nov 9, 2022

# COMMAND ----------

# MAGIC %md
# MAGIC #### Business Questions:
# MAGIC 
# MAGIC ##### EDA Questions:
# MAGIC 1. Business question: How will we measure “engagement” on Reddit? In what ways do people interact with a post?
# MAGIC - Technical proposal: We will categorize “engagement” as upvotes/downvotes and number of comments. The more of these interactions a post has, the more engagement it has received. We will measure these interactions by using pushshift.io Reddit API to call the variables num_comments and score.
# MAGIC 
# MAGIC 2. Business question: When are the best times to post on Reddit to get the most engagement?
# MAGIC - Technical proposal: We will measure the best times to post on Reddit by using pushshift.io Reddit API to call the created_utc variable. We will order the Reddit posts by most engagement to least engagement and plot the created_utc variable. This will allow us to observe the best times to post on Reddit to get the most engagement.
# MAGIC 
# MAGIC 3. Business question: How long should a Reddit title be to get the most engagement? How many sentences, words, or length of words?
# MAGIC - Technical proposal: We will use the pushshift.io Reddit API to call the title variable to get the text (string) of the post. We will use NLP to count the number of sentences in each post, count the number of words in each post, and count the lengths of the words in each post. We will then order the Reddit posts by most engagement to least engagement and these counts to observe how the length of a post on Reddit leads to engagement. 
# MAGIC 
# MAGIC ##### NLP Questions:
# MAGIC 4. Business question: Do posts with positive, negative, or neutral sentiment receive the most attention on Reddit?
# MAGIC - Technical proposal: We will use the pushshift.io Reddit API to call the title variable to get the text (string) of the post. We will use NLP to find the sentiment of each post. We will then order the Reddit posts by most engagement to least engagement and sentiment to observe how the sentiment of a post on Reddit leads to engagement.
# MAGIC 
# MAGIC 5. Business question:  How prevalent is misinformation/fake news on Reddit? Which subreddits seem to suffer from this problem more?
# MAGIC - Technical proposal: We will use the pushift.io Reddit API to call the title variable to get the text (string) of the post. We will combine this data with the external FACTOID dataset to get labeled Reddit data of “fake news” or “real news”. After training a model on this labeled dataset and fine-tuning hyperparameters we will test it on various random samples of text from different subreddits to observe how prevalent misinformation is.
# MAGIC 
# MAGIC 6. Business question: Sentiment analysis of different subreddits. Which subreddits produce the most/least positive and negative discourse?
# MAGIC - Technical proposal: We will use the pushift.io Reddit API to call the title and created_utc variables for a subset of subreddits. We will then employ a pre-trained sentiment classifier (VADER) to generate sentiment labels for each text blurb and map them to keywords and thread topics that appear most often. Finally, we will plot this over time to understand trends in sentiment.
# MAGIC 
# MAGIC 7. Business question: How do high profile political stunts (i.e. DeSantis migrant crisis) get discussed/how do they affect candidate sentiment?
# MAGIC - Technical proposal: We will use the pushift.io Reddit API to call the title, created_utc, and score variables for text in a handful of subreddits. We will filter for a posts that mention a subset of high profile politicians (i.e. Nancy Pelosi, Donald Trump, Mitch McConnell, Ron DeSantis, AOC, etc.) and again use the VADER sentiment labels to classify different posts. Ideally, we will be able to identify keywords and dates of some political stunts and track sentiment across relevant politicians before, during, and after.
# MAGIC 
# MAGIC 8. Business question: How can nonprofits, lobbyists, and politicians leverage Reddit to push legislative agendas?
# MAGIC - Technical proposal: We will use the pushift.io Reddit API to call the title to get the text of posts under specific subreddit and filter the posts of nonprofits, lobbyists, and politicians. We will then employ topic modeling techniques to find the most mentioned topics among these posts and identify the key words of each topic. We will create a new variable ‘topic cluster’ based on the classification of the topic. 
# MAGIC 
# MAGIC ##### ML Questions:
# MAGIC 
# MAGIC 9. Business question: Can Reddit data in the 10 days leading up to an election be leveraged to make election predictions?
# MAGIC - Technical proposal: The sample size for this may be too small but we will take title and created_utc variables for the 10 days leading up to each presidential and congressional election over the last 8 years. We plan to tokenize text and extract the frequency of each candidate being mentioned during the given period. We will also identify a candidate’s overall sentiment score and compare this to election results. 
# MAGIC 
# MAGIC 10. Business question: How polarized is political discourse these days? Which political topics produce the most polarized discourse? How has that changed over time?
# MAGIC - Technical proposal: We will use the pushift.io Reddit API to call the title and created_utc variables for a subset of subreddits. We will then employ a pre-trained polarity classifier (VADER) to generate sentiment labels/polarity scores for each text blurb. We will identify subreddits with the largest range in polarity scores, and plot this trend over time.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 1) Data Loading 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1A. Load in full Reddit parquet files (for both comments file and submissions file)

# COMMAND ----------

dbutils.fs.ls("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet")

# COMMAND ----------

comments = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/comments")
submissions = spark.read.parquet("abfss://anly502@marckvaismanblob.dfs.core.windows.net/reddit/parquet/submissions")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 1B. Create directories to store and save data/plots

# COMMAND ----------

## create a directory called data/plots and data/csv to save generated data
import os
PLOT_DIR = os.path.join("data", "plots")
CSV_DIR = os.path.join("data", "csv")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 2. Cleaning and joining dataframes

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### 2A. Filter datasets for relevant subreddits.
# MAGIC ###### Our primary business questions are in regards to political content and engagement thus our subreddits of choice are: r/politics, r/Republican, and r/democrats

# COMMAND ----------

from pyspark.sql.functions import col, asc,desc
submissions_politics = submissions.filter(submissions.subreddit == "politics")
submissions_Republican = submissions.filter(submissions.subreddit == "Republican")
submissions_democrats = submissions.filter(submissions.subreddit == "democrats")

# COMMAND ----------

print(submissions_politics.count())
print(submissions_Republican.count())
print(submissions_democrats.count())

# COMMAND ----------

comments_politics = comments.filter(comments.subreddit == "politics")
comments_Republican = comments.filter(comments.subreddit == "Republican")
comments_democrats = comments.filter(comments.subreddit == "democrats")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2B. The 'body' column in comments contains some deleted or removed comments. We are interested in the text from the 'body' column, so we remove these rows because they do not have text to analyze. We want to sample from each of these subreddits to create dataframes of equal size --> thus, we filter out these 'removed' rows beforehand. 

# COMMAND ----------

# drop the rows with [removed]
comments_politics = comments_politics[comments_politics['body'] != '[removed]']
comments_Republican = comments_Republican[comments_Republican['body'] != '[removed]']
comments_democrats = comments_democrats[comments_democrats['body'] != '[removed]']

# drop the rows with [deleted]
comments_politics = comments_politics[comments_politics['body'] != '[deleted]']
comments_Republican = comments_Republican[comments_Republican['body'] != '[deleted]']
comments_democrats = comments_democrats[comments_democrats['body'] != '[deleted]']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2C. Filter for first level comments (i.e. where 'link_id'=='parent_id' and remove comments ON other comments)
# MAGIC 
# MAGIC We did this as each submission has dozens of comments, of varying length and importance, so by filtering for each first level comments we extract immediate reactions and sentiments about original texts.

# COMMAND ----------

# only rows where link_id and parent_id are equal
comments_politics = comments_politics[comments_politics['link_id'] == comments_politics['parent_id']]
comments_Republican = comments_Republican[comments_Republican['link_id'] == comments_Republican['parent_id']]
comments_democrats = comments_democrats[comments_democrats['link_id'] == comments_democrats['parent_id']]

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2D. Select only the relevant variables for our analysis.  
# MAGIC 'author' - can check which users are most active.  
# MAGIC 'body' - use for NLP.  
# MAGIC 'created_utc' - use for time analysis.  
# MAGIC 'link_id' - use to link comments to submission posts.  
# MAGIC 'score' - use to determine engament of comment.  
# MAGIC 'subreddit' - use to group by subreddit.  

# COMMAND ----------

# select variables
com_pol_filtered = comments_politics.select('author',
'body',
'created_utc',
'link_id',
'score',
'subreddit')

com_Rep_filtered = comments_Republican.select('author',
'body',
'created_utc',
'link_id',
'score',
'subreddit')

com_dem_filtered = comments_democrats.select('author',
'body',
'created_utc',
'link_id',                              
'score',
'subreddit')

# COMMAND ----------

com_dem_filtered.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2E. Join different comments subreddits dataframes together

# COMMAND ----------

# join all
com_all = com_pol_filtered.union(com_Rep_filtered)
com_all = com_all.union(com_dem_filtered)

# COMMAND ----------

# remove "t3_" from link_id so it can match the id of the submission post
from pyspark.sql import functions as F
com_all = com_all.withColumn("link_id", F.regexp_replace("link_id", "t3_", ""))


# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2F. Remove rows that have ['deleted by user'] as the title in submissions dataframe
# MAGIC 
# MAGIC These rows contain no text to build an NLP or ML model on so we remove them

# COMMAND ----------

# drop the rows with [deleted by user]
submissions_politics = submissions_politics[submissions_politics['title'] != '[deleted by user]']
submissions_Republican = submissions_Republican[submissions_Republican['title'] != '[deleted by user]']
submissions_democrats = submissions_democrats[submissions_democrats['title'] != '[deleted by user]']

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2G. Sample from each subreddit (20,000 rows) to get equal row counts across all three

# COMMAND ----------

# 20,000 each
from pyspark.sql.functions import rand

n = 20000

sub_politics = submissions_politics.orderBy(rand()).limit(n)
sub_Republican = submissions_Republican.orderBy(rand()).limit(n)
sub_democrats = submissions_democrats.orderBy(rand()).limit(n)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 2H. Join different submissions subreddits dataframes together

# COMMAND ----------

# join all
sub_all = sub_politics.union(sub_Republican)
sub_all = sub_all.union(sub_democrats)

# COMMAND ----------

# Select relevant columns for analysis
### 'Subreddit' is our label for what subreddit each text came from, 'title' is important for text analysis, 'score and num_comments' are important for analysis regarding engagement/sentiment, 'created_utc' is important for timeseries analysis, and 'id' is relevant for joining the comments and submissions dataframes together.
sub_all_filtered = sub_all.select(
    "subreddit",
    "title",
    "score",
    "num_comments",
    "created_utc",
    "id")

# COMMAND ----------

# Save the cleaned submissions dataframe to DBFS
sub_all_filtered.write.format('com.databricks.spark.csv').save("/FileStore/sub_all_filtered")

# COMMAND ----------

## Get list of ids in submissions dataframe --> relevant for joining with comments ids
# We only want comments that are connected to a submissions post
post_id = sub_all.select("_c5").rdd.flatMap(lambda x: x).collect()
post_id

# COMMAND ----------

# Keep only comments rows where link_id is in post_id list
com_all_new = com_all.filter((com_all['link_id']).isin(post_id))

# COMMAND ----------

### Prepare comments dataframe for join: Deal with duplicate column names across two dfs

# add "com_" to all column names in com_all dataframe
com_all_new = com_all_new.select([F.col(c).alias("com_"+c) for c in com_all_new.columns])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 3: Join comments and submissions dataframes

# COMMAND ----------

sub_com = sub_all.join(com_all_new, sub_all._c5 ==  com_all_new.com_link_id, "left")

# COMMAND ----------

# Save joined submissions+comments dataframe to DBFS
sub_com.write.format('com.databricks.spark.csv').save("/FileStore/sub_com")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 4: Conduct basic EDA on joined dataframes

# COMMAND ----------

### Read in joined submissions+comments dataframe from DBFS
sub_com = spark.read.csv("/data/csv/sub_com.csv", header=False)
sub_com

# COMMAND ----------

import pandas as pd

## Individual comments and submissions dataframes available for dataloading
# Read in submissions only dataframe for EDA
sub_all= spark.read.csv("/FileStore/sub_all_filtered", header=False)
sub_all = sub_all.toPandas()
sub_all

# COMMAND ----------

### Rename columns of sub_all
sub_all.rename(columns={'_c0':'subreddit', '_c1':'title', '_c2':'score', '_c3':'num_comments', '_c4':'created_utc', '_c5':'id'}, inplace=True)
sub_all

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Basic Data Cleaning
# MAGIC 
# MAGIC Remove extraneous columns, rename columns

# COMMAND ----------

# Convert to Pandas DF
sub_com_df = sub_com.toPandas()
sub_com_df

# COMMAND ----------

### Rename columns
sub_com_df.rename(columns={'_c0':'subreddit', '_c1':'title', '_c2':'score', '_c3':'num_comments', '_c4':'created_utc', '_c5':'id', '_c6':'username', '_c7':'body', '_c8':'com_created_utc', '_c9':'com_id', '_c10':'com_score', '_c11':'com_subreddit'}, inplace=True)
sub_com_df

# COMMAND ----------

### Drop duplicate or extraneous columns
# All three columns are either irrelevant to analysis or already contained in other columns
sub_com_df = sub_com_df.drop(columns=['com_subreddit', 'com_id'])
sub_com_df

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4A. Report on basic info of data (schema, row numbers, etc.)
# MAGIC 
# MAGIC We're going to plot and examine both the joined submissions+comments dataframes as well as just the submissions dataframe so data cleaning and exploration will happen for both.

# COMMAND ----------

# Check the datatype of columns
#sub_all.drop(sub_all.columns[0], axis=1)
for col in sub_all.columns:
    print(col, sub_all[col].dtype)

# COMMAND ----------

for col in sub_com_df.columns:
    print(col, sub_com_df[col].dtype)

# COMMAND ----------

print("Number of submissions rows: ", len(sub_all)) # Get number of rows in the submissions dataframe
print("Number of submissions+comments rows: ", len(sub_com_df)) # Get number of rows in the joined dataframe

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4B. Conduct basic quality checks (null values, datatype conversions, dealing with outliers)

# COMMAND ----------

# change 'subreddit' column to categorical variable
sub_all["subreddit"] = sub_all["subreddit"].astype("category")
sub_com_df["subreddit"] = sub_com_df["subreddit"].astype("category")

# COMMAND ----------

# change to numeric variables
sub_all["score"] = sub_all["score"].astype(int)
sub_all["num_comments"] = sub_all["num_comments"].astype(int)
#sub_com_df["score"] = sub_com_df["score"].astype(int)

# COMMAND ----------

# check for missing values
print("Missing values in submissions")
print(sub_all.isnull().sum())
print(" ")
print("Missing values in joined df")
print(sub_com_df.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC So there are no missing values in the submissions dataframe --> but hundreds of thousands in the joined dataframe!
# MAGIC 
# MAGIC This is due to the nature of the df as there are many comments per submission. We've decided not to remove these rows as they may contain valuable information for our analysis. When we get to feature extraction and NLP/ML models we want as much data as possible, so we will remove duplicates and deal with null values for specific tasks at that point.

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4C. Convert columns to proper datatypes

# COMMAND ----------

# Convert the time variable to proper datetime type
sub_all['time'] = pd.to_datetime(sub_all['created_utc'],unit='s')
sub_all

# COMMAND ----------

## Convert created_utc columns to dates in sub_com_df as well
import numpy as np

sub_com_df["created_utc"].value_counts()
### For all 'created_utc' values of 0, the date gets generated as '1970-01-01 00:00'. These are useless so will be replaced with NAs.

## For non-integer values (i.e. corrupted values) --> replace with NAs
sub_com_df["created_utc"] = pd.to_numeric(sub_com_df.created_utc, errors='coerce').fillna(0).astype(int)
sub_com_df["com_created_utc"] = pd.to_numeric(sub_com_df.com_created_utc, errors='coerce').fillna(0).astype(int)

sub_com_df['submission_time'] = pd.to_datetime(sub_com_df['created_utc'],unit='s')
sub_com_df['comment_time'] = pd.to_datetime(sub_com_df['com_created_utc'],unit='s')

# Replace '1970-01-01 00:00' with NAs
sub_com_df.replace('1970-01-01 00:00:00', np.nan)

sub_com_df

## Print outcome and drop old time columns
#sub_com_df.drop(columns=["created_utc", "com_created_utc"])

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 4D. Feature generations
# MAGIC 
# MAGIC Create 5 new relevant variables for analysis with data transformations as well as 3 dummy variables using regex.

# COMMAND ----------

## Create new month, year, and time of day variables
sub_all['year'] = (pd.Series(pd.DatetimeIndex(sub_all['time']).year)).values
sub_all['month'] = (pd.Series(pd.DatetimeIndex(sub_all['time']).month)).values
sub_all['hour'] = (pd.Series(pd.DatetimeIndex(sub_all['time']).hour)).values
sub_all['time_of_day'] = pd.cut(sub_all['hour'], bins=[0, 6, 12, 18, 23], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

sub_com_df['sub_year'] = (pd.Series(pd.DatetimeIndex(sub_com_df['submission_time']).year)).values
sub_com_df['sub_month'] = (pd.Series(pd.DatetimeIndex(sub_com_df['submission_time']).month)).values
sub_com_df['sub_hour'] = (pd.Series(pd.DatetimeIndex(sub_com_df['submission_time']).hour)).values
sub_com_df['sub_time_of_day'] = pd.cut(sub_com_df['sub_hour'], bins=[0, 6, 12, 18, 23], labels=['Night', 'Morning', 'Afternoon', 'Evening'])

## View new time variables
print(sub_all[["year", "month", "hour", "time_of_day"]])
print(sub_com_df[["sub_year", "sub_month", "sub_hour", "sub_time_of_day"]])

# COMMAND ----------

# Check for unique values in new 'hour' column
sub_all['hour'].unique()

# COMMAND ----------

# View binned time of day variable - does it capture all 4 time periods?
print(sub_all['time_of_day'].unique())

# COMMAND ----------

# Create 'title_length' variable to calculate length of title and 'comment_length' to calcualte length of comment
sub_all["title_length"] = sub_all['title'].apply(len)

sub_com_df["title_length"] =  (sub_com_df[sub_com_df["title"].notnull()])["title"].apply(len)
sub_com_df["comment_length"] = (sub_com_df[sub_com_df["body"].notnull()])["body"].apply(len)

# View output
print(sub_com_df[["title_length", "comment_length"]])

# COMMAND ----------

## Create 1 more variable:
# election_year (T/F) with T for year == 2022 because dataset only runs from 2021 onwards
sub_all["year"].min() # 2021
sub_all["year"].max() # 2022
sub_all["election_year"] = np.where(sub_all["year"]==2022, True, False)
sub_all

# COMMAND ----------

#### Create dummy variables about topics using regex commands
import re

## 1) police
sub_all["dummy_police"] = sub_all.title.str.contains(r'(?=.*police)|(?=.*law)|(?=.*enforcement)|(?=.*brutality)|(?=.*order)')
sub_com_df["dummy_police"] = sub_com_df.title.str.contains(r'(?=.*police)|(?=.*law)|(?=.*enforcement)|(?=.*brutality)|(?=.*order)')

## 2) healthcare
sub_all["dummy_healthcare"] = sub_all.title.str.contains(r'(?=.*health)|(?=.*insur)|(?=.*pharma)|(?=.*diagnos)|(?=.*medicine)|(?=.*vaccin)')
sub_com_df["dummy_healthcare"] = sub_com_df.title.str.contains(r'(?=.*health)|(?=.*insur)|(?=.*pharma)|(?=.*diagnos)|(?=.*medicine)|(?=.*vaccin)')

## 3) climate
sub_all["dummy_climate"] = sub_all.title.str.contains(r'(?=.*climate)|(?=.*energy)|(?=.*renewable)|(?=.*deforest)|(?=.*global warming)')
sub_com_df["dummy_climate"] = sub_com_df.title.str.contains(r'(?=.*climate)|(?=.*energy)|(?=.*renewable)|(?=.*deforest)|(?=.*global warming)')

## 4) taxes
sub_all["dummy_economy"] = sub_all.title.str.contains(r'(?=.*tax)|(?=.*inflation)|(?=.*economy)|(?=.*costs)|(?=.*unemployment)')
sub_com_df["dummy_economy"] = sub_com_df.title.str.contains(r'(?=.*tax)|(?=.*inflation)|(?=.*economy)|(?=.*costs)|(?=.*unemployment)')

# View new variables
print(sub_all[["title", "dummy_police", "dummy_healthcare", "dummy_climate", "dummy_economy"]])
print(sub_com_df[["title", "dummy_police", "dummy_healthcare", "dummy_climate", "dummy_economy"]])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5: Generate summary statistics and summary tables for the dataset.
# MAGIC Explore basic summary statistics broken down by *subreddit*, *time_of_day*, *election_year*, and *month*.

# COMMAND ----------

### Summary statistics

# Table 1: Summary Statistics broken down by subreddit
table1 = sub_all.groupby(sub_all["subreddit"]).describe()
print(table1)

table1.to_csv("data/csv/table1.csv")

# COMMAND ----------

# Table 2: Summary Statistics broken down by time of day
table2 = sub_all.groupby(sub_all["time_of_day"]).describe()
print(table2)

table2.to_csv("data/csv/table2.csv")

# COMMAND ----------

# Table 3: Summary Statistics broken down by election year
table3 = sub_all.groupby(sub_all["election_year"]).describe()
print(table3)

table3.to_csv("data/csv/table3.csv")

# COMMAND ----------

# Table 4: Summary Statistics broken down by month
table4 = sub_all.groupby(sub_all["month"]).describe()
print(table4)

table4.to_csv("data/csv/table4.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 5: Visualizations about our dataset

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5A. Data subsetting and cleaning for plotting

# COMMAND ----------

# remove the data beyond 99 percentile
sub_all_change_comment = sub_all[sub_all['num_comments'] < sub_all['num_comments'].quantile(0.99)]
sub_all_change_comment['subreddit'].value_counts()

# COMMAND ----------

republican_df = sub_all_change_comment[sub_all_change_comment['subreddit'] == 'Republican'].sample(n = 19409)
democrats_df = sub_all_change_comment[sub_all_change_comment['subreddit'] == 'democrats'].sample(n = 19409)
sub_all_change_comment = republican_df.append(democrats_df).append(sub_all_change_comment[sub_all_change_comment['subreddit'] == 'politics'])

# COMMAND ----------

# prepare dataset for 5B
sub_all_subdf = sub_all_change_comment[['subreddit','hour','num_comments']]
sub_all_gb_hour = sub_all_subdf.groupby(['hour', 'subreddit']).sum()
sub_all_gb_hour = sub_all_gb_hour.reset_index()


# COMMAND ----------

# prepare dataset for 5C
sub_all_time = sub_all_change_comment[['subreddit','time_of_day','num_comments']]
sub_all_gb_time = sub_all_time.groupby(['time_of_day', 'subreddit']).sum()
sub_all_gb_time = sub_all_gb_time.reset_index()

# COMMAND ----------

# prepare dataset for 5D
republican = sub_all_change_comment[sub_all_change_comment['subreddit'] == 'Republican'].sample(n = 1000)
democrats = sub_all_change_comment[sub_all_change_comment['subreddit'] == 'democrats'].sample(n = 1000)
sub_all_change_size = republican.append(democrats).append(sub_all_change_comment[sub_all_change_comment['subreddit'] == 'politics'].sample(n = 1000))
#sub_all_change_size

# COMMAND ----------

sub_all_change_range = sub_all_change_size[(sub_all_change_size['score'] < 2000) & (sub_all_change_size['num_comments'] < 150)]

# COMMAND ----------

sub_all_change_size

# COMMAND ----------

# prepare dataset for 5F
sub_all_cnt = sub_all[['subreddit','hour']]
sub_all_cnt = sub_all_cnt.groupby(['hour','subreddit'])['subreddit'].count().to_frame()
sub_all_cnt = sub_all_cnt.rename(columns = {'subreddit':'counts'})
sub_all_cnt = sub_all_cnt.reset_index()

# COMMAND ----------

sub_all_cnt['num_comments'] = sub_all_gb_hour['num_comments']
sub_all_cnt['comments_post_ratio'] = sub_all_cnt['num_comments']/sub_all_cnt['counts'] 

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5B. Bar Chart of Distribution of Number of Comments against Hour of Post
# MAGIC 
# MAGIC 
# MAGIC This graph shows the distribution of sum of number of comments of reddit posts versus hours in a day. These three subreddit share same pattern of fluctuation of number of comments through out the day. The number of comments peaked at afternoon at around 15:00 to 16:00 and stay lowest at morning time, 6:00 to 9:00. The politics subreddit have much more comments than other two subreddits. Overally, democrats subbreddit has more number of comments than republican.

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn  as sns

# BAR CHART
sns.set_theme(style="whitegrid", palette="Set1")
sns.barplot(
    data=sub_all_gb_hour, 
    #kind="bar",
    x="hour", 
    y="num_comments", 
    hue="subreddit").set(title='Distribution of Sum of Number of Comments against Time Hour of Subreddits')

plt.show()
plt.savefig('data/plots/fig1.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5C. Bar Chart of Distribution of Number of Comments against Time of Day of Post
# MAGIC 
# MAGIC This graph shows the distribution of sum of number of comments of reddit posts versus time of the day. This plot matches the previous one. The sum of number of comments of all three subreddits peaks at afternoon and reach lowest at night and morning. 

# COMMAND ----------

sns.set_theme(style="whitegrid", palette="Set1")
sns.barplot(
    data=sub_all_gb_time, 
    x="time_of_day", 
    y="num_comments", 
    hue="subreddit").set(title='Distribution of Sum of Number of Comments against Time of Day of Subreddits')

plt.show()
plt.savefig('data/plots/fig2.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5D. Scatterplot of post score vs. number of comments
# MAGIC 
# MAGIC The first graph below has shown the relationship between the score and number of comments of all three subreddit. The dataset used in this plot is the subset of original dataset, 1000 records of each subreddit. The score of most posts lay within the range of 0 to 2000 and the number of comments lay within the range of 0 to 100. Both the score and number of comments of politics subreddits are much more expanded than other two subreddits. The post score has a roughly positive relationship with number of comments. The second graph shows the same two variables with altered range, it shows the same positive relationship between these two variables. 

# COMMAND ----------

sns.scatterplot(
    data=sub_all_change_size, 
    x='num_comments', 
    y='score',
    hue="subreddit",
    alpha = .4).set(title = 'Number of Comment VS Score')
plt.show()
plt.savefig('data/plots/fig3.png')

# COMMAND ----------

sns.scatterplot(
    data=sub_all_change_range, 
    x='num_comments', 
    y='score',
    hue="subreddit",
    alpha = .4).set(title = 'Number of Comment VS Score')
plt.show()
plt.savefig('data/plots/fig4.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5E. Correlation plot between length of titles and number of comment
# MAGIC 
# MAGIC This graph has shown the relationship between length of titles and number of comments. The dataset used in this graph is a subset of original dataset, which take 1000 row of each subreddit. According to the graph, the number of comments reaches the peak at around title length of 70 and then decrease after that length. This graph answers the business question 3 'How long should a Reddit title be to get the most engagement?'.

# COMMAND ----------

sns.scatterplot(
    data=sub_all_change_size, 
    x='title_length', 
    y='num_comments',
    hue="subreddit",
    alpha = .4).set(title = 'Length of Title VS Number of Comment')
plt.show()
plt.savefig('data/plots/fig5.png')

# COMMAND ----------

# MAGIC %md
# MAGIC ##### 5F. Histogram of time vs. count of posts
# MAGIC 
# MAGIC This graph has shown the distribution of number of Reddit post in time hour of day. Each subreddit has the same sum of post, 20000. Similiar to the pattern of plot in 5B, the post of all three subreddits peaks at afternoon and reach bottom at morning. Republican subreddit has much higher post than other 2 subreddits from 12:00 to 16:00 while democrats subreddits has highest number of posts from 22:00 to 4:00.

# COMMAND ----------

sns.set_theme(style="whitegrid", palette="Set1")
sns.barplot(
    data=sub_all_cnt, 
    #kind="bar",
    x="hour", 
    y="counts", 
    hue="subreddit").set(title='Distribution of Counts of Post against Time Hour of Subreddits')
plt.show()
plt.savefig('data/plots/fig6.png')

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC The graph of number of comments verus time alone cannot well answer the buisiness question 1 and 2, 'How will we measure “engagement” on Reddit? In what ways do people interact with a post?' and 'When are the best times to post on Reddit to get the most engagement?', even though number of comments is a good indicator of engagement on Reddit, since the number of comments is related to number of posts. In order to avoid this issue, here creates a new variable the ratio of count of posts to the number of total comments at specific time hour and generates the plot below. All the three subreddits have similiar fluctuations on the ratio among the day. They all have relatively higher ratio from 11:00 to 17:00 and relatively lower ratio from 5:00 to 9:00. The answer to these business question is the best time to post on Reddit to get the most engagement is around afternoon time. 

# COMMAND ----------

sns.set_theme(style="whitegrid", palette="Set1")
sns.barplot(
    data=sub_all_cnt, 
    x="hour", 
    y="comments_post_ratio", 
    hue="subreddit").set(title='Number of Comments and Number of Post Ratio VS Time Hour of Subreddits')
plt.show()
plt.savefig('data/plots/fig7.png')


# COMMAND ----------

# MAGIC %md
# MAGIC ### Task 7: Load in external dataset.
# MAGIC 
# MAGIC Paper with metadata: https://paperswithcode.com/paper/liar-liar-pants-on-fire-a-new-benchmark
# MAGIC Data source: https://github.com/tfs4/liar_dataset
# MAGIC 
# MAGIC ##### One of our business questions explores the existence of fraudulent text in different subreddits. Our plan with the LIAR dataset is to train a fraudulent text detection ML model and then predict fake text labels on our Reddit data with this model.

# COMMAND ----------

## Define function to read in and convert .tsv file to .csv file
def parse_tsv(data_file):
    """
    Reads a tab-separated tsv file and returns
    texts: list of texts (sentences)
    labels: list of labels (fake or real news)
    """
    labels = []
    texts = []
    subjects = []
    speakers = []
    parties = []

    with open(data_file, 'r') as dd:
        for line in dd:
            fields = line.strip().split("\t")
            labels.append(fields[1])
            texts.append(fields[2])
            subjects.append(fields[3])
            speakers.append(fields[4])
            parties.append(fields[7])

    return labels, texts, subjects, speakers, parties
 

## Load texts and labels
train_data = "data/csv/liar_data/train.tsv"
test_data = "data/csv/liar_data/test.tsv"
val_data = "data/csv/liar_data/valid.tsv"
train_labels, train_texts, train_subjects, train_speakers, train_parties = parse_tsv(train_data)
test_labels, test_texts, test_subjects, test_speakers, test_parties = parse_tsv(test_data)
val_labels, val_texts, val_subjects, val_speakers, val_parties = parse_tsv(val_data)

# Combine into dataframes
liar_train = pd.DataFrame(list(zip(train_labels, train_texts, train_subjects, train_speakers, train_parties)), columns = ['label', 'text', 'subject', 'speaker', 'political_party'])
liar_test = pd.DataFrame(list(zip(test_labels, test_texts, test_subjects, test_speakers, test_parties)), columns = ['label', 'text', 'subject', 'speaker', 'political_party'])
liar_val = pd.DataFrame(list(zip(val_labels, val_texts, val_subjects, val_speakers, val_parties)), columns = ['label', 'text', 'subject', 'speaker', 'political_party'])
