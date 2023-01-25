# Databricks notebook source
# MAGIC %md
# MAGIC %md
# MAGIC # Sentiment model
# MAGIC 
# MAGIC ### Project Group #27
# MAGIC #### Clara Richter, Elise Rust, Yujia Jin
# MAGIC ##### ANLY 502
# MAGIC ##### Project Deliverable #2
# MAGIC #####Nov 22, 2022
# MAGIC 
# MAGIC ######Adapted from Lab 9 Setup

# COMMAND ----------

# Install PySpark and Spark NLP
! pip install -q pyspark==3.1.2 spark-nlp

# COMMAND ----------

import pandas as pd
import numpy as np
import json
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from sparknlp.annotator import *
from sparknlp.base import *
import sparknlp
from sparknlp.pretrained import PretrainedPipeline
import seaborn as sns

# COMMAND ----------

spark = sparknlp.start()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Pre-trained model:

# COMMAND ----------

MODEL_NAME='sentimentdl_use_twitter'

# COMMAND ----------

## Read in comments and submissions data
com_all=spark.read.parquet("/tmp/out/com_kpis.parquet")
sub_all=spark.read.parquet("/tmp/out/sub_kpis.parquet")

# COMMAND ----------

com_all.show(5)

# COMMAND ----------

# get list of text
text_list = list(com_all.select('body').toPandas()['body'])

# COMMAND ----------

# Get list of titles
titles_list = list(sub_all.select('title').toPandas()['title'])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Define Spark NLP pipleline:

# COMMAND ----------

documentAssembler = DocumentAssembler()\
    .setInputCol("text")\
    .setOutputCol("document")
    
use = UniversalSentenceEncoder.pretrained(name="tfhub_use", lang="en")\
 .setInputCols(["document"])\
 .setOutputCol("sentence_embeddings")


sentimentdl = SentimentDLModel.pretrained(name=MODEL_NAME, lang="en")\
    .setInputCols(["sentence_embeddings"])\
    .setOutputCol("sentiment")

nlpPipeline = Pipeline(
      stages = [
          documentAssembler,
          use,
          sentimentdl
      ])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the pipeline:
# MAGIC 
# MAGIC ###### First, for our subreddit comments and then for our subreddit titles.
# MAGIC 
# MAGIC We are examing these two dataframes separately as we believe they contain different types of text data and connotations. The titles of posts in the political subreddits are largely headlines from newspaper articles, and thus examining the headlines reflects the sentiment/polarization of journalists and large newspapers in the united states across different political factions. The comments of posts in the political subreddits reflect how average Reddit users and average American citizens may be engaging with various political comments. Understanding how positively or negatively different commentators on the Republican vs Democratic subreddits are engaging may reflect what types of conversations being fostered within each party's base and how heated discussions may be getting.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1) Comments

# COMMAND ----------

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

df_new = spark.createDataFrame(pd.DataFrame({"text":text_list}))
result = pipelineModel.transform(df_new)

# COMMAND ----------

result.show()

# COMMAND ----------

result2 = result.select('text', F.explode('sentiment.result'))
result2.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Add sentiment column to main dataframe:

# COMMAND ----------

# sentiment column
df_sentiment = result2.select('col')
df_sentiment = df_sentiment.withColumnRenamed("col","sentiment")
df_sentiment.show(10)

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id, row_number
from pyspark.sql import Window

#add 'sequential' index and join both dataframe to get the final result
a = com_all.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
b = df_sentiment.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

comments_sentiment = a.join(b, a.row_idx == b.row_idx).\
             drop("row_idx")
comments_sentiment.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2) Titles
# MAGIC 
# MAGIC Repeat the steps above

# COMMAND ----------

empty_df = spark.createDataFrame([['']]).toDF("text")

pipelineModel = nlpPipeline.fit(empty_df)

df_new = spark.createDataFrame(pd.DataFrame({"text":titles_list}))

result = pipelineModel.transform(df_new)

# COMMAND ----------

result.show()

# COMMAND ----------

result2 = result.select('text', F.explode('sentiment.result'))
result2.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Add sentiment column to dataframe

# COMMAND ----------

# sentiment column
df_sentiment = result2.select('col')
df_sentiment = df_sentiment.withColumnRenamed("col","sentiment")
df_sentiment.show(10)

# COMMAND ----------

#add 'sequential' index and join both dataframe to get the final result
a = sub_all.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))
b = df_sentiment.withColumn("row_idx", row_number().over(Window.orderBy(monotonically_increasing_id())))

titles_sentiment = a.join(b, a.row_idx == b.row_idx).\
             drop("row_idx")
titles_sentiment.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save data to DBFS:

# COMMAND ----------

## 1) Comments
comments_sentiment.write.parquet("/tmp/out/com_kpis_sent.parquet")

## 2) Submissions
titles_sentiment.write.parquet("/tmp/out/sub_kpis_sent.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Exploring our Business Questions
# MAGIC 
# MAGIC As a refresher from last time, the basic NLP business questions we're interested in are:
# MAGIC 1) Which subreddits produce the most/least negative discourse in their comments? In their titles?
# MAGIC 2) Which topics are correlated to the highest rates of negative sentiment and/or polarization?
# MAGIC 3) Do posts with positive, negative, or neutral sentiment receive the most engagement on Reddit?
# MAGIC 4) How does the economy influence political sentiment? When KPIs like unemployment rate are high, how does that affect the types of media and comments that are commonly posted?

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary tables and Graphs to address our business questions:

# COMMAND ----------

#### ONLY RUN ONCE: Change directory to root so that packages can load and tables/plots can be saved
import os

os.chdir('../../')

print(os.getcwd()) # Confirm the directory change was successful

# COMMAND ----------

import pandas as pd

## Read in sentment dataframes again
## 1) Comments
comments_sentiment = spark.read.parquet("dbfs:/tmp/out/com_kpis_sent.parquet")

## 2) Submissions
titles_sentiment = spark.read.parquet("dbfs:/tmp/out/sub_kpis_sent.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Polarization Data and Plot

# COMMAND ----------

###### Generate "polarization" variable 
# Comments
polarization = comments_sentiment.groupBy("subreddit", "sentiment").count()
polarization = polarization.groupBy("subreddit").pivot("sentiment").sum("count")
polarization_com = polarization.withColumn("polarization_score", (polarization['positive']-polarization['negative'])).select('subreddit','polarization_score')

comments_sentiment = comments_sentiment.join(polarization_com, on="subreddit")
comments_sentiment.show()

# Submissions
polarization = titles_sentiment.groupBy("subreddit", "sentiment").count()
polarization = polarization.groupBy("subreddit").pivot("sentiment").sum("count")
polarization_sub = polarization.withColumn("polarization_score", (polarization['positive']-polarization['negative'])).select('subreddit','polarization_score')

titles_sentiment = titles_sentiment.join(polarization_sub, on="subreddit")

# COMMAND ----------

polarization_com.show()

# COMMAND ----------

# Polarization scores --> how many more "negative" titles there were on each subreddit
polarization_sub.show()

# COMMAND ----------

# Polarization scores --> how many more "positive" titles there were on each subreddit
polarization_com_pd = polarization_com.toPandas()
polarization_com_pd

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt

plt.bar(polarization_com_pd["subreddit"], polarization_com_pd['polarization_score'])
plt.xlabel("Subreddits")
plt.ylabel('Polarization Score')
plt.title("Polarization of Subreddit Comments")
plt.show()
#plt.savefig('data/plots/polarization_comments.png', bbox_inches='tight')# Save to data/plots folder

# COMMAND ----------

polarization_sub_pd = polarization_sub.toPandas()
polarization_sub_pd

# COMMAND ----------

plt.bar(polarization_sub_pd["subreddit"], polarization_sub_pd['polarization_score'])
plt.xlabel("Subreddits")
plt.ylabel('Polarization Score')
plt.title("Polarization of Subreddit Posts")
plt.show()
#plt.savefig('data/plots/polarization_subreddit.png', bbox_inches='tight')# Save to data/plots folder

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Which subreddits produce the most/least negative discourse in their comments? In their titles?

# COMMAND ----------

subreddits_sent_df = titles_sentiment.groupBy("subreddit","sentiment").count().toPandas()
subreddits_sent_df

# COMMAND ----------

N = 3
ind = np.arange(N) 
width = 0.25
  
xvals = subreddits_sent_df.loc[subreddits_sent_df['sentiment'] == 'negative']
bar1 = plt.bar(ind, xvals['count'], width, color = 'r')

yvals = subreddits_sent_df.loc[subreddits_sent_df['sentiment'] == 'positive']
bar2 = plt.bar(ind+width, yvals['count'], width, color = 'g')

zvals = subreddits_sent_df.loc[subreddits_sent_df['sentiment'] == 'neutral']
bar3 = plt.bar(ind+width*2, zvals['count'], width, color = 'b')

plt.xlabel("Subreddits")
plt.ylabel('Frequency')
plt.title("Sentiment of Subreddit Posts")
  
plt.xticks(ind+width,['r/Republican', 'r/politics', 'r/democrats'])
plt.legend( (bar1, bar2, bar3), ('negative', 'positive', 'neutral') )
plt.plot()
#plt.savefig('data/plots/sent_subreddits.png', bbox_inches='tight')# Save to data/plots folder

# COMMAND ----------

subreddits_sent_com_df = comments_sentiment.groupBy("subreddit","sentiment").count().toPandas()
subreddits_sent_com_df

# COMMAND ----------

comments_sentiment.show()

# COMMAND ----------

N = 3
ind = np.arange(N) 
width = 0.25
  
xvals = subreddits_sent_com_df.loc[subreddits_sent_com_df['sentiment'] == 'negative']
bar1 = plt.bar(ind, xvals['count'], width, color = 'r')

yvals = subreddits_sent_com_df.loc[subreddits_sent_com_df['sentiment'] == 'positive']
bar2 = plt.bar(ind+width, yvals['count'], width, color = 'g')

zvals = subreddits_sent_com_df.loc[subreddits_sent_com_df['sentiment'] == 'neutral']
bar3 = plt.bar(ind+width*2, zvals['count'], width, color = 'b')

plt.xlabel("Subreddits")
plt.ylabel('Frequency')
plt.title("Sentiment of Subreddit Comments")
  
plt.xticks(ind+width,['r/Republican', 'r/politics', 'r/democrats'])
plt.legend( (bar1, bar2, bar3), ('negative', 'positive', 'neutral') )
#plt.savefig('data/plots/sent_subreddits_com.png', bbox_inches='tight')# Save to data/plots folder

# COMMAND ----------

# MAGIC %md 
# MAGIC ##### Do posts with positive, negative, or neutral sentiment receive the most engagement on Reddit?

# COMMAND ----------

titles_sentiment.groupBy("num_comments","sentiment").count().show()

# COMMAND ----------

import pandas as pd
num_com_sent_df = titles_sentiment.groupBy("num_comments","sentiment").count().toPandas()
score_sent_df = titles_sentiment.groupBy("score","sentiment").count().toPandas()

# COMMAND ----------

num_com_sent_df

# COMMAND ----------

score_sent_df

# COMMAND ----------

bins = [-1, 5, 10, 20, 50, 100, np.inf]
names = ['<5', '5-10', '10-20', '20-50', '50-100','100+']

score_sent_df['scoreRange'] = pd.cut(score_sent_df['score'], bins, labels=names)
score_sent_df

# COMMAND ----------

score_sent_df_gb = score_sent_df.groupby(['scoreRange','sentiment'])['count'].sum().to_frame().reset_index()
score_sent_df_gb.to_csv("data/csv/nlp/table1.csv") # Save to data/csvs
score_sent_df_gb

# COMMAND ----------

num_com_sent_df['comRange'] = pd.cut(num_com_sent_df['num_comments'], bins, labels=names)
num_com_sent_df

# COMMAND ----------

num_com_sent_df_gb = num_com_sent_df.groupby(['comRange','sentiment'])['count'].sum().to_frame().reset_index()
num_com_sent_df_gb

# COMMAND ----------

import seaborn as sns
sns.barplot(data = score_sent_df_gb, 
                x = 'scoreRange', 
                y = 'count', 
                hue = 'sentiment',
                hue_order = ['negative', 'positive', 'neutral'],
               palette=['red','green','blue'])
FS = 18
plt.xlabel('Score of post', fontsize=FS)
plt.ylabel('Sum of Frequency', fontsize=FS)
plt.title("Sentiment of engament")
plt.show()
#plt.savefig('data/plots/sent_score.png', bbox_inches='tight')# Save to data/plots folder

# COMMAND ----------

sns.barplot(data = num_com_sent_df_gb, 
                x = 'comRange', 
                y = 'count', 
                hue = 'sentiment',
                hue_order = ['negative', 'positive', 'neutral'],
               palette=['red','green','blue'])
FS = 18
plt.xlabel('Number of comments for post', fontsize=FS)
plt.ylabel('Sum of Comment Frequency', fontsize=FS)
plt.title("Sentiment of engament")
plt.show()
#plt.savefig('data/plots/sent_comments.png', bbox_inches='tight')# Save to data/plots folder

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Which topics are correlated to the highest rates of negative sentiment and/or polarization?

# COMMAND ----------

comments_sentiment.groupBy("subreddit","dummy_police", "dummy_healthcare", "dummy_climate","dummy_economy","sentiment").count().show()

# COMMAND ----------

comments_sentiment.groupBy("dummy_police","sentiment").count().show()

# COMMAND ----------

## Prepare data for plotting
com_df = comments_sentiment.toPandas()
com_df = com_df.melt(id_vars="sentiment", value_vars=["dummy_police", "dummy_healthcare", "dummy_climate", "dummy_economy"], var_name="topic")
com_df = com_df[com_df.value != False]

# COMMAND ----------

com_df = com_df.groupby(["sentiment", "topic"]).count()
com_df = com_df.reset_index(level=["sentiment", "topic"])

com_df

# COMMAND ----------

# Create summary table #2
table2 = com_df.pivot(index="topic", columns="sentiment", values="value")
table2 = table2.reset_index()
table2["% negative"] = (table2["negative"]/(table2["negative"]+table2["neutral"]+table2["positive"]))*100
table2["% positive"] = (table2["positive"]/(table2["negative"]+table2["neutral"]+table2["positive"]))*100
table2["topic"] = table2["topic"].str[6:]
table2.to_csv("data/csv/nlp/table2.csv") # save to Git
table2

# COMMAND ----------

N = 4
ind = np.arange(N) 
width = 0.25
  
xvals = com_df.loc[com_df['sentiment'] == 'negative']
bar1 = plt.bar(ind, xvals['value'], width, color = 'r')

yvals = com_df.loc[com_df['sentiment'] == 'positive']
bar2 = plt.bar(ind+width, yvals['value'], width, color = 'g')

zvals = com_df.loc[com_df['sentiment'] == 'neutral']
bar3 = plt.bar(ind+width*2, zvals['value'], width, color = 'b')

plt.xlabel("Topics")
plt.ylabel('# of Posts of a Given Sentiment Label')
plt.title("Sentiment of Subreddit Comments by Topic")
  
plt.xticks(ind+width,['Climate', "Economy", "Healthcare", "Police"])
plt.legend( (bar1, bar2, bar3), ('negative', 'positive', 'neutral') )
plt.tight_layout()
#plt.savefig('data/plots/sent_topics_com.png')# Save to data/plots folder
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### How does the economy influence political sentiment? When KPIs like unemployment rate are high, how does that affect the types of media and comments that are commonly posted?

# COMMAND ----------

comments_sentiment.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Time Series Plot

# COMMAND ----------

#### Creating time series plot
date_sentiment = comments_sentiment.toPandas().groupby(["date", "sentiment"]).size().to_frame().reset_index()
date_sentiment.columns = ["date", "sentiment", "count"]

## Normalize data to 0-1 scale
date_sentiment["normalized_count"] = (date_sentiment["count"]-min(date_sentiment["count"]))/(max(date_sentiment["count"])-min(date_sentiment["count"]))
date_sentiment.head()

# COMMAND ----------

##### Get normalized KPIs by date too
date_kpis = comments_sentiment.toPandas()[["date", "CPI", "DOW", "Unemp_Rate"]]
date_kpis["normalized_CPI"] = (date_kpis["CPI"]-min(date_kpis["CPI"]))/(max(date_kpis["CPI"])-min(date_kpis["CPI"]))
date_kpis["normalized_DOW"] = (date_kpis["DOW"]-min(date_kpis["DOW"]))/(max(date_kpis["DOW"])-min(date_kpis["DOW"]))
date_kpis["normalized_Unemp"] = (date_kpis["Unemp_Rate"]-min(date_kpis["Unemp_Rate"]))/(max(date_kpis["Unemp_Rate"])-min(date_kpis["Unemp_Rate"]))

date_kpis.head()

# COMMAND ----------

## Join back
date_df = pd.merge(date_sentiment, date_kpis, on="date")
date_df.set_index('date', inplace=True)
date_df.columns=["sentiment", "count", "normalized_count", "CPI", "DOW", "Unemp_Rate", "CPI (Norm)", "DOW (Norm)", "Unemployment Rate (Norm)"]
date_df

# COMMAND ----------

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm

colors = {'negative':'red', 'positive':'green', 'neutral':'blue'}
#cmp1 = cm.get_cmap('Set1', 3)# visualize with the new_inferno colormaps
cmp1 = ListedColormap(['red', 'green', 'blue'])
cmp2 = ListedColormap(['black', 'gray', 'brown'])
fig, ax = plt.subplots()

date_df.groupby('sentiment')['normalized_count'].plot(y=["negative", "neutral", "positive"], legend='True', ax=ax, title="KPIs against Sentiment of Text over Time", color=colors)
date_df.plot(y=["CPI (Norm)", "DOW (Norm)", "Unemployment Rate (Norm)"], ax=ax, colormap=cmp2, linestyle='dashed')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
FS = 18
plt.xlabel('Date', fontsize=FS)
plt.ylabel('Frequency', fontsize=FS)
plt.show()
#plt.savefig('data/plots/kpis_sentiment.png')# Save to data/plots folder

# COMMAND ----------

# MAGIC %md
# MAGIC #### Close Spark Session:

# COMMAND ----------

spark.stop()
