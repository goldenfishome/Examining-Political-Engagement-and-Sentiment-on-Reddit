# Databricks notebook source
# MAGIC %md
# MAGIC # Testing Loading in Pre-Trained Model for Prediction
# MAGIC 
# MAGIC A pre-trained decision tree regressor was downloaded to DBFS in code/ml/ML_Model_Building.ipynb. This file loads it back into DataBricks and generates predictions.

# COMMAND ----------

# Load packages
from pyspark.ml.regression import DecisionTreeRegressionModel

# COMMAND ----------

## Load model
test_dt = DecisionTreeRegressionModel.load("/tmp/dt_model.h5")
test_dt

# COMMAND ----------

## Load data
test_subreddit_data = spark.read.parquet("dbfs:/tmp/out/title_KPI.parquet")
test_subreddit_DOW = test_subreddit_data.select("DOW", "features")
test_subreddit_DOW.show()

# COMMAND ----------

# Generate predictions
results = test_dt.transform(test_subreddit_DOW)
results.show()
