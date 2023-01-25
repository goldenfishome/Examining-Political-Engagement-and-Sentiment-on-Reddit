# Databricks notebook source
# MAGIC %md
# MAGIC # Machine Learning Models
# MAGIC 
# MAGIC # Project Group #27
# MAGIC #### Clara Richter, Elise Rust, Yujia Jin
# MAGIC ##### ANLY 502
# MAGIC ##### Project Deliverable #3
# MAGIC #####Dec 5, 2022
# MAGIC 
# MAGIC The original dataset for this notebook is described in [The Pushshift Reddit Dataset](https://arxiv.org/pdf/2001.08435.pdf) paper.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Review of the ML Business Questions
# MAGIC 
# MAGIC 1) Predict the state of the economy from our dataset using regression
# MAGIC 2) Classification of subreddits
# MAGIC 
# MAGIC ##### To answer these questions we will build a few supervised Machine Learning Models:
# MAGIC 
# MAGIC 1) Decision Tree and Gradient Boosted Tree Regression --> KPIs regression
# MAGIC 2) Random Forest --> subreddit classification
# MAGIC 
# MAGIC ##### Compare performance of two models/hyperparameter sets for each analysis

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer, IndexToString, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressionModel, GBTRegressor
from sklearn.ensemble import RandomForestRegressor
from pyspark.ml.linalg import Vectors
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from sklearn.metrics import confusion_matrix # Confusion Matrix
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, Model
from pyspark.sql import SparkSession
import pandas as pd

# COMMAND ----------

### Only run ONCE --> Change directory to root in order to load data and save plots, tables, models
import os
os.chdir("../..") # Move from code/ml up to root --> up two levels
print(os.getcwd())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Load in training and testing data
# MAGIC 
# MAGIC Data previously cleaned in *code/ml/ML_Data_Prep.ipynb*
# MAGIC * title_KPIs = dataframe from submissions with vectorized text data + KPIs labels
# MAGIC * com_subred = dataframe from comments with vectorized text data + subreddit label

# COMMAND ----------

## Read in cleaned ML dataframes
## 1) dataset for predicting KPIs from posts
title_KPIs = spark.read.parquet("dbfs:/tmp/out/title_KPI.parquet")

## 2) Dataset for predicting subreddits from commments
com_subred = spark.read.parquet("dbfs:/tmp/out/com_subred1.parquet")

## 3) Dataset for predicting subreddits from titles
title_subred = spark.read.parquet("dbfs:/tmp/out/title_subred_data.parquet")

# COMMAND ----------

title_KPIs.show(5)

# COMMAND ----------

com_subred.show(5)

# COMMAND ----------

title_subred.show(5)

# COMMAND ----------

### Ensure we're training on a balanced dataset --> equal numbers of each class to not skew the classification
com_subred.groupBy("subreddit").count().show()

# COMMAND ----------

title_subred.groupBy("subreddit").count().show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1) Regression to predict KPIs from Reddit Text
# MAGIC * Decision Tree
# MAGIC * Gradient-Boosted Tree Regression
# MAGIC 
# MAGIC Of the KPIs (DOW, CPI, and Unemployment Rate), we've chosen to focus on the DOW index given the intertwining of the stock market and politics, and preliminary findings from our NLP analysis. DOW is measured by month and year, thus is a continuous value meriting regression analysis. Some texts will have the same DOW, depending on the time posted, but we opted to use regression rather than classification given the high number of unique DOW values in the dataset.

# COMMAND ----------

## Print schema of election data
title_KPIs.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split data into training and testing

# COMMAND ----------

#title_KPIs_small = title_KPIs.sample(0.01)
title_DOW = title_KPIs.select(['DOW','features'])

# COMMAND ----------

train_data, test_data = title_DOW.randomSplit([0.8, 0.2], 24)

print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build the model
# MAGIC * Note: The features in this subset don't need further ML transformations (i.e. OneHotEncoding) as they are already type float. 
# MAGIC * No features need to be converted to categorical variables either as we are doing regression.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Decision Tree Regression Model
# MAGIC 
# MAGIC Hyperparameters:
# MAGIC * Impurity = 'variance' (only option)
# MAGIC * maxDepth = 10 
# MAGIC * maxBins = 25

# COMMAND ----------

# Initialize Decision Tree multi-output classifier
dt = DecisionTreeRegressor(impurity='variance', labelCol="DOW", maxDepth=10, maxBins = 25, featuresCol="features")

# COMMAND ----------

## Train model on training data
dt_model = dt.fit(train_data)
#dt_model = dt_model.transform(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Visualize Feature Importance

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt


# plot feature importance
token = []
importance_score = []

for i, val in enumerate(dt_model.featureImportances):
    token.append(i)
    importance_score.append(val)

feature_df = pd.DataFrame({'token':token, 'importance_score':importance_score})#.sort_value('importance_score')
feature_df = feature_df.nlargest(10, 'importance_score')
feature_df['token'] = pd.Categorical(feature_df.token)

feature_df.plot(kind='barh')
plt.xlabel('Feature Importance Score')
plt.ylabel('Token')
plt.title('Ten Most Significant Tokens in DT Regression')
plt.show()
plt.savefig("data/plots/ml/DT_Tokens.png") 

# COMMAND ----------

# Predict DOW values for testing data --> using the testing data for hyperparameter tuning
results = dt_model.transform(test_data)
results.show()

# Prepare dataframe for metrics evaluation
# Rename "DOW" to "label"
results = results.withColumnRenamed("DOW", "label")
results = results.select("features", "label", "prediction")

# COMMAND ----------

results_pd = results.toPandas()
plt.scatter(results_pd.label,results_pd.prediction)
plt.xlim((29800, 36010)) # restricts x axis 
plt.ylim((29800, 36010)) # restricts y axis 
plt.plot([29800, 36010], [29800, 36010], color="red") # plots line y = x
plt.xlabel('Test DOW')
plt.ylabel('Predicted DOW')
plt.title('Test VS Predicted DOW Values in DT Regression')
plt.show()
plt.savefig("data/plots/ml/DT_Result.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate Model Performance
# MAGIC 
# MAGIC - Run model evaluation metrics (R2, MSE, RMSE, MAE) for regression
# MAGIC - Evaluate models using at least 2 different metrics and compare and interpret results

# COMMAND ----------

# Initialize evaluator object
evaluator = RegressionEvaluator()

## 1) Get R^2
print("R^2: ")
print(evaluator.evaluate(results, {evaluator.metricName: "r2"}))

## 2) Get MSE
print("MSE: ")
print(evaluator.evaluate(results,{evaluator.metricName: "mse"}))

## 3) Get Root Mean Squared Error
print("RMSE: ")
print(evaluator.evaluate(results, {evaluator.metricName: "rmse"}))

## 4) Get Mean Absolute Error
print("MAE: ")
print(evaluator.evaluate(results,
{evaluator.metricName: "mae"})
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Gradient-Boosted Tree Regression
# MAGIC Hyperparameters:
# MAGIC * impurity: 'variance'
# MAGIC * lossType: 'squared'
# MAGIC * maxIter: 10 iterations
# MAGIC * stepSize: 0.15 (a.k.a. learning rate)
# MAGIC * maxDepth: 10

# COMMAND ----------

# Initialize Gradient Boosted Tree Regression Model
gb = GBTRegressor(labelCol="DOW", featuresCol="features", impurity="variance", lossType="squared", maxIter=10, stepSize=0.15, maxDepth=10)

# COMMAND ----------

## Train model on training data
gb_model = gb.fit(train_data)

# COMMAND ----------

# View key parameters of the model
print("Impurity: ", gb_model.getImpurity()) # Get impurity
print("Feature Subset Strategy: ", gb_model.getFeatureSubsetStrategy())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Visualize Feature Importance

# COMMAND ----------

title_DOW.select("features").show()

# COMMAND ----------

import matplotlib.pyplot as plt

# plot feature importance
token = []
importance_score = []

for i, val in enumerate(gb_model.featureImportances):
    token.append(i)
    importance_score.append(val)

feature_df = pd.DataFrame({'token':token, 'importance_score':importance_score})#.sort_value('importance_score')
feature_df = feature_df.nlargest(10, 'importance_score')
feature_df['token'] = pd.Categorical(feature_df.token)

feature_df.plot(kind='barh')
plt.xlabel('Feature Importance Score')
plt.ylabel('Token')
plt.title('Ten Most Significant Tokens in GB Regression')
plt.show()
plt.savefig("data/plots/ml/GB_Tokens.png")

# COMMAND ----------

# Predict DOW values for testing data --> using the testing data for hyperparameter tuning
results_gb = gb_model.transform(test_data)
results_gb.show()

# Prepare dataframe for metrics evaluation
# Rename "DOW" to "label"
results_gb = results_gb.withColumnRenamed("DOW", "label")
results_gb = results_gb.select("features", "label", "prediction")

# COMMAND ----------

results_gb_pd = results_gb.toPandas()
plt.scatter(results_gb_pd.label,results_gb_pd.prediction)
plt.xlim((29800, 36010)) # restricts x axis 
plt.ylim((29800, 36010)) # restricts y axis 
plt.plot([29800, 36010], [29800, 36010], color="red") # plots line y = x
plt.xlabel('Test DOW')
plt.ylabel('Predicted DOW')
plt.title('Test VS Predicted DOW in GB Regression')
plt.show()
plt.savefig("data/plots/ml/GB_Results.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Evaluate Model Performance
# MAGIC 
# MAGIC - Run model evaluation metrics (R2, MSE, RMSE, MAE) for regression
# MAGIC - Evaluate models using at least 2 different metrics and compare and interpret results

# COMMAND ----------

# Initialize evaluator object
evaluator2 = RegressionEvaluator()

## 1) Get R^2
print("R^2: ")
print(evaluator2.evaluate(results_gb, {evaluator2.metricName: "r2"}))

## 2) Get MSE
print("MSE: ")
print(evaluator2.evaluate(results_gb,{evaluator2.metricName: "mse"}))

## 3) Get Root Mean Squared Error
print("RMSE: ")
print(evaluator2.evaluate(results_gb, {evaluator2.metricName: "rmse"}))

## 4) Get Mean Absolute Error
print("MAE: ")
print(evaluator2.evaluate(results_gb,
{evaluator2.metricName: "mae"})
)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparison Across Regression Models --> Table #1

# COMMAND ----------

# Combine Regression Evaluation outcomes into a Table
r2 = [evaluator.evaluate(results, {evaluator.metricName: "r2"}), evaluator2.evaluate(results_gb, {evaluator2.metricName: "r2"})] 
mse = [evaluator.evaluate(results, {evaluator.metricName: "mse"}), evaluator2.evaluate(results_gb, {evaluator2.metricName: "mse"})]
rmse = [evaluator.evaluate(results, {evaluator.metricName: "rmse"}), evaluator2.evaluate(results_gb, {evaluator2.metricName: "rmse"})]
mae = [evaluator.evaluate(results, {evaluator.metricName: "mae"}), evaluator2.evaluate(results_gb, {evaluator2.metricName: "mae"})]
models=["Decision Tree", "Gradient Boost"]

# Combine into Table #1
regression_comp = pd.DataFrame(list(zip(r2, mse, rmse, mae)), columns=["R2", "MSE", "RMSE", "MAE"])
regression_comp.index = models

print(regression_comp)

# COMMAND ----------

## Save comparison table to data/csvs/ml
regression_comp.to_csv("data/csv/ml/regression_comparison.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2) Classification of subreddits using Random Forest Classifier
# MAGIC 
# MAGIC ##### NOTE: We are using a pipeline for extra credit. This model/analysis is also where it actually makes sense to use ML transformations like StringIndexer and OneHotEncoding so we use those as well.

# COMMAND ----------

# MAGIC %md
# MAGIC #### a) Subreddits with Comments

# COMMAND ----------

com_subred.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split data into training and testing

# COMMAND ----------

train_data, test_data = com_subred.randomSplit([0.8, 0.2], 24)

print("Number of training records: " + str(train_data.count()))
print("Number of testing records : " + str(test_data.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build the model

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ML Transformations -->
# MAGIC * 1) StringIndexer
# MAGIC * 2) OneHotEncoding

# COMMAND ----------

# String indexer --> for the label (subreddit)
stringIndexer_subreddit = StringIndexer(inputCol="subreddit", outputCol="subreddit_ix")

# COMMAND ----------

# Examine the labels
stringIndexer_subreddit = stringIndexer_subreddit.fit(com_subred)

print(stringIndexer_subreddit.labels)

# COMMAND ----------

# One hot encoding to convert our label subreddit with more than two levels
onehot_subreddit = OneHotEncoder(inputCol="subreddit_ix", outputCol="subreddit_vec")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initialize Random Forest Model:
# MAGIC 
# MAGIC ##### Three Hyperparameter Sets:
# MAGIC * Set 1: numTrees=1,000, impurity='gini', maxDepth=10, minInfoGain=5, seed=42
# MAGIC * Set 2: numTrees=500, impurity='entropy', maxDepth=NA, minInfoGain=10, seed=5

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter Set #1

# COMMAND ----------

# Instantiate Random Forest Classifier #1
rf_1 = RandomForestClassifier(labelCol="subreddit_ix", featuresCol="features", numTrees=1000, impurity='gini', seed=42)

# COMMAND ----------

labelConverter = IndexToString(inputCol="prediction", 
                               outputCol="predictedSubreddit",
                               #labels=["r/politics", "r/republican", "r/democrats"]
                               labels=['Republican', 'democrats', 'politics'])

# COMMAND ----------

# Initialize pipeline! Combine the string indexer, one hot encoding, random forest model, and label converter into one sequential pipeline
pipeline_rf = Pipeline(stages=[stringIndexer_subreddit, 
                               onehot_subreddit,
                               rf_1, labelConverter])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train Model --> Pipeline for extra credit

# COMMAND ----------

# First, fit the pipeline to the data. This step calculates the transformations, while transform() actually applies them to return a transformed dataframe
model_rf1 = pipeline_rf.fit(train_data)
model_rf1.transform(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate Model Performance
# MAGIC 
# MAGIC - Run model evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
# MAGIC - Evaluate models using at least 2 different metrics and compare and interpret results

# COMMAND ----------

## Predict new labels
predictions = model_rf1.transform(test_data)

# COMMAND ----------

# Show output prediction dataframe
predictions.show()

# COMMAND ----------

# Subset only relevant columns to examine the model outcomes
predictions_sub = predictions.select("subreddit_ix", "prediction")
predictions_sub

# COMMAND ----------

# Look at prediction distribution to ensure there's no biased skew
predictions_sub.groupBy("prediction").count().show()

# COMMAND ----------

### Build a class to visualize a ROC Curve
from pyspark.mllib.evaluation import BinaryClassificationMetrics

class CurveMetrics(BinaryClassificationMetrics):
    def __init__(self, *args):
        super(CurveMetrics, self).__init__(*args)

    def _to_list(self, rdd):
        points = []
        for row in rdd.collect():
            # Results are returned as type scala.Tuple2, 
            # which doesn't appear to have a py4j mapping
            points += [(float(row._1()), float(row._2()))]
        return points

    def get_curve(self, method):
        rdd = getattr(self._java_model, method)().toJavaRDD()
        return self._to_list(rdd)

# COMMAND ----------

import matplotlib.pyplot as plt
# Create a Pipeline estimator and fit on train DF, predict on test DF
#model = estimator.fit(train)
predictions = model_rf1.transform(test_data)

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('subreddit_ix','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['subreddit_ix'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('ROC for RF Binary Classification - Hyperparameter Set #1')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x_val, y_val)

# COMMAND ----------

## 1) Accuracy
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)
print("Accuracy of the Random Forest classification model is: ", accuracy)

## 2) Precision
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluatorRF.evaluate(predictions)
print("Precision of the Random Forest classification model is: ", precision)

## 3) Recall
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedRecall")
recall = evaluatorRF.evaluate(predictions)
print("Recall of the Random Forest classification model is: ", recall)

## 4) F-1 Score
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="f1")
f1 = evaluatorRF.evaluate(predictions)
print("F1 score of the Random Forest classification model is: ", f1)

# COMMAND ----------

## 5) Confusion Matrix

y_pred=predictions.select("prediction").collect()
y_orig=predictions.select("subreddit_ix").collect()

cm = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm)

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

labels=['Republican', 'democrats', 'politics']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: RF Binary Classification")
plt.show()
plt.savefig("data/plots/ml/RF_CM1.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter Set #2
# MAGIC * numTrees=10, impurity='entropy', maxDepth=NA, minInfoGain=10, seed=5

# COMMAND ----------

# Initialize new RF with new hyperparameter set
rf_2 = RandomForestClassifier(labelCol="subreddit_ix", featuresCol="features", numTrees=500, impurity='entropy', seed=5)

# COMMAND ----------

# Re-initialize pipeline with new RF model
pipeline_rf_2 = Pipeline(stages=[stringIndexer_subreddit, 
                               onehot_subreddit,
                               rf_2, labelConverter])

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train Model --> Pipeline for extra credit

# COMMAND ----------

# Fit the pipeline to the data. This step calculates the transformations, while transform() actually applies them to return a transformed dataframe
model_rf_2 = pipeline_rf_2.fit(train_data)
model_rf_2.transform(train_data)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate Model Performance
# MAGIC 
# MAGIC - Run model evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
# MAGIC - Evaluate models using at least 2 different metrics and compare and interpret results

# COMMAND ----------

## Predict new labels
predictions_2 = model_rf_2.transform(test_data)

# COMMAND ----------

### ROC Curve for Hyperparameter Set #2 --> Compare ROC Curves across two models

# Returns as a list (false positive rate, true positive rate)
preds = predictions_2.select('subreddit_ix','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['subreddit_ix'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val2 = [x[0] for x in points]
y_val2 = [x[1] for x in points]
plt.title('ROC For Random Forest Binary Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x_val, y_val, label="Hyperparameter Set #1") # From hyperparameter set #2
plt.plot(x_val2, y_val2, label="Hyperparameter Set #2")
plt.legend()
plt.show()
#plt.savefig("data/plots/ml/ROC.png")

# COMMAND ----------

## 1) Accuracy
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="accuracy")
accuracy_2 = evaluatorRF.evaluate(predictions_2)
print("Accuracy of the Random Forest classification model is: ", accuracy_2)

## 2) Precision
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedPrecision")
precision_2 = evaluatorRF.evaluate(predictions_2)
print("Precision of the Random Forest classification model is: ", precision_2)

## 3) Recall
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedRecall")
recall_2 = evaluatorRF.evaluate(predictions_2)
print("Recall of the Random Forest classification model is: ", recall_2)

## 4) F-1 Score
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="f1")
f1_2 = evaluatorRF.evaluate(predictions_2)
print("F1 score of the Random Forest classification model is: ", f1_2)

# COMMAND ----------

## 2) Confusion Matrix
y_pred=predictions_2.select("predictedSubreddit").collect()
y_orig=predictions_2.select("subreddit").collect()

cm_2 = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm_2)

# COMMAND ----------

labels=['Republican', 'democrats', 'politics']
disp = ConfusionMatrixDisplay(confusion_matrix=cm_2, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: RF Binary Classification")
plt.show()
plt.savefig("data/plots/ml/RF_CM2.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparison Across Classification Hyperparameter Sets --> Table #2

# COMMAND ----------

### Get table with accuracy, precision, recall, F1 for each model
accuracies = [accuracy, accuracy_2] 
recalls = [recall, recall_2]
precisions = [precision, precision_2]
f1s = [f1, f1_2]
models=["Hyperparameter Set #1", "Hyperparameter Set #2"]

# Combine into Table #1
classification_comp = pd.DataFrame(list(zip(accuracies, recalls, precisions, f1s)), columns=["Accuracy", "Recall", "Precision", "F1"])
classification_comp.index = models

print(classification_comp)

# COMMAND ----------

## Save table to data/csvs/ml/
classification_comp.to_csv("data/csv/ml/classification_comparison.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC #### b) Subreddits with titles

# COMMAND ----------

title_subred.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Split data into training and testing

# COMMAND ----------

train_data_title, test_data_title = title_subred.randomSplit([0.8, 0.2], 24)

print("Number of training records: " + str(train_data_title.count()))
print("Number of testing records : " + str(test_data_title.count()))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Build the model

# COMMAND ----------

# MAGIC %md
# MAGIC ##### ML Transformations -->
# MAGIC * 1) StringIndexer
# MAGIC * 2) OneHotEncoding

# COMMAND ----------

# String indexer --> for the label (subreddit)
stringIndexer_subreddit = StringIndexer(inputCol="subreddit", outputCol="subreddit_ix")

# COMMAND ----------

# Examine the labels
stringIndexer_subreddit = stringIndexer_subreddit.fit(title_subred)

print(stringIndexer_subreddit.labels)

# COMMAND ----------

# One hot encoding to convert our label subreddit with more than two levels
onehot_subreddit = OneHotEncoder(inputCol="subreddit_ix", outputCol="subreddit_vec")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Initialize Random Forest Model:
# MAGIC 
# MAGIC ##### Three Hyperparameter Sets:
# MAGIC * Set 1: numTrees=1,000, impurity='gini', maxDepth=10, minInfoGain=5, seed=42
# MAGIC * Set 2: numTrees=500, impurity='entropy', maxDepth=NA, minInfoGain=10, seed=5

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter Set #1

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train Model --> Pipeline for extra credit

# COMMAND ----------

# First, fit the pipeline to the data. This step calculates the transformations, while transform() actually applies them to return a transformed dataframe
model_rf1 = pipeline_rf.fit(train_data_title)
model_rf1.transform(train_data_title)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate Model Performance
# MAGIC 
# MAGIC - Run model evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
# MAGIC - Evaluate models using at least 2 different metrics and compare and interpret results

# COMMAND ----------

## Predict new labels
predictions = model_rf1.transform(test_data_title)

# COMMAND ----------

# Show output prediction dataframe
predictions.show()

# COMMAND ----------

# Subset only relevant columns to examine the model outcomes
predictions_sub = predictions.select("subreddit_ix", "prediction")
predictions_sub

# COMMAND ----------

# Look at prediction distribution to ensure there's no biased skew
predictions_sub.groupBy("prediction").count().show()

# COMMAND ----------

import matplotlib.pyplot as plt
# Create a Pipeline estimator and fit on train DF, predict on test DF
#model = estimator.fit(train)
predictions = model_rf1.transform(test_data_title)

# Returns as a list (false positive rate, true positive rate)
preds = predictions.select('subreddit_ix','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['subreddit_ix'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val = [x[0] for x in points]
y_val = [x[1] for x in points]
plt.title('ROC for RF Binary Classification - Hyperparameter Set #1')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x_val, y_val)

# COMMAND ----------

## 1) Accuracy
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="accuracy")
accuracy = evaluatorRF.evaluate(predictions)
print("Accuracy of the Random Forest classification model is: ", accuracy)

## 2) Precision
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedPrecision")
precision = evaluatorRF.evaluate(predictions)
print("Precision of the Random Forest classification model is: ", precision)

## 3) Recall
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedRecall")
recall = evaluatorRF.evaluate(predictions)
print("Recall of the Random Forest classification model is: ", recall)

## 4) F-1 Score
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="f1")
f1 = evaluatorRF.evaluate(predictions)
print("F1 score of the Random Forest classification model is: ", f1)

# COMMAND ----------

## 5) Confusion Matrix

y_pred=predictions.select("prediction").collect()
y_orig=predictions.select("subreddit_ix").collect()

cm_3 = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm_3)

# COMMAND ----------

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

labels=['Republican', 'democrats', 'politics']
disp = ConfusionMatrixDisplay(confusion_matrix=cm_3, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: RF Binary Classification")
plt.show()
plt.savefig("data/plots/ml/RF_TIT1.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Hyperparameter Set #2
# MAGIC * numTrees=10, impurity='entropy', maxDepth=NA, minInfoGain=10, seed=5

# COMMAND ----------

# MAGIC %md
# MAGIC #### Train Model --> Pipeline for extra credit

# COMMAND ----------

# Fit the pipeline to the data. This step calculates the transformations, while transform() actually applies them to return a transformed dataframe
model_rf_2 = pipeline_rf_2.fit(train_data_title)
model_rf_2.transform(train_data_title)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluate Model Performance
# MAGIC 
# MAGIC - Run model evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
# MAGIC - Evaluate models using at least 2 different metrics and compare and interpret results

# COMMAND ----------

## Predict new labels
predictions_2 = model_rf_2.transform(test_data_title)

# COMMAND ----------

### ROC Curve for Hyperparameter Set #2 --> Compare ROC Curves across two models

# Returns as a list (false positive rate, true positive rate)
preds = predictions_2.select('subreddit_ix','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['subreddit_ix'])))
points = CurveMetrics(preds).get_curve('roc')

plt.figure()
x_val2 = [x[0] for x in points]
y_val2 = [x[1] for x in points]
plt.title('ROC For Random Forest Binary Classification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.plot(x_val, y_val, label="Hyperparameter Set #1") # From hyperparameter set #2
plt.plot(x_val2, y_val2, label="Hyperparameter Set #2")
plt.legend()
plt.show()
#plt.savefig("data/plots/ml/ROC.png")

# COMMAND ----------

## 1) Accuracy
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="accuracy")
accuracy_2 = evaluatorRF.evaluate(predictions_2)
print("Accuracy of the Random Forest classification model is: ", accuracy_2)

## 2) Precision
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedPrecision")
precision_2 = evaluatorRF.evaluate(predictions_2)
print("Precision of the Random Forest classification model is: ", precision_2)

## 3) Recall
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="weightedRecall")
recall_2 = evaluatorRF.evaluate(predictions_2)
print("Recall of the Random Forest classification model is: ", recall_2)

## 4) F-1 Score
evaluatorRF = MulticlassClassificationEvaluator(labelCol="subreddit_ix", predictionCol="prediction", metricName="f1")
f1_2 = evaluatorRF.evaluate(predictions_2)
print("F1 score of the Random Forest classification model is: ", f1_2)

# COMMAND ----------

## 2) Confusion Matrix
y_pred=predictions_2.select("predictedSubreddit").collect()
y_orig=predictions_2.select("subreddit").collect()

cm_4 = confusion_matrix(y_orig, y_pred)
print("Confusion Matrix:")
print(cm_4)

# COMMAND ----------

labels=['Republican', 'democrats', 'politics']
disp = ConfusionMatrixDisplay(confusion_matrix=cm_4, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix: RF Binary Classification")
plt.show()
plt.savefig("data/plots/ml/RF_CM4.png")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Comparison Across Classification Hyperparameter Sets --> Table #2

# COMMAND ----------

### Get table with accuracy, precision, recall, F1 for each model
accuracies = [accuracy, accuracy_2] 
recalls = [recall, recall_2]
precisions = [precision, precision_2]
f1s = [f1, f1_2]
models=["Hyperparameter Set #1", "Hyperparameter Set #2"]

# Combine into Table #1
classification_comp = pd.DataFrame(list(zip(accuracies, recalls, precisions, f1s)), columns=["Accuracy", "Recall", "Precision", "F1"])
classification_comp.index = models

print(classification_comp)

# COMMAND ----------

## Save table to data/csvs/ml/
classification_comp.to_csv("data/csv/ml/classification_comparison_title.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Save model output and use for inference

# COMMAND ----------

dt_model.save("tmp/dt_model.h5")

# COMMAND ----------

# Display file in DBFS
display(dbutils.fs.ls("/tmp/dt_model.h5"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Test Loading Model Back In
# MAGIC And generate predictions with data

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressionModel

test_dt = DecisionTreeRegressionModel.load("/tmp/dt_model.h5")
test_dt

# COMMAND ----------

results = test_dt.transform(test_data)
results.show()
