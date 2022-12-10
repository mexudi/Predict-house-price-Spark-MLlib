// Databricks notebook source
// MAGIC %md
// MAGIC # Spark MLlib Real Example
// MAGIC ## Overview
// MAGIC To give an idea of how Spark MLlib works, I will use the San Francisco housing data set provided by 
// MAGIC Airbnb. The dataset contains information about Airbnb rentals in San Francisco, such as the number 
// MAGIC of bedrooms, location, review score…. The goal is to predict the nightly rental prices for listings in 
// MAGIC that city. This is a regression problem, which means during my illustration I will use a Regression 
// MAGIC Model.
// MAGIC 
// MAGIC ## Data Ingestion and Exploration
// MAGIC Let’s take a quick peek at the data set and the corresponding schema

// COMMAND ----------

val filePath ="/FileStore/tables/part_00000_tid.parquet"
val airbnbDF = spark.read.parquet(filePath)
airbnbDF.select("neighbourhood_cleansed", "room_type", "bedrooms", "bathrooms","number_of_reviews", "price").show(5)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Training & Testing dataset
// MAGIC Before we begin feature engineering and modeling, we will divide our data set into two groups: train and test. For our Airbnb data set, we will keep 80% for the training set and set aside 20% of our data for the test set. Further, we will set a random seed=42 for reproducibility.

// COMMAND ----------

val Array(trainDF, testDF) = airbnbDF.randomSplit(Array(.8, .2), seed=42)
println(f"""There are ${trainDF.count} rows in the training set, and
${testDF.count} in the test set""")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Preparing features with transformers
// MAGIC Linear regression (like many other algorithms in Spark) requires that all the input features are contained within a single vector in your DataFrame. Thus, we need to transform our data. **Transformers** in Spark accept a DataFrame as input and return a new DataFrame with one or more columns appended to it. They do not learn from your data, but apply rule-based transformations using the **transform()** method.
// MAGIC 
// MAGIC **VectorAssembler** takes a list of input columns and creates a new DataFrame with an additional column, which we will call features. It combines the values of those input columns into a single vector

// COMMAND ----------

import org.apache.spark.ml.feature.VectorAssembler
val vecAssembler = new VectorAssembler()
 .setInputCols(Array("bedrooms"))
 .setOutputCol("features")
val vecTrainDF = vecAssembler.transform(trainDF)
vecTrainDF.select("bedrooms", "features", "price").show(10)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Using estimators to build models
// MAGIC After setting up our **vectorAssembler**, we have our data prepared and transformed into a format that our linear regression model expects. In Spark, LinearRegression is a type of estimator it takes in a DataFrame and returns a Model. **Estimators** learn parameters from your data, have an estimator_name.fit() method, and are eagerly evaluated, whereas transformers are lazily evaluated. The output of an estimator’s fit() method is a transformer. Once the estimator has learned the parameters, the transformer can apply these parameters to new data points to generate predictions.

// COMMAND ----------

import org.apache.spark.ml.regression.LinearRegression
val lr = new LinearRegression()
 .setFeaturesCol("features")
 .setLabelCol("price")
val lrModel = lr.fit(vecTrainDF)
val m = lrModel.coefficients(0)
val b = lrModel.intercept
println(f"""The formula for the linear regression line is
price = $m%1.2f*bedrooms + $b%1.2f""")

// COMMAND ----------

// MAGIC %md
// MAGIC ## Creating a pipeline
// MAGIC If we want to apply our model to our test set, then we need to prepare that data in the same way as the training set. It becomes cumbersome to remember not 
// MAGIC only which steps to apply, but also the 
// MAGIC ordering of the steps.
// MAGIC 
// MAGIC Since pipelineModel is a transformer, it is straightforward to apply it to our test data set too

// COMMAND ----------

import org.apache.spark.ml.Pipeline
val pipeline = new Pipeline().setStages(Array(vecAssembler, lr))
val pipelineModel = pipeline.fit(trainDF)
val predDF = pipelineModel.transform(testDF)
predDF.select("bedrooms", "features", "price", "prediction").show(10)

// COMMAND ----------

// MAGIC %md
// MAGIC ## Conclusion
// MAGIC In this code we built a model using only a single feature, bedrooms. However, you may want to build a model using all of your features, some of which may be categorical.

// COMMAND ----------


