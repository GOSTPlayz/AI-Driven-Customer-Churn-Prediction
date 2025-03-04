import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


spark = SparkSession.builder.appName("ChurnPrediction").getOrCreate()


data = spark.read.csv('customer_churn.csv', header=True, inferSchema=True)

# Handling missing values
data = data.na.fill({'TotalCharges': 0})

# Encoding categorical features
indexers = [StringIndexer(inputCol=col, outputCol=col+"_index").fit(data) for col in ['Gender', 'SubscriptionType']]

# Feature transformation
assembler = VectorAssembler(inputCols=['Gender_index', 'SubscriptionType_index', 'MonthlyCharges', 'TotalCharges'], outputCol='features')
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

# Define pipeline
pipeline = Pipeline(stages=indexers + [assembler, scaler])
data_prepared = pipeline.fit(data).transform(data)

# Selecting final dataset
final_data = data_prepared.select(col("scaled_features").alias("features"), col("Churn").alias("label"))


train_data, test_data = final_data.randomSplit([0.8, 0.2], seed=42)


rf = RandomForestClassifier(featuresCol='features', labelCol='label', numTrees=100)
model = rf.fit(train_data)



predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction', metricName='accuracy')
accuracy = evaluator.evaluate(predictions)
print("Accuracy:", accuracy)


model.write().overwrite().save('churn_model_spark')


import boto3
s3 = boto3.client('s3')
s3.upload_file('churn_model_spark', 'my-s3-bucket', 'churn_model_spark')

