from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier


def main():
    spark = SparkSession.builder.appName(
        "WineQualityClassifier").getOrCreate()

    train_data = spark.read.option(
        "delimiter", ";").option("header", True).option("inferSchema", True).csv(f'./data/TrainingDataset.csv')
    val_data = spark.read.option("delimiter", ";").option("header", True).option("inferSchema", True).csv(
        f'./data/ValidationDataset.csv')

    data = train_data.union(val_data)
    print(data.show(5))

    label_indexer = StringIndexer().setInputCol(
        "quality").setOutputCol("label").fit(data)
    vectorAssembler = VectorAssembler().setInputCols(
        data.columns[:-1]).setOutputCol("features")
    pipeline = Pipeline().setStages([label_indexer, vectorAssembler])

    transformedData = pipeline.fit(data).transform(
        data).select("features", "label")
    print(transformedData.show(5))

    (trainingData, testData) = transformedData.randomSplit([0.8, 0.2])
    print("Training Data Count:", trainingData.count())
    print("Test Data Count:", testData.count())

    # Random Forest Classifier
    # rf = RandomForestClassifier(
    #     featuresCol='features', labelCol='label', maxDepth=17, numTrees=706)
    rf = RandomForestClassifier(
        featuresCol='features', labelCol='label', maxDepth=17, numTrees=7)
    rfModel = rf.fit(trainingData)
    predictions = rfModel.transform(testData)

    # evaluation
    evaluator = MulticlassClassificationEvaluator()
    print('F1 accuracy score:', evaluator.evaluate(predictions))

    # saving model
    rf.write().overwrite().save("./models/rfModel")

    # saving label indexer
    label_indexer.save('label_indexer')


if __name__ == "__main__":
    main()
