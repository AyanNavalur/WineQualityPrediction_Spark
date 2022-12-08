from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier


def main():
    spark = SparkSession.builder.appName(
        "WineQualityPredictor").getOrCreate()

    # load the model
    rf = RandomForestClassifier.load("./models/rfModel")

    # load the data
    test_data = spark.read.option(
        "delimiter", ";").option("header", True).option("inferSchema", True).csv(f'./data/TestDataset.csv')

    # this loads the label indexer from the training script.
    label_indexer_loaded = StringIndexer.load('label_indexer')
    # label_indexer = StringIndexer().setInputCol(
    #     "quality").setOutputCol("label")
    vectorAssembler = VectorAssembler().setInputCols(
        test_data.columns[:-1]).setOutputCol("features")
    pipeline = Pipeline().setStages([label_indexer_loaded, vectorAssembler])

    transformedData = pipeline.fit(test_data).transform(
        test_data).select("features", "label")
    print(transformedData.show(5))

    # predict
    predictions = rf.transform(transformedData)
    print(predictions.show(5))

    # save the predictions
    # predictions.write.csv("./data/predictions.csv")

    # evaluation
    evaluator = MulticlassClassificationEvaluator()
    print('F1 accuracy score:', evaluator.evaluate(predictions))

    # writing accuracy to file
    with open("./accuracy.txt", "w") as f:
        f.write('F1 accuracy score:', str(
            evaluator.evaluate(predictions)))  # check this once


if __name__ == "__main__":
    main()
