from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF
from pyspark.ml.classification import NaiveBayes
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer

spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .master("yarn") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

data_path = "hdfs:///project/sentiment/input/IMDB_Dataset.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True, multiLine=True, escape="\"")
df = df.repartition(20)
df = df.dropna()
print("Số lượng nhãn (Label) tìm thấy: ", df.select("sentiment").distinct().count())

label_indexer = StringIndexer(inputCol="sentiment", outputCol="label")

tokenizer = Tokenizer(inputCol="review", outputCol="words")

remover = StopWordsRemover(inputCol="words", outputCol="filtered")

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)
idf = IDF(inputCol="rawFeatures", outputCol="features")

nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

pipeline = Pipeline(stages=[label_indexer, tokenizer, remover, hashingTF, idf, nb])

(train_data, test_data) = df.randomSplit([0.8, 0.2], seed=42)

print("Đang huấn luyện mô hình...")
model = pipeline.fit(train_data)

predictions = model.transform(test_data)

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"Độ chính xác của mô hình: {accuracy * 100:.2f}%")

model.write().overwrite().save("hdfs:///project/sentiment/output/naive_bayes_model")

spark.stop()