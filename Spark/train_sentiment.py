from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, StopWordsRemover, HashingTF, IDF, StringIndexer
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import time

spark = SparkSession.builder \
    .appName("Sentiment_Train_LogisticRegression") \
    .master("yarn") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
start_time = time.time()
print("--- Đang đọc dữ liệu ---")
data_path = "hdfs:///btl/data/IMDB.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True, multiLine=True, escape="\"")
df = df.dropna()

indexer = StringIndexer(inputCol="sentiment", outputCol="label")
indexer_model = indexer.fit(df)
df_indexed = indexer_model.transform(df)

labels = indexer_model.labels
print(f"\n[INFO] Mapping nhãn: 0.0 = {labels[0]}, 1.0 = {labels[1]}")

# --- Pipeline xử lý Text ---
tokenizer = Tokenizer(inputCol="review", outputCol="words")
default_stop_words = StopWordsRemover.loadDefaultStopWords("english")

sentimental_words = ["like", "love", "good", "bad", "great", "terrible", 
                     "not", "no", "nor", "but", "best", "worst", "better"]

new_stop_words = [w for w in default_stop_words if w not in sentimental_words]

remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=new_stop_words)

hashingTF = HashingTF(inputCol="filtered", outputCol="rawFeatures", numFeatures=5000)

# regParam: tham số Regularization (chống overfitting)
# elasticNetParam: 0.0 là L2 (Ridge), 1.0 là L1 (Lasso). Để 0.0 cho chạy nhanh và ổn định.
lr = LogisticRegression(maxIter=20, regParam=0.01, elasticNetParam=0.0, labelCol="label", featuresCol="rawFeatures")

pipeline = Pipeline(stages=[tokenizer, remover, hashingTF, lr])

(train_data, test_data) = df_indexed.randomSplit([0.8, 0.2], seed=42)

print("--- Đang huấn luyện mô hình Logistic Regression ---")
model = pipeline.fit(train_data)
end_time = time.time()
training_duration = end_time - start_time
predictions = model.transform(test_data)
evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"--- Độ chính xác (Accuracy): {accuracy * 100:.2f}% ---")

#Lưu model vào đường dẫn mới
output_path = "hdfs:///btl/spark/output/logistic_regression_model"
model.write().overwrite().save(output_path)

print(f"Đã lưu model tại: {output_path}")
print("\n" + "="*40)
print(f"TỔNG THỜI GIAN CHẠY SPARK: {training_duration:.2f} giây")
print("="*40 + "\n")
spark.stop()