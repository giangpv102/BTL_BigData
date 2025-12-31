from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("ViewResults").master("yarn").getOrCreate()
spark.sparkContext.setLogLevel("ERROR") # Giảm bớt log rác

print(">>> Đang đọc Model từ HDFS...")
model_path = "hdfs:///project/sentiment/output/naive_bayes_model"
model = PipelineModel.load(model_path)

# Lưu ý: Phải dùng multiLine=True như lúc train
data_path = "hdfs:///project/sentiment/input/IMDB_Dataset.csv"
df = spark.read.csv(data_path, header=True, inferSchema=True, multiLine=True, escape="\"")

predictions = model.transform(df)

print("\n" + "="*50)
print("KẾT QUẢ DỰ ĐOÁN (Label 1.0 = Positive, 0.0 = Negative)")
print("="*50)

predictions.select("review", "prediction",'sentiment').show(20, truncate=50) 
spark.stop()