import sys
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import col, udf, when
from pyspark.sql.types import FloatType

# --- CẤU HÌNH ---
# Đường dẫn dữ liệu và model (phải khớp với lúc train)
DATA_PATH = "hdfs:///btl/data/IMDB.csv"  # Hoặc file test riêng nếu có
MODEL_PATH = "hdfs:///btl/spark/output/logistic_regression_model"

# Khởi tạo Spark
spark = SparkSession.builder \
    .appName("Spark_Result_Formatter") \
    .master("yarn") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("--- Đang tải dữ liệu và model... ---")

# 1. Đọc dữ liệu (Giống logic file train)
df = spark.read.csv(DATA_PATH, header=True, inferSchema=True, multiLine=True, escape="\"")
df = df.dropna()

# 2. Xử lý nhãn thủ công để tạo cột 'label' (0.0 hoặc 1.0)
# Lưu ý: Lúc train bạn dùng StringIndexer, nhưng để đơn giản lúc test ta map cứng
# Positive -> 1.0, Negative -> 0.0
df_clean = df.withColumn("label", 
    when(col("sentiment") == "positive", 1.0)
    .when(col("sentiment") == "negative", 0.0)
    .otherwise(None)
).dropna(subset=["label"])

# 3. Load Model đã lưu
try:
    model = PipelineModel.load(MODEL_PATH)
except Exception as e:
    print(f"Lỗi: Không tìm thấy model tại {MODEL_PATH}. Hãy chạy train_sentiment.py trước!")
    sys.exit(1)

# 4. Thực hiện dự đoán
# Chia lại tập test giống file train (hoặc dùng file test riêng nếu có)
# Lưu ý: Seed phải giống hệt file train để ra đúng tập test đó
(train_data, test_data) = df_clean.randomSplit([0.8, 0.2], seed=42)

predictions = model.transform(test_data)

# --- XỬ LÝ ĐỂ IN KẾT QUẢ ---

# UDF để lấy xác suất của class 1 (Positive) từ vector probability
# Vector output của Spark là [prob_neg, prob_pos]
get_pos_prob = udf(lambda v: float(v[1]), FloatType())

# Tạo cột pos_prob để dễ xử lý
results = predictions.select("label", "prediction", "probability") \
                     .withColumn("pos_prob", get_pos_prob("probability"))

# Lấy 10 dòng đầu tiên để in bảng chi tiết
sample_rows = results.take(10)

# Tính toán Confusion Matrix (TP, TN, FP, FN)
# Sử dụng GroupBy để tính toán nhanh trên Big Data thay vì loop
metrics = results.groupBy("label", "prediction").count().collect()

tp, tn, fp, fn = 0, 0, 0, 0
for row in metrics:
    lbl = row['label']
    pred = row['prediction']
    cnt = row['count']
    
    if lbl == 1.0 and pred == 1.0: tp = cnt
    elif lbl == 0.0 and pred == 0.0: tn = cnt
    elif lbl == 0.0 and pred == 1.0: fp = cnt
    elif lbl == 1.0 and pred == 0.0: fn = cnt

total = tp + tn + fp + fn
correct = tp + tn

# --- IN RA MÀN HÌNH (ĐỊNH DẠNG GIỐNG HỆT MAPREDUCE) ---

print(f"{'PREDICTED':<10} | {'LABEL':<10} | {'POS (%)':<10} | {'NEG (%)':<10} | {'RESULT':<10}")
print("-" * 65)

# In 10 dòng mẫu
for row in sample_rows:
    actual = int(row['label'])
    predicted = int(row['prediction'])
    prob_score = row['pos_prob'] # Xác suất là Positive
    
    is_correct = (actual == predicted)
    
    pos_percent = prob_score * 100
    neg_percent = (1.0 - prob_score) * 100
    
    pred_str = "Positive" if predicted == 1 else "Negative"
    act_str = "Positive" if actual == 1 else "Negative"
    result_str = "CORRECT" if is_correct else "WRONG"
    
    print(f"{pred_str:<10} | {act_str:<10} | {pos_percent:6.2f}%   | {neg_percent:6.2f}%   | {result_str:<10}")

print("-" * 65)
print("... (Xem tiếp thống kê tổng hợp bên dưới) ...\n")

if total > 0:
    accuracy = (correct / total) * 100
    print(f"TONG SO MAU: {total}")
    print(f"CHINH XAC  : {accuracy:.2f}% ({correct}/{total})")
    print(f"CONFUSION MATRIX:")
    print(f"   TP: {tp} | FP: {fp}")
    print(f"   FN: {fn} | TN: {tn}")
else:
    print("Khong co du lieu dau ra.")

spark.stop()