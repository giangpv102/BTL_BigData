import sys
import time
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Hàm nhập liệu đặc biệt để chạy được với spark-submit trên terminal
def nhap_du_lieu_tu_tty(prompt_text):
    sys.stdout.write(prompt_text)
    sys.stdout.flush()
    try:
        # Mở trực tiếp thiết bị terminal để đọc
        with open("/dev/tty") as tty:
            line = tty.readline().strip()
            return line
    except Exception:
        # Fallback nếu chạy trên IDE hoặc local thông thường
        return input(prompt_text)

# 1. Khởi tạo Spark
# Lưu ý: Khi chạy để test tương tác, nên dùng 'client' mode hoặc local
spark = SparkSession.builder \
    .appName("Interactive_Sentiment_Predictor") \
    .master("yarn") \
    .getOrCreate()

# Tắt log rác để màn hình console sạch sẽ
spark.sparkContext.setLogLevel("ERROR") 

# 2. Load Model Logistic Regression
# ĐƯỜNG DẪN NÀY PHẢI KHỚP VỚI LÚC TRAIN
model_path = "hdfs:///btl/spark/output/logistic_regression_model"
print(f"\n--- DANG TAI MODEL TU: {model_path} ---")

try:
    loaded_model = PipelineModel.load(model_path)
    print("-> Tai model thanh cong!")
except Exception as e:
    print(f"Loi load model: {e}")
    print("Hay kiem tra lai duong dan HDFS hoac chay lai file train.")
    sys.exit(1)

# [QUAN TRỌNG] Kiểm tra lại log lúc train để map đúng nhãn
# Thông thường StringIndexer sắp xếp theo tần suất xuất hiện.
# Nếu dữ liệu cân bằng, thường index 0 là class đầu tiên theo bảng chữ cái hoặc ngẫu nhiên.
# Bạn nên xem lại dòng log [INFO] Mapping nhãn ở file train.
LABELS_MAPPING = ["NEGATIVE", "POSITIVE"] 

def du_doan(cau_noi):
    # Tạo DataFrame từ input
    data = spark.createDataFrame([(cau_noi,)], ["review"])
    
    # Transform qua pipeline đã load
    result = loaded_model.transform(data)
    
    # Lấy kết quả đầu tiên
    row = result.select("prediction", "probability").collect()[0]
    
    pred_index = int(row.prediction)
    
    # Map index sang text (Positive/Negative)
    text_label = LABELS_MAPPING[pred_index] if pred_index < len(LABELS_MAPPING) else "Unknown"
    
    # Lấy độ tin cậy từ vector probability
    confidence = row.probability[pred_index]
    return text_label, confidence

# Warm-up (Chạy thử 1 lần để Spark khởi tạo các executor, lần sau sẽ nhanh hơn)
print("-> Dang khoi dong engine (Warm-up)...")
try:
    du_doan("warm up") 
    print("-> HE THONG SAN SANG!")
except Exception as e:
    print(f"Loi Warm-up: {e}")

# 3. Vòng lặp chính
print("="*60)
print(" CHUONG TRINH PHAN TICH CAM XUC LOGISTIC REGRESSION")
print(" (Go 'exit' de thoat)")
print("="*60)

while True:
    try:
        # Nhập liệu
        user_input = nhap_du_lieu_tu_tty("\nNhap review (tieng Anh): ")
        
        if not user_input or user_input.lower() in ['exit', 'quit', 'thoat']:
            print("Tam biet!")
            break
        
        t0 = time.time()
        nhan, do_tin_cay = du_doan(user_input)
        process_time = time.time() - t0
        
        # In kết quả
        print(f" --> KET QUA: {nhan}")
        print(f" --> Thoi gian xu ly: {process_time:.2f}s")
        
    except KeyboardInterrupt:
        print("\nDa dung chuong trinh.")
        break
    except Exception as e:
        print(f"Loi: {e}")

spark.stop()