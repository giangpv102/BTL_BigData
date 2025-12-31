#!/bin/bash

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn file dữ liệu test trên HDFS (có thể dùng chung file train nếu không có tập test riêng)
HDFS_INPUT="/btl/data/IMDB.csv" 
HDFS_OUTPUT="/btl/mapreducer/output/test"
WEIGHTS_FILE="weights.txt"
export PYTHONHASHSEED=42
# --- KIỂM TRA ĐIỀU KIỆN ---
if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "LỖI: Không tìm thấy file mô hình '$WEIGHTS_FILE'."
    echo "Vui lòng chạy train.sh trước để tạo file weights."
    exit 1
fi

echo "Đang chạy kiểm thử với mô hình Logistic Regression..."

# --- XÓA OUTPUT CŨ ---
hdfs dfs -rm -r $HDFS_OUTPUT 2>/dev/null

# --- CHẠY HADOOP STREAMING ---
# Quan trọng: Phải gửi kèm file weights.txt bằng tham số -files
hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
    -files $WEIGHTS_FILE,test_mapper.py,test_reducer.py \
    -mapper "python3 test_mapper.py" \
    -reducer "python3 test_reducer.py" \
    -input $HDFS_INPUT \
    -output $HDFS_OUTPUT

# --- HIỂN THỊ KẾT QUẢ ---
echo "----------------------------------------"
echo "KẾT QUẢ KIỂM THỬ:"
echo "----------------------------------------"
hdfs dfs -cat $HDFS_OUTPUT/*