#!/bin/bash

# --- CẤU HÌNH ĐƯỜNG DẪN ---
# Đường dẫn file dữ liệu test trên HDFS (có thể dùng chung file train nếu không có tập test riêng)
HDFS_INPUT="/btl/data/IMDB.csv" 
HDFS_OUTPUT="/btl/mapreducer/output/test"
WEIGHTS_FILE="weights.txt"
export PYTHONHASHSEED=42

rm -f weights.txt
hdfs dfs -getmerge /btl/mapreducer/output/gradient_epoch weights.txt

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