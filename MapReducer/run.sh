#!/bin/bash

EPOCHS=10  
HDFS_INPUT="/btl/data/IMDB.csv"
HDFS_OUTPUT="/btl/mapreducer/output/gradient_epoch"
WEIGHTS_FILE="weights.txt"
export PYTHONHASHSEED=42

rm -f .${WEIGHTS_FILE}.crc

if [ ! -f "$WEIGHTS_FILE" ]; then
    echo "Không tìm thấy weights cũ, tạo file mới..."
    touch $WEIGHTS_FILE
else
    echo "Tìm thấy weights cũ, sẽ tiếp tục train..."
fi

for (( i=1; i<=$EPOCHS; i++ ))
do
    echo "=================================================="
    echo "Starting Epoch $i / $EPOCHS"
    echo "=================================================="
    
    # Xóa output cũ trên HDFS
    hdfs dfs -rm -r $HDFS_OUTPUT > /dev/null 2>&1

    # Chạy MapReduce
    hadoop jar $HADOOP_HOME/share/hadoop/tools/lib/hadoop-streaming-*.jar \
        -files $WEIGHTS_FILE,mapper.py,reducer.py \
        -mapper "python3 mapper.py" \
        -reducer "python3 reducer.py" \
        -input $HDFS_INPUT \
        -output $HDFS_OUTPUT

    # Kiểm tra xem Job có thành công không
    if hdfs dfs -test -e $HDFS_OUTPUT/_SUCCESS; then
        echo "Job Epoch $i thành công. Đang cập nhật weights..."
        
        # Backup file cũ
        cp $WEIGHTS_FILE ${WEIGHTS_FILE}.bak
        
        # Lấy file mới về file tạm
        hdfs dfs -getmerge $HDFS_OUTPUT weights_new.txt
        
        # Kiểm tra file mới có dữ liệu không
        if [ -s "weights_new.txt" ]; then
            mv weights_new.txt $WEIGHTS_FILE
            
            rm -f .${WEIGHTS_FILE}.crc
            
            echo "Weights đã được cập nhật."
        else
            echo "LỖI: File weights mới bị rỗng! Giữ lại weights cũ và dừng."
            rm weights_new.txt
            exit 1
        fi
    else
        echo "LỖI: Hadoop Job thất bại ở Epoch $i! Dừng chương trình."
        exit 1
    fi
done

echo "Training finished!"