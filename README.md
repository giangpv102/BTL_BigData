# BTL_BigData
Bài tập lớn Big Data: xử lý dữ liệu đánh giá phim IMDB bằng **Hadoop MapReduce** và **Apache Spark**.


## 1. Prepare Dataset

File dữ liệu gốc (~50MB).  
Chạy script sau để tăng kích thước dữ liệu nhằm giả lập môi trường Big Data:

```bash
python replicate.py
```
## 2. Đẩy dữ liệu lên HDFS

### Tạo thư mục trên HDFS

```bash
hdfs dfs -mkdir /btl
hdfs dfs -mkdir /btl/data
```

### Đẩy file dữ liệu

```bash
hdfs dfs -put IMDB_Datasets.csv /btl/data
```

## 3. Chạy với MapReduce
### Train
```bash
cd MapReducer
./run.sh
```
### Test
```bash
./run_test.sh
```
## 4. Chạy với Spark 
### Train
```bash
spark-submit --master yarn --deploy-mode client train_sentiment.py
```
### Test
```bash
spark-submit result.py
```