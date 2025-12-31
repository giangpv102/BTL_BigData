#!/usr/bin/env python3
import sys
import math
import csv
import hashlib

# --- CẤU HÌNH ---
NUM_FEATURES = 10000
weights = [0.0] * NUM_FEATURES

# Danh sách từ dừng (Hardcoded để chạy nhanh trên Hadoop)
STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 
    'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 
    'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
    'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 
    'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', 
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't", 'br', 'movie', 'film'
}

try:
    with open('weights.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                weights[int(parts[0])] = float(parts[1])
except IOError:
    pass

def sigmoid(z):
    if z < -20: return 0
    if z > 20: return 1
    return 1 / (1 + math.exp(-z))

def get_features(text):
    features = {}
    # Bias term (luôn là 1)
    features[0] = 1.0
    
    clean_text = text.replace("<br />", " ").lower()
    words = clean_text.split()
    
    for w in words:
        # Làm sạch ký tự đặc biệt
        w = w.strip('.,!"\'?-:;()[]')
        if not w or len(w) < 2: continue
        if w in STOPWORDS: continue
        
        # Hash ổn định bằng MD5
        hash_object = hashlib.md5(w.encode('utf-8'))
        # Index từ 1 đến 9999
        idx = (int(hash_object.hexdigest(), 16) % (NUM_FEATURES - 1)) + 1
        features[idx] = features.get(idx, 0) + 1
    return features

# 2. Xử lý Batch Gradient
local_gradient = [0.0] * NUM_FEATURES
count = 0
reader = csv.reader(sys.stdin)

for row in reader:
    # Bỏ qua header hoặc dòng lỗi
    if len(row) < 2: continue
    if row[0] == "review" and row[1] == "sentiment": continue
    
    review_text = row[0]
    sentiment_str = row[1]
    
    if sentiment_str == "positive": label = 1.0
    elif sentiment_str == "negative": label = 0.0
    else: continue

    # Tính toán
    features = get_features(review_text)
    dot_product = 0.0
    for idx, val in features.items():
        dot_product += weights[idx] * val
        
    prediction = sigmoid(dot_product)
    error = prediction - label
    
    # Cộng dồn Gradient
    for idx, val in features.items():
        local_gradient[idx] += error * val
    count += 1

# 3. Emit kết quả
gradient_str_parts = []
for i in range(NUM_FEATURES):
    if local_gradient[i] != 0:
        gradient_str_parts.append(f"{i}:{local_gradient[i]}")

if count > 0:
    print(f"GRADIENT\t{count}\t{','.join(gradient_str_parts)}")