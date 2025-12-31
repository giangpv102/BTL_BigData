#!/usr/bin/env python3
import sys
import math
import csv
import hashlib

NUM_FEATURES = 10000
weights = [0.0] * NUM_FEATURES

# --- STOPWORDS (COPY Y HỆT TỪ MAPPER) ---
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
except IOError: pass

def sigmoid(z):
    if z < -20: return 0
    if z > 20: return 1
    return 1 / (1 + math.exp(-z))

def get_features(text):
    features = {}
    features[0] = 1.0 # Bias
    clean_text = text.replace("<br />", " ").lower()
    words = clean_text.split()
    for w in words:
        w = w.strip('.,!"\'?-:;()[]')
        if not w or len(w) < 2: continue
        if w in STOPWORDS: continue
        hash_object = hashlib.md5(w.encode('utf-8'))
        idx = (int(hash_object.hexdigest(), 16) % (NUM_FEATURES - 1)) + 1
        features[idx] = features.get(idx, 0) + 1
    return features

reader = csv.reader(sys.stdin)
for row in reader:
    if len(row) < 2: continue
    if row[0] == "review" and row[1] == "sentiment": continue

    review_text = row[0]
    sentiment_str = row[1]

    if sentiment_str == "positive": actual_label = 1
    elif sentiment_str == "negative": actual_label = 0
    else: continue

    features = get_features(review_text)
    dot_product = 0.0
    for idx, val in features.items():
        dot_product += weights[idx] * val

    probability = sigmoid(dot_product)
    predicted_label = 1 if probability >= 0.5 else 0
    
    print(f"{actual_label}\t{predicted_label}\t{probability}")