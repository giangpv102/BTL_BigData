#!/usr/bin/env python3
import sys
import math
import csv
import hashlib

NUM_FEATURES = 5000
weights = [0.0] * NUM_FEATURES

STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", 
    "be", "because", "been", "before", "being", "below", "between", "both", "by", "can't", "cannot", "could", 
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", 
    "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", 
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", 
    "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", 
    "my", "myself", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", 
    "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", 
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", 
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", 
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", 
    "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", 
    "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", 
    "yourselves"
}

# Load trọng số hiện tại (cho Gradient Descent)
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
    # Bias term 
    features[0] = 1.0
    
    clean_text = text.replace("<br />", " ").lower()
    
    for char in '.,!"\'?-:;()[]':
        clean_text = clean_text.replace(char, ' ')
        
    words = clean_text.split()
    
    for w in words:
        if not w: continue
        if w in STOPWORDS: continue
        
        # Hashing logic mô phỏng HashingTF
        hash_object = hashlib.md5(w.encode('utf-8'))
        # Mapping vào [1, NUM_FEATURES-1], dành index 0 cho Bias
        idx = (int(hash_object.hexdigest(), 16) % (NUM_FEATURES - 1)) + 1
        
        # TF (Term Frequency) count
        features[idx] = features.get(idx, 0) + 1
    return features

# --- Batch Gradient Calculation ---
local_gradient = [0.0] * NUM_FEATURES
count = 0
reader = csv.reader(sys.stdin)

for row in reader:
    if len(row) < 2: continue
    if row[0] == "review" and row[1] == "sentiment": continue
    
    review_text = row[0]
    sentiment_str = row[1]
    
    if sentiment_str == "positive": label = 1.0
    elif sentiment_str == "negative": label = 0.0
    else: continue

    features = get_features(review_text)
    
    # Dot product (w * x)
    dot_product = 0.0
    for idx, val in features.items():
        dot_product += weights[idx] * val
        
    prediction = sigmoid(dot_product)
    error = prediction - label
    
    # Tính Gradient: (h(x) - y) * x_j
    for idx, val in features.items():
        local_gradient[idx] += error * val
    count += 1

# Emit Gradient dạng sparse để tiết kiệm băng thông
gradient_str_parts = []
for i in range(NUM_FEATURES):
    if local_gradient[i] != 0:
        gradient_str_parts.append(f"{i}:{local_gradient[i]}")

if count > 0:
    print(f"GRADIENT\t{count}\t{','.join(gradient_str_parts)}")