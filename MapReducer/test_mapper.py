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
    for char in '.,!"\'?-:;()[]':
        clean_text = clean_text.replace(char, ' ')
        
    words = clean_text.split()
    for w in words:
        if not w: continue
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