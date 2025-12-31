#!/usr/bin/env python3
import sys

# --- CẤU HÌNH ---
NUM_FEATURES = 10000
LEARNING_RATE = 0.05
weights = [0.0] * NUM_FEATURES

# 1. Load trọng số cũ
try:
    with open('weights.txt', 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                weights[int(parts[0])] = float(parts[1])
except IOError:
    pass

total_gradient = [0.0] * NUM_FEATURES
total_count = 0

# 2. Tổng hợp Gradient từ Mapper
for line in sys.stdin:
    line = line.strip()
    parts = line.split('\t')
    
    if len(parts) < 3: continue
    if parts[0] != "GRADIENT": continue
    
    count = int(parts[1])
    grad_str = parts[2]
    
    total_count += count
    
    if grad_str:
        pairs = grad_str.split(',')
        for pair in pairs:
            idx, val = pair.split(':')
            total_gradient[int(idx)] += float(val)

# 3. Cập nhật trọng số (Gradient Descent)
if total_count > 0:
    for i in range(NUM_FEATURES):
        avg_gradient = total_gradient[i] / total_count
        weights[i] = weights[i] - (LEARNING_RATE * avg_gradient)

# 4. In trọng số mới (Sát lề trái)
for i in range(NUM_FEATURES):
    print(f"{i}\t{weights[i]}")