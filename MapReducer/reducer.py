#!/usr/bin/env python3
import sys

NUM_FEATURES = 5000
LEARNING_RATE = 0.1
REG_PARAM = 0.01    

weights = [0.0] * NUM_FEATURES

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

# Tổng hợp Gradient từ Mapper
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

# 3. Cập nhật trọng số (Gradient Descent với L2 Regularization)
if total_count > 0:
    for i in range(NUM_FEATURES):
        # Average gradient trên toàn bộ dataset
        avg_gradient = total_gradient[i] / total_count
        
        # Ở đây ta áp dụng L2 cho tất cả feature weights trừ Bias (index 0)
        reg_term = 0.0
        if i > 0: 
            reg_term = REG_PARAM * weights[i]
            
        # Update Rule: w_new = w_old - LR * (Gradient + Regularization)
        weights[i] = weights[i] - (LEARNING_RATE * (avg_gradient + reg_term))

# 4. In trọng số mới
for i in range(NUM_FEATURES):
    print(f"{i}\t{weights[i]}")