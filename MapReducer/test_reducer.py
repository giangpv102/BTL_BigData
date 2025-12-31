#!/usr/bin/env python3
import sys

total = 0
correct = 0
tp, tn, fp, fn = 0, 0, 0, 0
print_count = 0

print(f"{'PREDICTED':<10} | {'LABEL':<10} | {'POS (%)':<10} | {'NEG (%)':<10} | {'RESULT':<10}")
print("-" * 65)

for line in sys.stdin:
    try:
        parts = line.strip().split('\t')
        if len(parts) != 3: continue
        
        actual = int(parts[0])
        predicted = int(parts[1])
        prob_score = float(parts[2])
        
        is_correct = (actual == predicted)
        total += 1
        if is_correct:
            correct += 1
            if actual == 1: tp += 1
            else: tn += 1
        else:
            if actual == 0 and predicted == 1: fp += 1
            if actual == 1 and predicted == 0: fn += 1
            
        if print_count < 10:
            pos_percent = prob_score * 100
            neg_percent = (1.0 - prob_score) * 100
            pred_str = "Positive" if predicted == 1 else "Negative"
            act_str = "Positive" if actual == 1 else "Negative"
            result_str = "CORRECT" if is_correct else "WRONG"
            
            print(f"{pred_str:<10} | {act_str:<10} | {pos_percent:6.2f}%   | {neg_percent:6.2f}%   | {result_str:<10}")
            print_count += 1
            if print_count == 10:
                 print("-" * 65)
                 print("... (Xem tiếp thống kê tổng hợp bên dưới) ...\n")

    except ValueError:
        continue

if total > 0:
    accuracy = (correct / total) * 100
    print(f"TONG SO MAU: {total}")
    print(f"CHINH XAC  : {accuracy:.2f}% ({correct}/{total})")
    print(f"CONFUSION MATRIX:")
    print(f"   TP: {tp} | FP: {fp}")
    print(f"   FN: {fn} | TN: {tn}")
else:
    print("Khong co du lieu dau ra.")