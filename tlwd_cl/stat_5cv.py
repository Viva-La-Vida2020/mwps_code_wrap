import os
import re
import numpy as np


# 提取最佳的val_acc值
def extract_best_accuracy(log_file):
    best_acc = None
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(r'val_acc:([0-9\.]+)', line)
            if match:
                best_acc = float(match.group(1))
    return best_acc


# 获取所有实验的得分
def get_experiment_scores(base_path):
    experiment_scores = {}

    for exp_folder in os.listdir(base_path):
        if exp_folder.startswith("Train_AsDiv-A") and os.path.isdir(os.path.join(base_path, exp_folder)):
            scores = []
            for fold in range(5):  # fold0 to fold4
                log_path = os.path.join(base_path, exp_folder, f'fold{fold}_log.txt')
                if os.path.exists(log_path):
                    best_acc = extract_best_accuracy(log_path)
                    if best_acc is not None:
                        scores.append(best_acc)
            if len(scores) == 5:  # 确保所有的fold都有
                experiment_scores[exp_folder] = np.mean(scores)

    return experiment_scores


# 设定当前路径
# base_path = os.getcwd()
base_path = 'checkpoints'
experiment_results = get_experiment_scores(base_path)

# 输出结果
for exp, avg_score in experiment_results.items():
    print(f"{exp}: {avg_score:.4f}")
