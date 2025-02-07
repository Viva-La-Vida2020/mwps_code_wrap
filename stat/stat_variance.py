import matplotlib.pyplot as plt

# 数据
data = [
    [83.1, 83.9, 83.7, 84.3, 83.7, 84.1, 83.3],  # Group 1
    [85.1, 85.4, 84.0, 84.3, 85.6, 84.4, 84.6],  # Group 2
    [85.5, 86.0, 85.7, 86.1, 86.1, 85.6, 85.3]   # Group 3
]

# 绘制箱线图
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=["Textual-CL", "BERT-GTS+TLWD", "BERT-GTS+Simpler"], patch_artist=True)

# 设置图表标题和标签
plt.title("Boxplot of Three CL Methods")
plt.ylabel("Accuracy")
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 显示图表
plt.show()