import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 模型和数据的表现
models = ["Textual-CL", "BERT-GTS+TLWD", "BERT-GTS+Simpler"]
# 每组数据A, MathQA, C的表现
performance_data = {
    "Math23K": {
        "Textual-CL": [83.1, 83.9, 83.7, 84.3, 83.7, 84.1, 83.3],
        "BERT-GTS+TLWD": [85.1, 85.4, 84.0, 84.3, 85.6, 84.4, 84.6],
        "BERT-GTS+Simpler": [85.5, 86.0, 85.9, 86.3, 86.1, 86.4, 85.7]
    },
    "MathQA": {
        "Textual-CL": [],  # 模拟 MathQA 数据集中的 Textual-CL 数据缺失
        "BERT-GTS+TLWD": [73.9, 74.4, 75.1, 75.4, 75.5, 74.8, 75.1],
        "BERT-GTS+Simpler": [75.8, 75.7, 76, 75.6, 76, 75.6, 75.8]
    },
    "ASDiv-A": {
        "Textual-CL": [72.4, 71.5, 74.1, 73.1, 72.4, 73.1, 70.3],
        "BERT-GTS+TLWD": [75.4, 75.4, 75.5, 75.4, 76.2, 74.5, 73.9],
        "BERT-GTS+Simpler": [75.3, 74.9, 76.6, 75.4, 76.5, 76.5, 75.9]
    }
}

# 数据整理成DataFrame格式
data = []
for dataset in ["Math23K", "MathQA", "ASDiv-A"]:
    for model in models:
        if performance_data[dataset][model]:  # 只添加有数据的模型
            data.append([dataset, model] + performance_data[dataset][model])

# 创建DataFrame
df = pd.DataFrame(data, columns=["Dataset", "Model", "Performance1", "Performance2", "Performance3", "Performance4", "Performance5", "Performance6", "Performance7"])

# 将数据从宽格式转换为长格式，方便绘图
df_melted = df.melt(id_vars=["Dataset", "Model"], value_vars=["Performance1", "Performance2", "Performance3", "Performance4", "Performance5", "Performance6", "Performance7"], var_name="Performance Type", value_name="Performance")

# 创建图形，使用3个子图显示A、MathQA、C的数据集
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 绘制每个数据集的箱线图
for i, dataset in enumerate(["Math23K", "MathQA", "ASDiv-A"]):
    subset = df_melted[df_melted["Dataset"] == dataset]
    sns.boxplot(x="Dataset", y="Performance", hue="Model", data=subset, palette="Set2", width=0.4, ax=axes[i])
    axes[i].set_title(f'Model Performance on Dataset {dataset}')
    axes[i].set_xlabel('Dataset')
    if i == 0:
        axes[i].set_ylabel('Performance')

    # 设置每个子图的独立纵坐标范围
    y_min, y_max = subset["Performance"].min(), subset["Performance"].max()
    axes[i].set_ylim(y_min - 1, y_max + 1)  # 适当加一些缓冲空间

# 调整布局
plt.tight_layout()

# 展示图形
plt.savefig('variance.png', dpi=300, bbox_inches='tight')
plt.show()
