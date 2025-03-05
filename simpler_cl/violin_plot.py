import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.lines import Line2D  # 用于自定义图例
from matplotlib.ticker import MultipleLocator


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
        "Textual-CL": [73.4, 74.2, 74.4, 74.9, 73.6, 73.8, 74.1],  # 模拟 MathQA 数据集中的 Textual-CL 数据缺失
        # "BERT-GTS+TLWD": [73.9, 74.4, 75.1, 75.4, 75.5, 74.8, 75.1],
        "BERT-GTS+TLWD": [75.3, 75.5, 75.4, 75.4, 75.6, 75.3, 74.9],
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
fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# 定义固定颜色方案，确保每个模型的颜色一致
# color_palette = {"Textual-CL": "C3", "BERT-GTS+TLWD": "C1", "BERT-GTS+Simpler": "C0"}
color_palette = {"Textual-CL": "#8FAADC",
                 "BERT-GTS+TLWD": (255/255, 230/255, 153/255),
                 "BERT-GTS+Simpler": (169/255, 209/255, 142/255)}
# color_palette = {"Textual-CL": "#EC3E31",
#                  "BERT-GTS+TLWD": "#A6D0E6",
#                  "BERT-GTS+Simpler": "#A8D3A0"}
# color_palette = {"Textual-CL": "#6CA3D4",
#                  "BERT-GTS+TLWD": "#EDD283",
#                  "BERT-GTS+Simpler": "#A5D395"}


# 绘制每个数据集的Violin Plot
for i, dataset in enumerate(["Math23K", "MathQA", "ASDiv-A"]):
    subset = df_melted[df_melted["Dataset"] == dataset]
    if i == 2:
        sns.violinplot(x="Dataset", y="Performance", hue="Model", data=subset, palette=color_palette, width=0.6, ax=axes[i], split=True, legend=True)
        axes[i].legend(loc='lower right', fontsize=10, title="Models", title_fontsize=16)
    else:
        sns.violinplot(x="Dataset", y="Performance", hue="Model", data=subset, palette=color_palette, width=0.6, ax=axes[i], split=True, legend=False)

    if i == 1:
        axes[i].yaxis.set_major_locator(MultipleLocator(1))  # 使 y 轴刻度间隔为 1（可调整）
    axes[i].set_title('')
    axes[i].set_xlabel('')
    if i == 0:
        axes[i].set_ylabel('Accuracy', fontsize=16, fontweight='bold')  # 设置Y轴字体大小
    else:
        axes[i].set_ylabel('', fontsize=16)

    # 增大纵坐标数字的字体并加粗
    axes[i].tick_params(axis='y', labelsize=16)  # 增大纵坐标数字的字体大小
    # for label in axes[i].get_yticklabels():
    #     label.set_fontweight('bold')  # 加粗纵坐标数字

    # 增大横坐标Dataset字体并加粗
    axes[i].tick_params(axis='x', labelsize=16)  # 增大横坐标Dataset字体大小
    for label in axes[i].get_xticklabels():
        label.set_fontweight('bold')  # 加粗横坐标Dataset字体


# # 创建自定义图例
# custom_legend = [Line2D([0], [0], color="C0", lw=4, label="Textual-CL"),
#                  Line2D([0], [0], color="C1", lw=4, label="BERT-GTS+TLWD"),
#                  Line2D([0], [0], color="C2", lw=4, label="BERT-GTS+Simpler")]
#
# # 设置图例，放置在外部
# plt.legend(handles=custom_legend, title="Models", loc='center', bbox_to_anchor=(0.5, 1.15), fontsize=12, title_fontsize=14)

# 调整布局
plt.tight_layout()
plt.savefig('violin_plot_v6.png', dpi=2000)
# 展示图形
plt.show()
