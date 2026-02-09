import pandas as pd
import numpy as np

# 1. 读取 CSV 文件
df = pd.read_csv('exp/FD/proposed/non_adapted_results.csv')

# 你的 CSV 最后两行是统计行 (mean, std)，读取时最好把它们去掉，只保留实验数据
# 假设最后两行是 summary，可以通过判断 'run' 列是否是数字来过滤
df = df[pd.to_numeric(df['run'], errors='coerce').notnull()]

# 2. 核心步骤：按 scenario 分组，计算 f1_score 的均值和标准差
# 这里的 'scenario' 对应表格里的列 (比如 2->11, 12->16)
# 'f1_score' 是你要统计的指标
grouped = df.groupby('scenario')['f1_score'].agg(['mean', 'std'])

# 3. 转换为百分比格式 (可选，看论文要求是小数还是百分比)
# 表格图中是百分比 (e.g., 60.0)，所以乘以 100
grouped = grouped * 100

# 4. 格式化成字符串 "Mean ± Std"
# .2f 表示保留两位小数
grouped['formatted_result'] = grouped.apply(
    lambda row: f"{row['mean']:.2f}±{row['std']:.2f}", axis=1
)

# 5. 计算最后一列 AVG (所有任务均值的平均值)
total_avg = grouped['mean'].mean()
print(f"Total AVG Score: {total_avg:.2f}")

# 6. 打印结果，可以直接复制到论文
print("\n=== 论文表格数据 ===")
print(grouped[['formatted_result']])

# 如果需要保存成 Excel 方便复制
# grouped.to_excel("formatted_results.xlsx")