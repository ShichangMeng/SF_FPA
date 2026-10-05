import pandas as pd

# 1. 加载数据
df = pd.read_csv("/mnt/cache/mengshichang/exp_boiler_baselines_ep5_rerun2/Boiler/TemSR_ep5_3sc/best_results.csv")

# 2. 过滤掉末尾非 Scenario 的统计汇总行（mean 和 std）
df_clean = df[~df["scenario"].isin(["mean", "std"])].copy()

# 确保 f1_score 列为浮点数格式
df_clean["f1_score"] = pd.to_numeric(df_clean["f1_score"])

# 3. 按 scenario 分组计算 mean 和 std
stats = (
    df_clean.groupby("scenario", sort=False)["f1_score"]
    .agg(["mean", "std"])
    .reset_index()
)

# 4. 生成各个 Scenario 的 Mean±Std (百分比格式)
scenario_f1_strs = [
    f"{row['mean']*100:.2f}±{row['std']*100:.2f}" for _, row in stats.iterrows()
]

# 5. 计算 Overall 的 Avg (%) 平均值
overall_f1_mean = df_clean["f1_score"].mean() * 100
overall_str = f"{overall_f1_mean:.2f}"

# 6. 拼接为 LaTeX 行格式并打印
latex_row = " & ".join(scenario_f1_strs) + f" & {overall_str} \\\\"

print("\n=== LaTeX 复制文本 ===")
print(latex_row)