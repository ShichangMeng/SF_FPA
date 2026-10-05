from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import umap


@torch.no_grad()
def extract_features(algorithm, state_dict, dataloader):
  """根据给定的 state_dict 提取目标域数据的特征和真实标签"""
  current_state = deepcopy(algorithm.network.state_dict())

  algorithm.network.load_state_dict(state_dict)
  algorithm.feature_extractor.eval()
  algorithm.freq_adapter.eval()

  features_list = []
  labels_list = []

  for batch in dataloader:
    x = batch[0].float().to(algorithm.device)
    y = batch[1].long().cpu().numpy()

    x_adapted = algorithm.freq_adapter(x)
    feat, _ = algorithm.feature_extractor(x_adapted)

    features_list.append(feat.cpu().numpy())
    labels_list.append(y)

  algorithm.network.load_state_dict(current_state)

  return np.concatenate(features_list), np.concatenate(labels_list)


def plot_umap_before_after(
    algorithm,
    src_only_model,
    adapted_model,
    trg_dataloader,
    save_path="umap_comparison.png",
    num_classes=6,
    class_names=None,
):
  """绘制符合论文高顶刊出版标准的 UMAP 对比图"""
  print(">>> Extracting features for UMAP visualization... <<<")
  feat_pre, labels = extract_features(
      algorithm, src_only_model, trg_dataloader
  )
  feat_post, _ = extract_features(algorithm, adapted_model, trg_dataloader)

  if class_names is None:
    class_names = [str(i) for i in range(num_classes)]
  if len(class_names) != num_classes:
    raise ValueError(
        f"Expected {num_classes} class names, got {len(class_names)}"
    )
  label_names = np.asarray(class_names)[labels]

  print(">>> Computing UMAP embeddings... <<<")
  reducer_pre = umap.UMAP(
      n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
  )
  emb_pre = reducer_pre.fit_transform(feat_pre)

  reducer_post = umap.UMAP(
      n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
  )
  emb_post = reducer_post.fit_transform(feat_post)

  print(">>> Plotting paper-quality UMAP... <<<")
  # 样式设置：去除内部网格，保留干练边框
  sns.set_theme(style="ticks")
  fig, axes = plt.subplots(1, 2, figsize=(14, 6))

  # 1. 构建字典映射颜色板，彻底解决 Seaborn 颜色数量不匹配报错
  raw_colors = sns.color_palette("tab10", num_classes)
  palette_dict = {name: raw_colors[i] for i, name in enumerate(class_names)}

  # 2. 绘制自适应前 (Before Adaptation)
  sns.scatterplot(
      x=emb_pre[:, 0],
      y=emb_pre[:, 1],
      hue=label_names,
      palette=palette_dict,
      hue_order=class_names,
      ax=axes[0],
      s=45,
      alpha=0.85,
      edgecolor=None,
      legend="full",
  )
  axes[0].set_title(
      "Before Adaptation (Source-Only)", fontsize=13, fontweight="bold", pad=10
  )
  axes[0].set_xlabel("")
  axes[0].set_ylabel("")
  axes[0].set_xticks([])
  axes[0].set_yticks([])

  # 3. 绘制自适应后 (After Adaptation)
  sns.scatterplot(
      x=emb_post[:, 0],
      y=emb_post[:, 1],
      hue=label_names,
      palette=palette_dict,
      hue_order=class_names,
      ax=axes[1],
      s=45,
      alpha=0.85,
      edgecolor=None,
      legend=False,
  )
  axes[1].set_title(
      "After Adaptation (SF-FPA)", fontsize=13, fontweight="bold", pad=10
  )
  axes[1].set_xlabel("")
  axes[1].set_ylabel("")
  axes[1].set_xticks([])
  axes[1].set_yticks([])

  # 4. 提取图例并统一放置在底部正中央 (Bottom-Center Horizontal Legend)
  handles, leg_labels = axes[0].get_legend_handles_labels()
  if axes[0].get_legend():
    axes[0].get_legend().remove()

  fig.legend(
      handles=handles,
      labels=leg_labels,
      loc="upper center",
      bbox_to_anchor=(0.5, 0.02),
      ncol=min(num_classes, 10),
      title="Class Label",
      title_fontsize="11",
      fontsize="10",
      frameon=True,
      edgecolor="0.8",
  )

  # 调整布局留出底部图例空间
  plt.tight_layout(rect=[0, 0.05, 1, 1])
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f">>> UMAP visualization saved to {save_path} <<<")

  from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch


@torch.no_grad()
def plot_frequency_motivation_fig(
    algorithm,
    adapted_model,
    src_dataloader,
    trg_dataloader,
    save_path="motivation_freq_alignment.png",
    max_samples=200,
):
  """绘制高张力 Motivation 图：展示 Adaptation 前后频域幅度谱的真实差距与残差消融"""
  print(">>> Extracting Source and Target signals for Motivation Figure... <<<")
  device = algorithm.device

  # 加载最佳模型权重并备份
  current_state = deepcopy(algorithm.network.state_dict())
  algorithm.network.load_state_dict(adapted_model)
  algorithm.freq_adapter.eval()

  # 1. 提取 Source & Target 数据
  src_x_list, trg_x_list = [], []
  for batch in src_dataloader:
    src_x_list.append(batch[0])
    if sum(x.size(0) for x in src_x_list) >= max_samples:
      break
  for batch in trg_dataloader:
    trg_x_list.append(batch[0])
    if sum(x.size(0) for x in trg_x_list) >= max_samples:
      break

  src_x = torch.cat(src_x_list, dim=0)[:max_samples].to(device)
  trg_x = torch.cat(trg_x_list, dim=0)[:max_samples].to(device)

  # 2. 计算 Target Adaptation 后的信号
  trg_x_adapted = algorithm.freq_adapter(trg_x)

  # 恢复模型状态
  algorithm.network.load_state_dict(current_state)

  # 3. 计算频域幅度谱 (FFT Amplitude Spectrum)
  src_amp = torch.fft.rfft(src_x, dim=-1).abs().cpu().numpy()
  trg_amp_before = torch.fft.rfft(trg_x, dim=-1).abs().cpu().numpy()
  trg_amp_after = torch.fft.rfft(trg_x_adapted, dim=-1).abs().cpu().numpy()

  # 均值与标准差计算
  src_mean = np.mean(src_amp, axis=(0, 1))
  trg_pre_mean = np.mean(trg_amp_before, axis=(0, 1))
  trg_post_mean = np.mean(trg_amp_after, axis=(0, 1))

  # 核心改进 1: 忽略 f=0 (DC直流分量)，只展示 f >= 1 的有效交流频域信号，避免尺度被拉畸形
  freq_bins = np.arange(1, len(src_mean))
  src_mean = src_mean[1:]
  trg_pre_mean = trg_pre_mean[1:]
  trg_post_mean = trg_post_mean[1:]

  # 计算绝对残差/差距 (Absolute Spectral Gap)
  gap_before = np.abs(src_mean - trg_pre_mean)
  gap_after = np.abs(src_mean - trg_post_mean)

  print(">>> Plotting Improved Motivation Figure... <<<")
  sns.set_theme(style="ticks")
  fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

  # 专业学术配色
  c_src = "#1f77b4"  # 蓝
  c_trg_pre = "#d62728"  # 红
  c_trg_post = "#2ca02c"  # 绿

  # -------------------------------------------------------------
  # 图 (a): 时域原始波形对比 (Time Domain)
  # -------------------------------------------------------------
  sample_idx, channel_idx = 0, 0
  axes[0].plot(
      src_x[sample_idx, channel_idx].cpu().numpy(),
      label="Source Domain",
      color=c_src,
      alpha=0.85,
      linewidth=1.5,
  )
  axes[0].plot(
      trg_x[sample_idx, channel_idx].cpu().numpy(),
      label="Target Domain",
      color=c_trg_pre,
      alpha=0.85,
      linewidth=1.5,
  )
  axes[0].set_title(
      "(a) Raw Signals in Time Domain", fontsize=12, fontweight="bold", pad=10
  )
  axes[0].set_xlabel("Time Step", fontsize=11)
  axes[0].set_ylabel("Amplitude", fontsize=11)
  axes[0].legend(loc="upper right", frameon=True)
  axes[0].grid(True, linestyle="--", alpha=0.5)

  # -------------------------------------------------------------
  # 核心改进 2: 计算统一的 Y 轴范围上限 (Unified Y-limits)
  # -------------------------------------------------------------
  max_y_val = max(src_mean.max(), trg_pre_mean.max(), trg_post_mean.max()) * 1.1

  # 图 (b): 频域幅度谱对齐对比 (Overlay Spectrum: Before vs After)
  axes[1].plot(
      freq_bins,
      src_mean,
      label="Source Spectrum",
      color=c_src,
      linewidth=2.2,
  )
  axes[1].plot(
      freq_bins,
      trg_pre_mean,
      label="Target (Raw)",
      color=c_trg_pre,
      linewidth=1.8,
      linestyle="--",
  )
  axes[1].plot(
      freq_bins,
      trg_post_mean,
      label="Target (Adapted)",
      color=c_trg_post,
      linewidth=2.0,
  )

  axes[1].set_title(
      "(b) Amplitude Spectra Overlay", fontsize=12, fontweight="bold", pad=10
  )
  axes[1].set_xlabel("Frequency Bin (f >= 1)", fontsize=11)
  axes[1].set_ylabel("Magnitude |X(f)|", fontsize=11)
  axes[1].set_ylim(0, max_y_val)  # 强行统一 Y 轴
  axes[1].legend(loc="upper right", frameon=True)
  axes[1].grid(True, linestyle="--", alpha=0.5)

  # -------------------------------------------------------------
  # 核心改进 3: 图 (c) 绘制直接的残差消融差距曲线 (Spectral Discrepancy Gap)
  # -------------------------------------------------------------
  axes[2].plot(
      freq_bins,
      gap_before,
      label="Gap Before Adaptation",
      color=c_trg_pre,
      linewidth=2.0,
  )
  axes[2].fill_between(freq_bins, 0, gap_before, color=c_trg_pre, alpha=0.25)

  axes[2].plot(
      freq_bins,
      gap_after,
      label="Gap After Adaptation",
      color=c_trg_post,
      linewidth=2.0,
  )
  axes[2].fill_between(freq_bins, 0, gap_after, color=c_trg_post, alpha=0.4)

  axes[2].set_title(
      "(c) Spectral Discrepancy Reduction",
      fontsize=12,
      fontweight="bold",
      pad=10,
  )
  axes[2].set_xlabel("Frequency Bin (f >= 1)", fontsize=11)
  axes[2].set_ylabel(
      "Absolute Gap |X_{src} - X_{trg}|", fontsize=11
  )  # 绝对残差
  axes[2].set_ylim(0, max_y_val * 0.7)  # 残差图设置专属合理的 Y 轴
  axes[2].legend(loc="upper right", frameon=True)
  axes[2].grid(True, linestyle="--", alpha=0.5)

  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f">>> Motivation figure saved to {save_path} <<<")
