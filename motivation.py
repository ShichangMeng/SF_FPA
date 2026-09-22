import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_conceptual_motivation_fig(
    save_path="vis/motivation/conceptual_motivation_fig1.png",
):
  """生成论文 Figure 1 概念 Motivation 图 (修复文字超界与边距问题)

  展示：时域混淆 -> 频域解耦 -> 幅度偏置 vs 相位不变
  """
  sns.set_theme(style="ticks")
  t = np.linspace(0, 1, 200)

  # 1. 构造概念信号：Source 与 Target (带有振幅缩放、相位偏移与高频噪声)
  src_sig = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 12 * t)
  trg_sig = 1.8 * np.sin(2 * np.pi * 5 * (t - 0.03)) + 0.9 * np.sin(
      2 * np.pi * 12 * t
  )

  # 2. 计算傅里叶变换
  src_fft = np.fft.rfft(src_sig)
  trg_fft = np.fft.rfft(trg_sig)

  freqs = np.arange(len(src_fft))
  src_mag, trg_mag = np.abs(src_fft), np.abs(trg_fft)
  src_phase, trg_phase = np.angle(src_fft), np.angle(trg_fft)

  # 3. 绘图 (1 行 3 列架构)
  fig = plt.figure(figsize=(16, 4.8))
  c_src = "#1f77b4"  # 蓝 (Source)
  c_trg = "#d62728"  # 红 (Target)

  # --- 子图 (a): 时域（混淆难对齐） ---
  ax1 = fig.add_subplot(1, 3, 1)
  ax1.plot(
      t, src_sig, label="Source Domain", color=c_src, lw=2.0, alpha=0.85
  )
  ax1.plot(
      t,
      trg_sig,
      label="Target Domain",
      color=c_trg,
      lw=2.0,
      ls="--",
      alpha=0.85,
  )
  ax1.set_title(
      "(a) Time Domain: Entangled Shift", fontsize=12, fontweight="bold", pad=12
  )
  ax1.set_xlabel("Time", fontsize=11)
  ax1.set_ylabel("Amplitude", fontsize=11)
  ax1.legend(loc="upper right", frameon=True)
  ax1.grid(True, linestyle=":", alpha=0.6)

  # --- 子图 (b): 频域幅度谱（明确的 Amplitude Gap） ---
  ax2 = fig.add_subplot(1, 3, 2)
  ax2.plot(
      freqs[1:25], src_mag[1:25], label="Source Mag", color=c_src, lw=2.2
  )
  ax2.plot(
      freqs[1:25],
      trg_mag[1:25],
      label="Target Mag",
      color=c_trg,
      lw=2.2,
      ls="--",
  )

  # 【关键修复】扩展 Y 轴顶部范围 (留出 35% 空间)，放置标注和 Legend 不挤压
  max_mag = max(src_mag[1:25].max(), trg_mag[1:25].max())
  ax2.set_ylim(0, max_mag * 1.35)

  # 精确计算 Gap 标注的位置
  peak_bin = 5
  y_src_peak = src_mag[peak_bin]
  y_trg_peak = trg_mag[peak_bin]

  # 1) 在 Peak 右侧绘制垂直双向箭头 (<->)
  ax2.annotate(
      "",
      xy=(peak_bin + 0.3, y_trg_peak),
      xytext=(peak_bin + 0.3, y_src_peak),
      arrowprops=dict(
          arrowstyle="<->", color="black", lw=1.6, mutation_scale=12
      ),
  )

  # 2) 箭头右侧放置清晰说明文本
  ax2.text(
      peak_bin + 1.2,
      (y_src_peak + y_trg_peak) / 2,
      "Amplitude Gap\n(Domain Style Shift)",
      fontsize=10,
      fontweight="bold",
      color="black",
      va="center",
  )

  ax2.set_title(
      "(b) Magnitude Spectrum: Style Shift",
      fontsize=12,
      fontweight="bold",
      pad=12,
  )
  ax2.set_xlabel("Frequency Bin", fontsize=11)
  ax2.set_ylabel("Magnitude |X(f)|", fontsize=11)
  ax2.legend(loc="upper right", frameon=True)
  ax2.grid(True, linestyle=":", alpha=0.6)

  # --- 子图 (c): 频域相位谱（语义高度保留） ---
  ax3 = fig.add_subplot(1, 3, 3)
  ax3.scatter(
      freqs[1:25],
      src_phase[1:25],
      label="Source Phase",
      color=c_src,
      s=45,
      alpha=0.85,
  )
  ax3.scatter(
      freqs[1:25],
      trg_phase[1:25],
      label="Target Phase",
      color=c_trg,
      s=45,
      marker="x",
      alpha=0.85,
  )

  # 调整纵轴范围避免 Legend 压住数据点
  ax3.set_ylim(-3.5, 3.2)

  ax3.set_title(
      "(c) Phase Spectrum: Semantic Invariance",
      fontsize=12,
      fontweight="bold",
      pad=12,
  )
  ax3.set_xlabel("Frequency Bin", fontsize=11)
  ax3.set_ylabel("Phase Angle (rad)", fontsize=11)
  ax3.legend(loc="upper right", frameon=True)
  ax3.grid(True, linestyle=":", alpha=0.6)

  plt.tight_layout()
  plt.savefig(save_path, dpi=300, bbox_inches="tight")
  plt.close()
  print(f">>> Conceptual Motivation figure saved to {save_path} <<<")


if __name__ == "__main__":
  plot_conceptual_motivation_fig()