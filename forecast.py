import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 获取 PeMS 交通数据集 (PeMS04 -> PeMS08)
# ==========================================
print("=== 加载 PeMS 交通流数据集 (Source: PeMS04, Target: PeMS08) ===")

def load_pems_or_mock():
    # 尝试加载本地真实的 PeMS 数据文件
    if os.path.exists('PEMS04.npz') and os.path.exists('PEMS08.npz'):
        print("找到本地 PeMS 数据，正在加载真实数据...")
        # 取交通流量特征 (通常是第 0 个特征)，并计算所有传感器节点的平均流量代表全局趋势
        data_src = np.load('PEMS04.npz')['data'][:, :, 0].mean(axis=1).reshape(-1, 1)
        data_trg = np.load('PEMS08.npz')['data'][:, :, 0].mean(axis=1).reshape(-1, 1)
    else:
        print("未找到 PEMS04.npz/PEMS08.npz，正在生成具有跨域物理特征的模拟交通流数据...")
        # 模拟 30 天的数据 (5分钟一次，每天 288 个点)
        t = np.arange(288 * 30)
        # 模拟双峰交通流 (早晚高峰)
        base_pattern = np.sin(2 * np.pi * t / 288) + 0.5 * np.cos(4 * np.pi * t / 288)
        
        # Source 域 (PeMS04)
        data_src = (base_pattern * 150 + 300 + np.random.randn(len(t)) * 10).reshape(-1, 1)
        
        # Target 域 (PeMS08)：产生 Domain Shift
        # 1. 振幅偏移：车流量整体减少 (乘以 0.6)
        # 2. 相位偏移：高峰期推迟 (用 np.roll 平移 24 个点，相当于推迟 2 小时)
        shifted_pattern = np.roll(base_pattern, shift=24)
        data_trg = (shifted_pattern * 90 + 200 + np.random.randn(len(t)) * 10).reshape(-1, 1)
        
    return data_src, data_trg

data_src, data_trg = load_pems_or_mock()

# 标准化 (必须用 Source 的统计量去 Transform Target，以此保留/放大 Domain Shift)
scaler = StandardScaler()
data_src = scaler.fit_transform(data_src)
data_trg = scaler.transform(data_trg) 

def create_sliding_windows(series, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(series) - seq_len - pred_len):
        X.append(series[i : i + seq_len, 0])
        Y.append(series[i + seq_len : i + seq_len + pred_len, 0])
    return torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(1), \
           torch.tensor(np.array(Y), dtype=torch.float32)

# PeMS 经典设定：用过去 2 小时 (24步) 预测未来 1 小时 (12步)
SEQ_LEN = 24 
PRED_LEN = 12 

src_x, src_y = create_sliding_windows(data_src, SEQ_LEN, PRED_LEN)
trg_x, trg_y = create_sliding_windows(data_trg, SEQ_LEN, PRED_LEN)

# ==========================================
# 2. 网络定义 (Baseline + FAL)
# ==========================================
class FrequencyAdaptationLayer(nn.Module):
    """
    频域自适应层
    1. 将时域信号转换到频域 (FFT)
    2. 使用 MLP 预测 Δ幅值 和 Δ相位
    3. 叠加 Δ 并反变换回时域 (iFFT)
    """
    def __init__(self, seq_len, channels, amp_scale=0.5, phase_scale=0.5, hidden_dim=128):
        super(FrequencyAdaptationLayer, self).__init__()
        
        self.freq_len = seq_len // 2 + 1
        self.amp_scale = amp_scale
        self.phase_scale = phase_scale
        
        self.amp_mlp = nn.Sequential(
            nn.Linear(self.freq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len),
            nn.Tanh()
        )
        
        self.phase_mlp = nn.Sequential(
            nn.Linear(self.freq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len),
            nn.Tanh() 
        )

        for m in self.amp_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        
        for m in self.phase_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        fft_x = torch.fft.rfft(x, dim=-1) 
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        delta_amp = self.amp_mlp(amp) * self.amp_scale  
        delta_phase = self.phase_mlp(phase) * self.phase_scale 
        
        new_amp = amp * (1 + delta_amp)
        new_phase = phase * (1 + delta_phase)
        
        new_fft = torch.polar(new_amp, new_phase)
        x_adapted = torch.fft.irfft(new_fft, n=x.shape[-1], dim=-1)
        
        return x_adapted

class CNNForecaster(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * seq_len, pred_len)
        )
    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 实验开始
# ==========================================
model = CNNForecaster(SEQ_LEN, PRED_LEN)
criterion = nn.MSELoss()

# --- 阶段 A: Source 域预训练 ---
print("\n=== 阶段 A: Source 域 (PeMS04) 预训练 Baseline ===")
optimizer = optim.Adam(model.parameters(), lr=0.005)
model.train()
# 为了演示速度，这里迭代次数设为 50，真实实验中可调高
for epoch in range(50):
    optimizer.zero_grad()
    loss = criterion(model(src_x), src_y)
    loss.backward()
    optimizer.step()
print(f"预训练完成，Source 域 MSE Loss: {loss.item():.4f}")

# --- 阶段 B: Target 域 Zero-shot 跨域预测 ---
model.eval()
with torch.no_grad():
    base_pred = model(trg_x)
    base_mse = criterion(base_pred, trg_y)
print(f"\n=== 阶段 B: Baseline 跨城市直接预测 Target 域 (PeMS08) ===")
print(f"Baseline MSE (Zero-shot): {base_mse.item():.4f}")

# --- 阶段 C: 引入 FAL 的流式自适应 (Streaming TTA) ---
print("\n=== 阶段 C: Baseline + FAL 在 PeMS08 进行流式自适应 (Streaming TTA) ===")
fal = FrequencyAdaptationLayer(SEQ_LEN, 1)
fal_optimizer = optim.Adam(fal.parameters(), lr=0.005)

# 完全冻结预测骨干网络，只更新轻量级的 FAL 模块
for param in model.parameters(): 
    param.requires_grad = False

fal.train()
model.eval()

tta_predictions = []

# 模拟真实交通流场景下，数据不断随时间到来
batch_size = 32
for i in range(0, len(trg_x), batch_size):
    x_batch = trg_x[i : i+batch_size]
    y_batch = trg_y[i : i+batch_size]
    
    # 1. 用当前的 FAL 调整特征并预测当下交通流
    adapted_x = fal(x_batch)
    with torch.no_grad(): 
        pred = model(adapted_x)
        tta_predictions.append(pred)
    
    # 2. 真实交通量到达后，立刻计算 Loss 并反向传播只更新 FAL 模块，使其适应新的高峰规律
    fal_optimizer.zero_grad()
    train_pred = model(fal(x_batch)) 
    loss = criterion(train_pred, y_batch)
    loss.backward()
    fal_optimizer.step()

tta_predictions = torch.cat(tta_predictions, dim=0)
tta_mse = criterion(tta_predictions, trg_y)

print(f"Baseline + FAL 适应后的真实预测误差 (MSE): {tta_mse.item():.4f}")
print(f"误差下降幅度: {(base_mse.item() - tta_mse.item()):.4f} 🚀")