
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F
import math

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x_flat = self.aap(x).view(x.shape[0], -1)

        return x_flat, x
##  Classifier
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions

class FT_classifier(nn.Module):
    def __init__(self, configs):
        super(FT_classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels + configs.fourier_modes*2//configs.stride, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions
    
## Temporal Imputer
class Temporal_Imputer(nn.Module):
    def __init__(self, configs):
        super(Temporal_Imputer, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        # input size: batch_size, 128 channel, 18 seq_length
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)
        # take the last time step
        return out

# temporal masking
def masking(x, num_splits=8, num_masked=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, fl=128):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.pi = torch.acos(torch.zeros(1)).item() * 2
        
    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x = torch.cos(x)
        x_ft = torch.fft.rfft(x,norm='ortho')
        x_ft2 = torch.fft.fft(x).abs()
    
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1) 
        
        a = out_ft[:, :, :self.modes1].abs() # amplitude
        p = out_ft[:, :, :self.modes1].angle() # phase
        
        return torch.cat([a,p],-1), out_ft

class classifier2(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(2*model_output_dim * configs.final_out_channels, configs.num_classes)
        
    
    def forward(self, x):
        predictions = self.logits(x)
        return predictions

class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        self.modes1 = configs.fourier_modes   # Number of low-frequency modes to keep
        self.width = configs.input_channels
        self.length =  configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes*2)   # It doubles because frequency features contain both amplitude and phase
        self.cnn = CNN(configs).to('cuda')  # Time Feature Encoder
        self.cnn_f = CNN(configs).to('cuda')  # Frequency Feature Extractor
        self.TSlength_aligned = configs.final_out_channels * configs.features_len
        self.projector = nn.Sequential(
            nn.Linear(self.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.TSlength_aligned)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(self.TSlength_aligned, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, self.TSlength_aligned)
        )


    def forward(self, xt, xf):
        
        ef, _ = self.cnn_f(xf)
        et, etx = self.cnn(xt)
       
        z_time = self.projector(et)
        z_freq = self.projector_f(ef)
        
        return et, etx, ef, z_time, z_freq
    
class tf_decoder(nn.Module):
    def __init__(self, configs):
        super(tf_decoder, self).__init__()
        self.input_channels, self.sequence_len = configs.input_channels, configs.sequence_len
        self.bn1 = nn.BatchNorm1d(self.input_channels,self.sequence_len)
        self.bn2 = nn.BatchNorm1d(self.input_channels,self.sequence_len)
        self.convT = torch.nn.ConvTranspose1d(configs.final_out_channels * configs.features_len, self.sequence_len, self.input_channels, stride=1)
        self.modes = configs.fourier_modes
        self.stride = configs.stride

    def forward(self, f, out_ft):
        x_low = self.bn1(torch.fft.irfft(out_ft, n=self.sequence_len))   # reconstruct  time series by using low frequency frequency features
        et = f[:,self.modes*2//self.stride:]
        x_high = F.relu(self.bn2(self.convT(et.unsqueeze(2)).permute(0,2,1))) # reconstruct time series by using time features for high frequency patterns. 
        return x_low + x_high
    
class FrequencyAdaptationLayer(nn.Module):
    """
    频域自适应层 (Channel-Shared 修复版)
    """
    def __init__(self, seq_len, channels, amp_scale=0.1, phase_scale=0.1, hidden_dim=64):
        super(FrequencyAdaptationLayer, self).__init__()
        
        self.freq_len = seq_len // 2 + 1
        self.amp_scale = amp_scale
        self.phase_scale = phase_scale
        
        # 幅值调节 MLP
        self.amp_mlp = nn.Sequential(
            nn.Linear(self.freq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len),
            nn.Tanh()
        )
        
        # 相位调节 MLP
        self.phase_mlp = nn.Sequential(
            nn.Linear(self.freq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len),
            nn.Tanh()
        )

        # 最后一层权重和偏置置零，确保初始状态为恒等映射 (Identity)
        nn.init.zeros_(self.amp_mlp[2].weight)
        nn.init.zeros_(self.amp_mlp[2].bias)
        nn.init.zeros_(self.phase_mlp[2].weight)
        nn.init.zeros_(self.phase_mlp[2].bias)

    def forward(self, x):
        # x: [Batch, Channel, Seq_Len]
        fft_x = torch.fft.rfft(x, dim=-1) 
        
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 1. 计算 Delta
        delta_amp = self.amp_mlp(amp) * self.amp_scale  
        # 相位变化范围映射到 [-phase_scale * pi, phase_scale * pi]
        delta_phase = self.phase_mlp(phase) * self.phase_scale * math.pi 
        
        # 2. 叠加修正 (幅值乘法缩放，相位加法偏移)
        new_amp = amp * (1.0 + delta_amp)
        new_phase = phase + delta_phase
        
        # 3. iFFT 反变换
        new_fft = torch.polar(new_amp, new_phase)
        x_adapted = torch.fft.irfft(new_fft, n=x.shape[-1], dim=-1)
        
        return x_adapted
    
class ChannelIndependentLinear(nn.Module):
    def __init__(self, channels, in_features, out_features):
        super(ChannelIndependentLinear, self).__init__()
        self.channels = channels
        self.weight = nn.Parameter(torch.Tensor(channels, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(channels, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.einsum('bci,cio->bco', x, self.weight) + self.bias


class FrequencyAdaptationLayer_Independent(nn.Module):
    """
    频域自适应层 (Channel-Independent 完全独立修复版)
    """
    def __init__(self, seq_len, channels, amp_scale=0.1, phase_scale=0.1, hidden_dim=64):
        super(FrequencyAdaptationLayer_Independent, self).__init__()
        
        self.freq_len = seq_len // 2 + 1
        self.amp_scale = amp_scale
        self.phase_scale = phase_scale
        
        # 幅值调节 MLP (通道独立)
        self.amp_mlp = nn.Sequential(
            ChannelIndependentLinear(channels, self.freq_len, hidden_dim),
            nn.ReLU(),
            ChannelIndependentLinear(channels, hidden_dim, self.freq_len),
            nn.Tanh()
        )
        
        # 相位调节 MLP (通道独立，修复原代码误用普通 Linear 的问题)
        self.phase_mlp = nn.Sequential(
            ChannelIndependentLinear(channels, self.freq_len, hidden_dim),
            nn.ReLU(),
            ChannelIndependentLinear(channels, hidden_dim, self.freq_len),
            nn.Tanh()
        )

        # 最后一层初始化为 0
        nn.init.zeros_(self.amp_mlp[2].weight)
        nn.init.zeros_(self.amp_mlp[2].bias)
        nn.init.zeros_(self.phase_mlp[2].weight)
        nn.init.zeros_(self.phase_mlp[2].bias)

    def forward(self, x):
        fft_x = torch.fft.rfft(x, dim=-1) 
        
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        delta_amp = self.amp_mlp(amp) * self.amp_scale  
        delta_phase = self.phase_mlp(phase) * self.phase_scale * math.pi
        
        # 幅值乘法，相位加法
        new_amp = amp * (1.0 + delta_amp)
        new_phase = phase + delta_phase
        
        new_fft = torch.polar(new_amp, new_phase)
        x_adapted = torch.fft.irfft(new_fft, n=x.shape[-1], dim=-1)
        
        return x_adapted
    
class FrequencyAdaptationLayer_linear(nn.Module):
    """
    频域自适应层 (Linear 版本)
    将原来的 MLP 替换为单层 Linear 映射，用于消融实验测试。
    1. 将时域信号转换到频域 (FFT)
    2. 使用 Linear 预测 Δ幅值 和 Δ相位
    3. 叠加 Δ 并反变换回时域 (iFFT)
    """
    # 注意：这里删除了 hidden_dim 参数，因为单层 Linear 不需要隐藏层
    def __init__(self, seq_len, channels, amp_scale=0.1, phase_scale=0.1):
        super(FrequencyAdaptationLayer_linear, self).__init__()
        
        # rfft 后的频率维度长度
        self.freq_len = seq_len // 2 + 1
        self.amp_scale = amp_scale
        self.phase_scale = phase_scale
        
        # 幅值调节 Linear
        # 直接从 freq_len 映射到 freq_len，相当于一个最简单的线性组合变换
        self.amp_linear = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.Tanh() # 强烈建议保留 Tanh，限制幅度变化的范围，防止反变换时梯度爆炸
        )
        
        # 相位调节 Linear
        self.phase_linear = nn.Sequential(
            nn.Linear(self.freq_len, self.freq_len),
            nn.Tanh() # 限制相位变化的范围 (-1, 1)
        )

        # 初始化为 0，保证初始状态下是对等变换 (Identity)
        # 这样刚开始训练时不会破坏原始数据分布
        for m in self.amp_linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        
        for m in self.phase_linear.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [Batch, Channel, Seq_Len]
        
        # 1. FFT 变换
        fft_x = torch.fft.rfft(x, dim=-1) 
        
        # 提取幅值和相位
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 2. 计算 Delta (直接使用单层 Linear)
        delta_amp = self.amp_linear(amp) * self.amp_scale  
        delta_phase = self.phase_linear(phase) * self.phase_scale 
        
        # 3. 叠加修正
        new_amp = amp * (1 + delta_amp)
        new_phase = phase * (1 + delta_phase)
        
        # 4. 组合并反变换 iFFT
        new_fft = torch.polar(new_amp, new_phase)
        x_adapted = torch.fft.irfft(new_fft, n=x.shape[-1], dim=-1)
        
        return x_adapted

class TransformerAdapter(nn.Module):
    """
    频域自适应的 Transformer 核心组件
    把每一个频率点当作一个 Token，利用 Self-Attention 捕捉全局频率的非线性相关性
    """
    def __init__(self, freq_len, d_model=16, nhead=4, num_layers=1):
        super().__init__()
        # 1. 维度扩增：将单维度的标量特征投射到 d_model 维度
        self.proj_in = nn.Linear(1, d_model)
        
        # 2. 位置编码：由于频率本身具有绝对的顺序意义（低频到高频），加入可学习的位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, freq_len, d_model))
        
        # 3. Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 2, 
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. 降维输出：将 d_model 投射回 1，并用 Tanh 限制范围
        self.proj_out = nn.Linear(d_model, 1)
        self.tanh = nn.Tanh()

        # 初始化为 0，保证初始状态是对等变换 (Identity mapping)
        nn.init.constant_(self.proj_out.weight, 0)
        nn.init.constant_(self.proj_out.bias, 0)

    def forward(self, x):
        # x shape: [Batch, Channel, Freq_Len]
        B, C, F = x.shape
        
        # 将 Batch 和 Channel 展平，作为独立的样本序列输入
        x = x.view(B * C, F, 1)  # [B*C, Freq_Len, 1]
        
        # 投影并加上位置编码
        x = self.proj_in(x) + self.pos_embed
        
        # Self-Attention 交互
        x = self.transformer(x)
        
        # 映射回单维并激活
        x = self.proj_out(x).squeeze(-1)  # [B*C, Freq_Len]
        x = self.tanh(x)
        
        # 恢复原有形状
        return x.view(B, C, F)


class FrequencyAdaptationLayer_transformer(nn.Module):
    """
    频域自适应层 (Transformer / Self-Attention 版本)
    用于消融实验，证明 MLP 是性能与复杂度的最佳平衡点。
    """
    def __init__(self, seq_len, channels, amp_scale=0.1, phase_scale=0.1, d_model=16, nhead=4):
        super(FrequencyAdaptationLayer_transformer, self).__init__()
        
        self.freq_len = seq_len // 2 + 1
        self.amp_scale = amp_scale
        self.phase_scale = phase_scale
        
        # 使用 Transformer 替换原来的 MLP
        self.amp_adapter = TransformerAdapter(self.freq_len, d_model=d_model, nhead=nhead)
        self.phase_adapter = TransformerAdapter(self.freq_len, d_model=d_model, nhead=nhead)

    def forward(self, x):
        # 1. FFT 变换
        fft_x = torch.fft.rfft(x, dim=-1) 
        
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 2. 计算 Delta (使用 Transformer)
        delta_amp = self.amp_adapter(amp) * self.amp_scale
        delta_phase = self.phase_adapter(phase) * self.phase_scale
        
        # 3. 叠加修正
        new_amp = amp * (1 + delta_amp)
        new_phase = phase * (1 + delta_phase)
        
        # 4. iFFT 反变换
        new_fft = torch.polar(new_amp, new_phase)
        x_adapted = torch.fft.irfft(new_fft, n=x.shape[-1], dim=-1)
        
        return x_adapted
    
class Signal_Recover(nn.Module):
    def __init__(self, configs, hparams):
        super(Signal_Recover, self).__init__()
        self.seq_length = configs.sequence_len
        self.num_channels = configs.input_channels
        self.hid_dim = configs.AR_hid_dim_raw
        # input size: batch_size, 9 channel, 128 seq_length
        self.rnn_encoder = nn.LSTM(input_size=self.num_channels, hidden_size=hparams['AR_hid_dim'], batch_first=True)
        self.rnn_decoder = nn.LSTM(input_size=hparams['AR_hid_dim'], hidden_size=self.hid_dim, batch_first=True)

    def forward(self, x):
        ### x size (bs, N, L)

        x = x.transpose(-1,-2)
        out, (h, c) = self.rnn_encoder(x)
        out, (h, c) = self.rnn_decoder(out)

        out = out.transpose(-1,-2)
        # take the last time step
        return out
    
def masking2(x, num_splits=8, num_masked=4): ## also segment v1
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'b n (p l) -> b n p l', p=num_splits)
    masked_patches = patches.clone()

    # Generate random indices for masking
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    # Create a mask tensor
    mask = torch.zeros_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = True  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches[mask] = 0

    # Reshape the masked patches and the mask back to the original shape
    mask = rearrange(mask, 'b n p l -> b n (p l)')
    masked_x = rearrange(masked_patches, 'b n p l -> b n (p l)', p=num_splits)

    return masked_x, [mask,rand_indices]

def segment_mask_v1(x, mask_indices, num_segments, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)


    masked_patches = patches.clone()

    # Generate random indices for masking
    selected_indices = mask_indices[:, -num_removed:]
    # Create a mask tensor
    mask = torch.zeros_like(patches, dtype=torch.bool)

    # Create a batch index tensor
    batch_indices = torch.arange(x.shape[0]).unsqueeze(1).unsqueeze(2)

    # Apply the mask using advanced indexing
    mask[batch_indices, torch.arange(x.shape[1]).view(1, -1, 1), selected_indices.unsqueeze(0).expand(x.shape[0], -1, -1)] = True  # 32,1,1  1,6,1  32,6,1

    # Mask the selected patches
    masked_patches[mask] = 0

    # Reshape the masked patches and the mask back to the original shape
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_segments)

    return masked_x

def segment_mask_v3(x, posi, padding, num_segments, num_removed=2):
    # Reshape input tensor to create patches
    ### Input size (bs, num_channels, time_length)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_segments)


    masked_patches = patches.clone()

    if posi == 'first':
        if padding:
            masked_patches[:, :, :num_removed, :] = 0
        else:
            masked_patches = masked_patches[:,:,num_removed:,:]
    elif posi == 'last':
        if padding:
            masked_patches[:, :, -num_removed:, :] = 0
        else:
            masked_patches = masked_patches[:, :, :-num_removed, :]
    else:
        raise NotImplementedError('Not accurate position for segments')
    p = num_segments if padding else num_segments-num_removed
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=p)

    return masked_x

def get_distances(X, Y, dist_type="cosine"):
        if dist_type == "euclidean":
            distances = torch.cdist(X, Y)
        elif dist_type == "cosine":
            distances = 1 - torch.matmul(F.normalize(X, dim=1), F.normalize(Y, dim=1).T)
        else:
            raise NotImplementedError(f"{dist_type} distance not implemented.")

        return distances

@torch.no_grad()
def soft_k_nearest_neighbors(features, features_bank, probs_bank, num_neighbors):
    pred_probs = []
    pred_probs_all = []

    for feats in features.split(128):
        distances = get_distances(feats, features_bank)
        _, idxs = distances.sort()
        idxs = idxs[:, : num_neighbors]
        
        # (64, num_nbrs, num_classes), average over dim=1
        probs = probs_bank[idxs, :].mean(1)
        pred_probs.append(probs)
        # (64, num_nbrs, num_classes)
        probs_all = probs_bank[idxs, :]
        pred_probs_all.append(probs_all)

    pred_probs_all = torch.cat(pred_probs_all)
    pred_probs = torch.cat(pred_probs)
    
    #Pseudolabels
    _, pred_labels = pred_probs.max(dim=1)
    # (64, num_nbrs, num_classes), max over dim=2
    _, pred_labels_all = pred_probs_all.max(dim=2)
    #First keep maximum for all classes between neighbors and then keep max between classes
    _, pred_labels_hard = pred_probs_all.max(dim=1)[0].max(dim=1)

    return pred_labels, pred_probs, pred_labels_all, pred_labels_hard

def refine_predictions(
    features,
    probs,
    banks, num_neighbors):
    feature_bank = banks["features"]
    probs_bank = banks["probs"]
    pred_labels, probs, pred_labels_all, pred_labels_hard = soft_k_nearest_neighbors(features, feature_bank, probs_bank, num_neighbors)

    return pred_labels, probs, pred_labels_all, pred_labels_hard

@torch.no_grad()
def update_labels(banks, idxs, features, logits):
    probs = F.softmax(logits, dim=1)

    start = banks["ptr"]
    end = start + len(idxs)
    idxs_replace = torch.arange(start, end).cuda() % len(banks["features"])
    banks["features"][idxs_replace, :] = features
    banks["probs"][idxs_replace, :] = probs
    banks["ptr"] = end % len(banks["features"])

@torch.no_grad()
def eval_and_label_dataset(epoch, FE, classifier, banks, test_dataloader, train_dataloader, num_neighbors):
    print("Evaluating Dataset!")
    
    FE.eval()
    classifier.eval()
    logits, indices, gt_labels = [], [], []
    features = []           

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
        # for batch_idx, batch in enumerate(train_dataloader):
            test_inputs, test_targets, test_idxs, test_inputs_f = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[6].cuda()
            # inputs, targets, idxs = batch[0].cuda(), batch[2].cuda(), batch[3].cuda()
            feats, _, _, _, _ = FE(test_inputs, test_inputs_f)
            # print(f'batch_idx:{batch_idx}')
            # print(f'feats.shape:{feats.shape}')

            logits_cls = classifier(feats)

            features.append(feats)
            gt_labels.append(test_targets)
            logits.append(logits_cls)
            indices.append(test_idxs)  

    features = torch.cat(features)
    gt_labels = torch.cat(gt_labels)
    logits = torch.cat(logits)
    indices = torch.cat(indices)

    probs = F.softmax(logits, dim=1)
    rand_idxs = torch.randperm(len(features)).cuda()
    banks = {
        "features": features[rand_idxs][: 16384], #hanya diacak urutan indeksnya menggunakan rand_idxs
        "probs": probs[rand_idxs][: 16384],
        "ptr": 0,
    }
        
    FE.train()
    classifier.train()
    return banks