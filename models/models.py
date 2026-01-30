
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

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
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1) 
        r = out_ft[:, :, :self.modes1].abs()
        p = out_ft[:, :, :self.modes1].angle() 
        return torch.concat([r,p],-1), out_ft

class tf_encoder(nn.Module):
    def __init__(self, configs):
        super(tf_encoder, self).__init__()
        self.modes1 = configs.fourier_modes   # Number of low-frequency modes to keep
        self.width = configs.input_channels
        self.length =  configs.sequence_len
        self.freq_feature = SpectralConv1d(self.width, self.width, self.modes1,self.length)  # Frequency Feature Encoder
        self.bn_freq = nn.BatchNorm1d(configs.fourier_modes*2//configs.stride)   # It doubles because frequency features contain both amplitude and phase
        self.cnn = CNN(configs).to('cuda')  # Time Feature Encoder
        self.avg = nn.Conv1d(self.width, 1, kernel_size=3 ,
                  stride=configs.stride, bias=False, padding=(3 // 2))   


    def forward(self, x):
        ef, out_ft = self.freq_feature(x)
        ef = F.relu(self.bn_freq(self.avg(ef).squeeze(1)))
        et, out_tt = self.cnn(x)
        f = torch.concat([ef,et],-1)
        # out = torch.concat([out_ft,out_tt],-1)
        return f, out_tt, out_ft

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
    频域自适应层
    1. 将时域信号转换到频域 (FFT)
    2. 使用 MLP 预测 Δ幅值 和 Δ相位
    3. 叠加 Δ 并反变换回时域 (iFFT)
    """
    def __init__(self, seq_len, channels, amp_scale=0.1, phase_scale=0.1, hidden_dim=64):
        super(FrequencyAdaptationLayer, self).__init__()
        
        # rfft 后的频率维度长度
        self.freq_len = seq_len // 2 + 1
        
        # 幅值调节 MLP: Input: (freq_len), Output: (freq_len)
        # 这里设计为对每个 Channel 独立应用或者共享权重，
        # 为了简单起见，这里利用 Linear 的特性，对最后一维(频率维)进行全连接操作
        self.amp_mlp = nn.Sequential(
            nn.Linear(self.freq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len),
            nn.Tanh() # 限制幅度变化的范围，防止崩坏
        )
        
        # 相位调节 MLP
        self.phase_mlp = nn.Sequential(
            nn.Linear(self.freq_len, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.freq_len),
            nn.Tanh() # 限制相位变化的范围 (-1, 1) -> 乘以系数控制弧度
        )

        # 初始化为 0，保证初始状态下是对等变换 (Identity)
        # 这样刚开始训练时不会破坏原始数据分布
        for m in self.amp_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)
        
        for m in self.phase_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [Batch, Channel, Seq_Len]
        
        # 1. FFT 变换
        # rfft 返回的是复数张量
        fft_x = torch.fft.rfft(x, dim=-1) 
        
        # 提取幅值和相位
        amp = torch.abs(fft_x)
        phase = torch.angle(fft_x)
        
        # 2. 计算 Delta
        # MLP 输入是当前的幅值/相位分布
        # 注意：amp 和 phase 的形状是 [B, C, Freq_Len]
        # nn.Linear 默认作用在最后一个维度，所以正好

        delta_amp = self.amp_mlp(amp) * 0.2  # 乘以系数限制初始扰动大小
        delta_phase = self.phase_mlp(phase) * 0.2 # 限制相位改变量 (比如最多 0.1 弧度)
        
        # 3. 叠加修正
        new_amp = amp * (1 + delta_amp)
        new_phase = phase + (1 + delta_phase)
        
        # 4. 组合并反变换 iFFT
        # 构造复数
        new_fft = torch.polar(new_amp, new_phase)
        
        # 反变换回时域
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