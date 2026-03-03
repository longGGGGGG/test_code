"""
模型二消融实验 (Ablation Study for DualBranchFusionModel)

消融变体 (对应论文表5-4):
    1. 完整模型                     : 原始 DualBranchFusionModel
    2. w/o 交叉注意力（→直接拼接）  : 去除 CrossAttention，改为特征直接拼接+Linear
    3. w/o 自适应门控（→固定均值融合）: 去除 AdaptiveGatedFusion，改为 0.5*a + 0.5*b
    4. 仅CNN分支                    : 去除LSTM分支，仅用CNN特征
    5. 仅LSTM分支                   : 去除CNN分支，仅用LSTM特征

每个变体采用相同的5折分层交叉验证、超参数和随机种子。
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import numpy as np
import glob
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
def setup_logger(name, log_file, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


logger = setup_logger('ablation2', 'model2_ablation.log')


# ─────────────────────────────────────────────
# 随机种子
# ─────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# 数据加载（与 model2 完全一致）
# ─────────────────────────────────────────────
def load_gas_data(data_dir, gas_type='both', target_length=500, downsample_factor=10):
    concentration_mapping = {
        "010": 25.0, "020": 50.0, "030": 75.0, "040": 100.0, "050": 125.0,
        "060": 150.0, "070": 175.0, "080": 200.0, "090": 225.0, "100": 250.0
    }
    data_list, labels, gas_labels = [], [], []

    dirs_to_load = []
    if gas_type in ('Ac', 'both'):
        dirs_to_load.append(('Ac', 'GAC'))
    if gas_type in ('Ea', 'both'):
        dirs_to_load.append(('Ea', 'GEa'))

    for dir_name, gas_code in dirs_to_load:
        gas_dir = os.path.join(data_dir, dir_name)
        files = sorted(glob.glob(os.path.join(gas_dir, '*.txt')))
        logger.info(f"  [{dir_name}] 发现 {len(files)} 个文件, 开始加载...")

        for file_path in tqdm(files, desc=f"    加载{dir_name}", ncols=80):
            fname = os.path.basename(file_path)
            try:
                parts = fname.split('F')
                concentration_str = parts[1][:3]
                if concentration_str not in concentration_mapping:
                    continue
                concentration = concentration_mapping[concentration_str]
            except (IndexError, ValueError):
                continue
            try:
                raw_data = pd.read_csv(file_path, sep=r'\s+', header=None).values
            except Exception:
                continue
            if raw_data.ndim != 2 or raw_data.shape[1] < 5:
                continue

            time_col = raw_data[:, 0]
            sensor_data = raw_data[:, 1:5]

            baseline_mask = time_col < 40
            if baseline_mask.sum() < 50:
                baseline_mask = np.arange(len(time_col)) < 500
            baseline = sensor_data[baseline_mask].mean(axis=0)
            delta_r = (sensor_data - baseline) / (np.abs(baseline) + 1e-8)

            response_mask = (time_col >= 40) & (time_col <= 290)
            delta_r_resp = delta_r[response_mask]
            if delta_r_resp.shape[0] < 100:
                continue

            delta_r_ds = delta_r_resp[::downsample_factor]
            n_points = delta_r_ds.shape[0]

            feat_delta_r = delta_r_ds
            feat_deriv = np.gradient(delta_r_ds, axis=0)
            window = min(20, n_points // 5)
            if window < 3:
                window = 3
            feat_std = np.zeros_like(delta_r_ds)
            for i in range(n_points):
                s = max(0, i - window // 2)
                e = min(n_points, i + window // 2 + 1)
                feat_std[i] = delta_r_ds[s:e].std(axis=0)

            combined = np.hstack([feat_delta_r, feat_deriv, feat_std])
            tensor_data = torch.tensor(combined, dtype=torch.float32)

            if tensor_data.shape[0] > target_length:
                idx_sel = np.linspace(0, tensor_data.shape[0] - 1, target_length, dtype=int)
                tensor_data = tensor_data[idx_sel]
            elif tensor_data.shape[0] < target_length:
                pad = torch.zeros(target_length - tensor_data.shape[0], tensor_data.shape[1])
                tensor_data = torch.cat([tensor_data, pad], dim=0)

            data_list.append(tensor_data)
            labels.append(concentration)
            gas_labels.append(0 if gas_code == 'GAC' else 1)

    if len(data_list) > 0:
        all_data = torch.stack(data_list)
        global_mean = all_data.mean(dim=(0, 1), keepdim=True)
        global_std = all_data.std(dim=(0, 1), keepdim=True) + 1e-8
        all_data = (all_data - global_mean) / global_std
        data_list = [all_data[i] for i in range(all_data.shape[0])]

    logger.info(f"加载完成: {len(data_list)} 个样本")
    return data_list, labels, gas_labels


# ─────────────────────────────────────────────
# 数据增强 & Dataset
# ─────────────────────────────────────────────
def augment_data(sensor_data):
    augmented = sensor_data.clone()
    if np.random.random() > 0.5:
        noise = torch.randn_like(augmented) * 0.02 * torch.std(augmented)
        augmented = augmented + noise
    if np.random.random() > 0.5:
        scale = torch.rand(1).item() * 0.1 + 0.95
        augmented = augmented * scale
    if np.random.random() > 0.5:
        shift = np.random.randint(-50, 50)
        augmented = torch.roll(augmented, shifts=shift, dims=0)
    return augmented


class GasDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, labels, augment=False):
        self.data_list = data_list
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        label = self.labels[idx]
        if self.augment and torch.rand(1).item() > 0.5:
            data = augment_data(data)
        return data, label


# ─────────────────────────────────────────────
# 共用子模块（与 model2 完全一致）
# ─────────────────────────────────────────────
class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv_path = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(out_channels), nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm1d(out_channels),
        ) if in_channels != out_channels or stride != 1 else nn.Identity()
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.conv_path(x) + self.shortcut(x))


class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels * 2, channels // 2), nn.GELU(),
            nn.Linear(channels // 2, channels), nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        avg_w = self.avg_pool(x).view(b, c)
        max_w = self.max_pool(x).view(b, c)
        w = self.fc(torch.cat([avg_w, max_w], dim=1)).view(b, c, 1)
        return x * w


class TemporalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn_a2b = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                                     dropout=dropout, batch_first=True)
        self.cross_attn_b2a = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                                     dropout=dropout, batch_first=True)
        self.norm_a = nn.LayerNorm(d_model)
        self.norm_b = nn.LayerNorm(d_model)

    def forward(self, feat_a, feat_b):
        enhanced_a, _ = self.cross_attn_a2b(query=feat_a, key=feat_b, value=feat_b)
        enhanced_a = self.norm_a(feat_a + enhanced_a)
        enhanced_b, _ = self.cross_attn_b2a(query=feat_b, key=feat_a, value=feat_a)
        enhanced_b = self.norm_b(feat_b + enhanced_b)
        return enhanced_a, enhanced_b


class AdaptiveGatedFusion(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.Sigmoid()
        )
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(),
        )

    def forward(self, feat_a, feat_b):
        gate_weight = self.gate(torch.cat([feat_a, feat_b], dim=-1))
        fused = gate_weight * feat_a + (1 - gate_weight) * feat_b
        return self.projection(fused)


# ─────────────────────────────────────────────
# 完整模型（基准，与 model2 完全一致）
# ─────────────────────────────────────────────
class FullModel(nn.Module):
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 fusion_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            ResidualCNNBlock(input_dim, cnn_channels, kernel_size=7),
            nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels, cnn_channels * 2, kernel_size=5),
            nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels * 2, cnn_channels * 4, kernel_size=3),
            nn.MaxPool1d(2),
        )
        self.spatial_attn = SpatialAttention(cnn_channels * 4)
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(cnn_channels * 4, fusion_dim, kernel_size=1),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )
        self.lstm_branch = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                                   num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.2)
        self.temporal_attn = TemporalSelfAttention(lstm_hidden * 2, n_heads, dropout)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, fusion_dim)
        self.lstm_norm = nn.LayerNorm(fusion_dim)
        self.cross_attention = CrossAttention(fusion_dim, n_heads, dropout)
        self.pool_a = nn.AdaptiveAvgPool1d(1)
        self.pool_b = nn.AdaptiveAvgPool1d(1)
        self.fusion = AdaptiveGatedFusion(fusion_dim)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4), nn.BatchNorm1d(fusion_dim // 4),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = self.spatial_attn(x_cnn)
        x_cnn = self.cnn_proj(x_cnn)
        feat_a = x_cnn.permute(0, 2, 1)

        x_lstm, _ = self.lstm_branch(x)
        x_lstm = self.temporal_attn(x_lstm)
        feat_b = self.lstm_norm(self.lstm_proj(x_lstm))

        min_len = min(feat_a.size(1), feat_b.size(1))
        if feat_a.size(1) != min_len:
            feat_a = F.adaptive_avg_pool1d(feat_a.permute(0, 2, 1), min_len).permute(0, 2, 1)
        if feat_b.size(1) != min_len:
            feat_b = F.adaptive_avg_pool1d(feat_b.permute(0, 2, 1), min_len).permute(0, 2, 1)

        feat_a, feat_b = self.cross_attention(feat_a, feat_b)
        vec_a = self.pool_a(feat_a.permute(0, 2, 1)).squeeze(-1)
        vec_b = self.pool_b(feat_b.permute(0, 2, 1)).squeeze(-1)
        fused = self.fusion(vec_a, vec_b)
        return self.regressor(fused).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 1: w/o 交叉注意力（→直接拼接）
# ─────────────────────────────────────────────
class AblationNoCrossAttn(nn.Module):
    """去除CrossAttention，两分支特征各自池化后直接拼接，再经Linear投影到fusion_dim"""
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 fusion_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            ResidualCNNBlock(input_dim, cnn_channels, kernel_size=7), nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels, cnn_channels * 2, kernel_size=5), nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels * 2, cnn_channels * 4, kernel_size=3), nn.MaxPool1d(2),
        )
        self.spatial_attn = SpatialAttention(cnn_channels * 4)
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(cnn_channels * 4, fusion_dim, kernel_size=1),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )
        self.lstm_branch = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                                   num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.2)
        self.temporal_attn = TemporalSelfAttention(lstm_hidden * 2, n_heads, dropout)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, fusion_dim)
        self.lstm_norm = nn.LayerNorm(fusion_dim)
        self.pool_a = nn.AdaptiveAvgPool1d(1)
        self.pool_b = nn.AdaptiveAvgPool1d(1)
        # 直接拼接后投影回 fusion_dim（替代交叉注意力+门控）
        self.concat_proj = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim), nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4), nn.BatchNorm1d(fusion_dim // 4),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = self.spatial_attn(x_cnn)
        x_cnn = self.cnn_proj(x_cnn)
        feat_a = x_cnn  # (batch, fusion_dim, seq_len_a)

        x_lstm, _ = self.lstm_branch(x)
        x_lstm = self.temporal_attn(x_lstm)
        feat_b = self.lstm_norm(self.lstm_proj(x_lstm))  # (batch, seq_len, fusion_dim)

        vec_a = self.pool_a(feat_a).squeeze(-1)                     # (batch, fusion_dim)
        vec_b = self.pool_b(feat_b.permute(0, 2, 1)).squeeze(-1)    # (batch, fusion_dim)

        fused = self.concat_proj(torch.cat([vec_a, vec_b], dim=-1))
        return self.regressor(fused).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 2: w/o 自适应门控（→固定均值融合）
# ─────────────────────────────────────────────
class AblationFixedFusion(nn.Module):
    """保留交叉注意力，但将AdaptiveGatedFusion替换为简单的0.5*a + 0.5*b"""
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 fusion_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        self.cnn_branch = nn.Sequential(
            ResidualCNNBlock(input_dim, cnn_channels, kernel_size=7), nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels, cnn_channels * 2, kernel_size=5), nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels * 2, cnn_channels * 4, kernel_size=3), nn.MaxPool1d(2),
        )
        self.spatial_attn = SpatialAttention(cnn_channels * 4)
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(cnn_channels * 4, fusion_dim, kernel_size=1),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )
        self.lstm_branch = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                                   num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.2)
        self.temporal_attn = TemporalSelfAttention(lstm_hidden * 2, n_heads, dropout)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, fusion_dim)
        self.lstm_norm = nn.LayerNorm(fusion_dim)
        self.cross_attention = CrossAttention(fusion_dim, n_heads, dropout)
        self.pool_a = nn.AdaptiveAvgPool1d(1)
        self.pool_b = nn.AdaptiveAvgPool1d(1)
        # 固定均值融合后的投影（保持参数量接近，但无可学习门控）
        self.mean_proj = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU(),
        )
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4), nn.BatchNorm1d(fusion_dim // 4),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = self.spatial_attn(x_cnn)
        x_cnn = self.cnn_proj(x_cnn)
        feat_a = x_cnn.permute(0, 2, 1)

        x_lstm, _ = self.lstm_branch(x)
        x_lstm = self.temporal_attn(x_lstm)
        feat_b = self.lstm_norm(self.lstm_proj(x_lstm))

        min_len = min(feat_a.size(1), feat_b.size(1))
        if feat_a.size(1) != min_len:
            feat_a = F.adaptive_avg_pool1d(feat_a.permute(0, 2, 1), min_len).permute(0, 2, 1)
        if feat_b.size(1) != min_len:
            feat_b = F.adaptive_avg_pool1d(feat_b.permute(0, 2, 1), min_len).permute(0, 2, 1)

        feat_a, feat_b = self.cross_attention(feat_a, feat_b)
        vec_a = self.pool_a(feat_a.permute(0, 2, 1)).squeeze(-1)
        vec_b = self.pool_b(feat_b.permute(0, 2, 1)).squeeze(-1)

        # 固定均值融合（替代自适应门控）
        fused = self.mean_proj(0.5 * vec_a + 0.5 * vec_b)
        return self.regressor(fused).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 3: 仅CNN分支
# ─────────────────────────────────────────────
class AblationCNNOnly(nn.Module):
    """去除LSTM分支，仅使用CNN空间特征"""
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,  # noqa: W0613
                 fusion_dim=128, n_heads=4, dropout=0.3):  # lstm_hidden/n_heads kept for uniform signature
        super().__init__()
        self.cnn_branch = nn.Sequential(
            ResidualCNNBlock(input_dim, cnn_channels, kernel_size=7), nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels, cnn_channels * 2, kernel_size=5), nn.MaxPool1d(2),
            ResidualCNNBlock(cnn_channels * 2, cnn_channels * 4, kernel_size=3), nn.MaxPool1d(2),
        )
        self.spatial_attn = SpatialAttention(cnn_channels * 4)
        self.cnn_proj = nn.Sequential(
            nn.Conv1d(cnn_channels * 4, fusion_dim, kernel_size=1),
            nn.BatchNorm1d(fusion_dim), nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4), nn.BatchNorm1d(fusion_dim // 4),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, x):
        x_cnn = x.permute(0, 2, 1)
        x_cnn = self.cnn_branch(x_cnn)
        x_cnn = self.spatial_attn(x_cnn)
        x_cnn = self.cnn_proj(x_cnn)  # (batch, fusion_dim, seq_len_a)
        vec = self.pool(x_cnn).squeeze(-1)
        return self.regressor(vec).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 4: 仅LSTM分支
# ─────────────────────────────────────────────
class AblationLSTMOnly(nn.Module):
    """去除CNN分支，仅使用LSTM时序特征"""
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,  # noqa: W0613
                 fusion_dim=128, n_heads=4, dropout=0.3):  # cnn_channels kept for uniform signature
        super().__init__()
        self.lstm_branch = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden,
                                   num_layers=2, batch_first=True,
                                   bidirectional=True, dropout=0.2)
        self.temporal_attn = TemporalSelfAttention(lstm_hidden * 2, n_heads, dropout)
        self.lstm_proj = nn.Linear(lstm_hidden * 2, fusion_dim)
        self.lstm_norm = nn.LayerNorm(fusion_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.BatchNorm1d(fusion_dim // 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4), nn.BatchNorm1d(fusion_dim // 4),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(fusion_dim // 4, 1)
        )

    def forward(self, x):
        x_lstm, _ = self.lstm_branch(x)
        x_lstm = self.temporal_attn(x_lstm)
        feat = self.lstm_norm(self.lstm_proj(x_lstm))  # (batch, seq_len, fusion_dim)
        vec = self.pool(feat.permute(0, 2, 1)).squeeze(-1)
        return self.regressor(vec).squeeze(-1)


# ─────────────────────────────────────────────
# 训练 / 评估工具（与 model2 完全一致）
# ─────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=60, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        l2 = sum(p.pow(2).sum() for p in model.parameters())
        loss = loss + 5e-5 * l2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            total_loss += criterion(preds, y_batch).item()
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return total_loss / len(loader), mean_absolute_error(all_targets, all_preds), \
           r2_score(all_targets, all_preds), all_preds, all_targets


def train_model(model, train_loader, val_loader, device, epochs=500, lr=2e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=3e-3)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=60)

    best_r2 = -float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, _, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        if val_r2 > best_r2:
            best_r2 = val_r2
            best_epoch = epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if early_stopping(val_loss):
            logger.info(f"      Early stopping at epoch {epoch+1} | best_val_r2={best_r2:.4f} (ep {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_r2


# ─────────────────────────────────────────────
# 单个变体的 5 折交叉验证
# ─────────────────────────────────────────────
def run_variant(variant_name, model_fn, X_data, y_data, y_data_norm,
                y_min, y_max, device, n_folds=5, epochs=500):
    logger.info(f"\n{'='*70}")
    logger.info(f"  消融变体: {variant_name}")
    logger.info(f"{'='*70}")

    set_seed(42)
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = np.zeros(len(X_data))
    all_targets = np.zeros(len(X_data))
    fold_r2_list, fold_mae_list = [], []

    for fold_i, (train_val_idx, test_idx) in enumerate(skf.split(range(len(X_data)), y_data)):
        logger.info(f"    Fold {fold_i+1}/{n_folds} ...")

        tv_labels = [y_data[i] for i in train_val_idx]
        train_idx, val_idx = train_test_split(
            list(range(len(train_val_idx))), test_size=0.15,
            random_state=42, stratify=tv_labels)
        train_idx = [train_val_idx[i] for i in train_idx]
        val_idx   = [train_val_idx[i] for i in val_idx]

        train_dataset = GasDataset([X_data[i] for i in train_idx],
                                   [y_data_norm[i] for i in train_idx], augment=True)
        val_dataset   = GasDataset([X_data[i] for i in val_idx],
                                   [y_data_norm[i] for i in val_idx], augment=False)
        test_dataset  = GasDataset([X_data[i] for i in test_idx],
                                   [y_data_norm[i] for i in test_idx], augment=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16,
                                                   shuffle=True, num_workers=0)
        val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=16,
                                                   shuffle=False, num_workers=0)
        test_loader  = torch.utils.data.DataLoader(test_dataset,  batch_size=16,
                                                   shuffle=False, num_workers=0)

        model = model_fn().to(device)
        model, _ = train_model(model, train_loader, val_loader, device,
                               epochs=epochs, lr=2e-3)

        criterion = nn.HuberLoss(delta=1.0)
        _, _, _, preds_norm, targets_norm = evaluate(model, test_loader, criterion, device)

        preds_ppm   = preds_norm   * (y_max - y_min) + y_min
        targets_ppm = targets_norm * (y_max - y_min) + y_min

        fold_r2  = r2_score(targets_ppm, preds_ppm)
        fold_mae = mean_absolute_error(targets_ppm, preds_ppm)
        fold_r2_list.append(fold_r2)
        fold_mae_list.append(fold_mae)

        for j, idx in enumerate(test_idx):
            all_preds[idx]   = preds_ppm[j]
            all_targets[idx] = targets_ppm[j]

        logger.info(f"      Fold {fold_i+1} 测试: R²={fold_r2:.4f}, MAE={fold_mae:.2f} ppm")

    mean_r2  = float(np.mean(fold_r2_list))
    std_r2   = float(np.std(fold_r2_list, ddof=1))
    mean_mae = float(np.mean(fold_mae_list))
    std_mae  = float(np.std(fold_mae_list, ddof=1))
    logger.info(f"  [{variant_name}] 平均R²={mean_r2:.4f}±{std_r2:.4f}, "
                f"平均MAE={mean_mae:.2f}±{std_mae:.2f} ppm")

    return {
        'name': variant_name,
        'mean_r2': mean_r2, 'std_r2': std_r2,
        'mean_mae': mean_mae, 'std_mae': std_mae,
        'fold_r2': fold_r2_list, 'fold_mae': fold_mae_list,
        'all_preds': all_preds, 'all_targets': all_targets,
    }


# ─────────────────────────────────────────────
# 汇总表格
# ─────────────────────────────────────────────
def print_summary_table(results):
    baseline = results[0]
    logger.info("\n" + "=" * 80)
    logger.info("  消融实验汇总结果 (对应论文表5-4)")
    logger.info("=" * 80)
    header = f"{'变体':<34} {'平均R²':>10} {'平均MAE(ppm)':>14} {'ΔR²':>10}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results:
        delta = r['mean_r2'] - baseline['mean_r2']
        delta_str = f"{delta:+.4f}" if r['name'] != baseline['name'] else "—"
        logger.info(f"  {r['name']:<32} {r['mean_r2']:>10.4f} "
                    f"{r['mean_mae']:>14.2f} {delta_str:>10}")
    logger.info("=" * 80)


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ── 加载数据 ──
    logger.info("\n============================================================")
    logger.info("加载数据...")
    logger.info("============================================================")
    data_dir = "./大论文-数据集"
    X_data, y_data, gas_labels = load_gas_data(
        data_dir, gas_type='both', target_length=500, downsample_factor=10)

    y_min, y_max = min(y_data), max(y_data)
    y_data_norm = np.array([(y - y_min) / (y_max - y_min) for y in y_data])
    input_dim = X_data[0].shape[1]  # 12

    logger.info(f"数据集大小: {len(X_data)}, 特征维度: {input_dim}")
    logger.info(f"浓度范围: {y_min} ~ {y_max} ppm")

    # ── 定义各变体工厂函数 ──
    def make_full():
        return FullModel(input_dim=input_dim)

    def make_no_cross_attn():
        return AblationNoCrossAttn(input_dim=input_dim)

    def make_fixed_fusion():
        return AblationFixedFusion(input_dim=input_dim)

    def make_cnn_only():
        return AblationCNNOnly(input_dim=input_dim)

    def make_lstm_only():
        return AblationLSTMOnly(input_dim=input_dim)

    # ── 依次运行 5 个变体 ──
    ablation_variants = [
        ("完整模型（模型二）",            make_full),
        ("w/o 交叉注意力（→直接拼接）",  make_no_cross_attn),
        ("w/o 自适应门控（→固定均值融合）", make_fixed_fusion),
        ("仅CNN分支",                    make_cnn_only),
        ("仅LSTM分支",                   make_lstm_only),
    ]

    all_results = []
    total_start = time.time()

    for variant_name, model_fn in ablation_variants:
        result = run_variant(
            variant_name=variant_name,
            model_fn=model_fn,
            X_data=X_data,
            y_data=y_data,
            y_data_norm=y_data_norm,
            y_min=y_min, y_max=y_max,
            device=device,
            n_folds=5,
            epochs=500,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start
    logger.info(f"\n全部消融实验完成，总耗时: {total_elapsed/60:.1f} min")

    print_summary_table(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("消融实验全部完成!")
    logger.info("=" * 60)
