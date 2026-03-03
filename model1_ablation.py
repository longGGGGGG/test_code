"""
模型一消融实验 (Ablation Study for CNN-LSTM-Attention)

消融变体 (对应论文表4-5):
    1. 完整模型           : 原始 CNNLSTMAttentionModel
    2. w/o 多尺度卷积     : 仅保留 k=7 单一卷积核 (SingleScaleCNN)
    3. w/o SE 注意力      : 去除 ChannelAttention 模块
    4. w/o BiLSTM (→LSTM) : 双向 LSTM 改为单向
    5. w/o 数据增强       : 训练集不施加任何增强

每个变体均采用相同的 5 折分层交叉验证、超参数和随机种子，
最终汇总各变体的平均 R² / MAE 及与完整模型的差值 ΔR²。
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
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False


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


logger = setup_logger('ablation', 'model1_ablation.log')


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
# 数据加载（与 model1 完全一致）
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
        logger.info(f"  [{dir_name}] 发现 {len(files)} 个文件")

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
# 数据增强
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
# 共用子模块
# ─────────────────────────────────────────────
class ChannelAttention(nn.Module):
    """SE 通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(x + self.dropout(attn_out))
        return x, attn_weights


# ─────────────────────────────────────────────
# CNN 变体
# ─────────────────────────────────────────────
class MultiScaleCNN(nn.Module):
    """完整版：k=3/7/11 并行"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_small = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels), nn.GELU())
        self.conv_medium = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels), nn.GELU())
        self.conv_large = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5),
            nn.BatchNorm1d(out_channels), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Conv1d(out_channels * 3, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels), nn.GELU())

    def forward(self, x):
        s = self.conv_small(x)
        m = self.conv_medium(x)
        l = self.conv_large(x)
        return self.fusion(torch.cat([s, m, l], dim=1))


class SingleScaleCNN(nn.Module):
    """消融版：仅保留 k=7 单一卷积核"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels), nn.GELU())

    def forward(self, x):
        return self.conv(x)


# ─────────────────────────────────────────────
# 完整模型（基准）
# ─────────────────────────────────────────────
class FullModel(nn.Module):
    """完整 CNN-BiLSTM-Attention 模型"""
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 n_heads=4, mlp_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn_block1 = MultiScaleCNN(input_dim, cnn_channels)
        self.channel_attn1 = ChannelAttention(cnn_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn_block2 = MultiScaleCNN(cnn_channels, cnn_channels * 2)
        self.channel_attn2 = ChannelAttention(cnn_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.temporal_attention = TemporalMultiHeadAttention(lstm_hidden * 2, n_heads, dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        pool_dim = lstm_hidden * 2 * 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, mlp_hidden), nn.BatchNorm1d(mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.BatchNorm1d(mlp_hidden // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.channel_attn1(self.cnn_block1(x)))
        x = self.pool2(self.channel_attn2(self.cnn_block2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.temporal_attention(x)
        x_t = x.permute(0, 2, 1)
        x = torch.cat([self.global_avg_pool(x_t).squeeze(-1),
                        self.global_max_pool(x_t).squeeze(-1)], dim=1)
        return self.mlp(x).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 1: w/o 多尺度卷积（仅 k=7）
# ─────────────────────────────────────────────
class AblationNoMultiScale(nn.Module):
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 n_heads=4, mlp_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn_block1 = SingleScaleCNN(input_dim, cnn_channels)
        self.channel_attn1 = ChannelAttention(cnn_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn_block2 = SingleScaleCNN(cnn_channels, cnn_channels * 2)
        self.channel_attn2 = ChannelAttention(cnn_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.temporal_attention = TemporalMultiHeadAttention(lstm_hidden * 2, n_heads, dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        pool_dim = lstm_hidden * 2 * 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, mlp_hidden), nn.BatchNorm1d(mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.BatchNorm1d(mlp_hidden // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.channel_attn1(self.cnn_block1(x)))
        x = self.pool2(self.channel_attn2(self.cnn_block2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.temporal_attention(x)
        x_t = x.permute(0, 2, 1)
        x = torch.cat([self.global_avg_pool(x_t).squeeze(-1),
                        self.global_max_pool(x_t).squeeze(-1)], dim=1)
        return self.mlp(x).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 2: w/o SE 注意力（去除 ChannelAttention）
# ─────────────────────────────────────────────
class AblationNoSE(nn.Module):
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 n_heads=4, mlp_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn_block1 = MultiScaleCNN(input_dim, cnn_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn_block2 = MultiScaleCNN(cnn_channels, cnn_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.temporal_attention = TemporalMultiHeadAttention(lstm_hidden * 2, n_heads, dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        pool_dim = lstm_hidden * 2 * 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, mlp_hidden), nn.BatchNorm1d(mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.BatchNorm1d(mlp_hidden // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.cnn_block1(x))   # 无 SE
        x = self.pool2(self.cnn_block2(x))   # 无 SE
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.temporal_attention(x)
        x_t = x.permute(0, 2, 1)
        x = torch.cat([self.global_avg_pool(x_t).squeeze(-1),
                        self.global_max_pool(x_t).squeeze(-1)], dim=1)
        return self.mlp(x).squeeze(-1)


# ─────────────────────────────────────────────
# 消融变体 3: w/o BiLSTM（单向 LSTM）
# ─────────────────────────────────────────────
class AblationUniLSTM(nn.Module):
    def __init__(self, input_dim=12, cnn_channels=32, lstm_hidden=64,
                 n_heads=4, mlp_hidden=128, dropout=0.3):
        super().__init__()
        self.cnn_block1 = MultiScaleCNN(input_dim, cnn_channels)
        self.channel_attn1 = ChannelAttention(cnn_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.cnn_block2 = MultiScaleCNN(cnn_channels, cnn_channels * 2)
        self.channel_attn2 = ChannelAttention(cnn_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 单向 LSTM（bidirectional=False），输出维度为 lstm_hidden（非 *2）
        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2, hidden_size=lstm_hidden,
            num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)
        self.temporal_attention = TemporalMultiHeadAttention(lstm_hidden, n_heads, dropout)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

        pool_dim = lstm_hidden * 2  # avg + max，但 lstm_hidden 未乘 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_dim, mlp_hidden), nn.BatchNorm1d(mlp_hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden // 2), nn.BatchNorm1d(mlp_hidden // 2),
            nn.GELU(), nn.Dropout(dropout * 0.5),
            nn.Linear(mlp_hidden // 2, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.pool1(self.channel_attn1(self.cnn_block1(x)))
        x = self.pool2(self.channel_attn2(self.cnn_block2(x)))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x, _ = self.temporal_attention(x)
        x_t = x.permute(0, 2, 1)
        x = torch.cat([self.global_avg_pool(x_t).squeeze(-1),
                        self.global_max_pool(x_t).squeeze(-1)], dim=1)
        return self.mlp(x).squeeze(-1)


# ─────────────────────────────────────────────
# 训练 / 评估工具（与 model1 完全一致）
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
    mae = mean_absolute_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return total_loss / len(loader), mae, r2, all_preds, all_targets


def train_model(model, train_loader, val_loader, device, epochs=500, lr=3e-3):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    criterion = nn.HuberLoss(delta=1.0)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=60)

    best_r2 = -float('inf')
    best_state = None
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mae, val_r2, _, _ = evaluate(model, val_loader, criterion, device)
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
                y_min, y_max, device, use_augment=True, n_folds=5, epochs=500):
    """
    model_fn: 无参调用返回模型实例的工厂函数
    use_augment: False 对应 w/o 数据增强 变体
    """
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
                                   [y_data_norm[i] for i in train_idx],
                                   augment=use_augment)
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
                               epochs=epochs, lr=3e-3)

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

    mean_r2  = np.mean(fold_r2_list)
    std_r2   = np.std(fold_r2_list)
    mean_mae = np.mean(fold_mae_list)
    std_mae  = np.std(fold_mae_list)
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
# 汇总表格打印
# ─────────────────────────────────────────────
def print_summary_table(results):
    baseline = results[0]  # 完整模型
    logger.info("\n" + "=" * 80)
    logger.info("  消融实验汇总结果 (对应论文表4-5)")
    logger.info("=" * 80)
    header = f"{'变体':<30} {'平均R²':>10} {'平均MAE(ppm)':>14} {'ΔR²':>10}"
    logger.info(header)
    logger.info("-" * 80)
    for r in results:
        delta = r['mean_r2'] - baseline['mean_r2']
        delta_str = f"{delta:+.4f}" if r['name'] != baseline['name'] else "—"
        logger.info(f"  {r['name']:<28} {r['mean_r2']:>10.4f} "
                    f"{r['mean_mae']:>14.2f} {delta_str:>10}")
    logger.info("=" * 80)


# ─────────────────────────────────────────────
# 结果可视化
# ─────────────────────────────────────────────
def plot_ablation_results(results, save_path='model1_ablation_results.png'):
    names    = [r['name'] for r in results]
    mean_r2  = [r['mean_r2']  for r in results]
    mean_mae = [r['mean_mae'] for r in results]
    std_r2   = [r['std_r2']   for r in results]
    std_mae  = [r['std_mae']  for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model 1 Ablation Study', fontsize=14, fontweight='bold')

    # R² 柱状图
    bars1 = ax1.bar(x, mean_r2, width, yerr=std_r2, capsize=4,
                    color=['#2166AC'] + ['#92C5DE'] * (len(names) - 1),
                    edgecolor='white', linewidth=0.8)
    ax1.set_ylabel('Mean R²', fontsize=11)
    ax1.set_title('Test R² per Variant', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax1.set_ylim(max(0, min(mean_r2) - 0.08), min(1.0, max(mean_r2) + 0.04))
    for bar, val in zip(bars1, mean_r2):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # MAE 柱状图
    bars2 = ax2.bar(x, mean_mae, width, yerr=std_mae, capsize=4,
                    color=['#D6604D'] + ['#F4A582'] * (len(names) - 1),
                    edgecolor='white', linewidth=0.8)
    ax2.set_ylabel('Mean MAE (ppm)', fontsize=11)
    ax2.set_title('Test MAE per Variant', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    ax2.set_ylim(0, max(mean_mae) * 1.2)
    for bar, val in zip(bars2, mean_mae):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"\n消融实验结果图已保存: {save_path}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # ── 加载数据 ──
    logger.info("\n加载数据...")
    data_dir = "./大论文-数据集"
    X_data, y_data, gas_labels = load_gas_data(
        data_dir, gas_type='both', target_length=500, downsample_factor=10)

    y_min, y_max = min(y_data), max(y_data)
    y_data_norm = np.array([(y - y_min) / (y_max - y_min) for y in y_data])
    input_dim = X_data[0].shape[1]  # 12

    logger.info(f"数据集大小: {len(X_data)}, 特征维度: {input_dim}")

    # ── 定义各变体工厂函数 ──
    def make_full():
        return FullModel(input_dim=input_dim)

    def make_no_multiscale():
        return AblationNoMultiScale(input_dim=input_dim)

    def make_no_se():
        return AblationNoSE(input_dim=input_dim)

    def make_uni_lstm():
        return AblationUniLSTM(input_dim=input_dim)

    # ── 依次运行 5 个变体 ──
    # 注意：完整模型也在此处重跑（独立基准），与保存的最佳权重无关
    ablation_variants = [
        ("Full Model",           make_full,          True),
        ("w/o Multi-Scale CNN",  make_no_multiscale, True),
        ("w/o SE Attention",     make_no_se,         True),
        ("w/o BiLSTM (→LSTM)",   make_uni_lstm,      True),
        ("w/o Data Augmentation", make_full,         False),   # 同架构，关闭增强
    ]

    all_results = []
    total_start = time.time()

    for variant_name, model_fn, use_aug in ablation_variants:
        result = run_variant(
            variant_name=variant_name,
            model_fn=model_fn,
            X_data=X_data,
            y_data=y_data,
            y_data_norm=y_data_norm,
            y_min=y_min, y_max=y_max,
            device=device,
            use_augment=use_aug,
            n_folds=5,
            epochs=500,
        )
        all_results.append(result)

    total_elapsed = time.time() - total_start
    logger.info(f"\n全部消融实验完成，总耗时: {total_elapsed/60:.1f} min")

    # ── 汇总 & 可视化 ──
    print_summary_table(all_results)
    plot_ablation_results(all_results, save_path='model1_ablation_results.png')

    logger.info("\n" + "=" * 60)
    logger.info("消融实验全部完成!")
    logger.info("=" * 60)
