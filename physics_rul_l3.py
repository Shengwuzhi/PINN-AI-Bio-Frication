"""
Physics-Constrained Recurrent Model for Level 3 RUL Prediction
物理约束循环模型 - 用于L3剩余使用寿命预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import os

# 物理常数
R = 8.314  # 气体常数 (J/(mol·K))

# ============================================================================
# Physics-Constrained RUL Model
# ============================================================================

class PhysicsRULModel(nn.Module):
    """
    物理约束的RUL预测模型
    结合LSTM和Arrhenius动力学方程
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, 
                 physics_dim=32, static_feature_dim=3, dropout=0.2):
        """
        参数:
        - input_dim: 输入特征维度（温度、振动 = 2）
        - hidden_dim: LSTM隐藏层维度
        - num_layers: LSTM层数
        - physics_dim: 物理分支的隐藏层维度
        - static_feature_dim: 静态特征维度（用于估计Ea和Pre_factor）
        - dropout: Dropout比率
        """
        super(PhysicsRULModel, self).__init__()
        
        # LSTM分支：提取时间序列特征
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 系统状态提取
        self.state_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )
        
        # 物理分支：从静态特征估计活化能和预指数因子
        # 静态特征可以是：链长、对称性、极性等
        self.physics_branch = nn.Sequential(
            nn.Linear(static_feature_dim, physics_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(physics_dim, physics_dim // 2),
            nn.ReLU(),
            nn.Linear(physics_dim // 2, 2)  # 输出: [Ea, log_A]
        )
        
        # 退化率预测层
        # 输入：系统状态 + 当前温度 + 物理参数
        self.degradation_rate_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 1 + 2, 64),  # 状态 + 温度 + [Ea, log_A]
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出：退化率 dk/dt
        )
        
        # RUL预测层（从健康度预测RUL）
        self.rul_predictor = nn.Sequential(
            nn.Linear(hidden_dim // 2 + 1, 32),  # 状态 + 当前健康度
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 输出：RUL
        )
        
    def forward(self, time_series, static_features, return_intermediate=False):
        """
        前向传播
        
        参数:
        - time_series: [batch_size, seq_len, 2] (温度, 振动)
        - static_features: [batch_size, static_feature_dim] (静态特征)
        - return_intermediate: 是否返回中间结果
        
        返回:
        - rul_predictions: [batch_size, seq_len] RUL预测
        - degradation_rates: [batch_size, seq_len] 退化率预测
        - health_predictions: [batch_size, seq_len] 健康度预测
        """
        batch_size, seq_len, _ = time_series.shape
        
        # LSTM处理时间序列
        lstm_out, (h_n, c_n) = self.lstm(time_series)
        # lstm_out: [batch_size, seq_len, hidden_dim]
        
        # 提取系统状态（使用最后一个时间步的隐藏状态）
        # 也可以使用每个时间步的状态
        system_states = []
        for t in range(seq_len):
            state = self.state_extractor(lstm_out[:, t, :])
            system_states.append(state)
        system_states = torch.stack(system_states, dim=1)  # [batch_size, seq_len, hidden_dim//2]
        
        # 物理分支：估计活化能和预指数因子
        physics_params = self.physics_branch(static_features)  # [batch_size, 2]
        Ea = physics_params[:, 0:1]  # 活化能 (kJ/mol)
        log_A = physics_params[:, 1:2]  # log(预指数因子)
        A = torch.exp(log_A)  # 预指数因子
        
        # 提取温度序列
        temperatures = time_series[:, :, 0:1]  # [batch_size, seq_len, 1]
        temperatures_kelvin = temperatures + 273.15  # 转换为开尔文
        
        # 计算Arrhenius退化率
        # k = A * exp(-Ea / (R * T))
        # 这里我们使用物理分支预测的参数
        Ea_j_per_mol = Ea * 1000  # 转换为J/mol [batch_size, 1]
        # 扩展维度以匹配temperatures_kelvin [batch_size, seq_len, 1]
        Ea_expanded = Ea_j_per_mol.unsqueeze(1)  # [batch_size, 1, 1]
        A_expanded = A.unsqueeze(1)  # [batch_size, 1, 1]
        arrhenius_rate = A_expanded * torch.exp(-Ea_expanded / (R * temperatures_kelvin))
        # arrhenius_rate: [batch_size, seq_len, 1]
        
        # 预测退化率（结合LSTM状态和物理参数）
        degradation_rates = []
        health_predictions = []
        rul_predictions = []
        
        # 初始健康度（假设为1.0）
        health = torch.ones(batch_size, 1, device=time_series.device)
        health_predictions.append(health)
        
        for t in range(seq_len):
            # 当前系统状态
            current_state = system_states[:, t, :]  # [batch_size, hidden_dim//2]
            current_temp = temperatures[:, t, :]  # [batch_size, 1]
            
            # 预测退化率（结合LSTM状态和物理参数）
            rate_input = torch.cat([
                current_state,
                current_temp,
                Ea,
                log_A
            ], dim=1)
            
            predicted_rate = self.degradation_rate_predictor(rate_input)  # [batch_size, 1]
            degradation_rates.append(predicted_rate)
            
            # 使用物理约束的退化率更新健康度
            # 也可以使用Arrhenius速率
            dt = 1.0  # 时间步长（小时）
            # 结合预测的退化率和Arrhenius速率
            combined_rate = 0.7 * predicted_rate + 0.3 * arrhenius_rate[:, t, :]
            
            # 更新健康度（单调递减）
            health = health - combined_rate * dt
            health = torch.clamp(health, min=0.0, max=1.0)
            health_predictions.append(health)
            
            # 预测RUL（从当前健康度）
            rul_input = torch.cat([current_state, health], dim=1)
            rul = self.rul_predictor(rul_input)  # [batch_size, 1]
            rul = torch.clamp(rul, min=0.0)  # RUL不能为负
            rul_predictions.append(rul)
        
        # 堆叠结果
        degradation_rates = torch.stack(degradation_rates, dim=1)  # [batch_size, seq_len, 1]
        health_predictions = torch.stack(health_predictions[1:], dim=1)  # [batch_size, seq_len, 1]
        rul_predictions = torch.stack(rul_predictions, dim=1)  # [batch_size, seq_len, 1]
        
        if return_intermediate:
            return {
                'rul': rul_predictions,
                'degradation_rate': degradation_rates,
                'health': health_predictions,
                'Ea': Ea,
                'A': A,
                'arrhenius_rate': arrhenius_rate
            }
        else:
            return rul_predictions, degradation_rates, health_predictions

# ============================================================================
# Standard LSTM Model (for comparison)
# ============================================================================

class StandardLSTMModel(nn.Module):
    """
    标准LSTM模型（用于对比）
    不包含物理约束
    """
    
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=2, dropout=0.2):
        super(StandardLSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # 直接预测RUL
        )
        
    def forward(self, time_series):
        lstm_out, _ = self.lstm(time_series)
        # 使用每个时间步的隐藏状态预测RUL
        rul_predictions = self.predictor(lstm_out)  # [batch_size, seq_len, 1]
        return rul_predictions

# ============================================================================
# 数据加载和预处理
# ============================================================================

def load_rul_data(sensor_data_path='physics_sensor_data.npz', 
                  static_data_path='physics_lubricant_data.csv'):
    """
    加载RUL预测数据
    """
    # 加载时间序列数据
    sensor_data = np.load(sensor_data_path)
    temperature = sensor_data['temperature']  # [n_samples, n_steps]
    vibration = sensor_data['vibration']  # [n_samples, n_steps]
    health = sensor_data['health']  # [n_samples, n_steps]
    
    # 计算真实RUL（健康度降到阈值以下的时间步）
    threshold = 0.5
    true_rul = []
    for i in range(len(health)):
        below_threshold = np.where(health[i] < threshold)[0]
        if len(below_threshold) > 0:
            rul = below_threshold[0]
        else:
            rul = len(health[i])  # 如果整个周期内都未降到阈值以下
        true_rul.append(rul)
    true_rul = np.array(true_rul)
    
    # 加载静态特征
    static_df = pd.read_csv(static_data_path)
    static_features = static_df[['chain_length', 'symmetry', 'polarity']].values
    
    # 组合时间序列特征
    time_series = np.stack([temperature, vibration], axis=-1)  # [n_samples, n_steps, 2]
    
    # 为每个时间步计算RUL（从当前时间步到失效的时间）
    rul_sequences = []
    for i in range(len(health)):
        rul_seq = []
        for t in range(len(health[i])):
            # 从时间步t开始，找到健康度降到阈值以下的时间
            remaining_health = health[i][t:]
            below_threshold = np.where(remaining_health < threshold)[0]
            if len(below_threshold) > 0:
                rul_t = below_threshold[0]
            else:
                rul_t = len(remaining_health)
            rul_seq.append(rul_t)
        rul_sequences.append(rul_seq)
    rul_sequences = np.array(rul_sequences)  # [n_samples, n_steps]
    
    return time_series, static_features, rul_sequences, true_rul, health

# ============================================================================
# 训练函数
# ============================================================================

def train_physics_rul_model(model, train_loader, val_loader, num_epochs=100, 
                            lr=0.001, device='cpu'):
    """
    训练物理约束RUL模型
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_time_series, batch_static, batch_rul_true in train_loader:
            batch_time_series = batch_time_series.to(device)
            batch_static = batch_static.to(device)
            batch_rul_true = batch_rul_true.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            rul_pred, degradation_rate, health_pred = model(batch_time_series, batch_static)
            
            # RUL预测损失
            rul_loss = nn.MSELoss()(rul_pred, batch_rul_true)
            
            # 单调性约束损失（RUL不应该增加）
            # 计算相邻时间步的RUL差值
            rul_diff = rul_pred[:, 1:, :] - rul_pred[:, :-1, :]  # [batch_size, seq_len-1, 1]
            # RUL不应该增加（允许小幅波动，但不应该大幅增加）
            monotonicity_loss = torch.mean(torch.relu(rul_diff - 0.1))  # 允许0.1的容差
            
            # 总损失
            loss = rul_loss + 0.1 * monotonicity_loss
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_time_series, batch_static, batch_rul_true in val_loader:
                batch_time_series = batch_time_series.to(device)
                batch_static = batch_static.to(device)
                batch_rul_true = batch_rul_true.to(device)
                
                rul_pred, degradation_rate, health_pred = model(batch_time_series, batch_static)
                
                rul_loss = nn.MSELoss()(rul_pred, batch_rul_true)
                rul_diff = rul_pred[:, 1:, :] - rul_pred[:, :-1, :]
                monotonicity_loss = torch.mean(torch.relu(rul_diff - 0.1))
                loss = rul_loss + 0.1 * monotonicity_loss
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'physics_rul_best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

def train_standard_lstm(model, train_loader, val_loader, num_epochs=100, 
                        lr=0.001, device='cpu'):
    """
    训练标准LSTM模型（用于对比）
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for batch_time_series, batch_rul_true in train_loader:
            batch_time_series = batch_time_series.to(device)
            batch_rul_true = batch_rul_true.to(device)
            
            optimizer.zero_grad()
            
            rul_pred = model(batch_time_series)
            loss = nn.MSELoss()(rul_pred, batch_rul_true)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_time_series, batch_rul_true in val_loader:
                batch_time_series = batch_time_series.to(device)
                batch_rul_true = batch_rul_true.to(device)
                
                rul_pred = model(batch_time_series)
                loss = nn.MSELoss()(rul_pred, batch_rul_true)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'standard_lstm_best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
    
    return train_losses, val_losses

# ============================================================================
# 可视化函数
# ============================================================================

def plot_comparison(physics_model, standard_model, test_loader, device='cpu', n_samples=3):
    """
    对比物理约束模型和标准LSTM模型的预测结果
    """
    physics_model.eval()
    standard_model.eval()
    
    # 加载最佳模型
    if os.path.exists('physics_rul_best_model.pth'):
        physics_model.load_state_dict(torch.load('physics_rul_best_model.pth', map_location=device))
    if os.path.exists('standard_lstm_best_model.pth'):
        standard_model.load_state_dict(torch.load('standard_lstm_best_model.pth', map_location=device))
    
    physics_model = physics_model.to(device)
    standard_model = standard_model.to(device)
    
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_time_series, batch_static, batch_rul_true in test_loader:
            if sample_count >= n_samples:
                break
            
            batch_time_series = batch_time_series.to(device)
            batch_static = batch_static.to(device)
            batch_rul_true = batch_rul_true.to(device)
            
            # 物理约束模型预测
            rul_physics, _, _ = physics_model(batch_time_series, batch_static)
            
            # 标准LSTM模型预测
            rul_standard = standard_model(batch_time_series)
            
            # 绘制每个样本
            for i in range(min(batch_time_series.size(0), n_samples - sample_count)):
                ax = axes[sample_count]
                
                time_steps = np.arange(rul_physics.size(1))
                
                # 真实RUL
                true_rul = batch_rul_true[i, :, 0].cpu().numpy()
                ax.plot(time_steps, true_rul, 'g-', linewidth=2, label='True RUL', alpha=0.7)
                
                # 物理约束模型预测
                pred_physics = rul_physics[i, :, 0].cpu().numpy()
                ax.plot(time_steps, pred_physics, 'b-', linewidth=2, label='Physics-Constrained', alpha=0.7)
                
                # 标准LSTM模型预测
                pred_standard = rul_standard[i, :, 0].cpu().numpy()
                ax.plot(time_steps, pred_standard, 'r--', linewidth=2, label='Standard LSTM', alpha=0.7)
                
                ax.set_xlabel('Time Step', fontsize=11)
                ax.set_ylabel('RUL', fontsize=11)
                ax.set_title(f'Sample {sample_count + 1}: RUL Prediction Comparison', fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                sample_count += 1
                if sample_count >= n_samples:
                    break
    
    plt.tight_layout()
    plt.savefig('rul_prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("RUL预测对比图已保存为 'rul_prediction_comparison.png'")
    plt.close()

# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主训练程序
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print("加载数据...")
    time_series, static_features, rul_sequences, true_rul, health = load_rul_data()
    
    print(f"时间序列形状: {time_series.shape}")
    print(f"静态特征形状: {static_features.shape}")
    print(f"RUL序列形状: {rul_sequences.shape}")
    print(f"真实RUL范围: {true_rul.min()} - {true_rul.max()}")
    
    # 数据标准化
    # 时间序列标准化
    n_samples, n_steps, n_features = time_series.shape
    time_series_reshaped = time_series.reshape(-1, n_features)
    scaler_time = StandardScaler()
    time_series_scaled = scaler_time.fit_transform(time_series_reshaped)
    time_series_scaled = time_series_scaled.reshape(n_samples, n_steps, n_features)
    
    # 静态特征标准化
    scaler_static = StandardScaler()
    static_features_scaled = scaler_static.fit_transform(static_features)
    
    # RUL序列标准化
    rul_reshaped = rul_sequences.reshape(-1, 1)
    scaler_rul = StandardScaler()
    rul_scaled = scaler_rul.fit_transform(rul_reshaped)
    rul_scaled = rul_scaled.reshape(n_samples, n_steps, 1)
    
    # 划分训练集和验证集
    X_train, X_val, S_train, S_val, Y_train, Y_val = train_test_split(
        time_series_scaled, static_features_scaled, rul_scaled,
        test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    S_train = torch.FloatTensor(S_train)
    S_val = torch.FloatTensor(S_val)
    Y_train = torch.FloatTensor(Y_train)
    Y_val = torch.FloatTensor(Y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, S_train, Y_train)
    val_dataset = TensorDataset(X_val, S_val, Y_val)
    test_dataset = TensorDataset(X_val, S_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 训练物理约束模型
    print("\n训练物理约束RUL模型...")
    physics_model = PhysicsRULModel(
        input_dim=2, hidden_dim=64, num_layers=2,
        physics_dim=32, static_feature_dim=3, dropout=0.2
    )
    print(f"模型参数数量: {sum(p.numel() for p in physics_model.parameters()):,}")
    
    train_losses_physics, val_losses_physics = train_physics_rul_model(
        physics_model, train_loader, val_loader,
        num_epochs=50, lr=0.001, device=device
    )
    
    # 训练标准LSTM模型（用于对比）
    print("\n训练标准LSTM模型（用于对比）...")
    standard_model = StandardLSTMModel(input_dim=2, hidden_dim=64, num_layers=2, dropout=0.2)
    
    # 为标准LSTM创建数据加载器（不需要静态特征）
    train_dataset_std = TensorDataset(X_train, Y_train)
    val_dataset_std = TensorDataset(X_val, Y_val)
    train_loader_std = DataLoader(train_dataset_std, batch_size=16, shuffle=True)
    val_loader_std = DataLoader(val_dataset_std, batch_size=16, shuffle=False)
    
    train_losses_std, val_losses_std = train_standard_lstm(
        standard_model, train_loader_std, val_loader_std,
        num_epochs=50, lr=0.001, device=device
    )
    
    # 绘制训练曲线对比
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses_physics, label='Physics Model Train', color='blue')
    plt.plot(val_losses_physics, label='Physics Model Val', color='blue', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Physics-Constrained Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses_std, label='Standard LSTM Train', color='red')
    plt.plot(val_losses_std, label='Standard LSTM Val', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Standard LSTM Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rul_training_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("训练曲线对比图已保存为 'rul_training_curves_comparison.png'")
    plt.close()
    
    # 绘制预测对比图
    print("\n生成预测对比图...")
    plot_comparison(physics_model, standard_model, test_loader, device=device, n_samples=3)
    
    print("训练完成！")

if __name__ == "__main__":
    main()

