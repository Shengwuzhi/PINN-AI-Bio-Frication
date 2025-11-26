"""
Level 2.5 Bench Transfer Learning Layer
台架迁移学习层 - 从L2物理输出预测燃油经济性
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

# ============================================================================
# Bench Transfer Model
# ============================================================================

class BenchTransferModel(nn.Module):
    """
    台架迁移学习模型
    从L2物理输出（粘度）和发动机状态预测燃油经济性
    """
    
    def __init__(self, hidden_dims=[64, 128, 64], dropout=0.2):
        """
        参数:
        - hidden_dims: 隐藏层维度列表
        - dropout: Dropout比率
        """
        super(BenchTransferModel, self).__init__()
        
        # 输入特征维度：
        # - viscosity (1维)
        # - speed (1维)
        # - load (1维)
        # - stribeck_number (1维，显式计算)
        # - log_stribeck_number (1维，对数尺度)
        # 总共5维
        input_dim = 5
        
        # 构建MLP网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # 输出层：预测燃油经济性评分
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def calculate_stribeck_number_torch(self, viscosity, speed, load, contact_area=1e-4):
        """
        显式计算Stribeck数（PyTorch版本，支持梯度反向传播）
        S = (Viscosity * Speed) / (Load / Area)
        
        参数:
        - viscosity: 粘度 (mPa·s) [batch_size, 1] 或 [batch_size]
        - speed: 速度 (RPM) [batch_size, 1] 或 [batch_size]
        - load: 载荷 (N) [batch_size, 1] 或 [batch_size]
        - contact_area: 接触面积 (m^2)，默认1e-4
        
        返回:
        - stribeck: Stribeck数 [batch_size, 1]
        """
        # 确保是2D张量
        if viscosity.dim() == 1:
            viscosity = viscosity.unsqueeze(1)
        if speed.dim() == 1:
            speed = speed.unsqueeze(1)
        if load.dim() == 1:
            load = load.unsqueeze(1)
        
        # 将速度从RPM转换为m/s (假设曲轴半径为0.1m)
        speed_mps = speed * 2 * torch.tensor(np.pi, device=viscosity.device) * 0.1 / 60  # m/s
        
        # 粘度单位转换为Pa·s
        viscosity_pas = viscosity * 1e-3  # mPa·s -> Pa·s
        
        # 计算压力
        pressure = load / contact_area  # Pa
        
        # 计算Stribeck数
        # 添加epsilon避免除零
        epsilon = 1e-6
        stribeck = (viscosity_pas * speed_mps) / (pressure + epsilon)
        
        # 使用对数尺度（因为Stribeck数范围很小，对数尺度更稳定）
        log_stribeck = torch.log(stribeck + 1e-10)
        
        return stribeck, log_stribeck
    
    def forward(self, viscosity, speed, load):
        """
        前向传播
        
        参数:
        - viscosity: 粘度 [batch_size, 1] 或 [batch_size] (mPa·s)
        - speed: 速度 [batch_size, 1] 或 [batch_size] (RPM)
        - load: 载荷 [batch_size, 1] 或 [batch_size] (N)
        
        返回:
        - fuel_economy: 燃油经济性评分 [batch_size, 1]
        - stribeck: Stribeck数 [batch_size, 1]
        """
        # 确保输入是2D张量
        if viscosity.dim() == 1:
            viscosity = viscosity.unsqueeze(1)
        if speed.dim() == 1:
            speed = speed.unsqueeze(1)
        if load.dim() == 1:
            load = load.unsqueeze(1)
        
        # 显式计算Stribeck数（在模型内部，使用PyTorch以支持梯度）
        stribeck, log_stribeck = self.calculate_stribeck_number_torch(viscosity, speed, load)
        
        # 拼接特征：viscosity, speed, load, stribeck, log_stribeck
        features = torch.cat([viscosity, speed, load, stribeck, log_stribeck], dim=1)
        
        # 通过网络预测燃油经济性
        fuel_economy = self.network(features)
        
        return fuel_economy, (stribeck, log_stribeck)

# ============================================================================
# 数据加载和预处理
# ============================================================================

def load_bench_data(csv_path='physics_lubricant_data.csv', temperature=40.0):
    """
    加载台架测试数据
    
    参数:
    - csv_path: CSV文件路径
    - temperature: 使用的温度（40°C, 100°C, 或-20°C），默认40°C
    
    返回:
    - X: 输入特征 [viscosity, speed, load]
    - y: 目标值 [fuel_economy_score]
    - stribeck_true: 真实的Stribeck数（用于验证）
    """
    df = pd.read_csv(csv_path)
    
    # 根据温度选择粘度值
    if temperature == 40.0:
        viscosity_col = 'viscosity_40C'
    elif temperature == 100.0:
        viscosity_col = 'viscosity_100C'
    elif temperature == -20.0:
        viscosity_col = 'viscosity_minus20C'
    else:
        raise ValueError(f"不支持的温度: {temperature}")
    
    # 输入特征
    viscosity = df[viscosity_col].values
    speed = df['engine_speed_RPM'].values
    load = df['engine_load_N'].values
    
    # 目标值
    fuel_economy = df['fuel_economy_score'].values
    
    # 真实的Stribeck数（用于验证）
    stribeck_true = df['stribeck_number'].values
    
    # 组合输入特征
    X = np.column_stack([viscosity, speed, load])
    y = fuel_economy.reshape(-1, 1)
    
    return X, y, stribeck_true

# ============================================================================
# 训练函数
# ============================================================================

def train_bench_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """
    训练台架迁移模型
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # 分离输入特征
            viscosity = batch_X[:, 0:1]
            speed = batch_X[:, 1:2]
            load = batch_X[:, 2:3]
            
            optimizer.zero_grad()
            
            # 前向传播
            pred_fuel_economy, (stribeck, _) = model(viscosity, speed, load)
            
            # 计算损失
            loss = criterion(pred_fuel_economy, batch_y)
            
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
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                viscosity = batch_X[:, 0:1]
                speed = batch_X[:, 1:2]
                load = batch_X[:, 2:3]
                
                pred_fuel_economy, (stribeck, _) = model(viscosity, speed, load)
                loss = criterion(pred_fuel_economy, batch_y)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'bench_transfer_best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

# ============================================================================
# 可视化函数
# ============================================================================

def plot_stribeck_vs_fuel_economy(model, val_loader, stribeck_true_val, device='cpu', n_samples=500):
    """
    绘制Stribeck数 vs 预测燃油经济性的散点图
    用于验证模型是否学习到经典的"U形"摩擦曲线行为
    """
    model.eval()
    model = model.to(device)
    
    # 加载最佳模型
    if os.path.exists('bench_transfer_best_model.pth'):
        model.load_state_dict(torch.load('bench_transfer_best_model.pth', map_location=device))
    
    all_stribeck = []
    all_pred_fuel = []
    all_true_fuel = []
    
    with torch.no_grad():
        for i, (batch_X, batch_y) in enumerate(val_loader):
            if len(all_stribeck) >= n_samples:
                break
                
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            viscosity = batch_X[:, 0:1]
            speed = batch_X[:, 1:2]
            load = batch_X[:, 2:3]
            
            pred_fuel_economy, (stribeck, _) = model(viscosity, speed, load)
            
            all_stribeck.extend(stribeck.cpu().numpy().flatten())
            all_pred_fuel.extend(pred_fuel_economy.cpu().numpy().flatten())
            all_true_fuel.extend(batch_y.cpu().numpy().flatten())
    
    # 限制样本数量
    all_stribeck = np.array(all_stribeck[:n_samples])
    all_pred_fuel = np.array(all_pred_fuel[:n_samples])
    all_true_fuel = np.array(all_true_fuel[:n_samples])
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：预测值 vs Stribeck数
    axes[0].scatter(all_stribeck, all_pred_fuel, alpha=0.6, s=20, c='blue', label='Predicted')
    axes[0].set_xlabel('Stribeck Number', fontsize=12)
    axes[0].set_ylabel('Predicted Fuel Economy Score', fontsize=12)
    axes[0].set_title('Stribeck Number vs Predicted Fuel Economy', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # 右图：真实值 vs Stribeck数（用于对比）
    axes[1].scatter(all_stribeck, all_true_fuel, alpha=0.6, s=20, c='red', label='True')
    axes[1].set_xlabel('Stribeck Number', fontsize=12)
    axes[1].set_ylabel('True Fuel Economy Score', fontsize=12)
    axes[1].set_title('Stribeck Number vs True Fuel Economy', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('stribeck_fuel_economy_scatter.png', dpi=300, bbox_inches='tight')
    print("Stribeck数 vs 燃油经济性散点图已保存为 'stribeck_fuel_economy_scatter.png'")
    plt.close()
    
    # 创建另一个图：预测值 vs 真实值的对比
    plt.figure(figsize=(10, 8))
    plt.scatter(all_true_fuel, all_pred_fuel, alpha=0.6, s=20)
    plt.plot([all_true_fuel.min(), all_true_fuel.max()], 
             [all_true_fuel.min(), all_true_fuel.max()], 
             'r--', linewidth=2, label='Perfect Prediction')
    plt.xlabel('True Fuel Economy Score', fontsize=12)
    plt.ylabel('Predicted Fuel Economy Score', fontsize=12)
    plt.title('Predicted vs True Fuel Economy Score', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('fuel_economy_prediction_comparison.png', dpi=300, bbox_inches='tight')
    print("预测值 vs 真实值对比图已保存为 'fuel_economy_prediction_comparison.png'")
    plt.close()

# ============================================================================
# 主程序
# ============================================================================

def main():
    """
    主训练程序
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据（使用40°C的粘度）
    print("加载数据...")
    X, y, stribeck_true = load_bench_data(temperature=40.0)
    
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"Stribeck数范围: {stribeck_true.min():.6f} - {stribeck_true.max():.6f}")
    print(f"燃油经济性范围: {y.min():.2f} - {y.max():.2f}")
    
    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val, stribeck_train, stribeck_val = train_test_split(
        X_scaled, y, stribeck_true, test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    print("创建台架迁移模型...")
    model = BenchTransferModel(hidden_dims=[64, 128, 64], dropout=0.2)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses = train_bench_model(
        model, train_loader, val_loader, 
        num_epochs=100, lr=0.001, device=device
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('bench_transfer_training_curves.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存为 'bench_transfer_training_curves.png'")
    plt.close()
    
    # 验证和可视化
    print("生成Stribeck数 vs 燃油经济性散点图...")
    plot_stribeck_vs_fuel_economy(model, val_loader, stribeck_val, device=device, n_samples=500)
    
    print("训练完成！")

if __name__ == "__main__":
    main()

