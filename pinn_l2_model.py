"""
Physics-Informed Neural Network (PINN) for Level 2 Formulation Prediction
物理信息神经网络 - 用于L2配方预测
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# 物理常数
R = 8.314  # 气体常数 (J/(mol·K))

# ============================================================================
# PINN模型架构
# ============================================================================

class PINN_L2_Model(nn.Module):
    """
    物理信息神经网络模型
    用于预测润滑剂配方的物理参数
    """
    
    def __init__(self, input_dim=67, hidden_dims=[128, 256, 128], latent_dim=64):
        """
        参数:
        - input_dim: 输入维度 (3个Z参数 + 64维分子嵌入 = 67)
        - hidden_dims: 隐藏层维度列表
        - latent_dim: 潜在特征维度
        """
        super(PINN_L2_Model, self).__init__()
        
        # 构建骨干网络 (MLP)
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        self.backbone = nn.Sequential(*layers)
        
        # 提取潜在特征
        self.latent_layer = nn.Linear(prev_dim, latent_dim)
        
        # Head 1: VFT参数预测 (A, B, C)
        self.vft_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # 输出 A, B, C
        )
        
        # 初始化VFT头，使输出更接近合理范围
        # A通常在2-3之间，B在500-700之间，C在-80到-30之间（根据数据生成脚本）
        nn.init.normal_(self.vft_head[-1].weight, mean=0, std=0.1)
        nn.init.constant_(self.vft_head[-1].bias[0], 2.5)  # A的初始值
        nn.init.constant_(self.vft_head[-1].bias[1], 600.0)  # B的初始值
        nn.init.constant_(self.vft_head[-1].bias[2], -50.0)  # C的初始值（应该是负数）
        
        # Head 2: 表面物理参数预测 (Delta_G, Shear_Strength)
        self.surface_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 输出 Delta_G, Shear_Strength
        )
        
    def forward(self, x):
        """
        前向传播
        输入: x [batch_size, input_dim]
        输出: 
            - vft_params: [batch_size, 3] (A, B, C)
            - surface_params: [batch_size, 2] (Delta_G, Shear_Strength)
        """
        # 通过骨干网络
        features = self.backbone(x)
        
        # 提取潜在特征
        latent = torch.relu(self.latent_layer(features))
        
        # 通过两个头
        vft_params = self.vft_head(latent)
        surface_params = self.surface_head(latent)
        
        return vft_params, surface_params
    
    def calc_viscosity(self, vft_params, temperature):
        """
        显式物理层：使用VFT方程计算粘度
        ln(eta) = A + B / (T - C)
        eta = exp(A + B / (T - C))
        
        参数:
        - vft_params: [batch_size, 3] (A, B, C)
        - temperature: 标量或 [batch_size] 数组
        
        返回:
        - viscosity: [batch_size] 粘度值 (mPa·s)
        """
        A = vft_params[:, 0]
        B = vft_params[:, 1]
        C = vft_params[:, 2]
        
        # 确保温度大于C (避免除零和负值)
        # 对于每个样本，确保温度大于其对应的C+10
        if isinstance(temperature, torch.Tensor):
            if temperature.dim() == 0:
                T = torch.maximum(torch.full_like(C, temperature.item()), C + 10.0)
            else:
                T = torch.maximum(temperature, C + 10.0)
        else:
            T = torch.maximum(torch.full_like(C, float(temperature)), C + 10.0)
        
        # 计算ln(eta)
        ln_eta = A + B / (T - C)
        
        # 限制ln_eta的值以避免溢出
        ln_eta = torch.clamp(ln_eta, -2.3, 13.8)
        
        # 计算粘度
        viscosity = torch.exp(ln_eta)
        
        return viscosity
    
    def calc_friction(self, surface_params, temperature, concentration=1.0):
        """
        显式物理层：使用Langmuir等温线计算摩擦系数
        theta = (K*C) / (1 + K*C)
        其中 K = exp(-Delta_G / (R*T))
        
        参数:
        - surface_params: [batch_size, 2] (Delta_G, Shear_Strength)
        - temperature: 标量或 [batch_size] 数组
        - concentration: 浓度 (默认1.0)
        
        返回:
        - friction_coeff: [batch_size] 摩擦系数
        - surface_coverage: [batch_size] 表面覆盖率
        """
        Delta_G = surface_params[:, 0]  # kJ/mol
        Shear_Strength = surface_params[:, 1]  # 剪切强度 (未使用，但保留用于未来扩展)
        
        if isinstance(temperature, torch.Tensor):
            T = temperature
        else:
            T = torch.full_like(Delta_G, temperature)
        
        # 转换为开尔文
        T_kelvin = T + 273.15
        
        # 计算平衡常数 (Delta_G转换为J/mol)
        K = torch.exp(-Delta_G * 1000 / (R * T_kelvin))
        
        # 计算表面覆盖率
        theta = (K * concentration) / (1 + K * concentration)
        
        # 计算摩擦系数
        mu_0 = 0.3  # 无润滑剂时的摩擦系数
        mu_1 = 0.05  # 完全覆盖时的摩擦系数
        friction_coeff = mu_0 * (1 - theta) + mu_1 * theta
        
        return friction_coeff, theta

# ============================================================================
# 数据加载和预处理
# ============================================================================

def load_data(csv_path='physics_lubricant_data.csv'):
    """
    加载和预处理数据
    """
    df = pd.read_csv(csv_path)
    
    # 输入特征：Z参数 + 模拟分子嵌入
    # 由于CSV中没有存储64维嵌入，我们在这里生成（实际应用中应该从原始数据加载）
    n_samples = len(df)
    z_params = df[['chain_length', 'symmetry', 'polarity']].values
    
    # 生成模拟的分子嵌入（在实际应用中，这应该从分子结构计算）
    np.random.seed(42)
    molecular_embeddings = np.random.normal(0, 1, (n_samples, 64))
    
    # 合并输入特征
    X = np.hstack([z_params, molecular_embeddings])
    
    # 目标值：VFT参数和表面物理参数
    # 注意：我们不会直接预测这些参数，而是通过物理损失来学习
    # 但我们需要真实值来计算损失
    y_vft = df[['VFT_A', 'VFT_B', 'VFT_C']].values
    y_surface = df[['Delta_G']].values
    
    # 添加Shear_Strength（如果数据中没有，我们使用一个基于其他参数的估计值）
    # 这里我们使用一个简单的估计：基于极性和链长
    shear_strength = -10 - 5 * df['polarity'].values + 0.1 * df['chain_length'].values
    y_surface = np.column_stack([y_surface.flatten(), shear_strength])
    
    # 真实粘度值（用于计算物理损失）
    y_viscosity = df[['viscosity_40C', 'viscosity_100C', 'viscosity_minus20C']].values
    
    return X, y_vft, y_surface, y_viscosity

# ============================================================================
# 训练函数
# ============================================================================

def train_pinn_model(model, train_loader, val_loader, num_epochs=100, lr=0.001, device='cpu'):
    """
    训练PINN模型
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # 温度点（用于计算物理损失）
    temp_points = torch.tensor([40.0, 100.0, -20.0], device=device)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y_vft, batch_y_surface, batch_y_viscosity in train_loader:
            batch_X = batch_X.to(device)
            batch_y_viscosity = batch_y_viscosity.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            vft_params, surface_params = model(batch_X)
            
            # 计算物理损失：使用预测的VFT参数计算粘度，然后与真实粘度比较
            pred_viscosities = []
            for temp in temp_points:
                pred_vis = model.calc_viscosity(vft_params, temp.item())
                pred_viscosities.append(pred_vis)
            
            pred_viscosity_matrix = torch.stack(pred_viscosities, dim=1)  # [batch_size, 3]
            
            # 物理损失：使用对数尺度的MSE（因为粘度值范围很大）
            # 这样可以更好地处理不同数量级的粘度值
            log_pred = torch.log(pred_viscosity_matrix + 1e-6)
            log_true = torch.log(batch_y_viscosity + 1e-6)
            physics_loss = nn.MSELoss()(log_pred, log_true)
            
            # 也可以添加相对误差损失
            relative_loss = torch.mean(torch.abs(pred_viscosity_matrix - batch_y_viscosity) / (batch_y_viscosity + 1e-6))
            
            # 总损失：对数损失 + 相对误差损失
            physics_loss = physics_loss + 0.1 * relative_loss
            
            # 总损失
            loss = physics_loss
            
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
            for batch_X, batch_y_vft, batch_y_surface, batch_y_viscosity in val_loader:
                batch_X = batch_X.to(device)
                batch_y_viscosity = batch_y_viscosity.to(device)
                
                vft_params, surface_params = model(batch_X)
                
                pred_viscosities = []
                for temp in temp_points:
                    pred_vis = model.calc_viscosity(vft_params, temp.item())
                    pred_viscosities.append(pred_vis)
                
                pred_viscosity_matrix = torch.stack(pred_viscosities, dim=1)
                # 使用相同的损失函数
                log_pred = torch.log(pred_viscosity_matrix + 1e-6)
                log_true = torch.log(batch_y_viscosity + 1e-6)
                physics_loss = nn.MSELoss()(log_pred, log_true)
                relative_loss = torch.mean(torch.abs(pred_viscosity_matrix - batch_y_viscosity) / (batch_y_viscosity + 1e-6))
                physics_loss = physics_loss + 0.1 * relative_loss
                
                val_loss += physics_loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'pinn_l2_best_model.pth')
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 每5个epoch打印一次详细信息
        if (epoch + 1) % 5 == 0:
            with torch.no_grad():
                sample_X, _, _, sample_y_vis = next(iter(train_loader))
                sample_X = sample_X.to(device)
                sample_y_vis = sample_y_vis.to(device)
                sample_vft, _ = model(sample_X[:1])
                sample_pred_vis = []
                for temp in temp_points:
                    sample_pred_vis.append(model.calc_viscosity(sample_vft, temp.item())[0].item())
                print(f'  Sample: True Viscosity={sample_y_vis[0].cpu().numpy()}, Pred Viscosity={sample_pred_vis}')
                print(f'  Sample VFT: Pred={sample_vft[0].cpu().numpy()}')
    
    return train_losses, val_losses

# ============================================================================
# 验证和可视化
# ============================================================================

def plot_viscosity_curves(model, val_loader, device='cpu', n_samples=5):
    """
    绘制预测粘度曲线 vs 真实粘度曲线
    """
    model.eval()
    model = model.to(device)
    
    # 加载最佳模型
    if os.path.exists('pinn_l2_best_model.pth'):
        model.load_state_dict(torch.load('pinn_l2_best_model.pth', map_location=device))
    
    # 获取一批验证数据
    batch_X, batch_y_vft, batch_y_surface, batch_y_viscosity = next(iter(val_loader))
    batch_X = batch_X.to(device)
    batch_y_vft = batch_y_vft.to(device)
    
    # 限制样本数量
    n_samples = min(n_samples, batch_X.size(0))
    batch_X = batch_X[:n_samples]
    batch_y_vft = batch_y_vft[:n_samples]
    batch_y_viscosity = batch_y_viscosity[:n_samples]
    
    with torch.no_grad():
        # 预测VFT参数
        pred_vft, _ = model(batch_X)
        
        # 温度范围
        temperatures = np.linspace(-20, 120, 100)
        
        plt.figure(figsize=(14, 8))
        
        for i in range(n_samples):
            # 真实VFT参数
            true_A = batch_y_vft[i, 0].item()
            true_B = batch_y_vft[i, 1].item()
            true_C = batch_y_vft[i, 2].item()
            
            # 预测VFT参数
            pred_A = pred_vft[i, 0].item()
            pred_B = pred_vft[i, 1].item()
            pred_C = pred_vft[i, 2].item()
            
            # 计算真实粘度曲线
            true_viscosities = []
            for T in temperatures:
                true_vis = model.calc_viscosity(batch_y_vft[i:i+1], T)
                true_viscosities.append(true_vis.item())
            
            # 计算预测粘度曲线
            pred_viscosities = []
            for T in temperatures:
                pred_vis = model.calc_viscosity(pred_vft[i:i+1], T)
                pred_viscosities.append(pred_vis.item())
            
            # 绘制曲线
            plt.subplot(2, 3, i+1)
            plt.semilogy(temperatures, true_viscosities, 'b-', label='True Viscosity', linewidth=2)
            plt.semilogy(temperatures, pred_viscosities, 'r--', label='Predicted Viscosity', linewidth=2)
            
            # 标记三个温度点
            temp_points = [40.0, 100.0, -20.0]
            true_vis_points = batch_y_viscosity[i].cpu().numpy()
            pred_vis_points = []
            for temp in temp_points:
                pred_vis = model.calc_viscosity(pred_vft[i:i+1], temp)
                pred_vis_points.append(pred_vis.item())
            
            plt.semilogy(temp_points, true_vis_points, 'bo', markersize=8, label='True Points')
            plt.semilogy(temp_points, pred_vis_points, 'ro', markersize=8, label='Predicted Points')
            
            plt.xlabel('Temperature (°C)', fontsize=10)
            plt.ylabel('Viscosity (mPa·s)', fontsize=10)
            plt.title(f'Sample {i+1}\nTrue: A={true_A:.2f}, B={true_B:.1f}, C={true_C:.1f}\n'
                     f'Pred: A={pred_A:.2f}, B={pred_B:.1f}, C={pred_C:.1f}', fontsize=9)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
        
        plt.tight_layout()
        plt.savefig('pinn_viscosity_validation.png', dpi=300, bbox_inches='tight')
        print("验证图已保存为 'pinn_viscosity_validation.png'")
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
    
    # 加载数据
    print("加载数据...")
    X, y_vft, y_surface, y_viscosity = load_data()
    
    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_vft_train, y_vft_val, y_surface_train, y_surface_val, \
    y_viscosity_train, y_viscosity_val = train_test_split(
        X_scaled, y_vft, y_surface, y_viscosity, test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_vft_train = torch.FloatTensor(y_vft_train)
    y_vft_val = torch.FloatTensor(y_vft_val)
    y_surface_train = torch.FloatTensor(y_surface_train)
    y_surface_val = torch.FloatTensor(y_surface_val)
    y_viscosity_train = torch.FloatTensor(y_viscosity_train)
    y_viscosity_val = torch.FloatTensor(y_viscosity_val)
    
    # 创建数据加载器
    from torch.utils.data import TensorDataset, DataLoader
    
    train_dataset = TensorDataset(X_train, y_vft_train, y_surface_train, y_viscosity_train)
    val_dataset = TensorDataset(X_val, y_vft_val, y_surface_val, y_viscosity_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # 创建模型
    print("创建PINN模型...")
    model = PINN_L2_Model(input_dim=67, hidden_dims=[128, 256, 128], latent_dim=64)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses = train_pinn_model(
        model, train_loader, val_loader, 
        num_epochs=50, lr=0.001, device=device
    )
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pinn_training_curves.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存为 'pinn_training_curves.png'")
    plt.close()
    
    # 验证和可视化
    print("生成验证图...")
    plot_viscosity_curves(model, val_loader, device=device, n_samples=5)
    
    print("训练完成！")

if __name__ == "__main__":
    main()

