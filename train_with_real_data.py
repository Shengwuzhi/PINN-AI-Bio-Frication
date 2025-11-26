"""
Training Script with Real Data from AI-Bio-Frication
使用AI-Bio-Frication真实数据训练PINN模型
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 导入数据接口
from data_interface import (
    AIBioFricationInterface, 
    RealDataInterface, 
    UnifiedDataManager
)

# 导入模型
from pinn_l2_model import PINN_L2_Model, train_pinn_model, load_data as load_pinn_data
from bench_transfer_l25 import BenchTransferModel, train_bench_model
from physics_rul_l3 import PhysicsRULModel, train_physics_rul_model

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

# ============================================================================
# Training Functions with Real Data
# ============================================================================

def train_l2_with_real_data(data_source='ai_bio_frication', 
                            data_path=None,
                            num_epochs=100,
                            batch_size=32,
                            device='cpu'):
    """
    使用真实数据训练L2模型
    
    参数:
    - data_source: 数据源 ('ai_bio_frication', 'uploaded', 'combined')
    - data_path: 数据路径（可选）
    - num_epochs: 训练轮数
    - batch_size: 批次大小
    - device: 设备
    """
    print("="*60)
    print("训练L2模型 - 使用真实数据")
    print("="*60)
    
    # 加载数据
    if data_source == 'ai_bio_frication':
        interface = AIBioFricationInterface(data_path)
        data = interface.load_data()
    elif data_source == 'uploaded':
        interface = RealDataInterface()
        if data_path is None:
            raise ValueError("上传数据需要指定data_path")
        data = interface.upload_data(data_path, data_type='molecular')
    else:
        raise ValueError(f"不支持的数据源: {data_source}")
    
    molecular_features = data['molecular_features']
    targets = data.get('targets', None)
    
    # 如果没有目标值，需要生成（使用物理模型）
    if targets is None or 'vft_a' not in targets:
        print("警告: 未找到VFT参数目标值，将使用物理模型生成...")
        from physics_data_gen import generate_vft_parameters, calculate_viscosity_vft
        
        z_params = data['z_params']
        A, B, C = generate_vft_parameters(
            z_params[:, 0],  # chain_length
            z_params[:, 1],  # symmetry
            z_params[:, 2]   # polarity
        )
        
        # 计算不同温度下的粘度
        temp_40 = 40.0
        temp_100 = 100.0
        temp_minus20 = -20.0
        
        viscosity_40 = calculate_viscosity_vft(A, B, C, temp_40)
        viscosity_100 = calculate_viscosity_vft(A, B, C, temp_100)
        viscosity_minus20 = calculate_viscosity_vft(A, B, C, temp_minus20)
        
        targets = {
            'vft_a': A,
            'vft_b': B,
            'vft_c': C,
            'viscosity_40': viscosity_40,
            'viscosity_100': viscosity_100,
            'viscosity_minus20': viscosity_minus20
        }
    
    # 准备训练数据
    X = molecular_features
    y_vft = np.column_stack([
        targets['vft_a'],
        targets['vft_b'],
        targets['vft_c']
    ])
    y_viscosity = np.column_stack([
        targets['viscosity_40'],
        targets['viscosity_100'],
        targets['viscosity_minus20']
    ])
    
    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_vft_train, y_vft_val, y_vis_train, y_vis_val = train_test_split(
        X_scaled, y_vft, y_viscosity, test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_vft_train = torch.FloatTensor(y_vft_train)
    y_vft_val = torch.FloatTensor(y_vft_val)
    y_vis_train = torch.FloatTensor(y_vis_train)
    y_vis_val = torch.FloatTensor(y_vis_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_vft_train, torch.zeros_like(y_vft_train), y_vis_train)
    val_dataset = TensorDataset(X_val, y_vft_val, torch.zeros_like(y_vft_val), y_vis_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n创建PINN模型...")
    model = PINN_L2_Model(input_dim=X.shape[1], hidden_dims=[128, 256, 128], latent_dim=64)
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print(f"\n开始训练（{num_epochs}轮）...")
    train_losses, val_losses = train_pinn_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs, lr=0.001, device=device
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'pinn_l2_real_data_model.pth')
    print(f"\n模型已保存: pinn_l2_real_data_model.pth")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('L2 Model Training with Real Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('l2_real_data_training_curves.png', dpi=300, bbox_inches='tight')
    print("训练曲线已保存: l2_real_data_training_curves.png")
    plt.close()
    
    return model, train_losses, val_losses

def train_l25_with_real_data(data_source='ai_bio_frication',
                             data_path=None,
                             num_epochs=100,
                             batch_size=32,
                             device='cpu'):
    """
    使用真实数据训练L2.5模型
    """
    print("="*60)
    print("训练L2.5模型 - 使用真实数据")
    print("="*60)
    
    # 加载数据
    if data_source == 'ai_bio_frication':
        interface = AIBioFricationInterface(data_path)
        data = interface.load_data()
    else:
        raise ValueError(f"不支持的数据源: {data_source}")
    
    # 需要从AI-Bio-Frication数据中提取或生成台架测试数据
    # 这里我们使用物理模型生成
    from physics_data_gen import (
        generate_vft_parameters, calculate_viscosity_vft,
        generate_engine_conditions, calculate_stribeck_number,
        calculate_fuel_economy_score
    )
    
    z_params = data['z_params']
    n_samples = len(z_params)
    
    # 生成VFT参数和粘度
    A, B, C = generate_vft_parameters(
        z_params[:, 0], z_params[:, 1], z_params[:, 2]
    )
    viscosity_40 = calculate_viscosity_vft(A, B, C, 40.0)
    
    # 生成发动机条件
    speed, load = generate_engine_conditions(n_samples)
    
    # 计算Stribeck数和燃油经济性
    stribeck = calculate_stribeck_number(viscosity_40, speed, load)
    fuel_economy = calculate_fuel_economy_score(stribeck)
    
    # 准备数据
    X = np.column_stack([viscosity_40, speed, load])
    y = fuel_economy.reshape(-1, 1)
    
    # 数据标准化
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 转换为PyTorch张量
    X_train = torch.FloatTensor(X_train)
    X_val = torch.FloatTensor(X_val)
    y_train = torch.FloatTensor(y_train)
    y_val = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    print(f"\n创建L2.5模型...")
    model = BenchTransferModel(hidden_dims=[64, 128, 64], dropout=0.2)
    
    # 训练模型
    print(f"\n开始训练（{num_epochs}轮）...")
    train_losses, val_losses = train_bench_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs, lr=0.001, device=device
    )
    
    # 保存模型
    torch.save(model.state_dict(), 'bench_transfer_real_data_model.pth')
    print(f"\n模型已保存: bench_transfer_real_data_model.pth")
    
    return model, train_losses, val_losses

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    """
    主训练流程
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='使用真实数据训练PINN模型')
    parser.add_argument('--data_source', type=str, default='ai_bio_frication',
                       choices=['ai_bio_frication', 'uploaded'],
                       help='数据源类型')
    parser.add_argument('--data_path', type=str, default=None,
                       help='数据文件路径（可选）')
    parser.add_argument('--model', type=str, default='all',
                       choices=['l2', 'l25', 'l3', 'all'],
                       help='要训练的模型')
    parser.add_argument('--epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--device', type=str, default='cpu',
                       help='设备 (cpu/cuda)')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 训练L2模型
    if args.model in ['l2', 'all']:
        try:
            model_l2, train_losses_l2, val_losses_l2 = train_l2_with_real_data(
                data_source=args.data_source,
                data_path=args.data_path,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                device=device
            )
            print("\n✓ L2模型训练完成")
        except Exception as e:
            print(f"\n✗ L2模型训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 训练L2.5模型
    if args.model in ['l25', 'all']:
        try:
            model_l25, train_losses_l25, val_losses_l25 = train_l25_with_real_data(
                data_source=args.data_source,
                data_path=args.data_path,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                device=device
            )
            print("\n✓ L2.5模型训练完成")
        except Exception as e:
            print(f"\n✗ L2.5模型训练失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)

if __name__ == "__main__":
    main()

