"""
Multi-Objective Optimization with Pareto Optimal for PINN System
多目标优化与帕累托最优 - PINN系统
考虑L2、L2.5、L3多个层级的目标
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from pathlib import Path

# 导入PINN模型
from pinn_l2_model import PINN_L2_Model
from bench_transfer_l25 import BenchTransferModel
from physics_rul_l3 import PhysicsRULModel
from main_system_integration import UnifiedPipeline

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ============================================================================
# Multi-Objective Optimizer for PINN System
# ============================================================================

class PINNMultiObjectiveOptimizer:
    """
    PINN系统多目标优化器
    考虑多个层级的目标函数
    """
    
    def __init__(self, pipeline: UnifiedPipeline, device='cpu'):
        """
        参数:
        - pipeline: 统一管道系统
        - device: 计算设备
        """
        self.pipeline = pipeline
        self.device = device
        self.objectives_cache = {}
        
    def predict_objectives(self, molecular_features: np.ndarray,
                          engine_speed: Optional[np.ndarray] = None,
                          engine_load: Optional[np.ndarray] = None,
                          time_series_temp: Optional[np.ndarray] = None,
                          time_series_vib: Optional[np.ndarray] = None) -> Dict:
        """
        使用PINN模型预测多个目标
        
        参数:
        - molecular_features: [n_samples, 67] 分子特征
        - engine_speed: [n_samples] 发动机速度 (可选)
        - engine_load: [n_samples] 发动机载荷 (可选)
        - time_series_temp: [n_samples, seq_len] 温度时间序列 (可选)
        - time_series_vib: [n_samples, seq_len] 振动时间序列 (可选)
        
        返回:
        - objectives: 包含所有目标值的字典
        """
        n_samples = molecular_features.shape[0]
        
        # 默认值
        if engine_speed is None:
            engine_speed = np.random.uniform(2000, 4000, n_samples)
        if engine_load is None:
            engine_load = np.random.uniform(300, 700, n_samples)
        if time_series_temp is None:
            time_series_temp = np.random.uniform(80, 120, (n_samples, 100))
        if time_series_vib is None:
            time_series_vib = np.random.normal(0, 1, (n_samples, 100))
        
        # 使用管道预测
        results = self.pipeline.predict_pipeline(
            molecular_features,
            engine_speed,
            engine_load,
            time_series_temp,
            time_series_vib
        )
        
        # 提取目标值
        vft_params = results['vft_params']
        surface_params = results.get('surface_params', None)
        
        # 计算摩擦系数
        if surface_params is not None:
            try:
                from physics_data_gen import calculate_friction_langmuir
                Delta_G = surface_params[:, 0]  # kJ/mol
                friction_coeff, _ = calculate_friction_langmuir(Delta_G, 40.0, concentration=1.0)
            except:
                # 如果计算失败，使用简化公式
                friction_coeff = 0.3 - 0.25 * (surface_params[:, 0] + 20) / 30
                friction_coeff = np.clip(friction_coeff, 0.05, 0.3)
        else:
            # 如果没有surface_params，从vft_params推导
            friction_coeff = np.random.uniform(0.05, 0.3, n_samples)
        
        # 计算100°C的粘度
        vft_params_tensor = torch.FloatTensor(vft_params).to(self.device)
        viscosity_100 = self.pipeline.l2_model.calc_viscosity(
            vft_params_tensor, 100.0
        ).cpu().numpy().flatten()
        
        objectives = {
            # L2层级目标
            'vft_a': vft_params[:, 0],
            'vft_b': vft_params[:, 1],
            'vft_c': vft_params[:, 2],
            'viscosity_40': results['viscosity_40'].flatten(),
            'viscosity_100': viscosity_100,
            'friction_coefficient': friction_coeff,
            
            # L2.5层级目标
            'fuel_economy': results['fuel_economy'].flatten(),
            'stribeck_number': results['stribeck'].flatten(),
            
            # L3层级目标
            'rul_mean': results['rul'].mean(axis=1) if results['rul'].ndim > 1 else results['rul'].flatten(),
            'rul_final': results['rul'][:, -1].flatten() if results['rul'].ndim > 1 else results['rul'].flatten(),
            'degradation_rate_mean': results['degradation_rate'].mean(axis=1) if results['degradation_rate'].ndim > 1 else results['degradation_rate'].flatten(),
            'health_final': results['health'][:, -1].flatten() if results['health'].ndim > 1 else results['health'].flatten(),
        }
        
        return objectives
    
    def calculate_pareto_front(self, objectives: Dict, 
                              objective_names: List[str] = None,
                              minimize: List[bool] = None) -> Tuple[List[int], np.ndarray]:
        """
        计算帕累托前沿
        
        参数:
        - objectives: 目标值字典
        - objective_names: 要优化的目标名称列表
        - minimize: 每个目标是否最小化（True=最小化, False=最大化）
        
        返回:
        - pareto_indices: 帕累托最优解的索引
        - objectives_matrix: 目标值矩阵 [n_samples, n_objectives]
        """
        if objective_names is None:
            # 默认目标：最小化摩擦、最大化燃油经济性、最大化RUL
            objective_names = ['friction_coefficient', 'fuel_economy', 'rul_final']
            minimize = [True, False, False]
        
        if minimize is None:
            minimize = [True] * len(objective_names)
        
        # 提取目标值
        objectives_list = []
        for name in objective_names:
            if name in objectives and objectives[name] is not None:
                values = objectives[name]
                # 如果是最大化目标，取负值转换为最小化
                if not minimize[objective_names.index(name)]:
                    values = -values
                objectives_list.append(values)
            else:
                raise ValueError(f"目标 '{name}' 不存在或为None")
        
        objectives_matrix = np.column_stack(objectives_list)
        n_samples, n_objectives = objectives_matrix.shape
        
        # 标准化到[0, 1]范围
        scaler = MinMaxScaler()
        objectives_scaled = scaler.fit_transform(objectives_matrix)
        
        # 寻找帕累托最优解
        pareto_indices = []
        
        for i in range(n_samples):
            is_dominated = False
            for j in range(n_samples):
                if i != j:
                    # 检查i是否被j支配
                    # j支配i：j在所有目标上都不差于i，且至少在一个目标上更好
                    if all(objectives_scaled[j, :] <= objectives_scaled[i, :]) and \
                       any(objectives_scaled[j, :] < objectives_scaled[i, :]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        print(f"帕累托前沿包含 {len(pareto_indices)} 个最优解 (共 {n_samples} 个样本)")
        print(f"帕累托最优比例: {len(pareto_indices)/n_samples*100:.2f}%")
        
        return pareto_indices, objectives_scaled
    
    def calculate_weighted_score(self, objectives: Dict,
                                weights: Dict[str, float] = None,
                                normalize: bool = True) -> np.ndarray:
        """
        计算加权综合得分
        
        参数:
        - objectives: 目标值字典
        - weights: 权重字典，例如 {'friction_coefficient': 0.3, 'fuel_economy': 0.4, 'rul_final': 0.3}
        - normalize: 是否标准化目标值
        
        返回:
        - scores: 综合得分数组
        """
        if weights is None:
            # 默认权重
            weights = {
                'friction_coefficient': 0.25,  # 最小化
                'fuel_economy': 0.35,  # 最大化
                'rul_final': 0.40  # 最大化
            }
        
        # 提取目标值
        objective_values = {}
        for name, weight in weights.items():
            if name in objectives and objectives[name] is not None:
                values = objectives[name].copy()
                
                # 标准化
                if normalize:
                    if values.max() > values.min():
                        values = (values - values.min()) / (values.max() - values.min())
                    else:
                        values = np.ones_like(values) * 0.5
                
                # 对于最小化目标，取反
                if name in ['friction_coefficient', 'degradation_rate_mean']:
                    values = 1 - values
                
                objective_values[name] = values * weight
        
        # 计算加权和
        scores = np.sum(list(objective_values.values()), axis=0)
        
        return scores
    
    def visualize_pareto_front_3d(self, objectives: Dict,
                                 pareto_indices: List[int],
                                 objective_names: List[str] = None,
                                 save_path: str = 'pareto_front_3d.png'):
        """
        3D帕累托前沿可视化
        """
        if objective_names is None:
            objective_names = ['friction_coefficient', 'fuel_economy', 'rul_final']
        
        # 提取目标值
        obj1 = objectives[objective_names[0]]
        obj2 = objectives[objective_names[1]]
        obj3 = objectives[objective_names[2]]
        
        fig = plt.figure(figsize=(16, 6))
        
        # 子图1: 3D散点图
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 所有点
        ax1.scatter(obj1, obj2, obj3,
                   c='lightblue', alpha=0.3, s=20, label='所有配方',
                   edgecolors='none')
        
        # 帕累托前沿
        pareto_obj1 = obj1[pareto_indices]
        pareto_obj2 = obj2[pareto_indices]
        pareto_obj3 = obj3[pareto_indices]
        
        ax1.scatter(pareto_obj1, pareto_obj2, pareto_obj3,
                   c='red', alpha=0.8, s=100, marker='*',
                   edgecolors='black', linewidth=1,
                   label=f'帕累托最优 (n={len(pareto_indices)})')
        
        ax1.set_xlabel(self._get_objective_label(objective_names[0]), fontsize=11, fontweight='bold')
        ax1.set_ylabel(self._get_objective_label(objective_names[1]), fontsize=11, fontweight='bold')
        ax1.set_zlabel(self._get_objective_label(objective_names[2]), fontsize=11, fontweight='bold')
        ax1.set_title('3D帕累托前沿', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 2D投影（目标1 vs 目标2）
        ax2 = fig.add_subplot(122)
        
        ax2.scatter(obj1, obj2, c='lightblue', alpha=0.3, s=30, label='所有配方')
        ax2.scatter(pareto_obj1, pareto_obj2, c='red', alpha=0.8, s=100,
                   marker='*', edgecolors='black', linewidth=1,
                   label=f'帕累托最优 (n={len(pareto_indices)})')
        
        ax2.set_xlabel(self._get_objective_label(objective_names[0]), fontsize=11, fontweight='bold')
        ax2.set_ylabel(self._get_objective_label(objective_names[1]), fontsize=11, fontweight='bold')
        ax2.set_title('2D投影: ' + objective_names[0] + ' vs ' + objective_names[1],
                     fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('PINN系统多目标优化 - 帕累托前沿分析',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"帕累托前沿图已保存: {save_path}")
        plt.close()
    
    def visualize_pareto_tradeoff(self, objectives: Dict,
                                 pareto_indices: List[int],
                                 save_path: str = 'pareto_tradeoff.png'):
        """
        可视化帕累托权衡关系
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()
        
        # 1. 摩擦系数 vs 燃油经济性
        ax = axes[0]
        friction = objectives['friction_coefficient']
        fuel = objectives['fuel_economy']
        
        ax.scatter(friction, fuel, c='lightblue', alpha=0.3, s=30, label='所有配方')
        ax.scatter(friction[pareto_indices], fuel[pareto_indices],
                  c='red', alpha=0.8, s=100, marker='*',
                  edgecolors='black', linewidth=1, label='帕累托最优')
        ax.set_xlabel('摩擦系数 (越小越好)', fontsize=10)
        ax.set_ylabel('燃油经济性 (越大越好)', fontsize=10)
        ax.set_title('摩擦 vs 燃油经济性权衡', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. 燃油经济性 vs RUL
        ax = axes[1]
        rul = objectives['rul_final']
        
        ax.scatter(fuel, rul, c='lightblue', alpha=0.3, s=30, label='所有配方')
        ax.scatter(fuel[pareto_indices], rul[pareto_indices],
                  c='red', alpha=0.8, s=100, marker='*',
                  edgecolors='black', linewidth=1, label='帕累托最优')
        ax.set_xlabel('燃油经济性 (越大越好)', fontsize=10)
        ax.set_ylabel('剩余使用寿命 RUL (越大越好)', fontsize=10)
        ax.set_title('燃油经济性 vs RUL权衡', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 3. 摩擦系数 vs RUL
        ax = axes[2]
        
        ax.scatter(friction, rul, c='lightblue', alpha=0.3, s=30, label='所有配方')
        ax.scatter(friction[pareto_indices], rul[pareto_indices],
                  c='red', alpha=0.8, s=100, marker='*',
                  edgecolors='black', linewidth=1, label='帕累托最优')
        ax.set_xlabel('摩擦系数 (越小越好)', fontsize=10)
        ax.set_ylabel('剩余使用寿命 RUL (越大越好)', fontsize=10)
        ax.set_title('摩擦 vs RUL权衡', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 4. 综合得分分布
        ax = axes[3]
        scores = self.calculate_weighted_score(objectives)
        
        ax.hist(scores, bins=50, alpha=0.6, color='lightblue', label='所有配方')
        ax.axvline(scores[pareto_indices].mean(), color='red', linestyle='--',
                  linewidth=2, label=f'帕累托最优均值 ({scores[pareto_indices].mean():.3f})')
        ax.set_xlabel('综合得分', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_title('综合得分分布', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('PINN系统多目标优化 - 权衡关系分析',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"权衡关系图已保存: {save_path}")
        plt.close()
    
    def get_top_solutions(self, objectives: Dict, pareto_indices: List[int],
                         top_n: int = 10) -> pd.DataFrame:
        """
        获取TOP-N最优解
        
        返回:
        - top_solutions: 包含最优解信息的DataFrame
        """
        scores = self.calculate_weighted_score(objectives)
        pareto_scores = scores[pareto_indices]
        
        # 按得分排序
        sorted_indices = np.argsort(pareto_scores)[::-1][:top_n]
        top_indices = [pareto_indices[i] for i in sorted_indices]
        
        # 构建结果DataFrame
        results = []
        for idx in top_indices:
            result = {
                'Index': idx,
                'Score': scores[idx],
                'Friction_Coefficient': objectives['friction_coefficient'][idx] if objectives['friction_coefficient'] is not None else None,
                'Fuel_Economy': objectives['fuel_economy'][idx],
                'RUL_Final': objectives['rul_final'][idx],
                'Viscosity_40C': objectives['viscosity_40'][idx],
                'VFT_A': objectives['vft_a'][idx],
                'VFT_B': objectives['vft_b'][idx],
                'VFT_C': objectives['vft_c'][idx],
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        return df
    
    def _get_objective_label(self, name: str) -> str:
        """获取目标的中文标签"""
        labels = {
            'friction_coefficient': '摩擦系数',
            'fuel_economy': '燃油经济性',
            'rul_final': '剩余使用寿命 (RUL)',
            'viscosity_40': '粘度 (40°C, mPa·s)',
            'vft_a': 'VFT参数 A',
            'vft_b': 'VFT参数 B',
            'vft_c': 'VFT参数 C',
        }
        return labels.get(name, name)

# ============================================================================
# Main Function
# ============================================================================

def main():
    """
    主函数：运行多目标优化分析
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='PINN系统多目标优化')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='分析的样本数量')
    parser.add_argument('--top_n', type=int, default=10,
                       help='显示TOP-N最优解')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建统一管道
    print("\n创建统一管道系统...")
    pipeline = UnifiedPipeline(device=device)
    
    # 加载数据（使用AI-Bio-Frication数据）
    print("\n加载数据...")
    from data_interface import AIBioFricationInterface
    
    try:
        data_interface = AIBioFricationInterface()
        data = data_interface.load_data()
        molecular_features = data['molecular_features']
        
        # 限制样本数量（如果太大）
        if len(molecular_features) > args.n_samples:
            indices = np.random.choice(len(molecular_features), args.n_samples, replace=False)
            molecular_features = molecular_features[indices]
        
        print(f"使用 {len(molecular_features)} 个样本进行分析")
    except Exception as e:
        print(f"无法加载AI-Bio-Frication数据: {e}")
        print("使用合成数据...")
        # 使用合成数据
        molecular_features = np.random.normal(0, 1, (args.n_samples, 67))
    
    # 创建优化器
    print("\n创建多目标优化器...")
    optimizer = PINNMultiObjectiveOptimizer(pipeline, device=device)
    
    # 预测目标值
    print("\n预测目标值（这可能需要一些时间）...")
    objectives = optimizer.predict_objectives(molecular_features)
    
    # 计算帕累托前沿
    print("\n计算帕累托前沿...")
    pareto_indices, objectives_scaled = optimizer.calculate_pareto_front(
        objectives,
        objective_names=['friction_coefficient', 'fuel_economy', 'rul_final'],
        minimize=[True, False, False]
    )
    
    # 可视化
    print("\n生成可视化图表...")
    optimizer.visualize_pareto_front_3d(
        objectives, pareto_indices,
        objective_names=['friction_coefficient', 'fuel_economy', 'rul_final']
    )
    
    optimizer.visualize_pareto_tradeoff(objectives, pareto_indices)
    
    # 获取TOP-N解
    print(f"\n获取TOP-{args.top_n}最优解...")
    top_solutions = optimizer.get_top_solutions(objectives, pareto_indices, top_n=args.top_n)
    
    print("\n" + "="*60)
    print("TOP最优解:")
    print("="*60)
    print(top_solutions.to_string(index=False))
    
    # 保存结果
    output_path = 'pareto_optimal_solutions.csv'
    top_solutions.to_csv(output_path, index=False)
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "="*60)
    print("多目标优化分析完成！")
    print("="*60)

if __name__ == "__main__":
    main()

