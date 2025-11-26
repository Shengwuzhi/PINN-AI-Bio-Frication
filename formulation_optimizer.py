"""
Physics-Informed Multi-Objective Optimizer for Formulation Science
物理信息多目标优化器 - 配方科学集成
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

# 导入模型
from pinn_l2_model import PINN_L2_Model
from bench_transfer_l25 import BenchTransferModel

# 设置绘图风格
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# 物理常数
R = 8.314  # 气体常数 (J/(mol·K))

# ============================================================================
# Formulation Optimizer Class
# ============================================================================

class FormulationOptimizer:
    """
    物理信息多目标优化器
    集成配方科学到PINN框架
    """
    
    def __init__(self, device='cpu'):
        """
        初始化优化器
        
        参数:
        - device: 计算设备
        """
        self.device = device
        self.l2_model = None
        self.l25_model = None
        self.candidates_df = None
        self.evaluated_candidates = None
        
        # 加载模型
        self._load_models()
    
    def _load_models(self):
        """加载训练好的模型"""
        print("加载PINN模型...")
        
        # 加载L2模型
        self.l2_model = PINN_L2_Model(input_dim=67, hidden_dims=[128, 256, 128], latent_dim=64)
        if os.path.exists('pinn_l2_best_model.pth'):
            self.l2_model.load_state_dict(
                torch.load('pinn_l2_best_model.pth', map_location=self.device)
            )
            print("  [OK] L2模型加载成功")
        else:
            print("  警告: L2模型文件不存在，使用未训练模型")
        self.l2_model.eval()
        self.l2_model = self.l2_model.to(self.device)
        
        # 加载L2.5模型
        self.l25_model = BenchTransferModel(hidden_dims=[64, 128, 64], dropout=0.2)
        if os.path.exists('bench_transfer_best_model.pth'):
            try:
                self.l25_model.load_state_dict(
                    torch.load('bench_transfer_best_model.pth', map_location=self.device),
                    strict=False
                )
                print("  [OK] L2.5模型加载成功")
            except RuntimeError:
                print("  警告: L2.5模型结构不匹配，使用默认初始化")
        else:
            print("  警告: L2.5模型文件不存在，使用未训练模型")
        self.l25_model.eval()
        self.l25_model = self.l25_model.to(self.device)
    
    def load_candidates(self, data_path: str = 'physics_lubricant_data.csv',
                       ai_bio_data_path: str = None):
        """
        加载候选数据集
        
        参数:
        - data_path: PINN数据文件路径
        - ai_bio_data_path: AI-Bio-Frication数据路径（用于获取成本和可持续性）
        """
        print(f"\n加载候选数据: {data_path}")
        self.candidates_df = pd.read_csv(data_path)
        print(f"  加载了 {len(self.candidates_df)} 个候选配方")
        print(f"  特征数: {self.candidates_df.shape[1]}")
        
        # 尝试从AI-Bio-Frication数据加载成本和可持续性
        if ai_bio_data_path is None:
            ai_bio_data_path = r"C:\Users\weizh\AI-Bio-Frication\enhanced_lubricant_data.csv"
        
        if os.path.exists(ai_bio_data_path):
            try:
                print(f"\n从AI-Bio-Frication加载成本和可持续性数据...")
                ai_bio_df = pd.read_csv(ai_bio_data_path)
                
                # 如果数据量匹配，直接使用
                if len(ai_bio_df) >= len(self.candidates_df):
                    # 选择前N个样本
                    ai_bio_subset = ai_bio_df.head(len(self.candidates_df))
                    
                    if 'Cost_USD_kg' in ai_bio_subset.columns:
                        self.candidates_df['Cost_USD_kg'] = ai_bio_subset['Cost_USD_kg'].values
                        print("  [OK] 成本数据已加载")
                    
                    if 'Sustainability_Score' in ai_bio_subset.columns:
                        self.candidates_df['Sustainability_Score'] = ai_bio_subset['Sustainability_Score'].values
                        print("  [OK] 可持续性数据已加载")
                else:
                    # 如果数据量不匹配，使用随机采样或插值
                    if 'Cost_USD_kg' in ai_bio_df.columns:
                        cost_values = ai_bio_df['Cost_USD_kg'].values
                        # 随机采样或使用统计值
                        if len(cost_values) > 0:
                            self.candidates_df['Cost_USD_kg'] = np.random.choice(
                                cost_values, len(self.candidates_df), replace=True
                            )
                            print("  [OK] 成本数据已采样加载")
                    
                    if 'Sustainability_Score' in ai_bio_df.columns:
                        sustainability_values = ai_bio_df['Sustainability_Score'].values
                        if len(sustainability_values) > 0:
                            self.candidates_df['Sustainability_Score'] = np.random.choice(
                                sustainability_values, len(self.candidates_df), replace=True
                            )
                            print("  [OK] 可持续性数据已采样加载")
            except Exception as e:
                print(f"  警告: 无法从AI-Bio-Frication加载数据: {e}")
                print("  将使用生成的默认值")
        
        # 如果仍然没有成本和可持续性数据，生成合理的值
        if 'Cost_USD_kg' not in self.candidates_df.columns:
            # 基于分子特征生成成本（链长越长，成本越高）
            if 'chain_length' in self.candidates_df.columns:
                base_cost = 2.0
                cost = base_cost + 0.1 * self.candidates_df['chain_length'].values + np.random.normal(0, 0.5, len(self.candidates_df))
            else:
                cost = np.random.uniform(2, 5, len(self.candidates_df))
            self.candidates_df['Cost_USD_kg'] = np.clip(cost, 1.5, 8.0)
            print("  [OK] 成本数据已生成")
        
        if 'Sustainability_Score' not in self.candidates_df.columns:
            # 基于极性生成可持续性（极性越高，可持续性越好）
            if 'polarity' in self.candidates_df.columns:
                base_sustainability = 60.0
                sustainability = base_sustainability + 30 * self.candidates_df['polarity'].values + np.random.normal(0, 5, len(self.candidates_df))
            else:
                sustainability = np.random.uniform(50, 100, len(self.candidates_df))
            self.candidates_df['Sustainability_Score'] = np.clip(sustainability, 50, 100)
            print("  [OK] 可持续性数据已生成")
        
        return self.candidates_df
    
    def calculate_viscosity_vft(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, 
                                temperature: float) -> np.ndarray:
        """
        使用VFT公式显式计算粘度
        
        参数:
        - A, B, C: VFT参数
        - temperature: 温度 (°C)
        
        返回:
        - viscosity: 粘度 (mPa·s)
        """
        T = np.maximum(temperature, C + 10)  # 确保温度大于C
        ln_eta = A + B / (T - C)
        ln_eta = np.clip(ln_eta, -2.3, 13.8)  # 限制范围
        viscosity = np.exp(ln_eta)
        return viscosity
    
    def calculate_viscosity_index(self, vis_40: np.ndarray, vis_100: np.ndarray) -> np.ndarray:
        """
        计算粘度指数 (Viscosity Index, VI)
        使用ASTM D2270标准方法（简化版本）
        
        参数:
        - vis_40: 40°C下的粘度 (mPa·s)
        - vis_100: 100°C下的粘度 (mPa·s)
        
        返回:
        - vi: 粘度指数
        """
        # 转换为cSt（假设密度为0.9 g/cm³）
        vis_40_cst = vis_40 / 0.9
        vis_100_cst = vis_100 / 0.9
        
        # ASTM D2270简化公式
        # VI = ((L - U) / (L - H)) * 100
        # 其中L, H是标准参考值，U是实际值
        
        # 简化计算（基于经验公式）
        # VI ≈ 100 * (log(vis_40) - log(vis_100)) / (log(vis_40_ref) - log(vis_100_ref))
        # 使用近似公式
        L = 0.8353 * vis_40_cst + 14.67 * vis_40_cst - 216  # 参考值（简化）
        H = 0.1684 * vis_40_cst + 11.85 * vis_40_cst - 97  # 参考值（简化）
        U = vis_100_cst
        
        # 计算VI
        vi = ((L - U) / (L - H + 1e-10)) * 100
        vi = np.clip(vi, 0, 200)  # VI通常在0-200之间
        
        # 如果计算不合理，使用经验公式
        mask = (vi < 0) | (vi > 200) | np.isnan(vi)
        if np.any(mask):
            # 使用经验公式: VI ≈ 100 * (1 - (vis_100/vis_40)^0.3)
            vi_approx = 100 * (1 - np.power(vis_100_cst / (vis_40_cst + 1e-6), 0.3))
            vi[mask] = np.clip(vi_approx[mask], 0, 200)
        
        return vi
    
    def evaluate_candidates(self, standard_rpm: float = 2000.0, 
                          standard_load: float = 500.0) -> pd.DataFrame:
        """
        物理基础评估所有候选配方
        
        参数:
        - standard_rpm: 标准发动机转速 (RPM)
        - standard_load: 标准载荷 (N)
        
        返回:
        - evaluated_df: 评估结果DataFrame
        """
        print("\n开始物理基础评估...")
        
        if self.candidates_df is None:
            raise ValueError("请先加载候选数据")
        
        n_candidates = len(self.candidates_df)
        
        # 准备输入特征
        # 假设数据包含Z参数和分子嵌入，或需要从描述符生成
        if 'chain_length' in self.candidates_df.columns:
            z_params = self.candidates_df[['chain_length', 'symmetry', 'polarity']].values
        else:
            # 从其他特征推导Z参数
            print("  警告: 未找到Z参数，从描述符推导...")
            z_params = self._derive_z_params()
        
        # 生成分子嵌入（64维）
        if 'molecular_embeddings' in self.candidates_df.columns:
            # 如果已有嵌入，使用它
            embeddings = self.candidates_df['molecular_embeddings'].values
            if isinstance(embeddings[0], str):
                # 如果是字符串，需要解析
                embeddings = np.array([eval(e) for e in embeddings])
        else:
            # 从描述符生成嵌入
            print("  从描述符生成分子嵌入...")
            embeddings = self._generate_embeddings()
        
        # 组合特征
        molecular_features = np.hstack([z_params, embeddings])
        
        # 转换为张量
        features_tensor = torch.FloatTensor(molecular_features).to(self.device)
        
        # L2模型预测
        print("  运行L2模型预测...")
        with torch.no_grad():
            vft_params, surface_params = self.l2_model(features_tensor)
            vft_params = vft_params.cpu().numpy()
            surface_params = surface_params.cpu().numpy()
        
        A = vft_params[:, 0]
        B = vft_params[:, 1]
        C = vft_params[:, 2]
        
        # 显式计算不同温度下的粘度
        print("  计算不同温度下的粘度...")
        vis_40 = self.calculate_viscosity_vft(A, B, C, 40.0)
        vis_100 = self.calculate_viscosity_vft(A, B, C, 100.0)
        vis_minus20 = self.calculate_viscosity_vft(A, B, C, -20.0)
        
        # 计算粘度指数
        print("  计算粘度指数...")
        vi = self.calculate_viscosity_index(vis_40, vis_100)
        
        # 物理过滤：低温失效
        print("  应用物理过滤...")
        low_temp_pass = vis_minus20 <= 5000
        n_filtered = np.sum(~low_temp_pass)
        print(f"    过滤掉 {n_filtered} 个低温失效候选 ({n_filtered/n_candidates*100:.1f}%)")
        
        # L2.5模型预测燃油经济性
        print("  运行L2.5模型预测...")
        vis_40_tensor = torch.FloatTensor(vis_40).unsqueeze(1).to(self.device)
        speed_tensor = torch.FloatTensor(np.full(n_candidates, standard_rpm)).unsqueeze(1).to(self.device)
        load_tensor = torch.FloatTensor(np.full(n_candidates, standard_load)).unsqueeze(1).to(self.device)
        
        with torch.no_grad():
            fuel_economy, (stribeck, _) = self.l25_model(vis_40_tensor, speed_tensor, load_tensor)
            fuel_economy = fuel_economy.cpu().numpy().flatten()
            stribeck = stribeck.cpu().numpy().flatten()
        
        # 提取成本和可持续性（如果存在）
        cost = self.candidates_df['Cost_USD_kg'].values if 'Cost_USD_kg' in self.candidates_df.columns else np.random.uniform(2, 5, n_candidates)
        sustainability = self.candidates_df['Sustainability_Score'].values if 'Sustainability_Score' in self.candidates_df.columns else np.random.uniform(50, 100, n_candidates)
        
        # 构建评估结果DataFrame
        evaluated_df = pd.DataFrame({
            'Index': np.arange(n_candidates),
            'VFT_A': A,
            'VFT_B': B,
            'VFT_C': C,
            'Viscosity_40C': vis_40,
            'Viscosity_100C': vis_100,
            'Viscosity_minus20C': vis_minus20,
            'Viscosity_Index': vi,
            'Fuel_Economy': fuel_economy,
            'Stribeck_Number': stribeck,
            'Cost_USD_kg': cost,
            'Sustainability_Score': sustainability,
            'Low_Temp_Pass': low_temp_pass,
            'Delta_G': surface_params[:, 0] if surface_params.shape[1] > 0 else np.zeros(n_candidates),
        })
        
        # 添加原始数据列（如果存在）
        if 'Molecule_ID' in self.candidates_df.columns:
            evaluated_df['Molecule_ID'] = self.candidates_df['Molecule_ID'].values
        if 'SMILES' in self.candidates_df.columns:
            evaluated_df['SMILES'] = self.candidates_df['SMILES'].values
        
        self.evaluated_candidates = evaluated_df
        print(f"\n[OK] 评估完成: {len(evaluated_df)} 个候选配方")
        
        return evaluated_df
    
    def _derive_z_params(self) -> np.ndarray:
        """从描述符推导Z参数"""
        n_samples = len(self.candidates_df)
        
        # 尝试从现有列推导
        if 'MW' in self.candidates_df.columns:
            chain_length = (self.candidates_df['MW'].values / 50.0) * 10 + 10
            chain_length = np.clip(chain_length, 10, 50)
        else:
            chain_length = np.random.uniform(10, 50, n_samples)
        
        if 'FractionCSP3' in self.candidates_df.columns:
            symmetry = self.candidates_df['FractionCSP3'].values
        else:
            symmetry = np.random.uniform(0, 1, n_samples)
        
        if 'LogP' in self.candidates_df.columns:
            logp = self.candidates_df['LogP'].values
            polarity = 1.0 - (logp + 3) / 6.0
            polarity = np.clip(polarity, 0, 1)
        else:
            polarity = np.random.uniform(0, 1, n_samples)
        
        return np.column_stack([chain_length, symmetry, polarity])
    
    def _generate_embeddings(self) -> np.ndarray:
        """从描述符生成64维嵌入"""
        from sklearn.decomposition import PCA
        
        # 选择数值列作为描述符
        numeric_cols = self.candidates_df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['Cost_USD_kg', 'Sustainability_Score', 'VFT_A', 'VFT_B', 'VFT_C']
        descriptor_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(descriptor_cols) == 0:
            # 如果没有描述符，使用随机嵌入
            return np.random.normal(0, 1, (len(self.candidates_df), 64))
        
        descriptors = self.candidates_df[descriptor_cols].fillna(0).values
        
        # 使用PCA降维到64维
        pca = PCA(n_components=min(64, descriptors.shape[1]))
        embeddings = pca.fit_transform(descriptors)
        
        # 如果维度不足，填充
        if embeddings.shape[1] < 64:
            padding = np.zeros((embeddings.shape[0], 64 - embeddings.shape[1]))
            embeddings = np.hstack([embeddings, padding])
        
        return embeddings
    
    def identify_pareto_front(self, candidates_df: pd.DataFrame = None) -> Tuple[List[int], pd.DataFrame]:
        """
        识别帕累托前沿（非支配解）
        
        参数:
        - candidates_df: 候选DataFrame（如果为None，使用evaluated_candidates）
        
        返回:
        - pareto_indices: 帕累托最优解的索引列表
        - pareto_df: 帕累托最优解DataFrame
        """
        if candidates_df is None:
            if self.evaluated_candidates is None:
                raise ValueError("请先评估候选配方")
            candidates_df = self.evaluated_candidates
        
        # 只考虑通过物理过滤的候选
        valid_candidates = candidates_df[candidates_df['Low_Temp_Pass']].copy()
        
        if len(valid_candidates) == 0:
            print("警告: 没有通过物理过滤的候选")
            return [], pd.DataFrame()
        
        print(f"\n识别帕累托前沿（{len(valid_candidates)} 个有效候选）...")
        
        # 提取目标值
        # 目标1: 最大化燃油经济性
        fuel_economy = valid_candidates['Fuel_Economy'].values
        
        # 目标2: 最大化可持续性
        sustainability = valid_candidates['Sustainability_Score'].values
        
        # 目标3: 最小化成本
        cost = valid_candidates['Cost_USD_kg'].values
        
        # 标准化目标（转换为最小化问题）
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        
        objectives = np.column_stack([
            -fuel_economy,  # 最大化 -> 最小化（取负）
            -sustainability,  # 最大化 -> 最小化（取负）
            cost  # 已经是最小化
        ])
        
        objectives_scaled = scaler.fit_transform(objectives)
        
        # 寻找帕累托最优解
        pareto_indices = []
        n = len(objectives_scaled)
        
        for i in range(n):
            is_dominated = False
            for j in range(n):
                if i != j:
                    # 检查i是否被j支配
                    if all(objectives_scaled[j, :] <= objectives_scaled[i, :]) and \
                       any(objectives_scaled[j, :] < objectives_scaled[i, :]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(valid_candidates.index[i])
        
        pareto_df = valid_candidates.loc[pareto_indices].copy()
        pareto_df = pareto_df.sort_values('Fuel_Economy', ascending=False)
        
        print(f"[OK] 找到 {len(pareto_indices)} 个帕累托最优解")
        
        return pareto_indices, pareto_df
    
    def visualize_radar_chart(self, top_n: int = 3, save_path: str = 'formulation_radar_chart.png'):
        """
        生成雷达图：Top N配方 vs Baseline
        
        参数:
        - top_n: 显示前N个最优解
        - save_path: 保存路径
        """
        if self.evaluated_candidates is None:
            raise ValueError("请先评估候选配方")
        
        # 获取帕累托最优解
        pareto_indices, pareto_df = self.identify_pareto_front()
        
        if len(pareto_df) == 0:
            print("警告: 没有帕累托最优解，无法生成雷达图")
            return
        
        # 选择Top N
        top_solutions = pareto_df.head(top_n)
        
        # 计算Baseline（平均值）
        valid_candidates = self.evaluated_candidates[self.evaluated_candidates['Low_Temp_Pass']]
        
        # 计算Low_Temp_Performance（归一化到0-100）
        low_temp_perf = (5000 - valid_candidates['Viscosity_minus20C']) / 5000 * 100
        low_temp_perf = np.clip(low_temp_perf, 0, 100)
        
        # 计算Cost_Efficiency（归一化到0-100）
        cost_min = valid_candidates['Cost_USD_kg'].min()
        cost_max = valid_candidates['Cost_USD_kg'].max()
        cost_range = cost_max - cost_min
        if cost_range > 0:
            cost_efficiency = (1 - (valid_candidates['Cost_USD_kg'] - cost_min) / cost_range) * 100
        else:
            cost_efficiency = np.full(len(valid_candidates), 50.0)
        
        baseline = {
            'Fuel_Economy': float(valid_candidates['Fuel_Economy'].mean()),
            'Sustainability_Score': float(valid_candidates['Sustainability_Score'].mean()),
            'Low_Temp_Performance': float(low_temp_perf.mean()),
            'Cost_Efficiency': float(cost_efficiency.mean()),
            'Viscosity_Index': float(valid_candidates['Viscosity_Index'].mean())
        }
        
        # 准备数据
        categories = ['Fuel_Economy', 'Sustainability_Score', 'Low_Temp_Performance', 
                     'Cost_Efficiency', 'Viscosity_Index']
        n_categories = len(categories)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, n_categories, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
        
        # 绘制Baseline
        baseline_values = [baseline[cat] for cat in categories]
        baseline_values += baseline_values[:1]  # 闭合
        
        ax.plot(angles, baseline_values, 'o-', linewidth=2, label='Baseline (Average)', 
               color='gray', alpha=0.7)
        ax.fill(angles, baseline_values, alpha=0.1, color='gray')
        
        # 绘制Top N解决方案
        colors = plt.cm.Set3(np.linspace(0, 1, top_n))
        cost_min = valid_candidates['Cost_USD_kg'].min()
        cost_max = valid_candidates['Cost_USD_kg'].max()
        cost_range = cost_max - cost_min if cost_max > cost_min else 1.0
        
        for idx, (_, row) in enumerate(top_solutions.iterrows()):
            # 归一化各个值到0-100范围
            fuel_econ = float(row['Fuel_Economy'])
            sustainability = float(row['Sustainability_Score'])
            low_temp_perf = float(np.clip((5000 - row['Viscosity_minus20C']) / 5000 * 100, 0, 100))
            cost_eff = float((1 - (row['Cost_USD_kg'] - cost_min) / cost_range) * 100) if cost_range > 0 else 50.0
            vi = float(row['Viscosity_Index'])
            
            values = [fuel_econ, sustainability, low_temp_perf, cost_eff, vi]
            values += values[:1]  # 闭合
            
            label = f"Solution {idx+1}"
            if 'Molecule_ID' in row:
                label += f" ({row['Molecule_ID']})"
            
            ax.plot(angles, values, 'o-', linewidth=2, label=label, color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([
            'Fuel Economy\n(Score)',
            'Sustainability\n(Score)',
            'Low-Temp\nPerformance (%)',
            'Cost Efficiency\n(%)',
            'Viscosity Index\n(VI)'
        ], fontsize=10)
        
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        plt.title('Top Formulations vs Baseline - Radar Chart', 
                 fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"雷达图已保存: {save_path}")
        plt.close()
    
    def visualize_pareto_front_3d(self, save_path: str = 'pareto_front_3d_formulation.png'):
        """
        生成3D散点图：帕累托前沿
        
        参数:
        - save_path: 保存路径
        """
        if self.evaluated_candidates is None:
            raise ValueError("请先评估候选配方")
        
        # 获取帕累托最优解
        pareto_indices, pareto_df = self.identify_pareto_front()
        
        if len(pareto_df) == 0:
            print("警告: 没有帕累托最优解，无法生成3D图")
            return
        
        valid_candidates = self.evaluated_candidates[self.evaluated_candidates['Low_Temp_Pass']]
        
        fig = plt.figure(figsize=(16, 6))
        
        # 子图1: 3D散点图
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 所有有效候选
        ax1.scatter(valid_candidates['Cost_USD_kg'],
                   valid_candidates['Sustainability_Score'],
                   valid_candidates['Fuel_Economy'],
                   c='lightblue', alpha=0.3, s=20, label='All Valid Candidates',
                   edgecolors='none')
        
        # 帕累托前沿
        ax1.scatter(pareto_df['Cost_USD_kg'],
                   pareto_df['Sustainability_Score'],
                   pareto_df['Fuel_Economy'],
                   c='red', alpha=0.8, s=100, marker='*',
                   edgecolors='black', linewidth=1,
                   label=f'Pareto Front (n={len(pareto_df)})')
        
        ax1.set_xlabel('Cost ($/kg)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Sustainability Score', fontsize=11, fontweight='bold')
        ax1.set_zlabel('Fuel Economy Score', fontsize=11, fontweight='bold')
        ax1.set_title('3D Pareto Front', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 2D投影（成本 vs 燃油经济性）
        ax2 = fig.add_subplot(122)
        
        ax2.scatter(valid_candidates['Cost_USD_kg'], valid_candidates['Fuel_Economy'],
                   c='lightblue', alpha=0.3, s=30, label='All Valid Candidates')
        ax2.scatter(pareto_df['Cost_USD_kg'], pareto_df['Fuel_Economy'],
                   c='red', alpha=0.8, s=100, marker='*',
                   edgecolors='black', linewidth=1,
                   label=f'Pareto Front (n={len(pareto_df)})')
        
        ax2.set_xlabel('Cost ($/kg)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Fuel Economy Score', fontsize=11, fontweight='bold')
        ax2.set_title('2D Projection: Cost vs Fuel Economy', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Physics-Informed Multi-Objective Optimization - Pareto Front',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"3D帕累托前沿图已保存: {save_path}")
        plt.close()

# ============================================================================
# Main Execution
# ============================================================================

def main():
    """
    主执行函数
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='物理信息多目标优化器')
    parser.add_argument('--data_path', type=str, default='physics_lubricant_data.csv',
                       help='候选数据文件路径')
    parser.add_argument('--rpm', type=float, default=2000.0,
                       help='标准发动机转速 (RPM)')
    parser.add_argument('--load', type=float, default=500.0,
                       help='标准载荷 (N)')
    parser.add_argument('--top_n', type=int, default=3,
                       help='雷达图显示的Top N解')
    parser.add_argument('--device', type=str, default='cpu',
                       help='计算设备')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    print("="*70)
    print("物理信息多目标优化器")
    print("="*70)
    
    # 创建优化器
    optimizer = FormulationOptimizer(device=device)
    
    # 加载候选数据
    optimizer.load_candidates(data_path=args.data_path)
    
    # 评估候选配方
    evaluated_df = optimizer.evaluate_candidates(
        standard_rpm=args.rpm,
        standard_load=args.load
    )
    
    # 识别帕累托前沿
    pareto_indices, pareto_df = optimizer.identify_pareto_front()
    
    if len(pareto_df) > 0:
        print(f"\n帕累托最优解统计:")
        print(f"  数量: {len(pareto_df)}")
        print(f"  燃油经济性范围: {pareto_df['Fuel_Economy'].min():.2f} - {pareto_df['Fuel_Economy'].max():.2f}")
        print(f"  可持续性范围: {pareto_df['Sustainability_Score'].min():.2f} - {pareto_df['Sustainability_Score'].max():.2f}")
        print(f"  成本范围: ${pareto_df['Cost_USD_kg'].min():.2f} - ${pareto_df['Cost_USD_kg'].max():.2f}/kg")
        
        # 保存优化后的配方
        output_path = 'optimized_formulations.csv'
        pareto_df.to_csv(output_path, index=False)
        print(f"\n[OK] 优化配方已保存: {output_path}")
        
        # 生成可视化
        print("\n生成可视化图表...")
        optimizer.visualize_radar_chart(top_n=args.top_n)
        optimizer.visualize_pareto_front_3d()
        
        print("\n" + "="*70)
        print("优化完成！")
        print("="*70)
        print(f"\n生成的文件:")
        print(f"  - optimized_formulations.csv")
        print(f"  - formulation_radar_chart.png")
        print(f"  - pareto_front_3d_formulation.png")
    else:
        print("\n警告: 未找到帕累托最优解")

if __name__ == "__main__":
    main()

