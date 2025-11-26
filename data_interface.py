"""
Unified Data Interface for PINN-AI-Bio-Frication System
统一数据接口 - 连接AI-Bio-Frication和PINN-AI-Bio-Frication
支持多种数据源和真实数据上传
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from sklearn.preprocessing import StandardScaler
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Interface Base Class
# ============================================================================

class DataInterface:
    """
    统一数据接口基类
    支持多种数据源格式
    """
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = data_path
        self.scaler_X = None
        self.scaler_y = None
        self.feature_mapping = {}
        self.data_info = {}
        
    def load_data(self) -> Dict:
        """加载数据（子类实现）"""
        raise NotImplementedError
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据完整性"""
        raise NotImplementedError
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        raise NotImplementedError

# ============================================================================
# AI-Bio-Frication Data Interface
# ============================================================================

class AIBioFricationInterface(DataInterface):
    """
    AI-Bio-Frication项目数据接口
    读取10000个生物质分子的描述符数据
    """
    
    def __init__(self, data_path: str = None):
        if data_path is None:
            # 默认路径：AI-Bio-Frication项目
            default_path = Path(r"C:\Users\weizh\AI-Bio-Frication\enhanced_lubricant_data.csv")
            if default_path.exists():
                data_path = str(default_path)
            else:
                raise FileNotFoundError(f"未找到默认数据文件: {default_path}")
        
        super().__init__(data_path)
        self._init_feature_mapping()
    
    def _init_feature_mapping(self):
        """初始化特征映射（AI-Bio-Frication -> PINN-AI-Bio-Frication）"""
        # Z参数映射（如果AI-Bio-Frication中有Z参数）
        self.feature_mapping = {
            'Z_params': {
                'chain_length': ['Z_Param_Chain_Ratio', 'MW', 'NumAtoms'],  # 可能的映射
                'symmetry': ['Z_Param_Symmetry', 'FractionCSP3', 'NumAromaticRings'],
                'polarity': ['Z_Param_Polarity', 'LogP', 'TPSA']
            },
            'molecular_descriptors': [
                'MW', 'LogP', 'NumHDonors', 'NumHAcceptors', 'TPSA',
                'NumRotatableBonds', 'NumAromaticRings', 'FractionCSP3',
                'NumHeteroatoms', 'MolecularRefractivity', 'NumAtoms',
                'NumBonds', 'NumRings', 'HeavyAtomCount', 'BertzCT',
                'Chi0', 'Chi1', 'Chi0n', 'Chi1n', 'Kappa1', 'Kappa2'
            ]
        }
    
    def load_data(self) -> Dict:
        """
        加载AI-Bio-Frication数据并转换为PINN格式
        
        返回:
        - Dict包含:
            - 'molecular_features': 分子特征矩阵
            - 'z_params': Z参数（如果存在）
            - 'descriptors': 分子描述符
            - 'targets': 目标值（如果存在）
            - 'metadata': 元数据
        """
        print(f"从AI-Bio-Frication加载数据: {self.data_path}")
        df = pd.read_csv(self.data_path)
        
        # 验证数据
        if not self.validate_data(df):
            raise ValueError("数据验证失败")
        
        # 预处理
        df = self.preprocess_data(df)
        
        # 提取Z参数（如果存在）
        z_params = self._extract_z_params(df)
        
        # 提取分子描述符
        descriptors = self._extract_descriptors(df)
        
        # 提取目标值（如果存在）
        targets = self._extract_targets(df)
        
        # 生成分子嵌入（从描述符中提取64维）
        molecular_embeddings = self._generate_molecular_embeddings(descriptors)
        
        # 组合特征
        if z_params is not None:
            molecular_features = np.hstack([z_params, molecular_embeddings])
        else:
            # 如果没有Z参数，从描述符中推导
            z_params = self._derive_z_params(descriptors)
            molecular_features = np.hstack([z_params, molecular_embeddings])
        
        result = {
            'molecular_features': molecular_features,
            'z_params': z_params,
            'descriptors': descriptors,
            'molecular_embeddings': molecular_embeddings,
            'targets': targets,
            'metadata': {
                'n_samples': len(df),
                'n_features': molecular_features.shape[1],
                'n_descriptors': descriptors.shape[1],
                'source': 'AI-Bio-Frication',
                'molecule_ids': df['Molecule_ID'].values if 'Molecule_ID' in df.columns else None,
                'smiles': df['SMILES'].values if 'SMILES' in df.columns else None
            }
        }
        
        self.data_info = result['metadata']
        return result
    
    def _extract_z_params(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """提取Z参数"""
        z_param_cols = [
            'Z_Param_Chain_Ratio', 'Z_Param_Symmetry', 'Z_Param_Polarity',
            'Z_Param_Anion_Ring_Size', 'Z_Param_Density_Ratio'
        ]
        
        # 检查是否存在Z参数列
        available_cols = [col for col in z_param_cols if col in df.columns]
        
        if len(available_cols) >= 3:
            # 使用前3个Z参数（chain_length, symmetry, polarity）
            z_params = df[available_cols[:3]].values
            return z_params
        else:
            return None
    
    def _derive_z_params(self, descriptors: np.ndarray) -> np.ndarray:
        """
        从分子描述符推导Z参数
        使用物理关系进行推导
        """
        n_samples = descriptors.shape[0]
        
        # 从描述符中提取相关特征
        # 假设描述符顺序：MW, LogP, TPSA, NumRotatableBonds, ...
        if descriptors.shape[1] >= 10:
            # chain_length: 基于MW和原子数
            chain_length = (descriptors[:, 0] / 50.0) * 10 + 10  # MW归一化到10-50范围
            
            # symmetry: 基于FractionCSP3和芳香环数
            if descriptors.shape[1] > 7:
                symmetry = descriptors[:, 7]  # FractionCSP3
            else:
                symmetry = np.random.uniform(0, 1, n_samples)
            
            # polarity: 基于LogP和TPSA
            logp = descriptors[:, 1]
            if descriptors.shape[1] > 4:
                tpsa = descriptors[:, 4]
                # 归一化到0-1范围
                polarity = 1.0 - (logp + 3) / 6.0  # LogP通常在-3到3之间
                polarity = np.clip(polarity, 0, 1)
            else:
                polarity = np.random.uniform(0, 1, n_samples)
        else:
            # 如果描述符不足，使用默认值
            chain_length = np.random.uniform(10, 50, n_samples)
            symmetry = np.random.uniform(0, 1, n_samples)
            polarity = np.random.uniform(0, 1, n_samples)
        
        z_params = np.column_stack([chain_length, symmetry, polarity])
        return z_params
    
    def _extract_descriptors(self, df: pd.DataFrame) -> np.ndarray:
        """提取分子描述符"""
        descriptor_cols = self.feature_mapping['molecular_descriptors']
        available_cols = [col for col in descriptor_cols if col in df.columns]
        
        if len(available_cols) == 0:
            # 如果没有找到，使用所有数值列（排除ID和SMILES）
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['Molecule_ID'] if 'Molecule_ID' in numeric_cols else []
            available_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        descriptors = df[available_cols].fillna(0).values
        return descriptors
    
    def _extract_targets(self, df: pd.DataFrame) -> Optional[Dict]:
        """提取目标值（如果存在）"""
        target_cols = {
            'viscosity': ['Viscosity_cSt_40C', 'Viscosity_40C', 'viscosity_40C'],
            'friction': ['Friction_Coefficient', 'friction_coefficient'],
            'vft_a': ['VFT_A', 'VFT_A'],
            'vft_b': ['VFT_B', 'VFT_B'],
            'vft_c': ['VFT_C', 'VFT_C']
        }
        
        targets = {}
        for key, possible_cols in target_cols.items():
            for col in possible_cols:
                if col in df.columns:
                    targets[key] = df[col].values
                    break
        
        return targets if targets else None
    
    def _generate_molecular_embeddings(self, descriptors: np.ndarray) -> np.ndarray:
        """
        从分子描述符生成64维嵌入向量
        使用PCA或特征选择
        """
        from sklearn.decomposition import PCA
        
        n_samples, n_features = descriptors.shape
        
        if n_features >= 64:
            # 如果特征数足够，使用PCA降维到64维
            pca = PCA(n_components=64)
            embeddings = pca.fit_transform(descriptors)
        elif n_features > 0:
            # 如果特征数不足，使用填充
            pca = PCA(n_components=min(64, n_features))
            embeddings = pca.fit_transform(descriptors)
            # 填充到64维
            if embeddings.shape[1] < 64:
                padding = np.zeros((n_samples, 64 - embeddings.shape[1]))
                embeddings = np.hstack([embeddings, padding])
        else:
            # 如果没有特征，使用随机初始化
            embeddings = np.random.normal(0, 1, (n_samples, 64))
        
        return embeddings
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证数据完整性"""
        required_cols = ['SMILES'] if 'SMILES' in data.columns else []
        
        # 检查是否有足够的数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 10:
            print(f"警告: 数值列数量不足 ({len(numeric_cols)})")
            return False
        
        # 检查缺失值
        missing_ratio = data[numeric_cols].isnull().sum().sum() / (len(data) * len(numeric_cols))
        if missing_ratio > 0.5:
            print(f"警告: 缺失值比例过高 ({missing_ratio:.2%})")
            return False
        
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        # 填充缺失值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        # 处理无穷值
        data[numeric_cols] = data[numeric_cols].replace([np.inf, -np.inf], np.nan)
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        return data

# ============================================================================
# Real Data Upload Interface
# ============================================================================

class RealDataInterface(DataInterface):
    """
    真实数据上传接口
    支持用户上传真实实验数据进行训练和验证
    """
    
    def __init__(self, upload_path: str = None):
        if upload_path is None:
            upload_path = Path(__file__).parent / "uploaded_data"
            upload_path.mkdir(exist_ok=True)
        
        super().__init__(upload_path)
        self.upload_path = Path(upload_path)
        self.supported_formats = ['.csv', '.xlsx', '.xls', '.json']
    
    def upload_data(self, file_path: str, data_type: str = 'molecular') -> Dict:
        """
        上传真实数据
        
        参数:
        - file_path: 数据文件路径
        - data_type: 数据类型 ('molecular', 'sensor', 'bench_test')
        
        返回:
        - 处理后的数据字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"不支持的文件格式: {file_path.suffix}")
        
        # 复制文件到上传目录
        dest_path = self.upload_path / f"{data_type}_{file_path.name}"
        import shutil
        shutil.copy2(file_path, dest_path)
        
        # 加载数据
        if file_path.suffix == '.csv':
            df = pd.read_csv(file_path)
        elif file_path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path)
        elif file_path.suffix == '.json':
            df = pd.read_json(file_path)
        else:
            raise ValueError(f"无法读取文件格式: {file_path.suffix}")
        
        # 验证数据
        if not self.validate_data(df):
            raise ValueError("数据验证失败，请检查数据格式")
        
        # 预处理
        df = self.preprocess_data(df)
        
        # 保存处理后的数据
        processed_path = self.upload_path / f"{data_type}_processed_{file_path.name}"
        df.to_csv(processed_path, index=False)
        
        # 转换为PINN格式
        if data_type == 'molecular':
            result = self._convert_molecular_data(df)
        elif data_type == 'sensor':
            result = self._convert_sensor_data(df)
        elif data_type == 'bench_test':
            result = self._convert_bench_test_data(df)
        else:
            raise ValueError(f"不支持的数据类型: {data_type}")
        
        # 保存元数据
        metadata = {
            'source_file': str(file_path),
            'data_type': data_type,
            'n_samples': len(df),
            'columns': list(df.columns),
            'upload_time': pd.Timestamp.now().isoformat()
        }
        
        metadata_path = self.upload_path / f"{data_type}_metadata_{file_path.stem}.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        result['metadata'] = metadata
        return result
    
    def _convert_molecular_data(self, df: pd.DataFrame) -> Dict:
        """转换分子数据为PINN格式"""
        interface = AIBioFricationInterface()
        # 使用相同的方法处理
        df_processed = interface.preprocess_data(df)
        z_params = interface._extract_z_params(df_processed)
        if z_params is None:
            descriptors = interface._extract_descriptors(df_processed)
            z_params = interface._derive_z_params(descriptors)
        else:
            descriptors = interface._extract_descriptors(df_processed)
        
        molecular_embeddings = interface._generate_molecular_embeddings(descriptors)
        molecular_features = np.hstack([z_params, molecular_embeddings])
        
        targets = interface._extract_targets(df_processed)
        
        return {
            'molecular_features': molecular_features,
            'z_params': z_params,
            'descriptors': descriptors,
            'molecular_embeddings': molecular_embeddings,
            'targets': targets
        }
    
    def _convert_sensor_data(self, df: pd.DataFrame) -> Dict:
        """转换传感器数据为PINN格式"""
        # 查找温度、振动等列
        temp_cols = [col for col in df.columns if 'temp' in col.lower() or 'temperature' in col.lower()]
        vib_cols = [col for col in df.columns if 'vib' in col.lower() or 'vibration' in col.lower()]
        
        if not temp_cols or not vib_cols:
            raise ValueError("未找到温度或振动数据列")
        
        # 提取时间序列
        time_series = np.stack([
            df[temp_cols[0]].values,
            df[vib_cols[0]].values
        ], axis=-1)
        
        return {
            'time_series': time_series,
            'temperature': df[temp_cols[0]].values,
            'vibration': df[vib_cols[0]].values
        }
    
    def _convert_bench_test_data(self, df: pd.DataFrame) -> Dict:
        """转换台架测试数据为PINN格式"""
        # 查找相关列
        vis_cols = [col for col in df.columns if 'vis' in col.lower() or 'viscosity' in col.lower()]
        speed_cols = [col for col in df.columns if 'speed' in col.lower() or 'rpm' in col.lower()]
        load_cols = [col for col in df.columns if 'load' in col.lower()]
        fuel_cols = [col for col in df.columns if 'fuel' in col.lower() or 'economy' in col.lower()]
        
        result = {}
        if vis_cols:
            result['viscosity'] = df[vis_cols[0]].values
        if speed_cols:
            result['speed'] = df[speed_cols[0]].values
        if load_cols:
            result['load'] = df[load_cols[0]].values
        if fuel_cols:
            result['fuel_economy'] = df[fuel_cols[0]].values
        
        return result
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """验证真实数据"""
        if len(data) == 0:
            return False
        
        # 检查是否有足够的数值列
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 3:
            print("警告: 数值列数量不足")
            return False
        
        return True
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理真实数据"""
        # 填充缺失值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
        
        # 处理异常值（使用IQR方法）
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            data[col] = data[col].clip(lower_bound, upper_bound)
        
        return data

# ============================================================================
# Unified Data Manager
# ============================================================================

class UnifiedDataManager:
    """
    统一数据管理器
    整合多个数据源，提供统一接口
    """
    
    def __init__(self):
        self.data_sources = {}
        self.combined_data = None
    
    def add_data_source(self, name: str, interface: DataInterface):
        """添加数据源"""
        self.data_sources[name] = interface
    
    def load_all_data(self) -> Dict:
        """加载所有数据源并合并"""
        all_data = {}
        
        for name, interface in self.data_sources.items():
            print(f"\n加载数据源: {name}")
            try:
                data = interface.load_data()
                all_data[name] = data
            except Exception as e:
                print(f"警告: 加载数据源 {name} 失败: {e}")
                continue
        
        # 合并数据
        if len(all_data) > 0:
            self.combined_data = self._merge_data(all_data)
            return self.combined_data
        else:
            raise ValueError("没有成功加载任何数据源")
    
    def _merge_data(self, all_data: Dict) -> Dict:
        """合并多个数据源"""
        # 找到主要数据源（通常是AI-Bio-Frication）
        main_source = None
        for name, data in all_data.items():
            if 'molecular_features' in data:
                main_source = name
                break
        
        if main_source is None:
            raise ValueError("未找到主要数据源")
        
        main_data = all_data[main_source]
        
        # 合并其他数据源
        merged = {
            'molecular_features': main_data['molecular_features'],
            'z_params': main_data['z_params'],
            'descriptors': main_data['descriptors'],
            'molecular_embeddings': main_data['molecular_embeddings'],
            'metadata': {
                'sources': list(all_data.keys()),
                'n_samples': main_data['molecular_features'].shape[0],
                **main_data.get('metadata', {})
            }
        }
        
        # 添加目标值（如果存在）
        if 'targets' in main_data and main_data['targets']:
            merged['targets'] = main_data['targets']
        
        return merged
    
    def save_combined_data(self, output_path: str):
        """保存合并后的数据"""
        if self.combined_data is None:
            raise ValueError("没有可保存的数据，请先调用load_all_data()")
        
        # 保存为CSV格式（用于训练）
        df = pd.DataFrame(self.combined_data['molecular_features'])
        df.to_csv(output_path, index=False)
        print(f"合并数据已保存到: {output_path}")

# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # 示例：使用AI-Bio-Frication数据
    print("="*60)
    print("数据接口测试")
    print("="*60)
    
    # 1. 加载AI-Bio-Frication数据
    try:
        ai_bio_interface = AIBioFricationInterface()
        ai_bio_data = ai_bio_interface.load_data()
        print(f"\n✓ AI-Bio-Frication数据加载成功")
        print(f"  样本数: {ai_bio_data['molecular_features'].shape[0]}")
        print(f"  特征数: {ai_bio_data['molecular_features'].shape[1]}")
    except Exception as e:
        print(f"\n✗ AI-Bio-Frication数据加载失败: {e}")
        ai_bio_data = None
    
    # 2. 测试真实数据上传接口
    print("\n" + "="*60)
    print("真实数据上传接口已就绪")
    print("使用方法:")
    print("  from data_interface import RealDataInterface")
    print("  interface = RealDataInterface()")
    print("  data = interface.upload_data('your_data.csv', data_type='molecular')")
    print("="*60)

