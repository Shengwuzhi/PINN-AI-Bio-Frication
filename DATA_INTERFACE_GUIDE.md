# 数据接口使用指南
# Data Interface User Guide

## 概述

本系统提供了统一的数据接口，连接AI-Bio-Frication和PINN-AI-Bio-Frication两个项目，支持使用10000个生物质分子的真实描述符数据进行训练，并支持用户上传真实实验数据。

## 功能特点

1. **多数据源支持**
   - AI-Bio-Frication项目数据（10000个分子）
   - 用户上传的真实数据
   - 数据源合并

2. **自动数据转换**
   - 自动提取Z参数
   - 自动生成64维分子嵌入
   - 自动处理缺失值和异常值

3. **数据验证**
   - 完整性检查
   - 格式验证
   - 异常值检测

4. **真实数据上传**
   - 支持CSV、Excel、JSON格式
   - 自动预处理
   - 元数据保存

## 使用方法

### 1. 使用AI-Bio-Frication数据

```python
from data_interface import AIBioFricationInterface

# 创建接口（自动查找默认路径）
interface = AIBioFricationInterface()

# 加载数据
data = interface.load_data()

# 获取数据
molecular_features = data['molecular_features']  # [n_samples, 67]
z_params = data['z_params']  # [n_samples, 3]
descriptors = data['descriptors']  # [n_samples, n_descriptors]
targets = data.get('targets', None)  # 如果存在
metadata = data['metadata']
```

### 2. 上传真实数据

```python
from data_interface import RealDataInterface

# 创建接口
interface = RealDataInterface()

# 上传分子数据
molecular_data = interface.upload_data(
    file_path='your_molecular_data.csv',
    data_type='molecular'
)

# 上传传感器数据
sensor_data = interface.upload_data(
    file_path='your_sensor_data.csv',
    data_type='sensor'
)

# 上传台架测试数据
bench_data = interface.upload_data(
    file_path='your_bench_test_data.csv',
    data_type='bench_test'
)
```

### 3. 使用真实数据训练模型

#### 命令行方式

```bash
# 训练L2模型（使用AI-Bio-Frication数据）
python train_with_real_data.py --model l2 --data_source ai_bio_frication --epochs 100

# 训练L2.5模型
python train_with_real_data.py --model l25 --data_source ai_bio_frication --epochs 100

# 训练所有模型
python train_with_real_data.py --model all --data_source ai_bio_frication --epochs 100

# 使用上传的数据训练
python train_with_real_data.py --model l2 --data_source uploaded --data_path uploaded_data/molecular_data.csv
```

#### Python代码方式

```python
from train_with_real_data import train_l2_with_real_data

# 训练L2模型
model, train_losses, val_losses = train_l2_with_real_data(
    data_source='ai_bio_frication',
    num_epochs=100,
    batch_size=32,
    device='cuda'  # 或 'cpu'
)
```

### 4. 数据格式要求

#### 分子数据格式（CSV）

必需列：
- `SMILES`: 分子SMILES字符串
- `Molecule_ID`: 分子ID（可选）

推荐列（用于Z参数推导）：
- `MW`: 分子量
- `LogP`: 亲脂性系数
- `TPSA`: 拓扑极性表面积
- `FractionCSP3`: SP3碳比例
- `NumAromaticRings`: 芳香环数

如果存在Z参数列，将直接使用：
- `Z_Param_Chain_Ratio` 或 `chain_length`
- `Z_Param_Symmetry` 或 `symmetry`
- `Z_Param_Polarity` 或 `polarity`

#### 传感器数据格式（CSV）

必需列：
- 包含'temperature'或'temp'的列：温度数据
- 包含'vibration'或'vib'的列：振动数据

#### 台架测试数据格式（CSV）

推荐列：
- 包含'viscosity'或'vis'的列：粘度数据
- 包含'speed'或'rpm'的列：速度数据
- 包含'load'的列：载荷数据
- 包含'fuel'或'economy'的列：燃油经济性数据

## 数据接口类说明

### AIBioFricationInterface

**功能**: 读取AI-Bio-Frication项目的10000个分子数据

**自动处理**:
- 提取或推导Z参数
- 从50+维描述符生成64维嵌入
- 提取目标值（如果存在）

**方法**:
- `load_data()`: 加载并转换数据
- `validate_data(df)`: 验证数据完整性
- `preprocess_data(df)`: 预处理数据

### RealDataInterface

**功能**: 处理用户上传的真实数据

**支持格式**: CSV, Excel (.xlsx, .xls), JSON

**自动处理**:
- 数据验证
- 异常值处理（IQR方法）
- 缺失值填充
- 格式转换

**方法**:
- `upload_data(file_path, data_type)`: 上传并处理数据
- `validate_data(df)`: 验证数据
- `preprocess_data(df)`: 预处理数据

### UnifiedDataManager

**功能**: 管理多个数据源，合并数据

**方法**:
- `add_data_source(name, interface)`: 添加数据源
- `load_all_data()`: 加载所有数据源
- `save_combined_data(output_path)`: 保存合并数据

## 数据流程

```
AI-Bio-Frication数据
    ↓
AIBioFricationInterface
    ↓
提取Z参数 + 分子描述符
    ↓
生成64维嵌入
    ↓
组合为67维特征向量
    ↓
PINN模型训练
```

## 示例：完整训练流程

```python
# 1. 加载AI-Bio-Frication数据
from data_interface import AIBioFricationInterface
from train_with_real_data import train_l2_with_real_data

interface = AIBioFricationInterface()
data = interface.load_data()

print(f"加载了 {data['molecular_features'].shape[0]} 个样本")
print(f"特征维度: {data['molecular_features'].shape[1]}")

# 2. 训练模型
model, train_losses, val_losses = train_l2_with_real_data(
    data_source='ai_bio_frication',
    num_epochs=100,
    device='cuda'
)

# 3. 使用模型进行预测
model.eval()
sample_features = data['molecular_features'][:10]
sample_tensor = torch.FloatTensor(sample_features)
vft_params, surface_params = model(sample_tensor)
print(f"预测的VFT参数形状: {vft_params.shape}")
```

## 数据验证

系统会自动进行以下验证：

1. **完整性检查**
   - 必需列是否存在
   - 数据行数是否足够

2. **数值检查**
   - 缺失值比例
   - 异常值检测
   - 数值范围检查

3. **格式检查**
   - 文件格式支持
   - 数据类型正确性

## 故障排除

### 问题1: 找不到AI-Bio-Frication数据文件

**解决方案**:
```python
# 指定完整路径
interface = AIBioFricationInterface(
    data_path=r"C:\Users\weizh\AI-Bio-Frication\enhanced_lubricant_data.csv"
)
```

### 问题2: 数据验证失败

**可能原因**:
- 数据格式不正确
- 缺失值过多
- 必需列不存在

**解决方案**:
- 检查CSV文件格式
- 确保有足够的数值列（至少10列）
- 检查缺失值比例（应<50%）

### 问题3: Z参数缺失

**解决方案**:
系统会自动从分子描述符推导Z参数，无需手动处理。

## 高级用法

### 自定义特征映射

```python
# 修改特征映射
interface = AIBioFricationInterface()
interface.feature_mapping['Z_params']['chain_length'] = ['MW', 'NumAtoms']
```

### 合并多个数据源

```python
from data_interface import UnifiedDataManager, AIBioFricationInterface, RealDataInterface

manager = UnifiedDataManager()

# 添加AI-Bio-Frication数据
ai_bio_interface = AIBioFricationInterface()
manager.add_data_source('ai_bio', ai_bio_interface)

# 添加上传数据
upload_interface = RealDataInterface()
# ... 上传数据后
manager.add_data_source('uploaded', upload_interface)

# 加载并合并
combined_data = manager.load_all_data()
```

## 联系与支持

如有问题，请检查：
1. 数据文件路径是否正确
2. 数据格式是否符合要求
3. 依赖包是否已安装

## 更新日志

- v1.0: 初始版本，支持AI-Bio-Frication数据接口
- v1.1: 添加真实数据上传功能
- v1.2: 添加数据合并功能

