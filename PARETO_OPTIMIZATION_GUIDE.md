# 帕累托最优优化指南
# Pareto Optimization Guide

## 概述

本系统实现了基于PINN的多目标优化和帕累托最优分析，考虑L2、L2.5、L3多个层级的目标函数。

## 优化目标

### 主要目标（三维目标空间）

1. **最小化摩擦系数** (L2层级)
   - 目标：降低摩擦，提高润滑性能
   - 范围：0.05 - 0.3

2. **最大化燃油经济性** (L2.5层级)
   - 目标：提高燃油效率
   - 范围：0 - 100

3. **最大化剩余使用寿命RUL** (L3层级)
   - 目标：延长润滑剂寿命
   - 范围：0 - 100+ (时间步)

### 辅助目标

- **粘度** (40°C, 100°C)
- **VFT参数** (A, B, C)
- **Stribeck数**
- **退化率**
- **健康度**

## 使用方法

### 1. 基本使用

```python
from pareto_optimization import PINNMultiObjectiveOptimizer
from main_system_integration import UnifiedPipeline

# 创建管道和优化器
pipeline = UnifiedPipeline(device='cuda')
optimizer = PINNMultiObjectiveOptimizer(pipeline, device='cuda')

# 准备分子特征
molecular_features = ...  # [n_samples, 67]

# 预测目标值
objectives = optimizer.predict_objectives(molecular_features)

# 计算帕累托前沿
pareto_indices, objectives_scaled = optimizer.calculate_pareto_front(
    objectives,
    objective_names=['friction_coefficient', 'fuel_economy', 'rul_final'],
    minimize=[True, False, False]  # 摩擦最小化，其他最大化
)

# 可视化
optimizer.visualize_pareto_front_3d(objectives, pareto_indices)
optimizer.visualize_pareto_tradeoff(objectives, pareto_indices)

# 获取TOP-N解
top_solutions = optimizer.get_top_solutions(objectives, pareto_indices, top_n=10)
```

### 2. 命令行使用

```bash
# 使用默认设置（1000个样本）
python pareto_optimization.py

# 指定样本数量
python pareto_optimization.py --n_samples 2000

# 显示TOP-20最优解
python pareto_optimization.py --top_n 20

# 使用GPU
python pareto_optimization.py --device cuda
```

### 3. 自定义目标

```python
# 自定义目标组合
pareto_indices, _ = optimizer.calculate_pareto_front(
    objectives,
    objective_names=['viscosity_40', 'friction_coefficient', 'rul_final'],
    minimize=[False, True, False]  # 粘度最大化，摩擦最小化，RUL最大化
)
```

### 4. 加权综合得分

```python
# 计算加权综合得分
weights = {
    'friction_coefficient': 0.3,  # 30%权重
    'fuel_economy': 0.4,           # 40%权重
    'rul_final': 0.3               # 30%权重
}

scores = optimizer.calculate_weighted_score(objectives, weights=weights)

# 按得分排序
sorted_indices = np.argsort(scores)[::-1]  # 降序
top_10 = sorted_indices[:10]
```

## 输出结果

### 1. 可视化图表

- **pareto_front_3d.png**: 3D帕累托前沿图
  - 左图：3D散点图
  - 右图：2D投影

- **pareto_tradeoff.png**: 权衡关系分析图
  - 摩擦 vs 燃油经济性
  - 燃油经济性 vs RUL
  - 摩擦 vs RUL
  - 综合得分分布

### 2. 数据文件

- **pareto_optimal_solutions.csv**: TOP-N最优解详情
  - 包含所有目标值
  - 按综合得分排序

## 帕累托最优原理

### 定义

一个解是帕累托最优的，当且仅当：
- 不存在另一个解在所有目标上都不差于它
- 且至少在一个目标上优于它

### 数学表达

对于解 x*，如果不存在解 x 使得：
- f_i(x) ≤ f_i(x*) 对所有目标 i
- f_j(x) < f_j(x*) 至少对一个目标 j

则 x* 是帕累托最优解。

### 优势

1. **无需权重**: 不需要预先设定目标权重
2. **全面探索**: 发现所有可能的权衡方案
3. **决策支持**: 为决策者提供多种选择

## 应用场景

### 1. 配方设计

- 在多个性能指标间寻找平衡
- 识别最优配方候选
- 评估不同配方的权衡关系

### 2. 性能优化

- 同时优化多个性能指标
- 避免单一目标优化导致的性能损失
- 发现意外的性能组合

### 3. 成本效益分析

- 结合成本和性能
- 识别性价比最高的配方
- 支持决策制定

## 注意事项

1. **计算复杂度**
   - 帕累托前沿计算复杂度为 O(n²)
   - 对于大量样本，建议先采样

2. **目标选择**
   - 选择相互独立的目标
   - 避免高度相关的目标

3. **数据质量**
   - 确保目标值预测准确
   - 检查异常值

## 示例：完整工作流

```python
# 1. 加载数据
from data_interface import AIBioFricationInterface

data_interface = AIBioFricationInterface()
data = data_interface.load_data()
molecular_features = data['molecular_features'][:1000]  # 使用1000个样本

# 2. 创建优化器
from pareto_optimization import PINNMultiObjectiveOptimizer
from main_system_integration import UnifiedPipeline

pipeline = UnifiedPipeline(device='cpu')
optimizer = PINNMultiObjectiveOptimizer(pipeline, device='cpu')

# 3. 预测目标
objectives = optimizer.predict_objectives(molecular_features)

# 4. 计算帕累托前沿
pareto_indices, _ = optimizer.calculate_pareto_front(objectives)

# 5. 可视化
optimizer.visualize_pareto_front_3d(objectives, pareto_indices)
optimizer.visualize_pareto_tradeoff(objectives, pareto_indices)

# 6. 获取最优解
top_solutions = optimizer.get_top_solutions(objectives, pareto_indices, top_n=10)
print(top_solutions)

# 7. 保存结果
top_solutions.to_csv('optimal_solutions.csv', index=False)
```

## 与AI-Bio-Frication的对比

| 特性 | AI-Bio-Frication | PINN-AI-Bio-Frication |
|------|-----------------|----------------------|
| 目标数量 | 3 (成本、摩擦、可持续性) | 3+ (摩擦、燃油经济性、RUL等) |
| 数据来源 | 静态数据 | PINN模型预测 |
| 物理约束 | 无 | 有（VFT、Arrhenius等） |
| 层级考虑 | 单一层级 | 多层级（L2、L2.5、L3） |
| 可解释性 | 中等 | 高（物理参数） |

## 更新日志

- v1.0: 初始版本，实现基本帕累托最优计算
- v1.1: 添加多层级目标支持
- v1.2: 添加可视化功能
- v1.3: 集成PINN模型预测

