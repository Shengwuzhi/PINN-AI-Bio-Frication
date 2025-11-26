# Physics-Informed Neural Network System for Bio-based Lubricant Design
# 生物基润滑剂设计的物理信息神经网络系统

## 系统概述

本系统实现了一个多层级物理信息神经网络（PINN）架构，用于生物基润滑剂的配方设计和性能预测。

## 系统架构

### Level 1 (L1): 分子输入层
- **输入**: Z参数（链长、对称性、极性）+ 64维分子嵌入
- **功能**: 分子特征表示

### Level 2 (L2): 物理参数预测层
- **模型**: `pinn_l2_model.py` - Physics-Informed Neural Network
- **输入**: L1分子特征
- **输出**: 
  - VFT参数 (A, B, C)
  - 表面物理参数 (Delta_G, Shear_Strength)
- **物理约束**: VFT方程 `ln(η) = A + B/(T-C)`
- **特点**: 灰箱模型，显式物理层

### Level 2.5 (L2.5): 台架测试迁移学习层
- **模型**: `bench_transfer_l25.py` - Bench Transfer Model
- **输入**: L2输出的粘度 + 发动机状态（速度、载荷）
- **输出**: 燃油经济性评分
- **物理约束**: Stribeck数显式计算 `S = (η·v)/P`
- **特点**: 学习边界润滑vs流体动压润滑的摩擦行为

### Level 3 (L3): 剩余使用寿命预测层
- **模型**: `physics_rul_l3.py` - Physics-Constrained RUL Model
- **输入**: 时间序列（温度、振动）+ 静态特征
- **输出**: RUL（剩余使用寿命）、退化率、健康度
- **物理约束**: Arrhenius动力学 `k = A·exp(-Ea/RT)`
- **特点**: 单调性约束，物理引导的退化预测

## 生成的文件

### 核心模型文件
1. `physics_data_gen.py` - 物理数据生成器
2. `pinn_l2_model.py` - L2物理参数预测模型
3. `bench_transfer_l25.py` - L2.5台架测试模型
4. `physics_rul_l3.py` - L3 RUL预测模型
5. `main_system_integration.py` - 统一管道系统

### 发表级图表
1. **figure_1_vft_comparison.png** - VFT拟合对比图
   - 展示物理信息模型（灰箱）vs 黑箱MLP的平滑度对比
   - 证明科学优势

2. **figure_2_stribeck_mapping.png** - Stribeck映射图
   - 展示L2.5模型如何学习粘度、速度和效率的关系
   - 包含真实值、预测值和准确性分析

3. **figure_3_physics_constrained_rul.png** - 物理约束RUL图
   - 展示L3模型如何遵循Arrhenius定律
   - 高温下更快退化的物理行为

### 训练结果文件
- `pinn_training_curves.png` - L2模型训练曲线
- `pinn_viscosity_validation.png` - L2模型验证图
- `bench_transfer_training_curves.png` - L2.5模型训练曲线
- `stribeck_fuel_economy_scatter.png` - Stribeck数散点图
- `rul_training_curves_comparison.png` - L3模型训练对比
- `rul_prediction_comparison.png` - L3模型预测对比

## 使用方法

### 1. 生成数据
```bash
python physics_data_gen.py
```

### 2. 训练模型
```bash
# 训练L2模型
python pinn_l2_model.py

# 训练L2.5模型
python bench_transfer_l25.py

# 训练L3模型
python physics_rul_l3.py
```

### 3. 运行统一管道并生成图表
```bash
python main_system_integration.py
```

## 系统特点

1. **物理约束**: 所有模型都包含显式物理方程
2. **可解释性**: 输出物理参数（VFT参数、活化能等）
3. **稳定性**: 物理约束使预测更稳定
4. **科学性**: 遵循已知的物理定律

## 依赖包

见 `requirements.txt`

## 作者

AI-Bio-Frication Research Team

