# 文件快速参考
# File Quick Reference

## 您要找的文件位置

### 1. FORMULATION_OPTIMIZER_SUMMARY.md
**位置**: `C:\Users\weizh\PINN-AI-Bio-Frication\FORMULATION_OPTIMIZER_SUMMARY.md`
**状态**: ✅ 已创建
**内容**: 配方优化器的完整总结文档

### 2. pareto_front_3d.png
**位置**: `C:\Users\weizh\PINN-AI-Bio-Frication\pareto_front_3d.png`
**状态**: ✅ 已生成（677 KB）
**生成时间**: 2025/11/26 11:20:11
**生成脚本**: `pareto_optimization.py`
**内容**: 3D帕累托前沿图（摩擦系数、燃油经济性、RUL）

### 3. pareto_tradeoff.png
**位置**: `C:\Users\weizh\PINN-AI-Bio-Frication\pareto_tradeoff.png`
**状态**: ✅ 已生成（452 KB）
**生成时间**: 2025/11/26 11:20:12
**生成脚本**: `pareto_optimization.py`
**内容**: 权衡关系分析图（4个子图）

### 4. pareto_optimal_solutions.csv
**位置**: `C:\Users\weizh\PINN-AI-Bio-Frication\pareto_optimal_solutions.csv`
**状态**: ✅ 已生成（402 字节）
**生成时间**: 2025/11/26 11:20:12
**生成脚本**: `pareto_optimization.py`
**内容**: TOP-N帕累托最优解数据

## 相关文件说明

### 配方优化相关

| 文件名 | 说明 | 生成脚本 |
|--------|------|----------|
| `formulation_optimizer.py` | 配方优化器主程序 | - |
| `optimized_formulations.csv` | 优化配方数据 | `formulation_optimizer.py` |
| `formulation_radar_chart.png` | 雷达图（Top N vs Baseline） | `formulation_optimizer.py` |
| `pareto_front_3d_formulation.png` | 3D帕累托前沿（配方优化） | `formulation_optimizer.py` |

### 多目标优化相关

| 文件名 | 说明 | 生成脚本 |
|--------|------|----------|
| `pareto_optimization.py` | 多目标优化主程序 | - |
| `pareto_optimal_solutions.csv` | 帕累托最优解数据 | `pareto_optimization.py` |
| `pareto_front_3d.png` | 3D帕累托前沿图 | `pareto_optimization.py` |
| `pareto_tradeoff.png` | 权衡关系分析图 | `pareto_optimization.py` |

## 如何重新生成文件

### 生成pareto_optimization.py的文件

```bash
cd C:\Users\weizh\PINN-AI-Bio-Frication
python pareto_optimization.py --n_samples 500 --top_n 10
```

**生成的文件**:
- `pareto_front_3d.png`
- `pareto_tradeoff.png`
- `pareto_optimal_solutions.csv`

### 生成formulation_optimizer.py的文件

```bash
cd C:\Users\weizh\PINN-AI-Bio-Frication
python formulation_optimizer.py --top_n 3
```

**生成的文件**:
- `optimized_formulations.csv`
- `formulation_radar_chart.png`
- `pareto_front_3d_formulation.png`

## 文件内容预览

### pareto_optimal_solutions.csv

包含列：
- Index: 样本索引
- Score: 综合得分
- Friction_Coefficient: 摩擦系数
- Fuel_Economy: 燃油经济性
- RUL_Final: 最终RUL
- Viscosity_40C: 40°C粘度
- VFT_A, VFT_B, VFT_C: VFT参数

### optimized_formulations.csv

包含列：
- Index: 样本索引
- VFT_A, VFT_B, VFT_C: VFT参数
- Viscosity_40C, Viscosity_100C, Viscosity_minus20C: 不同温度下的粘度
- Viscosity_Index: 粘度指数
- Fuel_Economy: 燃油经济性
- Cost_USD_kg: 成本
- Sustainability_Score: 可持续性评分
- Low_Temp_Pass: 是否通过低温测试

## 文件验证

所有文件已确认存在：

```
✅ FORMULATION_OPTIMIZER_SUMMARY.md
✅ pareto_front_3d.png (677 KB)
✅ pareto_tradeoff.png (452 KB)
✅ pareto_optimal_solutions.csv (402 bytes)
```

## 如果文件丢失

如果文件丢失，可以运行以下命令重新生成：

```bash
# 生成pareto_optimization.py的文件
python pareto_optimization.py

# 生成formulation_optimizer.py的文件
python formulation_optimizer.py
```

---

**最后更新**: 2025-11-26 11:20

