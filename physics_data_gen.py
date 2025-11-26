"""
Physics-Based Synthetic Dataset Generator for Bio-based Lubricant System
生成基于物理定律的生物基润滑剂系统合成数据集
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# 设置随机种子以确保可重复性
np.random.seed(42)

# 物理常数
R = 8.314  # 气体常数 (J/(mol·K))

# ============================================================================
# L1: 分子输入 (Molecular Inputs)
# ============================================================================

def generate_molecular_inputs(n_samples=1000):
    """
    生成分子输入数据
    - Z参数: chain_length, symmetry, polarity
    - 分子嵌入: 64维向量
    """
    # Z参数生成
    chain_length = np.random.uniform(10, 50, n_samples)  # 链长
    symmetry = np.random.uniform(0, 1, n_samples)  # 对称性
    polarity = np.random.uniform(0, 1, n_samples)  # 极性
    
    # 分子嵌入 (64维向量)
    molecular_embeddings = np.random.normal(0, 1, (n_samples, 64))
    
    return {
        'chain_length': chain_length,
        'symmetry': symmetry,
        'polarity': polarity,
        'molecular_embeddings': molecular_embeddings
    }

# ============================================================================
# L2: 基于物理的标签生成 (Physics-Based Label Generation)
# ============================================================================

def generate_vft_parameters(chain_length, symmetry, polarity):
    """
    基于Z参数生成VFT参数 (A, B, C)
    VFT方程: ln(eta) = A + B / (T - C)
    
    物理关系:
    - 链长越长 -> A值越高 (更高的基础粘度)
    - 对称性影响B值
    - 极性影响C值 (玻璃化转变温度相关)
    """
    # A参数: 与链长正相关
    A = 2.0 + 0.5 * (chain_length / 50.0) + np.random.normal(0, 0.1, len(chain_length))
    
    # B参数: 与对称性和链长相关
    B = 500 + 200 * (1 - symmetry) + 100 * (chain_length / 50.0) + np.random.normal(0, 20, len(chain_length))
    
    # C参数: 与极性相关 (玻璃化转变温度)
    # 确保C值不会太大，避免在低温下出现问题
    # C值应该远小于工作温度（如40°C和-20°C），所以设置为-80到-30之间
    # 这样-20°C > C+10，40°C > C+10，100°C > C+10
    C = -50 - 20 * polarity + np.random.normal(0, 5, len(chain_length))
    C = np.clip(C, -80, -30)  # 限制C值范围，确保所有工作温度 > C+10
    
    return A, B, C

def calculate_viscosity_vft(A, B, C, temperature):
    """
    使用VFT方程计算粘度
    ln(eta) = A + B / (T - C)
    eta = exp(A + B / (T - C))
    """
    # 确保温度大于C (避免除零和负值)
    # 对于每个样本，确保温度大于其对应的C+10
    if isinstance(temperature, (int, float)):
        T_array = np.full_like(C, temperature)
    else:
        T_array = np.asarray(temperature)
    
    # 对于每个样本，如果温度小于C+10，则使用C+10
    # 但这样会导致所有样本的粘度相同，所以我们应该确保C值不会太大
    # 或者使用更合理的温度范围
    T = np.maximum(T_array, C + 10)
    
    # 计算ln(eta)
    ln_eta = A + B / (T - C)
    
    # 限制ln_eta的值以避免溢出 (粘度范围: 0.1 - 1e6 mPa·s)
    ln_eta = np.clip(ln_eta, -2.3, 13.8)
    eta = np.exp(ln_eta)  # 单位: mPa·s
    return eta

def generate_adsorption_energy(polarity):
    """
    基于极性生成吸附能 (Delta_G)
    极性越高 -> 吸附能越低 (更容易吸附)
    """
    # Delta_G单位为 kJ/mol
    # 极性越高，Delta_G越负 (更稳定)
    Delta_G = -20 - 30 * polarity + np.random.normal(0, 2, len(polarity))
    return Delta_G

def calculate_friction_langmuir(Delta_G, temperature, concentration=1.0):
    """
    使用Langmuir等温线计算摩擦系数
    theta = (K*C) / (1 + K*C)
    其中 K = exp(-Delta_G / (R*T))
    
    theta: 表面覆盖率
    摩擦系数与覆盖率相关: mu = mu_0 * (1 - theta) + mu_1 * theta
    """
    # 计算平衡常数
    K = np.exp(-Delta_G * 1000 / (R * temperature))  # Delta_G转换为J/mol
    
    # 计算表面覆盖率
    theta = (K * concentration) / (1 + K * concentration)
    
    # 计算摩擦系数
    # mu_0: 无润滑剂时的摩擦系数
    # mu_1: 完全覆盖时的摩擦系数
    mu_0 = 0.3
    mu_1 = 0.05
    mu = mu_0 * (1 - theta) + mu_1 * theta
    
    return mu, theta

# ============================================================================
# L2.5: 台架测试数据 (Bench Test Data)
# ============================================================================

def generate_engine_conditions(n_samples):
    """
    生成发动机条件
    """
    speed = np.random.uniform(1000, 5000, n_samples)  # RPM
    load = np.random.uniform(100, 1000, n_samples)  # N
    return speed, load

def calculate_stribeck_number(viscosity, speed, load):
    """
    计算Stribeck数
    Stribeck = (Viscosity * Speed) / (Load / Area)
    其中Area是接触面积，这里假设为常数
    """
    # 将速度从RPM转换为m/s (假设曲轴半径为0.1m)
    speed_mps = speed * 2 * np.pi * 0.1 / 60  # m/s
    # 粘度单位转换为Pa·s
    viscosity_pas = viscosity * 1e-3  # mPa·s -> Pa·s
    
    # 假设接触面积为1e-4 m^2 (1 cm^2)
    contact_area = 1e-4  # m^2
    pressure = load / contact_area  # Pa (压力 = 载荷/面积)
    
    # Stribeck数 = (粘度 * 速度) / 压力
    stribeck = (viscosity_pas * speed_mps) / pressure
    return stribeck

def calculate_fuel_economy_score(stribeck):
    """
    基于Stribeck数计算燃油经济性评分
    最优值在混合润滑区域 (Stribeck数适中)
    """
    # 使用对数尺度的Stribeck数 (更符合实际物理情况)
    log_stribeck = np.log10(np.maximum(stribeck, 1e-6))
    
    # 最优对数Stribeck数范围 (混合润滑区域)
    # 根据实际数据范围，调整最优值到合理位置
    optimal_log_stribeck = -2.2  # 对应Stribeck数约0.0063
    
    # 使用高斯型函数，在最优值附近得分最高
    # 标准差设为0.8，使得在较宽范围内得分较高
    exponent = ((log_stribeck - optimal_log_stribeck) / 0.8) ** 2
    exponent = np.clip(exponent, 0, 10)  # 限制指数范围
    score = 100 * np.exp(-exponent)
    
    # 确保分数在合理范围内
    score = np.clip(score, 0, 100)
    
    return score

# ============================================================================
# L3: 运行数据 (Operational Data)
# ============================================================================

def generate_time_series_data(n_samples, n_steps=100):
    """
    生成时间序列数据
    - 温度
    - 振动
    """
    time_series = []
    
    for i in range(n_samples):
        # 温度时间序列 (基础温度 + 波动)
        base_temp = np.random.uniform(80, 120)  # 基础温度 (°C)
        temp_fluctuation = np.random.normal(0, 5, n_steps)
        temperature = base_temp + temp_fluctuation
        temperature = np.cumsum(temperature - base_temp) * 0.1 + base_temp  # 添加趋势
        
        # 振动时间序列
        vibration = np.random.normal(0, 1, n_steps)
        vibration = np.abs(vibration)  # 振动幅度
        
        time_series.append({
            'temperature': temperature,
            'vibration': vibration
        })
    
    return time_series

def simulate_degradation_arrhenius(temperature_series, initial_health=1.0, 
                                   A_ox=1e10, Ea=80000, threshold=0.5):
    """
    使用Arrhenius动力学模拟退化
    d(Health)/dt = -A_ox * exp(-Ea / (R * T))
    
    参数:
    - A_ox: 预指数因子
    - Ea: 活化能 (J/mol)
    - threshold: 健康度阈值
    """
    n_steps = len(temperature_series)
    health = np.zeros(n_steps)
    health[0] = initial_health
    
    dt = 1.0  # 时间步长 (小时)
    
    for t in range(1, n_steps):
        T_kelvin = temperature_series[t] + 273.15  # 转换为开尔文
        degradation_rate = A_ox * np.exp(-Ea / (R * T_kelvin))
        health[t] = health[t-1] - degradation_rate * dt
        health[t] = np.maximum(health[t], 0)  # 健康度不能为负
    
    # 计算RUL (剩余使用寿命)
    # RUL是健康度降到阈值以下的时间步
    below_threshold = np.where(health < threshold)[0]
    if len(below_threshold) > 0:
        rul = below_threshold[0]
    else:
        rul = n_steps  # 如果整个周期内都未降到阈值以下
    
    return health, rul

# ============================================================================
# 主函数: 生成完整数据集
# ============================================================================

def generate_complete_dataset(n_samples=1000, n_steps=100):
    """
    生成完整的数据集
    """
    print(f"生成 {n_samples} 个样本的数据集...")
    
    # L1: 分子输入
    print("生成L1: 分子输入...")
    mol_data = generate_molecular_inputs(n_samples)
    
    # L2: 基于物理的标签
    print("生成L2: 基于物理的标签...")
    A, B, C = generate_vft_parameters(
        mol_data['chain_length'],
        mol_data['symmetry'],
        mol_data['polarity']
    )
    
    # 计算不同温度下的粘度
    temp_40 = 40.0
    temp_100 = 100.0
    temp_minus20 = -20.0
    
    viscosity_40 = calculate_viscosity_vft(A, B, C, temp_40)
    viscosity_100 = calculate_viscosity_vft(A, B, C, temp_100)
    viscosity_minus20 = calculate_viscosity_vft(A, B, C, temp_minus20)
    
    # 生成吸附能和摩擦系数
    Delta_G = generate_adsorption_energy(mol_data['polarity'])
    friction_coeff, surface_coverage = calculate_friction_langmuir(
        Delta_G, temp_40, concentration=1.0
    )
    
    # L2.5: 台架测试数据
    print("生成L2.5: 台架测试数据...")
    speed, load = generate_engine_conditions(n_samples)
    stribeck = calculate_stribeck_number(viscosity_40, speed, load)
    fuel_economy = calculate_fuel_economy_score(stribeck)
    
    # L3: 运行数据
    print("生成L3: 运行数据...")
    time_series_data = generate_time_series_data(n_samples, n_steps)
    
    # 为每个样本计算退化和RUL
    health_series = []
    rul_values = []
    
    for i in range(n_samples):
        health, rul = simulate_degradation_arrhenius(
            time_series_data[i]['temperature'],
            initial_health=1.0,
            A_ox=1e10,
            Ea=80000,
            threshold=0.5
        )
        health_series.append(health)
        rul_values.append(rul)
    
    # 组装静态数据 (L1/L2/L2.5)
    static_data = pd.DataFrame({
        # L1: 分子输入
        'chain_length': mol_data['chain_length'],
        'symmetry': mol_data['symmetry'],
        'polarity': mol_data['polarity'],
        
        # L2: VFT参数和粘度
        'VFT_A': A,
        'VFT_B': B,
        'VFT_C': C,
        'viscosity_40C': viscosity_40,
        'viscosity_100C': viscosity_100,
        'viscosity_minus20C': viscosity_minus20,
        
        # L2: 吸附能和摩擦
        'Delta_G': Delta_G,
        'friction_coefficient': friction_coeff,
        'surface_coverage': surface_coverage,
        
        # L2.5: 台架测试
        'engine_speed_RPM': speed,
        'engine_load_N': load,
        'stribeck_number': stribeck,
        'fuel_economy_score': fuel_economy,
        
        # L3: RUL
        'RUL': rul_values
    })
    
    # 组装时间序列数据 (L3)
    temperature_matrix = np.array([ts['temperature'] for ts in time_series_data])
    vibration_matrix = np.array([ts['vibration'] for ts in time_series_data])
    health_matrix = np.array(health_series)
    
    return static_data, temperature_matrix, vibration_matrix, health_matrix

# ============================================================================
# 可视化函数
# ============================================================================

def plot_viscosity_curves(static_data, n_samples_to_plot=5):
    """
    绘制粘度-温度曲线样本
    """
    temperatures = np.linspace(-20, 150, 100)
    
    plt.figure(figsize=(12, 8))
    
    # 随机选择几个样本进行绘制
    sample_indices = np.random.choice(len(static_data), n_samples_to_plot, replace=False)
    
    for idx in sample_indices:
        A = static_data.iloc[idx]['VFT_A']
        B = static_data.iloc[idx]['VFT_B']
        C = static_data.iloc[idx]['VFT_C']
        
        # 计算粘度曲线
        viscosities = calculate_viscosity_vft(A, B, C, temperatures)
        
        plt.semilogy(temperatures, viscosities, 
                    label=f'Sample {idx}: A={A:.2f}, B={B:.1f}, C={C:.1f}')
    
    plt.xlabel('Temperature (°C)', fontsize=12)
    plt.ylabel('Viscosity (mPa·s)', fontsize=12)
    plt.title('Viscosity-Temperature Curves (VFT Model)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('viscosity_temperature_curves.png', dpi=300, bbox_inches='tight')
    print("粘度-温度曲线已保存为 'viscosity_temperature_curves.png'")
    plt.close()

# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    # 生成数据集
    static_data, temp_matrix, vib_matrix, health_matrix = generate_complete_dataset(
        n_samples=1000,
        n_steps=100
    )
    
    # 保存静态数据
    static_data.to_csv('physics_lubricant_data.csv', index=False)
    print(f"\n静态数据已保存: physics_lubricant_data.csv")
    print(f"  样本数: {len(static_data)}")
    print(f"  特征数: {len(static_data.columns)}")
    
    # 保存时间序列数据
    np.savez('physics_sensor_data.npz',
             temperature=temp_matrix,
             vibration=vib_matrix,
             health=health_matrix)
    print(f"\n时间序列数据已保存: physics_sensor_data.npz")
    print(f"  温度数据形状: {temp_matrix.shape}")
    print(f"  振动数据形状: {vib_matrix.shape}")
    print(f"  健康度数据形状: {health_matrix.shape}")
    
    # 绘制粘度-温度曲线
    print("\n绘制粘度-温度曲线...")
    plot_viscosity_curves(static_data, n_samples_to_plot=5)
    
    # 显示数据统计信息
    print("\n=== 数据统计信息 ===")
    print(static_data.describe())
    
    print("\n数据集生成完成！")

