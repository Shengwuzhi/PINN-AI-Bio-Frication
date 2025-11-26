"""
测试数据接口功能
"""

import sys
from pathlib import Path

print("="*60)
print("测试数据接口")
print("="*60)

try:
    from data_interface import AIBioFricationInterface, RealDataInterface
    
    # 测试1: AI-Bio-Frication接口
    print("\n[测试1] AI-Bio-Frication数据接口")
    print("-" * 60)
    try:
        interface = AIBioFricationInterface()
        print("[OK] 接口创建成功")
        
        # 尝试加载数据（只加载前100个样本进行测试）
        print("正在加载数据（这可能需要一些时间）...")
        data = interface.load_data()
        
        print(f"[OK] 数据加载成功")
        print(f"  样本数: {data['molecular_features'].shape[0]}")
        print(f"  特征数: {data['molecular_features'].shape[1]}")
        print(f"  Z参数形状: {data['z_params'].shape}")
        print(f"  描述符形状: {data['descriptors'].shape}")
        print(f"  嵌入形状: {data['molecular_embeddings'].shape}")
        
        if data.get('targets'):
            print(f"  目标值: {list(data['targets'].keys())}")
        else:
            print("  目标值: 无（将使用物理模型生成）")
        
        print(f"  元数据: {data['metadata']}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] 文件未找到: {e}")
        print("  提示: 请确保AI-Bio-Frication项目中有enhanced_lubricant_data.csv文件")
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试2: 真实数据上传接口
    print("\n[测试2] 真实数据上传接口")
    print("-" * 60)
    try:
        upload_interface = RealDataInterface()
        print("[OK] 上传接口创建成功")
        print(f"  上传目录: {upload_interface.upload_path}")
        print("  支持格式: CSV, Excel, JSON")
        print("  提示: 使用 upload_data() 方法上传数据")
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)
    
except ImportError as e:
    print(f"[ERROR] 导入错误: {e}")
    print("  请确保所有依赖包已安装")

