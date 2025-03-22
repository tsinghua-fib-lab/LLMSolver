import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(r'/data1/shy/zgc/llm_solver/routefinder')

from routefinder.envs.mtvrp import MTVRPEnv, MTVRPGenerator

# 变体列表
# plist = ['cvrp', 'cvrpl', 'cvrptw', 'ovrp', 'vrpb']
plist = ['CVRP', 'CVRPL', 'CVRPTW', 'OVRP', 'VRPB']  # Example problem types

# 生成100个实例的函数
def generate_vrp_instances(plist, num_instances=100, save_dir='./td_new'):
    """Generate instances for each problem type in the plist."""
    
    # 遍历每个变体类型
    for problem_type in plist:
        # 设置变体生成预设
        variant_preset = problem_type.lower()
        
        # 初始化MTVRP生成器
        generator = MTVRPGenerator(
            variant_preset=variant_preset,  # 使用变体预设
            num_loc=20,  # 假设我们每个实例有20个地点
        )
        
        # 生成100个实例并保存
        td = generator._generate(batch_size=(num_instances,))
        save_path = f"{save_dir}/{problem_type}.npz"
        generator.save_data(td, save_path)
        print(f"Saved {problem_type} instance to {save_path}")

# 调用函数生成并保存实例
generate_vrp_instances(plist)
