import torch
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule
from models.dqn_agent import DQNAgent
from envs.carla_environment import CarlaEnvironment
import carla

class IntegratedSystem:
    def __init__(self, device='cpu'):
        self.device = device
        self.perception = PerceptionModule().to(self.device)
        self.attention = CrossDomainAttention(num_blocks=6).to(self.device)
        self.decision = DecisionModule().to(self.device)

    def forward(self, image, lidar_data, imu_data):
        scene_info, segmentation, odometry, obstacles, boundary = self.perception(imu_data, image, lidar_data)
        fused_features = self.attention(scene_info, segmentation, odometry, obstacles, boundary)
        policy, value = self.decision(fused_features)
        return policy, value

def run_simulation():
    env = CarlaEnv()  # 开始处理 CARLA 仿真
    system = IntegratedSystem(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    for _ in range(100):  # 运行100个模拟步骤
        image = torch.randn(3, 256, 256).unsqueeze(0).to(system.device)  
        lidar_data = torch.randn(1, 256, 256).unsqueeze(0).to(system.device)
        imu_data = torch.randn(1, 6).to(system.device)

        policy, value = system.forward(image, lidar_data, imu_data)
        # 将策略转换为CARLA控制信号
        control = carla.VehicleControl(throttle=float(policy[0][0]), steer=float(policy[0][1]))
        env.vehicle.apply_control(control)

        # Simulate some delay between steps
        time.sleep(0.1)

if __name__ == "__main__":
    run_simulation()
