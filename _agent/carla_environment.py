import carla
import numpy as np
import gym

class CarlaEnvironment(gym.Env):
    def __init__(self):
        super(CarlaEnvironment, self).__init__()
        self.client = carla.Client('localhost', 2000)  # 替换为您的IP和端口
        self.world = self.client.get_world()
        self.action_space = gym.spaces.Discrete(4)  # 4个动作:前进、左转、右转、后退）
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)  # 使用图像作为状态

        # 初始化车辆
        self.vehicle = None

    def reset(self):
        if self.vehicle is not None:
            self.vehicle.destroy()  # 清理上一个车辆实例
        blueprint = self.world.get_blueprint_library().filter('vehicle.*')[0]
        spawn_point = carla.Transform(carla.Location(x=0, y=0, z=0))  # 替换为您的起始位置
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)
        self.world.tick()

        return self.get_observation()

    def get_observation(self):
        # 获取车辆周围的图像
        image = np.zeros((128, 128, 3), dtype=np.uint8)  # 替换为实际图像获取
        return image

    def step(self, action):
        # 执行动作
        if action == 0:  # 向前
            self.vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
        elif action == 1:  # 左转
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=-1.0))
        elif action == 2:  # 右转
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=1.0))
        elif action == 3:  # 后退
            self.vehicle.apply_control(carla.VehicleControl(throttle=-1.0, steer=0.0))

        self.world.tick()
        next_state = self.get_observation()
        reward = 1  # 实际奖励计算
        done = False  # 设定完成标志

        return next_state, reward, done, {}

    def close(self):
        if self.vehicle is not None:
            self.vehicle.destroy()
