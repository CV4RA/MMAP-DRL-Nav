import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.dqn_agent import DQNAgent
from models.pruning import ModelPruner
from models.quantization import quantize_model
from envs.carla_environment import CarlaEnvironment 
import yaml

def load_config(config_path='configs/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def train_model(config):
    # 初始化环境
    env = CarlaEnvironment()

    # 初始化 DQN 代理
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size=state_size, action_size=action_size, config=config)

    # 优化器和损失函数
    optimizer = optim.Adam(agent.model.parameters(), lr=config['train']['learning_rate'])
    criterion = nn.MSELoss()

    episodes = config['train']['episodes']
    for e in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # 记忆
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            # 经验回放
            if len(agent.memory) > config['train']['batch_size']:
                agent.replay(config['train']['batch_size'])

        print(f"Episode {e+1}/{episodes}, Total Reward: {total_reward}")

    # 剪枝和量化
    pruner = ModelPruner(agent.model)
    pruner.prune_model(amount=0.2)  # 剪枝
    agent.model = quantize_model(agent.model)  # 量化

    # 导出模型为 ONNX 格式
    export_to_onnx(agent.model)

def export_to_onnx(model, file_path='model.onnx'):
    dummy_input = torch.randn(1, 128)  # 根据模型的输入维度调整
    torch.onnx.export(model, dummy_input, file_path, opset_version=11)

if __name__ == "__main__":
    config = load_config()
    train_model(config)
