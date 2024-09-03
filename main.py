import torch
from torch.optim import Adam
from utils.dataloader import get_dataloader
from models.perception_module import PerceptionModule
from models.attention_module import CrossDomainAttention
from models.decision_module import DecisionModule

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

def train_model(model, dataloader, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (image, lidar_data, imu_data, target_action) in enumerate(dataloader):
            image, lidar_data, imu_data, target_action = image.to(device), lidar_data.to(device), imu_data.to(device), target_action.to(device)
            optimizer.zero_grad()
            policy_output, value_output = model(image, lidar_data, imu_data)
            loss = nn.MSELoss()(policy_output, target_action) + nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(dataloader)}], Loss: {running_loss / 10:.4f}')
                running_loss = 0.0
    print('Training complete')

def test_model(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for image, lidar_data, imu_data, target_action in dataloader:
            image, lidar_data, imu_data, target_action = image.to(device), lidar_data.to(device), imu_data.to(device), target_action.to(device)
            policy_output, value_output = model(image, lidar_data, imu_data)
            loss = nn.MSELoss()(policy_output, target_action) + nn.MSELoss()(value_output, target_action.sum(dim=1, keepdim=True))
            total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    print(f'Test
