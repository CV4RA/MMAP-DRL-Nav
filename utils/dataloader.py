import torch
from torch.utils.data import Dataset, DataLoader

class CarlaDataset(Dataset):
    def __init__(self, num_samples=1000):
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        image = torch.randn(3, 256, 256)  # Example image data
        lidar_data = torch.randn(1, 256, 256)  # Example LiDAR data
        imu_data = torch.randn(6)  # Example IMU data
        action = torch.randn(2)  # Example action data

        return image, lidar_data, imu_data, action

def get_dataloader(batch_size=32, num_samples=1000):
    dataset = CarlaDataset(num_samples=num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
