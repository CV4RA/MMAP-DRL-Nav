import torch
import torch.nn as nn

class CrossDomainAttention(nn.Module):
    def __init__(self, num_blocks=6):
        super(CrossDomainAttention, self).__init__()
        self.attention_blocks = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=256, num_heads=8) for _ in range(num_blocks)
        ])

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        for block in self.attention_blocks:
            x, _ = block(x, x, x)
        
        return x.mean(dim=0)  # Example output of fused features
