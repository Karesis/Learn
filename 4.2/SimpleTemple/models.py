import torch
import torch.nn as nn
import torch.nn.functional as F


class VisionBlock(nn.Module):
    """Basic vision block with convolution, batch norm, and pooling"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.batch = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.batch(x)
        x = F.gelu(x)
        x = self.pool(x)
        return x


class VisionExpert(nn.Module):
    """Expert for extracting features from images"""
    
    def __init__(self, name, in_channels=3):
        super().__init__()
        self.name = name
        
        # Feature extraction blocks
        self.block1 = VisionBlock(in_channels, 32)  # [batch, 3, 32, 32] -> [batch, 32, 16, 16]
        self.block2 = VisionBlock(32, 64)           # [batch, 32, 16, 16] -> [batch, 64, 8, 8]
        self.block3 = VisionBlock(64, 128)          # [batch, 64, 8, 8] -> [batch, 128, 4, 4]
        
        # Feature size after flattening
        self.feature_size = 128 * 4 * 4
        
        # Feature transformation
        self.fc = nn.Linear(self.feature_size, 256)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Feature extraction
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Flatten features
        x = x.view(x.size(0), -1)
        
        # Feature transformation
        x = F.gelu(self.fc(x))
        x = self.dropout(x)
        return x

class MultiFashionExpertWithLSTM(nn.Module):
    """Multi-expert model with LSTM for combining expert opinions"""
    
    def __init__(self, num_classes=10, in_channels=1, num_experts=5):
        super().__init__()
        
        # Create multiple experts
        self.experts = nn.ModuleList([
            VisionExpert(f"Expert{i+1}", in_channels)
            for i in range(num_experts)
        ])
        
        # LSTM for processing expert sequence
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Final classifier
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # Collect opinions from each expert
        expert_features = []
        for expert in self.experts:
            features = expert(x)
            expert_features.append(features)
        
        # Stack expert opinions as sequence [batch, num_experts, 256]
        expert_sequence = torch.stack(expert_features, dim=1)
        
        # Process with LSTM
        lstm_out, _ = self.lstm(expert_sequence)
        
        # Take the last time step output [batch, 128]
        lstm_features = lstm_out[:, -1, :]
        
        # Feature fusion
        aggregated = self.fusion(lstm_features)
        
        # Final classification
        logits = self.classifier(aggregated)
        
        return logits
