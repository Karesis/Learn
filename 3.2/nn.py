import torch
import torch.nn as nn
import torch.nn.functional as F

class VisionBlock(nn.Module):
    """基础视觉模块 - CNN的构建块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)  # 填充保持尺寸不变
        self.batch = nn.BatchNorm2d(out_channels)  # 批归一化提高训练稳定性
        self.pool = nn.MaxPool2d(2)  # 2×2池化，将特征图尺寸减半
        
    def forward(self, x):
        x = self.conv(x)  # 卷积提取特征
        x = self.batch(x)  # 批归一化
        x = F.relu(x)  # 激活函数
        x = self.pool(x)  # 池化降维
        return x

class VisionExpert(nn.Module):
    """视觉专家 - 负责提取图像特征"""
    def __init__(self, name, in_channels=1):
        super().__init__()
        self.name = name
        # 专家的特征提取层
        self.block1 = VisionBlock(in_channels, 32)  # 输入: [batch, 1, 28, 28] -> 输出: [batch, 32, 14, 14]
        self.block2 = VisionBlock(32, 64)  # 输入: [batch, 32, 14, 14] -> 输出: [batch, 64, 7, 7]
        self.block3 = VisionBlock(64, 128)  # 输入: [batch, 64, 7, 7] -> 输出: [batch, 128, 3, 3]
        # 计算展平后的特征尺寸: 128 * 3 * 3 = 1152
        self.feature_size = 128 * 3 * 3
        # 特征变换层
        self.fc = nn.Linear(self.feature_size, 256)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # 特征提取
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        # 添加这一行：展平操作 - 将4D张量 [batch, 128, 3, 3] 转为 2D张量 [batch, 1152]
        x = x.view(x.size(0), -1)
        # 特征变换: [batch, 256]
        x = F.relu(self.fc(x))
        x = self.dropout(x)  # 防止过拟合
        return x  # 返回专家特征向量 [batch, 256]

class MultiFashionExpertWithLSTM(nn.Module):
    """使用LSTM整合专家意见的多专家时尚服装分类模型"""
    def __init__(self, num_classes=10, in_channels=1, num_experts=3):
        super().__init__()
        # 创建多个专家
        self.experts = nn.ModuleList([
            VisionExpert(f"专家{i+1}", in_channels)
            for i in range(num_experts)
        ])
        
        # LSTM层用于处理专家意见序列
        self.lstm = nn.LSTM(256, 128, batch_first=True)
        
        # 特征聚合层 - 注意输入维度是128，因为LSTM输出是128维
        self.fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 最终分类器
        self.classifier = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # 收集每个专家的意见
        expert_features = []
        for expert in self.experts:
            features = expert(x)
            expert_features.append(features)
        
        # 将专家意见堆叠为序列 [batch, num_experts, 256]
        expert_sequence = torch.stack(expert_features, dim=1)
        
        # 使用LSTM处理专家意见序列
        lstm_out, _ = self.lstm(expert_sequence)
        # 取最后一个时间步的输出 [batch, 128]
        lstm_features = lstm_out[:, -1, :]
        
        # 特征融合
        aggregated = self.fusion(lstm_features)
        
        # 最终分类
        logits = self.classifier(aggregated)
        return logits