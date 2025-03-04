# 具体模型实现
import os
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Tuple, Any
from abc import ABC, abstractmethod

from ..config.config import ModelConfig


class BaseModel(nn.Module, ABC):
    """基础模型抽象类"""
    
    def __init__(self, config: ModelConfig):
        """
        初始化基础模型
        
        Args:
            config: 模型配置
        """
        super().__init__()
        self.config = config
    
    @abstractmethod
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            **inputs: 模型输入
            
        Returns:
            字典，包含损失、logits等输出
        """
        pass
    
    def save_pretrained(self, save_dir: str) -> None:
        """
        保存模型到指定目录
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存模型权重
        model_path = os.path.join(save_dir, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # 保存模型配置
        config_path = os.path.join(save_dir, "config.json")
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(vars(self.config), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs) -> "BaseModel":
        """
        从预训练目录加载模型
        
        Args:
            model_dir: 模型目录
            **kwargs: 额外参数，可覆盖配置
            
        Returns:
            加载的模型
        """
        # 加载配置
        config_path = os.path.join(model_dir, "config.json")
        import json
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # 创建配置
        config = ModelConfig(**config_dict)
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建模型
        model = cls(config)
        
        # 加载权重
        model_path = os.path.join(model_dir, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        
        return model


class HuggingFaceModelWrapper(BaseModel):
    """Hugging Face 模型包装器"""
    
    def __init__(self, config: ModelConfig):
        """
        初始化HuggingFace模型包装器
        
        Args:
            config: 模型配置
        """
        super().__init__(config)
        
        # 导入相关库
        from transformers import AutoConfig, AutoModelForSequenceClassification
        
        # 创建或加载模型
        if config.pretrained_model:
            # 从预训练模型加载
            self.model = AutoModelForSequenceClassification.from_pretrained(
                config.pretrained_model
            )
        else:
            # 创建新模型
            hf_config = AutoConfig.from_pretrained(
                config.model_type,
                hidden_size=config.hidden_size,
                num_hidden_layers=config.num_layers,
                hidden_dropout_prob=config.dropout
            )
            self.model = AutoModelForSequenceClassification.from_config(hf_config)
    
    def forward(self, **inputs) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            **inputs: 模型输入，如input_ids, attention_mask, labels等
            
        Returns:
            字典，包含损失、logits等输出
        """
        outputs = self.model(**inputs)
        
        result = {
            "logits": outputs.logits
        }
        
        if "labels" in inputs:
            result["loss"] = outputs.loss
            
        return result
    
    def save_pretrained(self, save_dir: str) -> None:
        """
        保存模型到指定目录
        
        Args:
            save_dir: 保存目录
        """
        self.model.save_pretrained(save_dir)
    
    @classmethod
    def from_pretrained(cls, model_dir: str, **kwargs) -> "HuggingFaceModelWrapper":
        """
        从预训练目录加载模型
        
        Args:
            model_dir: 模型目录
            **kwargs: 额外参数
            
        Returns:
            加载的模型
        """
        from transformers import AutoConfig, AutoModelForSequenceClassification
        
        # 加载Hugging Face配置
        hf_config = AutoConfig.from_pretrained(model_dir)
        
        # 创建我们的配置
        config = ModelConfig(
            model_type=hf_config.model_type,
            hidden_size=hf_config.hidden_size,
            num_layers=getattr(hf_config, "num_hidden_layers", 12),
            dropout=getattr(hf_config, "hidden_dropout_prob", 0.1)
        )
        
        # 更新配置
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # 创建模型实例
        model_wrapper = cls(config)
        
        # 加载Hugging Face模型
        model_wrapper.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        return model_wrapper