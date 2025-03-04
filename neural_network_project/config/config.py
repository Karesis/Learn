# 配置类和默认配置
import os
import yaml
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ModelConfig:
    """模型配置"""
    model_name: str = "my_model"
    model_type: str = "custom"  # 或 "huggingface"
    pretrained_model: str = ""  # 如果使用预训练模型
    hidden_size: int = 768
    num_layers: int = 12
    dropout: float = 0.1
    activation: str = "gelu"
    # 添加其他模型参数


@dataclass
class DataConfig:
    """数据配置"""
    dataset_name: str = "my_dataset"
    train_path: str = "data/train.csv"
    val_path: str = "data/val.csv"
    test_path: str = "data/test.csv"
    batch_size: int = 32
    max_seq_length: int = 512
    num_workers: int = 4
    # 添加其他数据参数


@dataclass
class TrainingConfig:
    """训练配置"""
    output_dir: str = "outputs"
    seed: int = 42
    epochs: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    eval_strategy: str = "epoch"  # ["no", "steps", "epoch"]
    save_strategy: str = "epoch"  # ["no", "steps", "epoch"]
    # 添加其他训练参数


@dataclass
class Config:
    """总配置类"""
    project_name: str = "neural_network_project"
    run_name: str = "run_1"
    device: str = "cuda"  # 或 "cpu"
    
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> "Config":
        """从YAML文件加载配置"""
        if not os.path.exists(yaml_file):
            print(f"配置文件 {yaml_file} 不存在，使用默认配置")
            return cls()
        
        with open(yaml_file, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        
        # 更新顶级配置
        for key, value in config_dict.items():
            if key not in ["model", "data", "training"] and hasattr(config, key):
                setattr(config, key, value)
        
        # 更新嵌套配置
        if "model" in config_dict:
            for key, value in config_dict["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)
        
        if "data" in config_dict:
            for key, value in config_dict["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)
        
        if "training" in config_dict:
            for key, value in config_dict["training"].items():
                if hasattr(config.training, key):
                    setattr(config.training, key, value)
        
        return config
    
    def save_yaml(self, yaml_file: str) -> None:
        """保存配置到YAML文件"""
        os.makedirs(os.path.dirname(yaml_file), exist_ok=True)
        
        config_dict = {
            "project_name": self.project_name,
            "run_name": self.run_name,
            "device": self.device,
            "model": {k: v for k, v in vars(self.model).items()},
            "data": {k: v for k, v in vars(self.data).items()},
            "training": {k: v for k, v in vars(self.training).items()},
        }
        
        with open(yaml_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# 创建默认配置示例
def get_default_config() -> Config:
    """获取默认配置"""
    return Config()


# 从配置文件加载或使用默认配置
def load_config(config_path: Optional[str] = None) -> Config:
    """加载配置"""
    if config_path and os.path.exists(config_path):
        return Config.from_yaml(config_path)
    return get_default_config()