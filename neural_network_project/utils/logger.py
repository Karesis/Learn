# 日志工具
import os
import logging
import random
import numpy as np
import torch
from typing import Optional


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    获取日志器
    
    Args:
        name: 日志器名称
        level: 日志级别
        
    Returns:
        日志器
    """
    logger = logging.getLogger(name)
    
    # 避免重复设置处理器
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def set_seed(seed: int) -> None:
    """
    设置随机种子
    
    Args:
        seed: 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    设置日志
    
    Args:
        log_dir: 日志目录
        level: 日志级别
    """
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除所有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # 格式化器
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join(log_dir, "log.txt"), encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)