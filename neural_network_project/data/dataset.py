# 数据集类
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple, Any
from transformers import PreTrainedTokenizer

from ..config.config import DataConfig


class BaseDataset(Dataset):
    """基础数据集类"""
    
    def __init__(self, 
                 file_path: str, 
                 tokenizer: Optional[PreTrainedTokenizer] = None,
                 max_seq_length: int = 512,
                 **kwargs):
        """
        初始化数据集
        
        Args:
            file_path: 数据文件路径
            tokenizer: HuggingFace分词器(如果需要)
            max_seq_length: 最大序列长度
            **kwargs: 额外参数
        """
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # 读取数据
        self.data = self._load_data(file_path)
        
        # 预处理数据
        self.processed_data = self._preprocess_data(self.data)
    
    def _load_data(self, file_path: str) -> List[Dict]:
        """
        加载数据文件
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据列表
        """
        # 根据文件类型选择加载方式
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.csv':
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif ext == '.json':
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return [{'text': line.strip()} for line in f]
        else:
            raise ValueError(f"不支持的文件类型: {ext}")
    
    def _preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """
        预处理数据
        
        Args:
            data: 原始数据列表
            
        Returns:
            预处理后的数据列表
        """
        # 默认不做任何处理，由子类实现具体预处理逻辑
        return data
    
    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            处理后的样本字典，包含模型所需的所有输入
        """
        # 由子类实现具体的样本处理逻辑
        raise NotImplementedError("由子类实现")


class TextClassificationDataset(BaseDataset):
    """文本分类数据集示例"""
    
    def __init__(self, 
                 file_path: str, 
                 tokenizer: PreTrainedTokenizer,
                 max_seq_length: int = 512,
                 label_map: Optional[Dict[str, int]] = None,
                 text_col: str = "text",
                 label_col: str = "label",
                 **kwargs):
        """
        初始化文本分类数据集
        
        Args:
            file_path: 数据文件路径
            tokenizer: HuggingFace分词器
            max_seq_length: 最大序列长度
            label_map: 标签映射字典，将文本标签映射到整数
            text_col: 文本列名
            label_col: 标签列名
            **kwargs: 额外参数
        """
        self.text_col = text_col
        self.label_col = label_col
        self.label_map = label_map or {}
        
        super().__init__(file_path, tokenizer, max_seq_length, **kwargs)
        
        # 如果没有提供label_map，自动生成
        if not self.label_map and self.label_col in self.data[0]:
            unique_labels = sorted(set(item[self.label_col] for item in self.data))
            self.label_map = {label: i for i, label in enumerate(unique_labels)}
    
    def _preprocess_data(self, data: List[Dict]) -> List[Dict]:
        """预处理分类数据"""
        processed = []
        
        for item in data:
            if self.text_col not in item:
                continue
                
            text = item[self.text_col]
            
            # 处理标签
            label = -1  # 默认值，用于推理时没有标签的情况
            if self.label_col in item:
                label_text = item[self.label_col]
                label = self.label_map.get(label_text, -1)
            
            processed.append({
                "text": text,
                "label": label,
                "original": item  # 保留原始数据
            })
            
        return processed
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """获取一个分类样本"""
        item = self.processed_data[idx]
        text = item["text"]
        label = item["label"]
        
        # 使用tokenizer处理文本
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 去掉批次维度
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        
        # 添加标签
        if label != -1:
            encoding["labels"] = torch.tensor(label, dtype=torch.long)
            
        return encoding


def create_dataloader(dataset: Dataset, 
                     batch_size: int = 32, 
                     shuffle: bool = True, 
                     num_workers: int = 4) -> DataLoader:
    """
    创建DataLoader
    
    Args:
        dataset: 数据集
        batch_size: 批次大小
        shuffle: 是否打乱数据
        num_workers: 工作进程数
        
    Returns:
        数据加载器
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )