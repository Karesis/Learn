# 开发入口(训练和评估)
import os
import argparse
from typing import Dict, Optional

import torch
from transformers import AutoTokenizer

from neural_network_project.config.config import Config, load_config
from neural_network_project.data.dataset import TextClassificationDataset, create_dataloader
from neural_network_project.models.my_model import create_model
from neural_network_project.trainers.base_trainer import (
    BaseTrainer, EarlyStopping, ModelCheckpoint, TensorboardCallback
)
from neural_network_project.evaluators.base_evaluator import ClassificationEvaluator
from neural_network_project.utils.logger import get_logger, set_seed


logger = get_logger(__name__)


def train(config_path: Optional[str] = None) -> Dict[str, float]:
    """
    训练模型
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        训练结果
    """
    # 加载配置
    config = load_config(config_path)
    
    # 设置随机种子
    set_seed(config.training.seed)
    
    # 创建输出目录
    os.makedirs(config.training.output_dir, exist_ok=True)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model) if config.model.pretrained_model else None
    
    # 创建数据集
    train_dataset = TextClassificationDataset(
        file_path=config.data.train_path,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length
    )
    
    val_dataset = None
    if config.data.val_path and os.path.exists(config.data.val_path):
        val_dataset = TextClassificationDataset(
            file_path=config.data.val_path,
            tokenizer=tokenizer,
            max_seq_length=config.data.max_seq_length,
            label_map=train_dataset.label_map  # 使用与训练集相同的标签映射
        )
    
    # 创建数据加载器
    train_dataloader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    val_dataloader = None
    if val_dataset:
        val_dataloader = create_dataloader(
            dataset=val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers
        )
    
    # 创建模型
    model = create_model(
        config=config.model,
        vocab_size=len(tokenizer.vocab) if tokenizer else 30000,
        num_classes=len(train_dataset.label_map)
    )
    
    # 配置回调
    callbacks = [
        EarlyStopping(patience=3, monitor="val_loss" if val_dataloader else "train_loss"),
        ModelCheckpoint(
            checkpoint_dir=os.path.join(config.training.output_dir, "checkpoints"),
            monitor="val_loss" if val_dataloader else "train_loss"
        ),
        TensorboardCallback(log_dir=os.path.join(config.training.output_dir, "logs"))
    ]
    
    # 创建训练器
    trainer = BaseTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        config=config,
        callbacks=callbacks
    )
    
    # 训练模型
    logger.info("开始训练...")
    results = trainer.train()
    logger.info(f"训练完成: {results}")
    
    # 保存最终模型
    final_model_path = os.path.join(config.training.output_dir, "final_model")
    model.save_pretrained(final_model_path)
    if tokenizer:
        tokenizer.save_pretrained(final_model_path)
    logger.info(f"模型保存至: {final_model_path}")
    
    return results


def evaluate(model_path: str, test_path: str, config_path: Optional[str] = None) -> Dict[str, float]:
    """
    评估模型
    
    Args:
        model_path: 模型路径
        test_path: 测试数据路径
        config_path: 配置文件路径
        
    Returns:
        评估结果
    """
    # 加载配置
    config = load_config(config_path)
    
    # 加载分词器
    tokenizer_path = model_path  # 假设分词器和模型在同一目录
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # 加载标签映射
    import json
    label_map_path = os.path.join(model_path, "label_map.json")
    if os.path.exists(label_map_path):
        with open(label_map_path, "r", encoding="utf-8") as f:
            label_map = json.load(f)
    else:
        logger.warning("未找到标签映射文件，将从测试数据中推断")
        label_map = None
    
    # 创建测试数据集
    test_dataset = TextClassificationDataset(
        file_path=test_path,
        tokenizer=tokenizer,
        max_seq_length=config.data.max_seq_length,
        label_map=label_map
    )
    
    # 如果没有预设的标签映射，保存推断出的映射
    if label_map is None:
        label_map = test_dataset.label_map
        os.makedirs(model_path, exist_ok=True)
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    # 获取标签名称
    label_names = [None] * len(label_map)
    for name, idx in label_map.items():
        label_names[idx] = name
    
    # 创建数据加载器
    test_dataloader = create_dataloader(
        dataset=test_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # 加载模型
    from neural_network_project.models.base_model import BaseModel
    model = BaseModel.from_pretrained(model_path)
    
    # 创建评估器
    evaluator = ClassificationEvaluator(
        model=model,
        dataloader=test_dataloader,
        config=config,
        label_names=label_names
    )
    
    # 评估模型
    logger.info("开始评估...")
    results = evaluator.evaluate()
    
    # 打印结果
    logger.info("评估结果:")
    for key, value in results.items():
        logger.info(f"{key}: {value:.4f}")
    
    # 获取混淆矩阵
    cm = evaluator.get_confusion_matrix()
    logger.info(f"混淆矩阵: \n{cm}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="神经网络模型训练和评估")
    
    subparsers = parser.add_subparsers(dest="command", help="子命令")
    
    # 训练子命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    
    # 评估子命令
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument("--model", "-m", type=str, required=True, help="模型路径")
    eval_parser.add_argument("--test", "-t", type=str, required=True, help="测试数据路径")
    eval_parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train(args.config)
    elif args.command == "eval":
        evaluate(args.model, args.test, args.config)
    else:
        parser.print_help()