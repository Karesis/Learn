# 基础训练器类
import os
import time
import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import numpy as np
from tqdm import tqdm

from ..config.config import Config, TrainingConfig
from ..models.base_model import BaseModel
from ..utils.logger import get_logger

logger = get_logger(__name__)


class Callback:
    """训练回调基类"""
    
    def on_train_begin(self, trainer: "BaseTrainer") -> None:
        """训练开始时调用"""
        pass
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """训练结束时调用"""
        pass
    
    def on_epoch_begin(self, trainer: "BaseTrainer", epoch: int) -> None:
        """每个epoch开始时调用"""
        pass
    
    def on_epoch_end(self, trainer: "BaseTrainer", epoch: int, metrics: Dict[str, float]) -> None:
        """每个epoch结束时调用"""
        pass
    
    def on_batch_begin(self, trainer: "BaseTrainer", batch: Dict[str, torch.Tensor]) -> None:
        """每个批次开始时调用"""
        pass
    
    def on_batch_end(self, trainer: "BaseTrainer", batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], loss: float) -> None:
        """每个批次结束时调用"""
        pass
    
    def on_evaluate(self, trainer: "BaseTrainer", metrics: Dict[str, float]) -> None:
        """评估时调用"""
        pass


class EarlyStopping(Callback):
    """早停回调"""
    
    def __init__(self, patience: int = 3, min_delta: float = 0.0, monitor: str = "val_loss", mode: str = "min"):
        """
        初始化早停回调
        
        Args:
            patience: 容忍的epoch数量
            min_delta: 最小变化量
            monitor: 监控的指标
            mode: 模式，'min'表示监控指标越小越好，'max'表示越大越好
        """
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode
        
        self.counter = 0
        self.best_score = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
    
    def on_epoch_end(self, trainer: "BaseTrainer", epoch: int, metrics: Dict[str, float]) -> None:
        """每个epoch结束时检查是否应该早停"""
        score = metrics.get(self.monitor)
        if score is None:
            return
        
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            logger.info(f"早停! {self.monitor} 未改善 {self.patience} 个epoch")
            self.early_stop = True
            trainer.should_stop = True


class ModelCheckpoint(Callback):
    """模型检查点回调"""
    
    def __init__(self, 
                checkpoint_dir: str, 
                monitor: str = "val_loss", 
                mode: str = "min",
                save_best_only: bool = True,
                save_last: bool = True):
        """
        初始化模型检查点回调
        
        Args:
            checkpoint_dir: 检查点保存目录
            monitor: 监控的指标
            mode: 模式，'min'表示监控指标越小越好，'max'表示越大越好
            save_best_only: 是否只保存最佳模型
            save_last: 是否保存最后一个模型
        """
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_last = save_last
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        self.best_score = float('inf') if mode == 'min' else float('-inf')
    
    def on_epoch_end(self, trainer: "BaseTrainer", epoch: int, metrics: Dict[str, float]) -> None:
        """每个epoch结束时保存模型"""
        if self.save_last:
            # 保存最后的模型
            last_path = os.path.join(self.checkpoint_dir, "last_model.pt")
            trainer.save_checkpoint(last_path)
        
        score = metrics.get(self.monitor)
        if score is None or not self.save_best_only:
            return
        
        improved = (self.mode == 'min' and score < self.best_score) or \
                  (self.mode == 'max' and score > self.best_score)
        
        if improved:
            self.best_score = score
            # 保存最佳模型
            best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            trainer.save_checkpoint(best_path)
            logger.info(f"保存最佳模型，{self.monitor}: {score:.4f}")


class TensorboardCallback(Callback):
    """Tensorboard回调"""
    
    def __init__(self, log_dir: str):
        """
        初始化Tensorboard回调
        
        Args:
            log_dir: 日志目录
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            logger.warning("无法导入tensorboard，请安装: pip install tensorboard")
            self.enabled = False
    
    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """训练结束时关闭writer"""
        if self.enabled:
            self.writer.close()
    
    def on_batch_end(self, trainer: "BaseTrainer", batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor], loss: float) -> None:
        """记录每个批次的损失"""
        if self.enabled:
            self.writer.add_scalar("train/batch_loss", loss, trainer.global_step)
    
    def on_epoch_end(self, trainer: "BaseTrainer", epoch: int, metrics: Dict[str, float]) -> None:
        """记录每个epoch的指标"""
        if not self.enabled:
            return
            
        for name, value in metrics.items():
            self.writer.add_scalar(f"train/{name}", value, epoch)


class BaseTrainer:
    """基础训练器"""
    
    def __init__(self, 
                model: BaseModel, 
                train_dataloader: DataLoader,
                val_dataloader: Optional[DataLoader] = None,
                config: Optional[Config] = None,
                optimizer: Optional[Optimizer] = None,
                lr_scheduler: Optional[LambdaLR] = None,
                callbacks: Optional[List[Callback]] = None):
        """
        初始化训练器
        
        Args:
            model: 模型
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            config: 配置
            optimizer: 优化器
            lr_scheduler: 学习率调度器
            callbacks: 回调列表
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or Config()
        
        # 设置设备
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # 创建优化器
        self.optimizer = optimizer or self._create_optimizer()
        
        # 创建学习率调度器
        self.lr_scheduler = lr_scheduler or self._create_lr_scheduler()
        
        # 回调
        self.callbacks = callbacks or []
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.should_stop = False
        self.best_metrics = {}
    
    def _create_optimizer(self) -> Optimizer:
        """创建优化器"""
        return AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _create_lr_scheduler(self) -> Optional[LambdaLR]:
        """创建学习率调度器"""
        if self.config.training.warmup_steps > 0:
            # 线性预热，然后线性衰减
            def lr_lambda(current_step: int) -> float:
                if current_step < self.config.training.warmup_steps:
                    return float(current_step) / float(max(1, self.config.training.warmup_steps))
                
                total_steps = len(self.train_dataloader) * self.config.training.epochs
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - self.config.training.warmup_steps)))
            
            return LambdaLR(self.optimizer, lr_lambda)
        
        return None
    
    def train(self) -> Dict[str, float]:
        """
        训练模型
        
        Returns:
            最佳指标
        """
        # 调用训练开始回调
        for callback in self.callbacks:
            callback.on_train_begin(self)
        
        # 训练循环
        for epoch in range(self.config.training.epochs):
            self.current_epoch = epoch
            
            # 调用epoch开始回调
            for callback in self.callbacks:
                callback.on_epoch_begin(self, epoch)
            
            # 训练一个epoch
            train_metrics = self._train_epoch()
            
            # 评估
            val_metrics = {}
            if self.val_dataloader is not None:
                val_metrics = self._evaluate()
                # 调用评估回调
                for callback in self.callbacks:
                    callback.on_evaluate(self, val_metrics)
            
            # 合并指标
            metrics = {**train_metrics, **val_metrics}
            
            # 打印指标
            metrics_str = ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch+1}/{self.config.training.epochs} - {metrics_str}")
            
            # 调用epoch结束回调
            for callback in self.callbacks:
                callback.on_epoch_end(self, epoch, metrics)
            
            # 检查是否应该停止
            if self.should_stop:
                logger.info(f"训练在epoch {epoch+1}提前停止")
                break
        
        # 调用训练结束回调
        for callback in self.callbacks:
            callback.on_train_end(self)
        
        return self.best_metrics
    
    def _train_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            训练指标
        """
        self.model.train()
        
        total_loss = 0.0
        total_samples = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch in progress_bar:
            # 调用批次开始回调
            for callback in self.callbacks:
                callback.on_batch_begin(self, batch)
            
            # 将数据移到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(**batch)
            
            # 获取损失
            loss = outputs["loss"]
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            if self.config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.max_grad_norm
                )
            
            # 优化器步进
            self.optimizer.step()
            
            # 学习率调度器步进
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # 累积损失
            total_loss += loss.item() * batch["input_ids"].size(0)
            total_samples += batch["input_ids"].size(0)
            
            # 更新进度条
            progress_bar.set_postfix({
                "loss": loss.item()
            })
            
            # 调用批次结束回调
            for callback in self.callbacks:
                callback.on_batch_end(self, batch, outputs, loss.item())
            
            self.global_step += 1
        
        # 计算平均损失
        avg_loss = total_loss / total_samples
        
        return {"train_loss": avg_loss}
    
    def _evaluate(self) -> Dict[str, float]:
        """
        评估模型
        
        Returns:
            评估指标
        """
        self.model.eval()
        
        total_loss = 0.0
        total_samples = 0
        
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating"):
                # 将数据移到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播
                outputs = self.model(**batch)
                
                # 获取损失
                if "loss" in outputs:
                    loss = outputs["loss"]
                    total_loss += loss.item() * batch["input_ids"].size(0)
                    total_samples += batch["input_ids"].size(0)
                
                # 收集预测和标签
                if "labels" in batch and "logits" in outputs:
                    preds = torch.argmax(outputs["logits"], dim=-1)
                    all_preds.append(preds.cpu().numpy())
                    all_labels.append(batch["labels"].cpu().numpy())
        
        # 计算指标
        metrics = {}
        
        # 损失
        if total_samples > 0:
            metrics["val_loss"] = total_loss / total_samples
        
        # 准确率
        if all_preds and all_labels:
            all_preds = np.concatenate(all_preds)
            all_labels = np.concatenate(all_labels)
            accuracy = np.mean(all_preds == all_labels)
            metrics["val_accuracy"] = accuracy
        
        # 更新最佳指标
        for name, value in metrics.items():
            if name not in self.best_metrics or value > self.best_metrics[name]:
                self.best_metrics[name] = value
        
        return metrics
    
    def save_checkpoint(self, path: str) -> None:
        """
        保存检查点
        
        Args:
            path: 保存路径
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metrics": self.best_metrics
        }
        
        if self.lr_scheduler is not None:
            checkpoint["lr_scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载检查点
        
        Args:
            path: 加载路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["current_epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metrics = checkpoint["best_metrics"]
        
        if self.lr_scheduler is not None and "lr_scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])