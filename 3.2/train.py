import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os

def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练模型一个完整的epoch
    
    参数:
    - model: 要训练的模型
    - train_loader: 训练数据加载器
    - optimizer: 优化器
    - criterion: 损失函数
    - device: 计算设备(CPU/GPU)
    
    返回:
    - epoch_loss: 整个epoch的平均损失
    - accuracy: 训练准确率
    """
    model.train()  # 设置为训练模式
    total_loss = 0.0
    correct = 0
    total = 0
    
    # 创建进度条
    progress_bar = tqdm.tqdm(train_loader, desc="Training")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 提取数据和标签
        inputs, targets = batch['image'], batch['label']
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = inputs.float()      # 将图像转为float类型
        targets = targets.long()     # 将标签转为long类型
            
        # 清零梯度
        optimizer.zero_grad()
        
        # 获取模型输出
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    # 计算整个epoch的平均损失和准确率
    epoch_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    print(f'Training Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return epoch_loss, accuracy

def train_model(model, train_loader, val_loss_fn, num_epochs=10, lr=0.001, save_path=None):
    """
    完整的训练流程，但不包含评估逻辑
    
    参数:
    - model: 要训练的模型
    - train_loader: 训练数据加载器
    - val_loss_fn: 一个函数，用于获取验证损失（传入模型作为参数）
    - num_epochs: 训练的epoch数
    - lr: 初始学习率
    - save_path: 模型保存路径（如果为None则不保存）
    
    返回:
    - model: 训练后的模型
    - train_stats: 训练统计信息
    """
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # 训练统计
    train_stats = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'learning_rate': []
    }
    
    # 如果提供了保存路径，确保目录存在
    if save_path:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 记录训练统计
        train_stats['epoch'].append(epoch + 1)
        train_stats['train_loss'].append(train_loss)
        train_stats['train_acc'].append(train_acc)
        train_stats['learning_rate'].append(optimizer.param_groups[0]['lr'])
        
        # 获取验证损失（使用传入的函数）
        val_loss = val_loss_fn(model)
        
        # 更新学习率调度器
        scheduler.step(val_loss)
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Current Learning Rate: {current_lr:.6f}')
        
        # 保存模型（如果指定了保存路径）
        if save_path:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss
            }
            if epoch == num_epochs - 1:
                torch.save(checkpoint, f"{save_path}_final.pt")
            else:
                torch.save(checkpoint, f"{save_path}_epoch{epoch+1}.pt")
    
    return model, train_stats