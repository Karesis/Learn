import torch
import tqdm
import numpy as np

def evaluate_model(model, test_loader, criterion, device=None):
    """
    评估模型性能
    
    参数:
    - model: 要评估的模型
    - test_loader: 测试数据加载器
    - criterion: 损失函数
    - device: 计算设备(CPU/GPU)，如果为None则自动检测
    
    返回:
    - val_loss: 验证损失
    - accuracy: 验证准确率
    - all_preds: 所有预测结果
    - all_targets: 所有真实标签
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    # 创建进度条
    progress_bar = tqdm.tqdm(test_loader, desc="Evaluating")
    
    # 使用torch.no_grad()防止梯度计算，节省内存
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # 提取数据和标签
            inputs, targets = batch['image'], batch['label']
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()      # 将图像转为float类型
            targets = targets.long()     # 将标签转为long类型
            
            # 前向传播
            outputs = model(inputs)
            
            # 计算损失
            loss = criterion(outputs, targets)
            
            # 统计
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # 收集预测和标签
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': total_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    # 计算平均损失和准确率
    val_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Validation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy, np.array(all_preds), np.array(all_targets)

def get_validation_loss(model, test_loader, criterion=None, device=None):
    """
    获取验证损失的简单函数，用于学习率调度器
    
    参数:
    - model: 要评估的模型
    - test_loader: 测试数据加载器
    - criterion: 损失函数，如果为None则使用交叉熵损失
    - device: 计算设备(CPU/GPU)，如果为None则自动检测
    
    返回:
    - val_loss: 验证损失
    """
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    val_loss, _, _, _ = evaluate_model(model, test_loader, criterion, device)
    return val_loss

def detailed_evaluation(model, test_loader, device=None):
    """
    详细评估模型性能，包括各类别准确率
    
    参数:
    - model: 要评估的模型
    - test_loader: 测试数据加载器
    - device: 计算设备(CPU/GPU)，如果为None则自动检测
    
    返回:
    - results: 包含详细评估结果的字典
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = torch.nn.CrossEntropyLoss()
    
    # 获取评估结果
    val_loss, accuracy, all_preds, all_targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # Fashion MNIST类别名称
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                  'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 计算每个类别的准确率
    class_correct = np.zeros(10)
    class_total = np.zeros(10)
    
    for i in range(len(all_targets)):
        label = all_targets[i]
        pred = all_preds[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    # 打印每个类别的准确率
    print("\n类别准确率:")
    for i in range(10):
        if class_total[i] > 0:
            print(f'{class_names[i]}: {100 * class_correct[i] / class_total[i]:.2f}%')
    
    # 返回详细结果
    results = {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'class_correct': class_correct,
        'class_total': class_total,
        'all_preds': all_preds,
        'all_targets': all_targets
    }
    
    return results