"""
Evaluation utilities for neural network project.
Contains functions for evaluating models and visualizing results.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
from sklearn.metrics import confusion_matrix

from data import prepare_batch


def evaluate_model(model, test_loader, criterion, device=None):
    """
    Evaluate model performance on test set
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
        criterion: Loss function
        device: Device to use (cuda/cpu)
        
    Returns:
        val_loss: Test loss
        accuracy: Test accuracy
        all_preds: All predictions
        all_targets: All true labels
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_targets = []
    
    # Create progress bar
    progress_bar = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Prepare data
            inputs, targets = prepare_batch(batch, device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets)
            
            # Update statistics
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Collect predictions and targets
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / total,
                'acc': 100. * correct / total
            })
    
    # Calculate final statistics
    val_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    print(f'Evaluation Loss: {val_loss:.4f} | Accuracy: {accuracy:.2f}%')
    
    return val_loss, accuracy, np.array(all_preds), np.array(all_targets)


def get_validation_loss(model, test_loader, criterion=None, device=None):
    """
    Get validation loss only (helper function for learning rate scheduler)
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
        criterion: Loss function (defaults to CrossEntropyLoss)
        device: Device to use (cuda/cpu)
        
    Returns:
        val_loss: Validation loss
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    val_loss, _, _, _ = evaluate_model(model, test_loader, criterion, device)
    return val_loss


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=True, figsize=(12, 10)):
    """
    Enhanced confusion matrix plot using seaborn for better visualization
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the confusion matrix
        figsize: Figure size
    """
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not installed. Installing now...")
        import pip
        pip.main(['install', 'seaborn'])
        import seaborn as sns
        
    from sklearn.metrics import confusion_matrix
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # 设置标准化选项
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.1%'  # 百分比格式
        vmin, vmax = 0, 1
        title = "Normalized Confusion Matrix"
    else:
        fmt = 'd'  # 整数格式
        vmin, vmax = None, None
        title = "Confusion Matrix (counts)"
    
    # 创建带有网格的子图
    plt.figure(figsize=figsize)
    
    # 设置类别名称
    if class_names is None:
        class_names = [str(i) for i in range(cm.shape[0])]
    
    # 使用seaborn的热图函数创建更美观的混淆矩阵
    ax = sns.heatmap(
        cm, 
        annot=True,           # 在每个单元格中显示数值
        fmt=fmt,              # 设置显示格式
        cmap='Blues',         # 使用蓝色调色板
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=vmin, vmax=vmax, # 设置颜色范围
        cbar=True,            # 显示颜色条
        square=True,          # 确保单元格为正方形
        linewidths=.5,        # 添加网格线
        linecolor='lightgray',# 网格线颜色
        annot_kws={"size": 9 if len(class_names) > 10 else 11}  # 注释字体大小
    )
    
    # 设置标题和标签
    plt.title(title, fontsize=14, pad=20)
    ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, labelpad=10)
    
    # 调整 x 轴标签的角度，以便更好地阅读
    plt.xticks(rotation=45, ha="right")
    
    # 突出显示对角线（正确分类）
    for i in range(cm.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='darkblue', lw=2))
    
    plt.tight_layout()
    plt.show()
    
    return cm  # 返回混淆矩阵以供进一步分析


def plot_class_performance(class_correct, class_total, class_names=None, figsize=(12, 6)):
    """
    Visualize per-class performance with detailed metrics
    
    Args:
        class_correct: List of correctly classified samples per class
        class_total: List of total samples per class
        class_names: List of class names
        figsize: Figure size
    """
    try:
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("Required packages not installed. Installing now...")
        import pip
        pip.main(['install', 'seaborn', 'pandas'])
        import seaborn as sns
        import pandas as pd
    
    # 如果没有提供类别名称，则创建默认名称
    if class_names is None:
        class_names = [f"Class {i}" for i in range(len(class_total))]
    
    # 计算每个类别的准确率
    accuracies = [100 * correct / total if total > 0 else 0 
                 for correct, total in zip(class_correct, class_total)]
    
    # 创建数据框以便于使用seaborn
    df = pd.DataFrame({
        'Class': class_names,
        'Accuracy': accuracies,
        'Samples': class_total,
        'Correct': class_correct,
        'Incorrect': [total - correct for correct, total in zip(class_correct, class_total)]
    })
    
    # 按准确率排序
    df = df.sort_values('Accuracy', ascending=False)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 1. 准确率条形图
    sns.barplot(x='Accuracy', y='Class', data=df, ax=ax1, 
                palette='viridis', saturation=0.8, hue='Accuracy', dodge=False, legend=False)
    
    # 添加数值标签
    for i, v in enumerate(df['Accuracy']):
        ax1.text(max(v + 1, 5), i, f"{v:.1f}%", va='center')
    
    # 设置标题和标签
    ax1.set_title('Per-Class Accuracy (%)', fontsize=13)
    ax1.set_xlim(0, 105)  # 限制x轴范围以便显示标签
    ax1.grid(axis='x', linestyle='--', alpha=0.7)
    
    # 2. 样本计数堆叠条形图
    df_stack = df.set_index('Class')[['Correct', 'Incorrect']]
    df_stack.plot(kind='barh', stacked=True, ax=ax2, 
                 color=['#2ecc71', '#e74c3c'], width=0.6)
    
    # 设置标题和标签
    ax2.set_title('Correct vs. Incorrect Samples', fontsize=13)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    ax2.legend(loc='upper right')
    
    # 添加样本总数标签
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.text(row['Samples'] + 0.5, i, f"{int(row['Samples'])}", va='center')
    
    plt.tight_layout()
    plt.show()


def show_random_prediction(model, test_loader, class_names=None, num_samples=1, device=None):
    """
    Show random predictions from the model with enhanced visualization for CIFAR-10
    
    Args:
        model: Model to use for predictions
        test_loader: DataLoader with test data
        class_names: List of class names
        num_samples: Number of samples to show
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model is None:
        print("Please provide a model!")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch of data
    dataiter = iter(test_loader)
    batch = next(dataiter)
    
    # Get inputs and targets
    if isinstance(batch, dict):
        images, labels = batch['img'], batch['label']
    else:
        images, labels = batch
    
    # Choose random samples
    indices = random.sample(range(len(images)), min(num_samples, len(images)))
    
    # Create figure
    plt.figure(figsize=(15, 4 * num_samples))
    
    for i, idx in enumerate(indices):
        # Get image and label
        image = images[idx]
        label = labels[idx].item()
        
        # Prepare for display
        display_image = image.numpy()
        if len(display_image.shape) == 3:  # [channels, height, width]
            # 对于CIFAR-10的RGB图像，需要转置为[height, width, channels]才能正确显示
            display_image = np.transpose(display_image, (1, 2, 0))
            
            # 可选：规范化图像显示（特别是对于模型预处理过的图像）
            if display_image.max() > 1.0 or display_image.min() < 0.0:
                display_image = (display_image - display_image.min()) / (display_image.max() - display_image.min())
        
        # Prepare model input
        input_tensor = image.float().unsqueeze(0).to(device)  # Add batch dimension
        
        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            pred_idx = predicted.item()
            probs = torch.nn.functional.softmax(output, dim=1)[0].cpu().numpy()
        
        # Display image
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(display_image)  # 移除cmap参数，让matplotlib自动处理彩色图像
        
        # Set title
        if class_names is not None:
            true_label = class_names[label]
            pred_label = class_names[pred_idx]
        else:
            true_label = f"Class {label}"
            pred_label = f"Class {pred_idx}"
        
        color = 'green' if label == pred_idx else 'red'
        plt.title(f"True: {true_label}\nPred: {pred_label}", color=color)
        plt.axis('off')
        
        # Display probability bar chart with enhanced styling
        plt.subplot(num_samples, 2, i*2 + 2)
        
        # 创建水平条形图并应用颜色映射
        colors = ['lightgray'] * len(probs)
        colors[label] = 'royalblue'  # 真实标签用蓝色
        if label != pred_idx:
            colors[pred_idx] = 'tomato'  # 错误预测用红色
            
        y_pos = np.arange(len(probs))
        
        bars = plt.barh(y_pos, probs, color=colors, alpha=0.8, height=0.5)
        
        # 添加概率值标签
        for j, bar in enumerate(bars):
            width = bar.get_width()
            if width > 0.01:  # 只有当概率大于1%时才显示标签
                plt.text(max(width + 0.01, 0.02), bar.get_y() + bar.get_height()/2, 
                        f"{width:.2f}", va='center', fontsize=9)
        
        # Set ticks
        if class_names is not None:
            plt.yticks(range(len(class_names)), class_names)
        else:
            plt.yticks(range(len(probs)), [f"Class {j}" for j in range(len(probs))])
        
        plt.xlim(0, 1.0)  # 限制x轴范围在0到1之间
        plt.xlabel('Probability')
        plt.title("Class Probabilities")
        plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()


def detailed_evaluation(model, test_loader, class_names=None, device=None):
    """
    Perform detailed evaluation of a model with enhanced visualizations
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader with test data
        class_names: List of class names
        device: Device to use (cuda/cpu)
        
    Returns:
        results: Dictionary with evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    criterion = nn.CrossEntropyLoss()
    
    # 默认的CIFAR-10类别名称（如果未提供）
    if class_names is None:
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    # Get evaluation results
    val_loss, accuracy, all_preds, all_targets = evaluate_model(
        model, test_loader, criterion, device
    )
    
    # 打印总体评估结果
    print(f"\n{'='*50}")
    print(f"Overall Results:")
    print(f"{'='*50}")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    
    # Calculate per-class accuracy
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    for i in range(len(all_targets)):
        label = all_targets[i]
        pred = all_preds[i]
        if label == pred:
            class_correct[label] += 1
        class_total[label] += 1
    
    # 打印每个类别的准确率
    print(f"\n{'='*50}")
    print(f"Per-Class Results:")
    print(f"{'='*50}")
    for i in range(len(class_names)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'{class_names[i]}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})')
    
    # 可视化每个类别的性能
    plot_class_performance(class_correct, class_total, class_names)
    
    # 绘制混淆矩阵
    print(f"\n{'='*50}")
    print(f"Confusion Matrix:")
    plot_confusion_matrix(all_targets, all_preds, class_names)
    
    # 随机显示一些预测样本
    print(f"\n{'='*50}")
    print(f"Sample Predictions:")
    show_random_prediction(model, test_loader, class_names, num_samples=3, device=device)
    
    # 返回详细结果
    results = {
        'val_loss': val_loss,
        'accuracy': accuracy,
        'class_correct': class_correct,
        'class_total': class_total,
        'class_accuracy': [100 * correct / total if total > 0 else 0 
                          for correct, total in zip(class_correct, class_total)],
        'all_preds': all_preds,
        'all_targets': all_targets
    }
    
    return results


def analyze_model_performance(model, test_loader, class_names=None, device=None, num_samples=10):
    """
    Perform advanced analysis of model performance with visualizations
    
    Args:
        model: Model to analyze
        test_loader: DataLoader with test data
        class_names: List of class names
        device: Device to use (cuda/cpu)
        num_samples: Number of samples to analyze per category
        
    Returns:
        None (displays visualizations)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if class_names is None:
        class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
    
    print("Analyzing model performance...")
    
    # 收集预测结果
    all_images = []
    all_labels = []
    all_preds = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Collecting predictions"):
            # 准备数据
            inputs, targets = prepare_batch(batch, device)
            
            # 前向传播
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)
            
            # 获取预测
            _, preds = outputs.max(1)
            
            # 存储结果
            all_images.extend(inputs.cpu())
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    all_images = torch.stack(all_images)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    # 1. 找出最容易混淆的类别对
    cm = confusion_matrix(all_labels, all_preds)
    np.fill_diagonal(cm, 0)  # 忽略对角线（正确分类）
    
    most_confused_idx = np.unravel_index(np.argmax(cm), cm.shape)
    class1, class2 = most_confused_idx
    
    print(f"\nMost confused classes: {class_names[class1]} often predicted as {class_names[class2]}")
    print(f"Confusion count: {cm[class1, class2]} samples")
    
    # 2. 可视化最具挑战性的样本（正确类别但低置信度）
    correct_mask = (all_labels == all_preds)
    correct_indices = np.where(correct_mask)[0]
    
    if len(correct_indices) > 0:
        # 获取每个样本的预测置信度
        confidences = np.array([all_probs[i][all_preds[i]] for i in correct_indices])
        
        # 找出置信度最低的正确预测
        low_conf_indices = correct_indices[np.argsort(confidences)[:num_samples]]
        
        # 显示低置信度但正确的预测
        plt.figure(figsize=(15, 3*min(len(low_conf_indices), 5)))
        plt.suptitle("Challenging samples (correctly classified with low confidence)", 
                    fontsize=14, y=0.95)
        
        for i, idx in enumerate(low_conf_indices[:5]):  # 限制显示5个
            image = all_images[idx].numpy()
            label = all_labels[idx]
            pred = all_preds[idx]
            prob = all_probs[idx][pred]
            
            # 转换图像格式
            if len(image.shape) == 3:  # [channels, height, width]
                image = np.transpose(image, (1, 2, 0))  # 转为 [height, width, channels]
            
            plt.subplot(min(len(low_conf_indices), 5), 3, i*3 + 1)
            plt.imshow(image)
            plt.title(f"True & Pred: {class_names[label]}\nConf: {prob:.2f}", fontsize=10)
            plt.axis('off')
            
            # 显示类别概率分布
            plt.subplot(min(len(low_conf_indices), 5), 3, i*3 + 2)
            top5_idx = np.argsort(all_probs[idx])[-5:][::-1]
            top5_probs = [all_probs[idx][j] for j in top5_idx]
            top5_names = [class_names[j] for j in top5_idx]
            
            colors = ['green' if j == label else 'gray' for j in top5_idx]
            plt.barh(range(5), top5_probs, color=colors)
            plt.yticks(range(5), top5_names)
            plt.xlim(0, 1)
            plt.title("Top-5 Probabilities", fontsize=10)
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            
            # 空白子图，保持布局一致
            plt.subplot(min(len(low_conf_indices), 5), 3, i*3 + 3)
            plt.axis('off')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    # 3. 可视化错误预测样本
    incorrect_mask = (all_labels != all_preds)
    incorrect_indices = np.where(incorrect_mask)[0]
    
    if len(incorrect_indices) > 0:
        # 按置信度排序（找出模型最确信但错误的预测）
        incorrect_confidences = np.array([all_probs[i][all_preds[i]] for i in incorrect_indices])
        sorted_indices = incorrect_indices[np.argsort(incorrect_confidences)[::-1][:num_samples]]
        
        # 显示高置信度但错误的预测
        plt.figure(figsize=(15, 3*min(len(sorted_indices), 5)))
        plt.suptitle("Misclassified samples with high confidence", fontsize=14, y=0.95)
        
        for i, idx in enumerate(sorted_indices[:5]):  # 限制显示5个
            image = all_images[idx].numpy()
            label = all_labels[idx]
            pred = all_preds[idx]
            prob = all_probs[idx][pred]
            
            # 转换图像格式
            if len(image.shape) == 3:  # [channels, height, width]
                image = np.transpose(image, (1, 2, 0))  # 转为 [height, width, channels]
            
            plt.subplot(min(len(sorted_indices), 5), 3, i*3 + 1)
            plt.imshow(image)
            plt.title(f"True: {class_names[label]}\nPred: {class_names[pred]} ({prob:.2f})", 
                     fontsize=10, color='red')
            plt.axis('off')
            
            # 显示类别概率分布
            plt.subplot(min(len(sorted_indices), 5), 3, i*3 + 2)
            top5_idx = np.argsort(all_probs[idx])[-5:][::-1]
            top5_probs = [all_probs[idx][j] for j in top5_idx]
            top5_names = [class_names[j] for j in top5_idx]
            
            colors = ['blue' if j == label else ('red' if j == pred else 'gray') 
                     for j in top5_idx]
            plt.barh(range(5), top5_probs, color=colors)
            plt.yticks(range(5), top5_names)
            plt.xlim(0, 1)
            plt.title("Top-5 Probabilities", fontsize=10)
            plt.grid(axis='x', linestyle='--', alpha=0.5)
            
            # 类别之间的混淆趋势
            plt.subplot(min(len(sorted_indices), 5), 3, i*3 + 3)
            conf_subset = cm[label, :]
            top_confused = np.argsort(conf_subset)[-5:][::-1]
            confused_counts = [conf_subset[j] for j in top_confused]
            confused_names = [class_names[j] for j in top_confused]
            
            plt.barh(range(5), confused_counts)
            plt.yticks(range(5), confused_names)
            plt.title(f"Top confusions for {class_names[label]}", fontsize=10)
            plt.grid(axis='x', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
    
    # 4. 类别间相似度热图（基于模型预测）
    try:
        import seaborn as sns
    except ImportError:
        print("Seaborn not installed. Installing now...")
        import pip
        pip.main(['install', 'seaborn'])
        import seaborn as sns
        
    class_similarity = np.zeros((len(class_names), len(class_names)))
    
    for i in range(len(class_names)):
        # 获取真实类别为i的所有样本
        class_i_indices = np.where(all_labels == i)[0]
        
        if len(class_i_indices) > 0:
            # 获取这些样本的预测概率分布
            class_i_probs = all_probs[class_i_indices]
            
            # 计算平均概率分布
            avg_probs = np.mean(class_i_probs, axis=0)
            class_similarity[i] = avg_probs
    
    # 绘制类别相似度热图
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(class_similarity, annot=True, fmt='.2f', 
               xticklabels=class_names, yticklabels=class_names, cmap='viridis')
    
    plt.title("Class Similarity Matrix (based on model probabilities)", fontsize=14)
    plt.xlabel("Predicted Class")
    plt.ylabel("True Class")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
    
    print("\nAnalysis complete. Use the visualizations to identify areas for model improvement.")