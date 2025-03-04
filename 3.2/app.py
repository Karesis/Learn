import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from pathlib import Path
import sys

# 导入项目模块
from nn import MultiFashionExpertWithLSTM
from data import test_loader

# Fashion MNIST类别名称
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

def clear_screen():
    """清除终端屏幕"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_model(model_path):
    """加载已训练好的模型"""
    print(f"加载模型: {model_path}")
    
    # 创建模型实例
    model = MultiFashionExpertWithLSTM(num_classes=10, in_channels=1, num_experts=3)
    
    # 加载权重
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型! 训练轮次: {checkpoint['epoch']}")
        print(f"验证损失: {checkpoint['val_loss']:.4f}, 训练准确率: {checkpoint['train_acc']:.2f}%")
        model.eval()  # 设为评估模式
        return model
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None

def show_random_prediction(model, num_samples=1):
    """从测试集随机选择样本并显示模型预测"""
    if model is None:
        print("请先加载模型!")
        return
    
    # 获取随机样本
    samples = []
    labels = []
    
    # 从测试加载器中获取一批数据
    dataiter = iter(test_loader)
    batch = next(dataiter)
    
    # 从批次中随机选择样本
    indices = random.sample(range(len(batch['image'])), min(num_samples, len(batch['image'])))
    
    for idx in indices:
        image = batch['image'][idx]
        label = batch['label'][idx].item()
        
        # 准备输入 - 保留原始图像用于显示
        display_image = image.numpy()
        
        # 如果图像有通道维度，需要去除它以便显示
        if len(display_image.shape) == 3:  # 如果形状是 [1, 28, 28]
            display_image = display_image.squeeze(0)  # 变成 [28, 28]
        
        # 准备模型输入
        image_tensor = image.float().unsqueeze(0)  # 添加批次维度
        if len(image_tensor.shape) == 3:  # 如果是 [batch, height, width]
            image_tensor = image_tensor.unsqueeze(1)  # 添加通道维度
        
        # 获取预测
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1)
            pred_idx = predicted.item()
            
        # 获取预测概率
        probs = torch.nn.functional.softmax(output, dim=1)[0]
        
        # 添加到列表
        samples.append((display_image, label, pred_idx, probs.numpy()))
    
    # 显示样本和预测
    plt.figure(figsize=(12, 4 * num_samples))
    
    for i, (image, true_label, pred_label, probs) in enumerate(samples):
        # 显示图像
        plt.subplot(num_samples, 2, i*2 + 1)
        plt.imshow(image, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        plt.title(f"真实: {class_names[true_label]}\n预测: {class_names[pred_label]}", color=color)
        plt.axis('off')
        
        # 显示概率条形图
        plt.subplot(num_samples, 2, i*2 + 2)
        bars = plt.barh(range(10), probs)
        plt.yticks(range(10), class_names)
        plt.xlabel('概率')
        plt.tight_layout()
        
        # 高亮真实类别和预测类别
        bars[true_label].set_color('blue')
        if true_label != pred_label:
            bars[pred_label].set_color('red')
    
    plt.tight_layout()
    plt.show()

def evaluate_model(model):
    """在测试集上评估模型性能"""
    if model is None:
        print("请先加载模型!")
        return
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10
    
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch['image'], batch['label']
            
            # 数据预处理
            images = images.float()
            labels = labels.long()
            if len(images.shape) == 3:
                images = images.unsqueeze(1)
            
            # 获取预测
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # 更新统计
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 按类别统计
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i].item() == label:
                    class_correct[label] += 1
    
    # 显示总体准确率
    print(f"\n总体准确率: {100 * correct / total:.2f}%")
    
    # 显示每个类别的准确率
    print("\n按类别准确率:")
    for i in range(10):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            print(f"{class_names[i]}: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    # 识别最容易混淆的类别
    print("\n模型表现分析:")
    min_acc_idx = np.argmin([c/t if t > 0 else 1.0 for c, t in zip(class_correct, class_total)])
    print(f"最难识别的类别: {class_names[min_acc_idx]} "
          f"(准确率: {100 * class_correct[min_acc_idx] / class_total[min_acc_idx]:.2f}%)")

def display_menu():
    """显示主菜单"""
    clear_screen()
    print("=" * 50)
    print("    Fashion MNIST 模型演示")
    print("=" * 50)
    print("1. 加载模型")
    print("2. 随机样本预测")
    print("3. 预测多个样本 (5个)")
    print("4. 模型性能评估")
    print("5. 退出")
    print("=" * 50)

def main():
    """主函数"""
    model = None
    default_model_path = "./models/fashion_mnist_model_final.pt"
    
    # 检查默认模型是否存在，如果存在则自动加载
    if os.path.exists(default_model_path):
        model = load_model(default_model_path)
    
    while True:
        display_menu()
        
        # 显示当前模型状态
        if model is not None:
            print(f"当前已加载模型: {default_model_path}")
        else:
            print("当前未加载模型")
            
        # 获取用户选择
        choice = input("\n请选择操作 (1-5): ").strip()
        
        if choice == '1':
            # 加载模型
            model_path = input("请输入模型路径 (按Enter使用默认路径): ").strip()
            if not model_path:
                model_path = default_model_path
            
            model = load_model(model_path)
            input("\n按Enter继续...")
            
        elif choice == '2':
            # 随机样本预测
            if model is not None:
                print("正在生成预测，请稍候...")
                try:
                    show_random_prediction(model, num_samples=1)
                    input("\n关闭图表后按Enter继续...")
                except Exception as e:
                    print(f"预测时出错: {e}")
                    input("\n按Enter继续...")
            else:
                print("请先加载模型!")
                input("\n按Enter继续...")
                
        elif choice == '3':
            # 多个样本预测
            if model is not None:
                print("正在生成多个预测，请稍候...")
                try:
                    show_random_prediction(model, num_samples=5)
                    input("\n关闭图表后按Enter继续...")
                except Exception as e:
                    print(f"预测时出错: {e}")
                    input("\n按Enter继续...")
            else:
                print("请先加载模型!")
                input("\n按Enter继续...")
                
        elif choice == '4':
            # 模型性能评估
            if model is not None:
                print("正在评估模型，请稍候...")
                try:
                    evaluate_model(model)
                    input("\n按Enter继续...")
                except Exception as e:
                    print(f"评估时出错: {e}")
                    input("\n按Enter继续...")
            else:
                print("请先加载模型!")
                input("\n按Enter继续...")
                
        elif choice == '5':
            # 退出
            print("谢谢使用!")
            sys.exit(0)
            
        else:
            print("无效选择，请重试。")
            input("\n按Enter继续...")

if __name__ == "__main__":
    main()