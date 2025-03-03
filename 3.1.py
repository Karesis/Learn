import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 设置随机种子以确保结果可重复
torch.manual_seed(42)
np.random.seed(42)

# 定义数据集类 - 将数字转换为序列
class DigitSequenceDataset(Dataset):
    def __init__(self, size=10000, max_value=100000):
        self.size = size
        # 生成随机数字
        self.numbers = np.random.randint(0, max_value, size=size)
        # 从数字中提取尾数作为标签
        self.labels = np.array([int(str(num)[-1]) for num in self.numbers])
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # 将数字转换为数字序列
        number = self.numbers[idx]
        num_str = str(number)
        
        # 将每个数字字符转换为独热编码
        # 我们使用11个特征：10个用于数字0-9，1个用于填充
        seq_length = 20  # 固定序列长度
        sequence = np.zeros((seq_length, 10))
        
        # 从右向左填充数字（低位优先）
        for i in range(min(len(num_str), seq_length)):
            digit = int(num_str[-(i+1)])  # 从最后一位开始
            sequence[i, digit] = 1.0
            
        return torch.tensor(sequence, dtype=torch.float32), self.labels[idx]

# 定义LSTM模型
class LSTMDigitClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=128, num_layers=2, output_size=10):
        super(LSTMDigitClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        # LSTM输出
        out, _ = self.lstm(x)
        
        # 我们只需要最后一个时间步的输出
        out = out[:, -1, :]
        
        # 全连接层
        out = self.fc(out)
        return out

# 训练函数
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        # 每5个epoch输出一次进度
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%')
            print(f'Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%')
            
            # 提前停止：如果训练准确率达到99%以上，提前结束
            if epoch_acc > 99.0 and val_epoch_acc > 98.0:
                print(f"训练准确率已达到 {epoch_acc:.2f}%，提前停止训练")
                break
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# 评估函数
def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    # 计算分类报告
    report = classification_report(all_labels, all_preds)
    
    return cm, report, all_preds, all_labels

# 可视化函数
def plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Over Time')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Over Time')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

# 测试函数：传入一组数字，返回预测结果
def test_with_examples(model, numbers):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    results = []
    
    for number in numbers:
        # 将数字转换为序列表示
        num_str = str(number)
        seq_length = 20  # 与训练时相同
        sequence = np.zeros((seq_length, 10))
        
        # 从右向左填充数字（低位优先）
        for i in range(min(len(num_str), seq_length)):
            digit = int(num_str[-(i+1)])  # 从最后一位开始
            sequence[i, digit] = 1.0
            
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
        
        # 预测
        with torch.no_grad():
            output = model(sequence_tensor)
            _, predicted = torch.max(output.data, 1)
            predicted_digit = predicted.item()
        
        # 实际尾数
        actual_digit = int(str(number)[-1])
        
        results.append({
            'number': number,
            'actual_digit': actual_digit,
            'predicted_digit': predicted_digit,
            'correct': actual_digit == predicted_digit
        })
    
    return results

# 主函数
def main():
    # 创建数据集
    train_dataset = DigitSequenceDataset(size=50000, max_value=1000000)
    val_dataset = DigitSequenceDataset(size=5000, max_value=1000000)
    test_dataset = DigitSequenceDataset(size=10000, max_value=1000000)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # 创建模型、损失函数和优化器
    model = LSTMDigitClassifier(input_size=10, hidden_size=128, num_layers=2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型
    print("开始训练模型...")
    train_losses, val_losses, train_accuracies, val_accuracies = train_model(
        model, train_loader, val_loader, criterion, optimizer, epochs=50
    )
    
    # 可视化训练过程
    plot_training_progress(train_losses, val_losses, train_accuracies, val_accuracies)
    
    # 评估模型
    print("\n开始评估模型...")
    cm, report, all_preds, all_labels = evaluate_model(model, test_loader)
    print("\n分类报告:")
    print(report)
    
    # 可视化混淆矩阵
    plot_confusion_matrix(cm)
    
    # 测试一些示例数据
    print("\n使用示例数字测试模型:")
    test_examples = [42, 123, 7890, 5555, 9999, 10, 2025, 888, 
                     54321, 98765, 1000000, 999999, 500000, 
                     76543, 87654, 512435, 812793, 333777]
    
    results = test_with_examples(model, test_examples)
    
    correct_count = 0
    for result in results:
        status = "✓" if result['correct'] else "✗"
        print(f"数字: {result['number']}, 实际尾数: {result['actual_digit']}, "
              f"预测尾数: {result['predicted_digit']} {status}")
        if result['correct']:
            correct_count += 1
    
    print(f"\n示例准确率: {correct_count/len(results):.2%}")
    
    # 保存模型
    torch.save(model.state_dict(), "lstm_last_digit_classifier.pth")
    print("\n模型已保存为 'lstm_last_digit_classifier.pth'")
    
    # 交互式测试
    print("\n开始交互式测试 (输入'q'退出):")
    while True:
        user_input = input("请输入一个数字: ")
        if user_input.lower() == 'q':
            break
            
        try:
            number = int(user_input)
            # 预测
            result = test_with_examples(model, [number])[0]
            status = "✓" if result['correct'] else "✗"
            print(f"数字: {result['number']}, 实际尾数: {result['actual_digit']}, "
                 f"预测尾数: {result['predicted_digit']} {status}")
        except ValueError:
            print("请输入有效的整数")

if __name__ == "__main__":
    main()