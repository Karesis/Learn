#1.1
def ptv(x, value):
    print(f"{x}\nData Type: {value}")  # 添加了空格，使输出更美观

model_version = 2
ptv(model_version, 'int')
learning_rate = 0.01
ptv(learning_rate, 'float')
model_name = "SimpleNN"
ptv(model_name, 'string')
is_trained = False
ptv(is_trained, 'bool')
layers = [784, 128, 10]
ptv(layers, 'list')

#1.2
def evaluate_accuracy(accuracy):
    if accuracy >= 0.9:
        return "Excellent"
    elif accuracy >= 0.8:
        return "Great"
    elif accuracy >= 0.6:
        return "Average"
    else:
        return "Need Progress"

#1.3
accuracies = [0.5, 0.62, 0.68, 0.75, 0.79, 0.8, 0.82, 0.85, 0.89, 0.91]
average_accuracy = sum(accuracies) / len(accuracies)
max_accuracy = max(accuracies)
new_numbers = [acc for acc in accuracies if acc >= 0.8]
print(all(acc > 0.5 for acc in accuracies))

#2.1
def calculate_metrics(predictions, true_values):
    actual_number = 0
    for i in range(len(predictions)):
        if predictions[i] == true_values[i]:
            actual_number += 1
    return actual_number / len(predictions)

# 测试
preds = [1, 0, 1, 1, 0, 1]
actuals = [1, 0, 0, 1, 0, 1]
print(calculate_metrics(preds, actuals))  # 应返回 0.8333... (5/6)

#2.2
import math  # 移到函数外部，避免重复导入

def apply_activation(values, activation_type):
    def relu(x):
        return max(0, x)
    def sigmoid(x):
        return 1/(1+math.exp(-x))
    
    if activation_type == "relu":
        return [relu(x) for x in values]
    elif activation_type == "sigmoid":  # 改为elif，更符合逻辑
        return [sigmoid(x) for x in values]
    else:
        print('apply activation false')
        return []

# 测试
test_values = [-2, -1, 0, 1, 2]
print(apply_activation(test_values, "sigmoid")) # 应返回接近 [0.12, 0.27, 0.5, 0.73, 0.88] 的值
print(apply_activation(test_values, "relu"))    # 应返回 [0, 0, 0, 1, 2]

#3.1
import numpy as np
array1 = np.zeros(10)
print(array1)
array2 = np.eye(3)
print(array2)
array3 = np.random.randint(9, size=10)  # 修正了参数格式，添加了=号
print(array3)
array4 = np.random.rand(2,3)
print(array4)

#3.2
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A + B)  # 逐个元素加法
print(A * B)  # 逐个元素乘法
print(np.dot(A, B))  # A与B点积
print(A.T)

#3.3
def forward_layer(inputs, weights, bias):
    output = np.dot(inputs, weights) + bias
    # 正确的ReLU实现
    output = np.maximum(0, output)
    return output

# 测试数据
x = np.array([0.5, 0.3, 0.2])
W = np.array([[0.1, 0.2, 0.3], 
              [0.4, 0.5, 0.6], 
              [0.7, 0.8, 0.9]])
b = np.array([0.1, 0.2, 0.3])

# 输出结果
output = forward_layer(x, W, b)
print(output)

#4.1
import torch
tensor1 = torch.tensor([3.0, 4.0, 5.0])
tensor2 = torch.ones(2, 2)
tensor3 = torch.rand(3, 3)  # torch.rand生成[0,1)均匀分布的随机数
# torch.randn生成均值为0，标准差为1的正态分布随机数
tensor4 = torch.from_numpy(np.array([1, 2, 3]))

#4.2
x = torch.tensor([1.0, 2.0, 3.0])
y = torch.tensor([4.0, 5.0, 6.0])
print(x + y)  # 逐个元素相加
print(x * y)  # 逐个元素相乘
print(torch.dot(x, y))  # x和y点积
print(x.mean())  # 便捷函数，取得平均值

#4.3
def pytorch_forward_layer(inputs, weights, bias):
    output = inputs @ weights + bias  # 使用@操作符进行矩阵乘法
    output = torch.relu(output)
    return output  # 移除了多余的括号

# 测试数据
x = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
W = torch.tensor([[0.1, 0.2, 0.3], 
                  [0.4, 0.5, 0.6], 
                  [0.7, 0.8, 0.9]], dtype=torch.float32)
b = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)

# 输出结果
output = pytorch_forward_layer(x, W, b)
print(output)

#5.1
print("第一题：引入非线性关系，使得神经网络能够拟合复杂的行为")
print("第二题：前向传播就是给定输入，让神经网络计算输出，并记录数据流动方向；反向传播就是计算预测值与实际值之间的差距，然后根据数据流动的反方向，逐层求导，计算梯度")
print("第三题：权重表示后一层神经元接受前一层神经元的数据的程度，偏置则是防止数据过大或过小，方便处理")

#5.2
import torch.nn as nn

class ImprovedNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ImprovedNN, self).__init__()
        self.models = nn.Sequential(nn.Linear(input_size, hidden_size),
                                   nn.ReLU(),
                                   nn.Linear(hidden_size, output_size))
        
    def forward(self, x):
        return self.models(x)
        
    def predict(self, x):
        # 实现预测方法
        self.eval()  # 设置为评估模式
        with torch.no_grad():
            output = self.forward(x)
            _, predicted = torch.max(output, 1)
            return predicted

# 测试代码
model = ImprovedNN(784, 128, 10)
fake_image = torch.rand(1, 784)
model.eval()
with torch.no_grad():
    prediction = model(fake_image)
print("预测结果:", prediction)

#综合挑战: 预测数字的最后一位
import random
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

class DigitExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # 使用更深的网络结构，以便学习复杂的数学关系
        self.network = nn.Sequential(
            nn.Linear(1, 128),         # 输入就是原始数字
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 600),
            nn.GELU(),
            nn.Linear(600, 1000),
            nn.GELU(),
            nn.Linear(1000, 1000),
            nn.GELU(),
            nn.Linear(1000, 600),
            nn.GELU(),
            nn.Linear(600, 128),
            nn.GELU(),
            nn.Linear(128, 10)          # 输出10个类别
        )
        
        # 初始化权重，使用特殊初始化帮助学习模运算
        self._initialize_weights()
    
    def _initialize_weights(self):
        # 特殊初始化权重以帮助学习模运算
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用Xavier初始化，有助于处理大范围的数值
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        return self.network(x)
    
    def predict(self, number):
        self.eval()
        with torch.no_grad():
            if isinstance(number, (int, float)):
                # 直接输入数字
                x = torch.tensor([[float(number)]], dtype=torch.float32)
            else:
                # 已经是张量
                x = number
            output = self(x)
            return torch.argmax(output, dim=1)

# 数据准备函数 - 使用多个数据集进行渐进学习
def prepare_data(stages=3):
    datasets = []
    
    # 阶段1：小范围数字（0-100）
    size1 = 5000
    data1 = [random.randint(0, 100) for _ in range(size1)]
    targets1 = [x % 10 for x in data1]
    X1 = torch.tensor([[float(x)] for x in data1], dtype=torch.float32)
    y1 = torch.tensor(targets1, dtype=torch.long)
    
    # 阶段2：中等范围数字（0-10000）
    size2 = 10000
    data2 = [random.randint(0, 10000) for _ in range(size2)]
    targets2 = [x % 10 for x in data2]
    X2 = torch.tensor([[float(x)] for x in data2], dtype=torch.float32)
    y2 = torch.tensor(targets2, dtype=torch.long)
    
    # 阶段3：大范围数字（0-1000000）
    size3 = 20000
    data3 = [random.randint(0, 1000000) for _ in range(size3)]
    targets3 = [x % 10 for x in data3]
    X3 = torch.tensor([[float(x)] for x in data3], dtype=torch.float32)
    y3 = torch.tensor(targets3, dtype=torch.long)
    
    datasets = [(X1, y1, data1), (X2, y2, data2), (X3, y3, data3)]
    
    # 为每个阶段创建训练集和测试集
    train_test_splits = []
    for X, y, orig_data in datasets:
        # 80%/20% 分割
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        orig_train, orig_test = orig_data[:split], orig_data[split:]
        
        train_test_splits.append((X_train, y_train, X_test, y_test, orig_train, orig_test))
    
    return train_test_splits

# 优化函数：使用数据归一化
def normalize_data(X_train, X_test, normalization_type='minmax'):
    """
    对数据进行归一化处理，有助于模型学习
    
    normalization_type: 
        'minmax' - 将数据缩放到[0,1]范围
        'standard' - 标准化（均值0，标准差1）
        'log' - 对数变换后的MinMax
    """
    if normalization_type == 'minmax':
        # 查找训练集的最大值和最小值
        min_val = X_train.min()
        max_val = X_train.max()
        
        # 应用MinMax归一化
        X_train_normalized = (X_train - min_val) / (max_val - min_val + 1e-10)
        X_test_normalized = (X_test - min_val) / (max_val - min_val + 1e-10)
        
    elif normalization_type == 'standard':
        # 计算训练集的均值和标准差
        mean_val = X_train.mean()
        std_val = X_train.std()
        
        # 应用标准化
        X_train_normalized = (X_train - mean_val) / (std_val + 1e-10)
        X_test_normalized = (X_test - mean_val) / (std_val + 1e-10)
        
    elif normalization_type == 'log':
        # 对数变换（处理大范围的数值）
        X_train_log = torch.log1p(X_train)  # log(1+x)避免对0取对数
        X_test_log = torch.log1p(X_test)
        
        # 然后应用MinMax归一化
        min_val = X_train_log.min()
        max_val = X_train_log.max()
        
        X_train_normalized = (X_train_log - min_val) / (max_val - min_val + 1e-10)
        X_test_normalized = (X_test_log - min_val) / (max_val - min_val + 1e-10)
    
    else:
        # 不进行归一化
        X_train_normalized = X_train
        X_test_normalized = X_test
    
    return X_train_normalized, X_test_normalized

# 创建批次加载器
def create_batches(X, y, batch_size=128):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练函数
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001, 
               scheduler_type='cosine', patience=5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 设置学习率调度器
    if scheduler_type == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif scheduler_type == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience//2, factor=0.5)
    else:
        scheduler = None
    
    # 早停
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    
    train_losses = []
    val_accuracies = []
    learning_rates = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, labels in progress_bar:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item(), 'lr': optimizer.param_groups[0]['lr']})
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 验证
        val_accuracy = evaluate(model, val_loader)
        val_accuracies.append(val_accuracy)
        
        # 保存当前学习率
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Val Accuracy: {val_accuracy:.4f} - LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 更新学习率
        if scheduler_type == 'cosine':
            scheduler.step()
        elif scheduler_type == 'plateau':
            scheduler.step(epoch_loss)
        
        # 检查是否需要早停
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs.')
                break
    
    # 如果有最佳模型状态，加载它
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_accuracies, learning_rates

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# 测试具体的例子
def test_examples(model, test_numbers):
    print("\n测试特定的例子:")
    correct = 0
    
    for num in test_numbers:
        # 获取预测
        prediction = model.predict(num).item()
        actual = num % 10
        
        result = "✓" if prediction == actual else "✗"
        if prediction == actual:
            correct += 1
            
        print(f"数字: {num}, 预测最后一位: {prediction}, 实际最后一位: {actual}, {result}")
    
    print(f"总体正确率: {correct}/{len(test_numbers)} = {correct/len(test_numbers):.2%}")
    
    return correct / len(test_numbers)

# 渐进式训练函数
def progressive_training(normalization='log'):
    # 准备多阶段数据
    stages = prepare_data(stages=3)
    
    # 初始化模型
    model = DigitExtractor()
    
    # 存储所有训练阶段的指标
    all_train_losses = []
    all_val_accuracies = []
    
    # 阶段性训练
    for stage, (X_train, y_train, X_test, y_test, _, _) in enumerate(stages):
        print(f"\n=== 训练阶段 {stage+1} ===")
        print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
        print(f"数据范围: {X_train.min().item()} 到 {X_train.max().item()}")
        
        # 数据归一化
        X_train_norm, X_test_norm = normalize_data(X_train, X_test, normalization)
        
        # 创建数据加载器
        train_loader = create_batches(X_train_norm, y_train)
        test_loader = create_batches(X_test_norm, y_test)
        
        # 训练模型
        epochs = 20 if stage == 0 else 10  # 第一阶段训练更长时间
        lr = 0.001 / (stage + 1)  # 随着阶段增加，降低学习率
        
        model, train_losses, val_accuracies, _ = train_model(
            model, train_loader, test_loader, 
            epochs=epochs, 
            lr=lr,
            scheduler_type='cosine',
            patience=5
        )
        
        all_train_losses.extend(train_losses)
        all_val_accuracies.extend(val_accuracies)
        
        # 评估当前阶段
        test_accuracy = evaluate(model, test_loader)
        print(f'阶段 {stage+1} 最终测试集准确率: {test_accuracy:.4f}')
    
    # 在完整测试集上进行最终评估
    final_stage = stages[-1]  # 使用最后一个阶段的数据
    X_test, y_test = final_stage[2], final_stage[3]
    X_test_norm, _ = normalize_data(X_test, X_test, normalization)
    
    test_loader = create_batches(X_test_norm, y_test)
    final_accuracy = evaluate(model, test_loader)
    print(f'\n最终模型在完整测试集上的准确率: {final_accuracy:.4f}')
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(all_train_losses)
    plt.title('训练损失')
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    
    # 验证准确率
    plt.subplot(1, 2, 2)
    plt.plot(all_val_accuracies)
    plt.title('验证准确率')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    print('训练过程图已保存为 training_progress.png')
    
    # 测试特定例子
    test_numbers = [12345, 9, 80, 427, 1000, 9876543]
    
    # 预处理测试例子
    normalized_test_nums = []
    for num in test_numbers:
        # 使用与训练相同的方法归一化
        if normalization == 'log':
            x = torch.log1p(torch.tensor([[float(num)]], dtype=torch.float32))
            min_val = torch.log1p(X_train).min()
            max_val = torch.log1p(X_train).max()
            x_norm = (x - min_val) / (max_val - min_val + 1e-10)
        else:
            # 其他归一化方式...
            x_norm = torch.tensor([[float(num)]], dtype=torch.float32)  # 临时占位
            
        normalized_test_nums.append(x_norm)
    
    # 测试
    test_examples_accuracy = test_examples(model, test_numbers)
    
    return model, final_accuracy, test_examples_accuracy

# 执行渐进式训练
model, final_accuracy, test_examples_accuracy = progressive_training(normalization='log')

# 分析模型性能
print("\n模型性能分析:")
print(f"1. 数学公式模型 (x % 10): 100% 准确率")
print(f"2. 我们的神经网络模型: {final_accuracy:.2%} 准确率")
print(f"3. 特定测试例子的准确率: {test_examples_accuracy:.2%}")

if final_accuracy < 0.9:
    print("\n为什么模型准确率不够高:")
    print("1. 数学运算学习难度: 神经网络很难学习精确的模运算")
    print("2. 表示问题: 即使使用归一化，大数字的表示仍然具有挑战性")
    print("3. 网络限制: 全连接网络可能不是学习模运算的最佳结构")
    
    print("\n可能的进一步改进:")
    print("1. 使用特殊网络结构: 考虑使用循环神经网络(RNN)处理数字的各个位")
    print("2. 进一步的数据表示: 考虑将数字表示为数字序列而非单一数值")
    print("3. 更大的模型: 使用更多的隐藏层和神经元")
else:
    print("\n模型成功学习了如何提取最后一位！")
    
##改进版：
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

class CyclicModNet(nn.Module):
    """
    周期性感知神经网络 - 专门设计用于模10运算(提取最后一位数字)
    核心理念: 利用周期性三角函数精确捕捉模运算的周期性质
    """
    def __init__(self, num_frequencies=5):
        super().__init__()
        
        # 初始化频率参数 - 关键创新点
        # 包含一组可学习的频率，其中一些接近于模10的自然频率(0.1)
        initial_freqs = torch.tensor([0.1, 0.01, 0.001, 0.2, 0.05])[:num_frequencies]
        self.frequencies = nn.Parameter(initial_freqs)
        
        # 周期特征分类器
        self.classifier = nn.Sequential(
            nn.Linear(num_frequencies * 2, 128),  # 每个频率产生sin和cos两个特征
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)  # 输出10个类别(0-9)
        )
    
    def forward(self, x):
        # 生成周期特征
        cyclic_features = []
        
        for freq in self.frequencies:
            # 使用三角函数捕捉周期性
            # 关键直觉: 频率为0.1的正弦和余弦波可以完美捕捉模10的周期性
            sin_encoding = torch.sin(2 * math.pi * x * freq)
            cos_encoding = torch.cos(2 * math.pi * x * freq)
            cyclic_features.append(sin_encoding)
            cyclic_features.append(cos_encoding)
        
        # 组合所有周期特征
        combined_features = torch.cat(cyclic_features, dim=1)
        
        # 通过分类器网络
        logits = self.classifier(combined_features)
        
        return logits
    
    def predict(self, x):
        """用于单个数字预测"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, (int, float)):
                x = torch.tensor([[float(x)]], dtype=torch.float32)
            output = self(x)
            return torch.argmax(output, dim=1)

# 数据准备函数
def prepare_data(size=50000, max_num=1000000):
    """创建训练和测试数据"""
    # 生成包含广泛范围的随机数
    data = np.random.randint(0, max_num, size=size)
    
    # 计算每个数的最后一位(模10)
    targets = data % 10
    
    # 转换为PyTorch张量
    X = torch.tensor(data, dtype=torch.float32).reshape(-1, 1)
    y = torch.tensor(targets, dtype=torch.long)
    
    # 划分训练集和测试集 (80%/20%)
    split = int(0.8 * size)
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    return X_train, y_train, X_test, y_test

# 创建数据加载器
def create_batches(X, y, batch_size=256):
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练函数
def train_model(model, train_loader, test_loader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # 使用学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                    factor=0.5, patience=2)
    
    # 记录训练指标
    train_losses = []
    test_accuracies = []
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # 使用tqdm创建进度条
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for inputs, labels in progress_bar:
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 裁剪梯度
            optimizer.step()
            
            # 更新统计
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # 计算平均训练损失
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # 评估测试集
        test_accuracy = evaluate(model, test_loader)
        test_accuracies.append(test_accuracy)
        
        # 调整学习率
        scheduler.step(epoch_loss)
        
        # 打印当前学习的频率和性能
        print(f'Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f} - Accuracy: {test_accuracy:.4f}')
        print(f'Current frequencies: {model.frequencies.data}')
        
        # 如果已经达到很高的准确率，可以提前结束
        if test_accuracy > 0.99:
            print(f'已达到高准确率 {test_accuracy:.4f}，提前结束训练。')
            break
    
    return train_losses, test_accuracies

# 评估函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

# 测试特定例子
def test_examples(model, test_numbers):
    print("\n测试特定的例子:")
    correct = 0
    
    for num in test_numbers:
        prediction = model.predict(num).item()
        actual = num % 10
        
        result = "✓" if prediction == actual else "✗"
        if prediction == actual:
            correct += 1
            
        print(f"数字: {num}, 预测最后一位: {prediction}, 实际最后一位: {actual}, {result}")
    
    print(f"准确率: {correct}/{len(test_numbers)} = {correct/len(test_numbers):.2%}")

# 主函数
def main():
    # 设置随机种子以确保可重复性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 准备数据
    X_train, y_train, X_test, y_test = prepare_data()
    train_loader = create_batches(X_train, y_train)
    test_loader = create_batches(X_test, y_test)
    
    # 创建模型
    model = CyclicModNet(num_frequencies=5)
    
    # 训练模型
    train_losses, test_accuracies = train_model(model, train_loader, test_loader, epochs=10)
    
    # 测试特定例子
    test_numbers = [12345, 9, 80, 427, 1000, 9876543]
    test_examples(model, test_numbers)
    
    # 分析最终学习的频率参数
    print("\n最终学习的频率参数:")
    print(model.frequencies.data)
    
    # 绘制训练过程
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_progress.png')
    print("训练过程图已保存为 'training_progress.png'")

if __name__ == "__main__":
    main()