import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import seaborn as sns
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# 设置随机种子以确保结果可重复
np.random.seed(42)
torch.manual_seed(42)

# 绘图风格设置
plt.style.use('ggplot')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN']


# 1. 可视化全连接网络和LSTM的结构差异
def visualize_network_structures():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # 绘制全连接网络结构
    def draw_mlp(ax):
        # 绘制层
        layer_sizes = [4, 8, 8, 4]
        layer_positions = np.linspace(0.1, 0.9, len(layer_sizes))
        
        # 绘制节点
        for i, (pos, size) in enumerate(zip(layer_positions, layer_sizes)):
            y_positions = np.linspace(0.1, 0.9, size)
            
            for y in y_positions:
                circle = plt.Circle((pos, y), 0.02, color='skyblue', ec='blue', zorder=2)
                ax.add_artist(circle)
                
                # 如果不是第一层，连接到前一层的所有节点
                if i > 0:
                    prev_y_positions = np.linspace(0.1, 0.9, layer_sizes[i-1])
                    for prev_y in prev_y_positions:
                        ax.plot([layer_positions[i-1], pos], [prev_y, y], 'gray', alpha=0.5, zorder=1)
        
        # 添加输入和输出标签
        ax.text(0.05, 0.5, "输入\n数字", ha='center', va='center', fontsize=14)
        ax.text(0.95, 0.5, "预测\n尾数", ha='center', va='center', fontsize=14)
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("全连接神经网络 (MLP)", fontsize=16)
        
    # 绘制LSTM结构
    def draw_lstm(ax):
        # 绘制时间步
        timesteps = 5
        t_positions = np.linspace(0.1, 0.9, timesteps)
        
        # 绘制LSTM单元和状态
        for i, pos in enumerate(t_positions):
            # LSTM单元（框）
            rect = plt.Rectangle((pos-0.06, 0.4), 0.12, 0.2, color='lightgreen', ec='green', zorder=1)
            ax.add_artist(rect)
            ax.text(pos, 0.5, "LSTM", ha='center', va='center', fontsize=10)
            
            # 细胞状态线
            if i > 0:
                ax.arrow(t_positions[i-1]+0.06, 0.55, pos-t_positions[i-1]-0.12, 0, 
                         head_width=0.01, head_length=0.01, fc='orange', ec='orange', zorder=2)
                ax.text((t_positions[i-1]+0.06+pos-0.06)/2, 0.57, "C", color='orange', ha='center', fontsize=10)
            
            # 隐藏状态线
            if i > 0:
                ax.arrow(t_positions[i-1]+0.06, 0.45, pos-t_positions[i-1]-0.12, 0, 
                         head_width=0.01, head_length=0.01, fc='purple', ec='purple', zorder=2)
                ax.text((t_positions[i-1]+0.06+pos-0.06)/2, 0.43, "h", color='purple', ha='center', fontsize=10)
            
            # 输入
            if i < timesteps - 1:
                ax.arrow(pos, 0.35, 0, -0.1, head_width=0.01, head_length=0.01, fc='blue', ec='blue')
                ax.text(pos, 0.3, f"x{timesteps-i-1}", color='blue', ha='center')
            else:
                # 最后一个时间步连接到输出
                ax.arrow(pos, 0.6, 0, 0.1, head_width=0.01, head_length=0.01, fc='red', ec='red')
                ax.text(pos, 0.75, "预测尾数", color='red', ha='center')
                
        # 添加输入标签
        ax.text(0.05, 0.2, "输入数字序列", ha='center', fontsize=14)
        
        # 移除坐标轴
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("LSTM网络结构", fontsize=16)
    
    # 绘制两种网络结构
    draw_mlp(ax1)
    draw_lstm(ax2)
    
    plt.tight_layout()
    plt.savefig('network_structure_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 2. 可视化优化空间和局部最优点
def visualize_optimization_landscape():
    # 创建具有多个局部最小值的函数
    def complex_landscape(x, y):
        return np.sin(5*x)*np.cos(5*y) + np.sin(3*x)*np.cos(3*y) + x**2 + y**2
    
    def lstm_landscape(x, y):
        # 更简单的函数，代表LSTM的搜索空间
        return x**2 + y**2 + 0.2*np.sin(3*x)*np.cos(3*y)
    
    # 创建网格
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算Z值
    Z_complex = complex_landscape(X, Y)
    Z_lstm = lstm_landscape(X, Y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), subplot_kw={"projection": "3d"})
    
    # 绘制全连接网络的复杂优化空间
    surf1 = ax1.plot_surface(X, Y, Z_complex, cmap='viridis', alpha=0.8, linewidth=0)
    ax1.set_title("全连接网络的优化空间", fontsize=16)
    ax1.set_xlabel('参数1')
    ax1.set_ylabel('参数2')
    ax1.set_zlabel('损失')
    
    # 标注一些局部最优点
    local_minima = [(-1.5, -1.5), (-1, 1), (0.8, -0.5), (1.2, 1.2)]
    for lm in local_minima:
        ax1.scatter(lm[0], lm[1], complex_landscape(lm[0], lm[1]), color='red', s=50, marker='o')
    
    # 绘制LSTM的优化空间
    surf2 = ax2.plot_surface(X, Y, Z_lstm, cmap='plasma', alpha=0.8, linewidth=0)
    ax2.set_title("LSTM的优化空间（归纳偏置后）", fontsize=16)
    ax2.set_xlabel('参数1')
    ax2.set_ylabel('参数2')
    ax2.set_zlabel('损失')
    
    # 标注全局最优点
    ax2.scatter(0, 0, lstm_landscape(0, 0), color='red', s=100, marker='*')
    
    plt.tight_layout()
    plt.savefig('optimization_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()

# 3. 模拟训练过程，显示全连接网络和LSTM在序列数据上的学习曲线
def simulate_training_process():
    # 创建数据
    np.random.seed(42)
    epochs = 100
    
    # 模拟全连接网络在数字尾数问题上的训练过程
    mlp_acc = np.zeros(epochs)
    mlp_acc[:30] = np.linspace(10, 20, 30) + np.random.normal(0, 1, 30)
    mlp_acc[30:70] = np.linspace(20, 30, 40) + np.random.normal(0, 1.5, 40)
    mlp_acc[70:] = np.linspace(30, 35, 30) + np.random.normal(0, 1, 30)
    
    # 模拟LSTM在数字尾数问题上的训练过程
    lstm_acc = np.zeros(epochs)
    lstm_acc[:20] = np.linspace(10, 30, 20) + np.random.normal(0, 2, 20)
    lstm_acc[20:40] = np.linspace(30, 70, 20) + np.random.normal(0, 3, 20)
    lstm_acc[40:60] = np.linspace(70, 95, 20) + np.random.normal(0, 2, 20)
    lstm_acc[60:] = np.linspace(95, 100, 40) + np.random.normal(0, 0.5, 40)
    lstm_acc = np.clip(lstm_acc, 0, 100)  # 确保不超过100%
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    plt.plot(mlp_acc, 'b-', label='全连接网络', linewidth=2)
    plt.plot(lstm_acc, 'g-', label='LSTM', linewidth=2)
    plt.xlabel('训练轮次 (Epochs)')
    plt.ylabel('准确率 (%)')
    plt.title('模拟训练过程：全连接网络 vs LSTM')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mlp_acc, lstm_acc

# 4. 可视化模型对新数据的泛化能力
def visualize_generalization():
    # 创建测试数字
    test_numbers = np.array([42, 123, 7890, 5555, 9999, 10, 2025, 888, 54321, 98765])
    true_last_digits = np.array([2, 3, 0, 5, 9, 0, 5, 8, 1, 5])
    
    # 模拟全连接网络的预测（不太准确）
    mlp_predictions = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5])
    
    # 模拟LSTM的预测（接近完美）
    lstm_predictions = true_last_digits.copy()
    
    # 创建可视化
    plt.figure(figsize=(14, 6))
    
    # 数据点位置
    x = np.arange(len(test_numbers))
    width = 0.2
    
    # 绘制真实尾数、MLP预测和LSTM预测
    plt.bar(x - width, true_last_digits, width, label='真实尾数', color='gray')
    plt.bar(x, mlp_predictions, width, label='全连接网络预测', color='skyblue')
    plt.bar(x + width, lstm_predictions, width, label='LSTM预测', color='lightgreen')
    
    # 添加数字标签
    plt.xticks(x, [str(num) for num in test_numbers], rotation=45)
    
    plt.xlabel('测试数字')
    plt.ylabel('尾数 (0-9)')
    plt.title('全连接网络与LSTM在测试数字上的表现对比')
    plt.legend()
    plt.tight_layout()
    plt.savefig('generalization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 5. 可视化LSTM内部状态对序列处理的影响
def visualize_lstm_internal_state():
    # 模拟LSTM处理数字序列的内部状态变化
    # 假设我们处理数字"54321"，按照从右到左的顺序
    sequence = [1, 2, 3, 4, 5]
    seq_len = len(sequence)
    
    # 模拟简化的LSTM内部状态
    forget_gates = [0.2, 0.3, 0.4, 0.5, 0.6]  # 遗忘门激活值
    input_gates = [0.9, 0.8, 0.7, 0.6, 0.5]   # 输入门激活值
    cell_states = [0.1, 0.3, 0.4, 0.5, 0.6]   # 细胞状态
    hidden_states = [0.9, 0.7, 0.5, 0.3, 0.1] # 隐藏状态
    outputs = [0.9, 0.2, 0.1, 0.05, 0.01]     # 输出值（关注第一个时间步）
    
    # 创建图形
    fig, axs = plt.subplots(5, 1, figsize=(12, 10), sharex=True)
    
    # 绘制每个时间步的值
    timesteps = range(1, seq_len + 1)
    
    # 为了美观，时间步反转显示（从右到左的处理顺序）
    display_seq = sequence.copy()
    
    axs[0].bar(timesteps, forget_gates, color='red', alpha=0.7)
    axs[0].set_ylabel('遗忘门')
    axs[0].set_title('LSTM处理数字54321的内部状态变化（从右到左处理：1→2→3→4→5）')
    
    axs[1].bar(timesteps, input_gates, color='green', alpha=0.7)
    axs[1].set_ylabel('输入门')
    
    axs[2].bar(timesteps, cell_states, color='orange', alpha=0.7)
    axs[2].set_ylabel('细胞状态')
    
    axs[3].bar(timesteps, hidden_states, color='purple', alpha=0.7)
    axs[3].set_ylabel('隐藏状态')
    
    axs[4].bar(timesteps, outputs, color='blue', alpha=0.7)
    axs[4].set_ylabel('输出激活')
    axs[4].set_xticks(timesteps)
    axs[4].set_xticklabels(display_seq)
    axs[4].set_xlabel('处理的数字（时间步）')
    
    # 添加垂直线突出显示第一个时间步（尾数）
    for ax in axs:
        ax.axvline(x=1, color='black', linestyle='--', alpha=0.5)
        ax.text(1.1, ax.get_ylim()[1]*0.9, '尾数位置', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('lstm_internal_state.png', dpi=300, bbox_inches='tight')
    plt.show()

# 6. 演示LSTM如何通过结构避免局部最优
def visualize_training_paths():
    # 定义优化路径
    np.random.seed(42)
    steps = 20
    
    # MLP的多个训练路径（大多陷入局部最优）
    mlp_paths = []
    # 路径1：陷入局部最优
    path1_x = np.linspace(-1.5, -1.4, steps) + np.random.normal(0, 0.03, steps)
    path1_y = np.linspace(-1.5, -1.4, steps) + np.random.normal(0, 0.03, steps)
    mlp_paths.append((path1_x, path1_y))
    
    # 路径2：陷入另一个局部最优
    path2_x = np.linspace(-1.0, -0.9, steps) + np.random.normal(0, 0.02, steps)
    path2_y = np.linspace(1.0, 0.9, steps) + np.random.normal(0, 0.02, steps)
    mlp_paths.append((path2_x, path2_y))
    
    # 路径3：陷入另一个局部最优
    path3_x = np.linspace(0.8, 0.7, steps) + np.random.normal(0, 0.02, steps)
    path3_y = np.linspace(-0.5, -0.4, steps) + np.random.normal(0, 0.02, steps)
    mlp_paths.append((path3_x, path3_y))
    
    # LSTM训练路径（通常收敛到全局最优）
    lstm_path_x = np.linspace(0.8, 0.1, steps) + np.random.normal(0, 0.05, steps)
    lstm_path_y = np.linspace(0.8, 0.1, steps) + np.random.normal(0, 0.05, steps)
    
    # 创建网格
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    
    # 计算Z值（使用前面定义的函数）
    def complex_landscape(x, y):
        return np.sin(5*x)*np.cos(5*y) + np.sin(3*x)*np.cos(3*y) + x**2 + y**2
    
    def lstm_landscape(x, y):
        return x**2 + y**2 + 0.2*np.sin(3*x)*np.cos(3*y)
    
    Z_complex = complex_landscape(X, Y)
    Z_lstm = lstm_landscape(X, Y)
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 绘制MLP优化曲面和路径
    contour1 = ax1.contourf(X, Y, Z_complex, 50, cmap='viridis', alpha=0.8)
    for i, (path_x, path_y) in enumerate(mlp_paths):
        ax1.plot(path_x, path_y, 'o-', color=f'C{i}', 
                 label=f'训练路径 {i+1}', linewidth=2, markersize=5)
        # 标记终点
        ax1.plot(path_x[-1], path_y[-1], 'X', color=f'C{i}', markersize=10)
    
    ax1.set_title("全连接网络：多个训练路径陷入局部最优", fontsize=14)
    ax1.set_xlabel('参数维度1')
    ax1.set_ylabel('参数维度2')
    ax1.legend()
    fig.colorbar(contour1, ax=ax1, label='损失函数值')
    
    # 绘制LSTM优化曲面和路径
    contour2 = ax2.contourf(X, Y, Z_lstm, 50, cmap='plasma', alpha=0.8)
    ax2.plot(lstm_path_x, lstm_path_y, 'o-', color='red', 
             label='LSTM训练路径', linewidth=2, markersize=5)
    # 标记终点
    ax2.plot(lstm_path_x[-1], lstm_path_y[-1], 'X', color='red', markersize=10)
    ax2.set_title("LSTM：结构化偏置引导训练路径避开局部最优", fontsize=14)
    ax2.set_xlabel('参数维度1')
    ax2.set_ylabel('参数维度2')
    ax2.legend()
    fig.colorbar(contour2, ax=ax2, label='损失函数值')
    
    plt.tight_layout()
    plt.savefig('training_paths.png', dpi=300, bbox_inches='tight')
    plt.show()

# 7. 创建实际的小型MLP和LSTM模型，比较其在尾数任务上的表现
def build_and_compare_models():
    # 创建数据集
    n_samples = 1000
    
    # 随机生成数字
    numbers = np.random.randint(0, 100000, size=n_samples)
    # 提取尾数
    last_digits = np.array([int(str(num)[-1]) for num in numbers])
    
    # 分为训练集和测试集
    train_size = int(0.8 * n_samples)
    train_numbers, test_numbers = numbers[:train_size], numbers[train_size:]
    train_digits, test_digits = last_digits[:train_size], last_digits[train_size:]
    
    # 准备MLP输入（归一化数字）
    X_train_mlp = torch.tensor(train_numbers, dtype=torch.float32).view(-1, 1) / 100000.0
    X_test_mlp = torch.tensor(test_numbers, dtype=torch.float32).view(-1, 1) / 100000.0
    y_train = torch.tensor(train_digits, dtype=torch.long)
    y_test = torch.tensor(test_digits, dtype=torch.long)
    
    # 准备LSTM输入（数字序列，独热编码）
    def create_sequence_data(numbers):
        max_len = 5  # 考虑数字的最后5位
        sequences = []
        
        for num in numbers:
            num_str = str(num).zfill(max_len)[-max_len:]  # 确保至少5位数，取最后5位
            seq = np.zeros((max_len, 10))  # 10个类别（0-9）的独热编码
            
            for i, digit in enumerate(num_str):
                seq[i, int(digit)] = 1.0
                
            sequences.append(seq)
            
        return torch.tensor(np.array(sequences), dtype=torch.float32)
    
    X_train_lstm = create_sequence_data(train_numbers)
    X_test_lstm = create_sequence_data(test_numbers)
    
    # 定义MLP模型
    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 10)
            )
            
        def forward(self, x):
            return self.layers(x)
    
    # 定义LSTM模型
    class SimpleLSTM(nn.Module):
        def __init__(self):
            super(SimpleLSTM, self).__init__()
            self.lstm = nn.LSTM(10, 32, batch_first=True)
            self.fc = nn.Linear(32, 10)
            
        def forward(self, x):
            lstm_out, _ = self.lstm(x)
            # 只使用最后一个时间步的输出
            return self.fc(lstm_out[:, -1, :])
    
    # 初始化模型
    mlp_model = MLP()
    lstm_model = SimpleLSTM()
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
    lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)
    
    # 训练模型
    def train_model(model, X, y, optimizer, epochs=30, batch_size=32):
        accuracies = []
        for epoch in range(epochs):
            running_loss = 0.0
            correct = 0
            
            # 批处理训练
            permutation = torch.randperm(X.size()[0])
            for i in range(0, X.size()[0], batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X[indices], y[indices]
                
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # 计算准确率
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == batch_y).sum().item()
            
            accuracy = 100 * correct / X.size()[0]
            accuracies.append(accuracy)
            
        return accuracies
    
    # 训练模型
    mlp_accuracies = train_model(mlp_model, X_train_mlp, y_train, mlp_optimizer)
    lstm_accuracies = train_model(lstm_model, X_train_lstm, y_train, lstm_optimizer)
    
    # 评估模型
    def evaluate_model(model, X, y):
        with torch.no_grad():
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = 100 * (predicted == y).sum().item() / y.size(0)
        return accuracy, predicted.numpy()
    
    mlp_accuracy, mlp_predictions = evaluate_model(mlp_model, X_test_mlp, y_test)
    lstm_accuracy, lstm_predictions = evaluate_model(lstm_model, X_test_lstm, y_test)
    
    print(f"MLP测试准确率: {mlp_accuracy:.2f}%")
    print(f"LSTM测试准确率: {lstm_accuracy:.2f}%")
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(mlp_accuracies) + 1)
    plt.plot(epochs, mlp_accuracies, 'b-', label='MLP准确率')
    plt.plot(epochs, lstm_accuracies, 'g-', label='LSTM准确率')
    plt.xlabel('训练轮次')
    plt.ylabel('准确率 (%)')
    plt.title('实际模型训练过程对比')
    plt.legend()
    plt.grid(True)
    plt.savefig('real_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return mlp_predictions, lstm_predictions, test_digits, test_numbers

# 执行所有可视化函数
def run_all_visualizations():
    print("1. 可视化网络结构差异...")
    visualize_network_structures()
    
    print("\n2. 可视化优化空间和局部最优点...")
    visualize_optimization_landscape()
    
    print("\n3. 模拟训练过程...")
    simulate_training_process()
    
    print("\n4. 可视化泛化能力...")
    visualize_generalization()
    
    print("\n5. 可视化LSTM内部状态...")
    visualize_lstm_internal_state()
    
    print("\n6. 可视化训练路径...")
    visualize_training_paths()
    
    print("\n7. 构建并比较实际模型...")
    mlp_preds, lstm_preds, true_digits, test_nums = build_and_compare_models()
    
    print("\n所有可视化完成！")

if __name__ == "__main__":
    run_all_visualizations()
    