import torch
import torch.nn as nn
from data import train_loader, test_loader
from nn import MultiFashionExpertWithLSTM
from train import train_model
from eval import get_validation_loss, detailed_evaluation

def main():
    # 设置随机种子，保证结果可复现
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # 创建模型
    model = MultiFashionExpertWithLSTM(num_classes=10, in_channels=1, num_experts=3)
    
    # 定义一个用于获取验证损失的函数
    criterion = nn.CrossEntropyLoss()
    
    def val_loss_fn(model):
        return get_validation_loss(model, test_loader, criterion)
    
    # 训练模型（完全分离训练和评估过程）
    trained_model, train_stats = train_model(
        model=model,
        train_loader=train_loader,
        val_loss_fn=val_loss_fn,  # 传入验证函数以获取验证损失
        num_epochs=10,
        lr=0.001,
        save_path="./models/fashion_mnist_model"  # 可选，设置为None则不保存模型
    )
    
    # 训练完成后进行详细评估
    print("\n=== 最终模型评估 ===")
    eval_results = detailed_evaluation(trained_model, test_loader)
    
    # 打印最终准确率
    print(f"\n最终测试集准确率: {eval_results['accuracy']:.2f}%")
    
if __name__ == "__main__":
    main()