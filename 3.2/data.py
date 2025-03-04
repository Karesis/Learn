from datasets import load_dataset
from torch.utils.data import DataLoader

# 一步加载并配置
ds = load_dataset("zalando-datasets/fashion_mnist")

# 设置PyTorch格式
ds['train'].set_format(type='torch', columns=['image', 'label'])
ds['test'].set_format(type='torch', columns=['image', 'label'])
print(ds['train'])
print(ds['test'])

# 直接创建DataLoader
train_loader = DataLoader(ds['train'], batch_size=64, shuffle=True, num_workers=8)
test_loader = DataLoader(ds['test'], batch_size=64, shuffle=False, num_workers=8)










































