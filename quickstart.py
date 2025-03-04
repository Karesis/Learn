#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动创建神经网络项目模板结构的脚本。
运行此脚本将创建完整的目录结构和必要的空文件。
"""

import os
import argparse
import shutil
from typing import List, Dict, Tuple, Optional, Union


def create_file(file_path: str, content: str = "") -> None:
    """创建文件并写入内容"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"创建文件: {file_path}")


def create_directory(dir_path: str) -> None:
    """创建目录"""
    os.makedirs(dir_path, exist_ok=True)
    print(f"创建目录: {dir_path}")


def create_init_file(dir_path: str) -> None:
    """在指定目录创建__init__.py文件"""
    init_path = os.path.join(dir_path, "__init__.py")
    create_file(init_path, "# -*- coding: utf-8 -*-\n")


def create_project_structure(project_name: str, base_dir: str = ".") -> None:
    """创建完整的项目结构"""
    
    # 项目根目录
    project_dir = os.path.join(base_dir, project_name)
    
    # 首先清除已存在的目录(如果指定了)
    if os.path.exists(project_dir):
        print(f"警告: 项目目录 '{project_dir}' 已存在")
        choice = input("是否删除现有项目目录? [y/N]: ").strip().lower()
        if choice == 'y':
            shutil.rmtree(project_dir)
            print(f"已删除: {project_dir}")
        else:
            print("取消操作")
            return
    
    # 创建项目根目录
    create_directory(project_dir)
    
    # 创建项目子目录
    directories = [
        "",  # 项目根目录
        "config",
        "data",
        "models",
        "models/layers",
        "trainers",
        "evaluators",
        "utils",
        "app",
    ]
    
    # 创建所有目录并添加__init__.py
    for dir_path in directories:
        full_path = os.path.join(project_dir, dir_path)
        create_directory(full_path)
        create_init_file(full_path)
    
    # 创建必要的文件
    files = [
        # 配置文件
        ("config/config.py", "# 配置类和默认配置\n"),
        ("config/config.yaml", "# 项目配置\n"),
        
        # 数据处理
        ("data/dataset.py", "# 数据集类\n"),
        ("data/dataloader.py", "# 数据加载器\n"),
        ("data/preprocessor.py", "# 数据预处理\n"),
        
        # 模型定义
        ("models/base_model.py", "# 基础模型类\n"),
        ("models/my_model.py", "# 具体模型实现\n"),
        ("models/layers/custom_layers.py", "# 自定义层\n"),
        
        # 训练相关
        ("trainers/base_trainer.py", "# 基础训练器类\n"),
        ("trainers/my_trainer.py", "# 具体训练实现\n"),
        ("trainers/callbacks.py", "# 训练回调函数\n"),
        
        # 评估相关
        ("evaluators/base_evaluator.py", "# 基础评估器类\n"),
        ("evaluators/my_evaluator.py", "# 具体评估实现\n"),
        ("evaluators/metrics.py", "# 评估指标\n"),
        
        # 工具函数
        ("utils/logger.py", "# 日志工具\n"),
        ("utils/visualization.py", "# 可视化工具\n"),
        ("utils/utils.py", "# 通用工具函数\n"),
        
        # 应用入口
        ("app/api.py", "# API接口\n"),
        ("app/serve.py", "# 模型部署服务\n"),
        ("app/ui.py", "# 用户界面(如果有)\n"),
        
        # 主入口
        ("main.py", "# 开发入口(训练和评估)\n"),
        
        # 项目文档
        ("README.md", f"# {project_name}\n\n项目描述\n"),
        ("requirements.txt", "torch>=1.10.0\ntransformers>=4.12.0\n"),
    ]
    
    # 创建所有文件
    for file_path, content in files:
        full_path = os.path.join(project_dir, file_path)
        create_file(full_path, content)
    
    print(f"\n项目结构创建完成: {project_dir}")
    print(f"共创建 {len(directories)} 个目录和 {len(files)} 个文件")


def main():
    """脚本主函数"""
    parser = argparse.ArgumentParser(description='创建神经网络项目模板结构')
    parser.add_argument('--name', '-n', type=str, default='neural_network_project', 
                        help='项目名称(默认: neural_network_project)')
    parser.add_argument('--dir', '-d', type=str, default='.', 
                        help='基础目录，将在其中创建项目(默认: 当前目录)')
    
    args = parser.parse_args()
    create_project_structure(args.name, args.dir)


if __name__ == "__main__":
    main()