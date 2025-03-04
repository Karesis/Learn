# 模型部署服务
import os
import json
import torch
import argparse
from typing import Dict, List, Any, Union

from transformers import AutoTokenizer

from neural_network_project.models.base_model import BaseModel
from neural_network_project.config.config import Config, load_config
from neural_network_project.utils.logger import get_logger


logger = get_logger(__name__)


class ModelServer:
    """模型服务器"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化模型服务器
        
        Args:
            model_path: 模型路径
            device: 设备，'cpu'或'cuda'
        """
        self.model_path = model_path
        self.device = device
        
        # 加载模型
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # 加载标签映射
        self.label_map = self._load_label_map()
        
        # 反转标签映射
        self.id_to_label = {v: k for k, v in self.label_map.items()}
    
    def _load_model(self) -> BaseModel:
        """加载模型"""
        return BaseModel.from_pretrained(self.model_path)
    
    def _load_label_map(self) -> Dict[str, int]:
        """加载标签映射"""
        label_map_path = os.path.join(self.model_path, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            logger.warning("未找到标签映射文件")
            return {}
    
    def predict(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        单个文本预测
        
        Args:
            text: 输入文本
            **kwargs: 额外参数
            
        Returns:
            预测结果
        """
        # 使用tokenizer处理文本
        inputs = self.tokenizer(
            text,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 将输入移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取logits
        logits = outputs["logits"][0].cpu().numpy()
        
        # 获取预测类别
        pred_id = int(logits.argmax())
        
        # 获取预测概率
        probs = torch.nn.functional.softmax(outputs["logits"][0], dim=0).cpu().numpy()
        
        # 获取标签名称
        pred_label = self.id_to_label.get(pred_id, str(pred_id))
        
        # 整理所有标签的概率
        all_probs = {
            self.id_to_label.get(i, str(i)): float(prob)
            for i, prob in enumerate(probs)
        }
        
        return {
            "text": text,
            "prediction": pred_label,
            "confidence": float(probs[pred_id]),
            "probabilities": all_probs
        }
    
    def batch_predict(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        批量文本预测
        
        Args:
            texts: 输入文本列表
            **kwargs: 额外参数
            
        Returns:
            预测结果列表
        """
        # 使用tokenizer处理文本批次
        inputs = self.tokenizer(
            texts,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # 将输入移到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # 获取logits
        logits = outputs["logits"].cpu().numpy()
        
        # 获取预测类别
        pred_ids = logits.argmax(axis=1)
        
        # 获取预测概率
        probs = torch.nn.functional.softmax(outputs["logits"], dim=1).cpu().numpy()
        
        # 整理结果
        results = []
        for i, text in enumerate(texts):
            pred_id = int(pred_ids[i])
            pred_label = self.id_to_label.get(pred_id, str(pred_id))
            
            # 整理所有标签的概率
            all_probs = {
                self.id_to_label.get(j, str(j)): float(prob)
                for j, prob in enumerate(probs[i])
            }
            
            results.append({
                "text": text,
                "prediction": pred_label,
                "confidence": float(probs[i, pred_id]),
                "probabilities": all_probs
            })
        
        return results


def start_http_server(model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """
    启动HTTP服务器
    
    Args:
        model_path: 模型路径
        host: 主机地址
        port: 端口号
    """
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        logger.error("未安装Flask，请使用pip install flask安装")
        return
    
    app = Flask(__name__)
    
    # 初始化模型服务器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    server = ModelServer(model_path, device)
    
    @app.route("/predict", methods=["POST"])
    def predict():
        """单个文本预测接口"""
        data = request.json
        
        if not data or "text" not in data:
            return jsonify({"error": "缺少text字段"}), 400
        
        result = server.predict(data["text"])
        return jsonify(result)
    
    @app.route("/batch_predict", methods=["POST"])
    def batch_predict():
        """批量文本预测接口"""
        data = request.json
        
        if not data or "texts" not in data:
            return jsonify({"error": "缺少texts字段"}), 400
        
        results = server.batch_predict(data["texts"])
        return jsonify({"results": results})
    
    @app.route("/health", methods=["GET"])
    def health():
        """健康检查接口"""
        return jsonify({"status": "ok"})
    
    logger.info(f"启动HTTP服务器，监听 {host}:{port}")
    app.run(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型服务器")
    parser.add_argument("--model", "-m", type=str, required=True, help="模型路径")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="主机地址")
    parser.add_argument("--port", "-p", type=int, default=8000, help="端口号")
    
    args = parser.parse_args()
    
    start_http_server(args.model, args.host, args.port)