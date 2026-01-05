"""
模型评估模块 - 增强版（多核优化）
提供全方位的模型性能评估，包括细分指标、错误分析和模型对比
支持多核并行处理，大幅提升评估速度
"""

import sys
from pathlib import Path
import json
import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from defense.config import DefenseConfig


def _process_single_sample(args):
    """
    处理单个样本的预测（用于多进程）
    
    Args:
        args: (样本索引, 样本数据, 预测函数)
        
    Returns:
        预测结果字典
    """
    idx, sample, predict_func = args
    prompt = sample["prompt"]
    true_label = sample["label"]
    category = sample.get("category", "未分类")
    difficulty = sample.get("difficulty", "medium")
    
    try:
        # 调用预测函数
        pred_label, confidence = predict_func(prompt)
        
        return {
            "idx": idx,
            "prompt": prompt,
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": confidence,
            "category": category,
            "difficulty": difficulty,
            "correct": pred_label == true_label,
            "error": None
        }
        
    except Exception as e:
        return {
            "idx": idx,
            "prompt": prompt,
            "true_label": true_label,
            "pred_label": 0,  # 默认SAFE
            "confidence": 0.0,
            "category": category,
            "difficulty": difficulty,
            "correct": False,
            "error": str(e)
        }


class ImprovedModelEvaluator:
    """增强版模型评估器（支持多核并行）"""
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        初始化评估器
        
        Args:
            num_workers: 并行工作进程数，默认为CPU核心数
        """
        self.test_data_path = DefenseConfig.TEST_DATA_PATH
        self.refusal_keywords = DefenseConfig.EVALUATION_CONFIG["refusal_keywords"]
        self.results_dir = DefenseConfig.EVAL_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置工作进程数（静默设置，在实际使用时才显示）
        self.num_workers = num_workers if num_workers else cpu_count()
        
        # 存储评估结果
        self.test_data = []
        self.predictions = []
    
    def _clean_json_string(self, text: str) -> str:
        """清洗JSON字符串中的无效转义字符"""
        # 移除ANSI转义序列
        text = re.sub(r'\\e\[[0-9;]*m', '', text)
        text = re.sub(r'\\x1b\[[0-9;]*m', '', text)
        text = re.sub(r'\\033\[[0-9;]*m', '', text)
        
        # 移除其他无效的转义序列
        def replace_invalid_escape(match):
            full_match = match.group(0)
            escaped_char = match.group(1)
            
            # 检查是否是有效的JSON转义字符
            if escaped_char in ['"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u']:
                return full_match
            
            # 对于无效的转义，移除反斜杠
            return escaped_char
        
        text = re.sub(r'\\(.)', replace_invalid_escape, text)
        return text
    
    def load_test_data(self) -> List[Dict]:
        """加载测试数据"""
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {self.test_data_path}")
        
        test_data = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cleaned_line = self._clean_json_string(line)
                    data = json.loads(cleaned_line)
                    
                    # 确保必要字段存在
                    if "category" not in data:
                        data["category"] = "未分类"
                    if "difficulty" not in data:
                        data["difficulty"] = "medium"
                    
                    test_data.append(data)
                except json.JSONDecodeError as e:
                    print(f"⚠ 第 {line_num} 行JSON解析失败: {e}")
                    continue
        
        print(f"✓ 已加载 {len(test_data)} 条测试数据")
        return test_data
    
    def evaluate_model(self, model_name: str, predict_func, 
                      save_predictions: bool = True,
                      use_multiprocessing: bool = True) -> Dict:
        """
        评估单个模型（支持多核并行）
        
        Args:
            model_name: 模型名称（如"SFT"或"DPO"）
            predict_func: 预测函数，接收prompt返回(pred_label, confidence)
            save_predictions: 是否保存预测结果
            use_multiprocessing: 是否使用多进程并行（默认True）
            
        Returns:
            评估结果字典
        """
        print("\n" + "=" * 70)
        print(f"评估模型: {model_name}")
        print("=" * 70)
        
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        if use_multiprocessing and self.num_workers > 1:
            print(f"🚀 使用多核并行处理 ({self.num_workers} 个进程)")
            predictions = self._evaluate_parallel(test_data, predict_func)
        else:
            print(f"📝 使用单进程处理")
            predictions = self._evaluate_sequential(test_data, predict_func)
        
        print(f"✓ {model_name} 模型评估完成")
        
        # 检查错误
        error_count = sum(1 for p in predictions if p.get("error"))
        if error_count > 0:
            print(f"⚠️ 警告: {error_count} 个样本预测失败")
        
        # 计算各类指标
        results = self._calculate_all_metrics(model_name, predictions)
        
        # 保存预测结果
        if save_predictions:
            self._save_predictions(model_name, predictions)
        
        return results
    
    def _evaluate_sequential(self, test_data: List[Dict], predict_func) -> List[Dict]:
        """
        单进程顺序评估
        
        Args:
            test_data: 测试数据列表
            predict_func: 预测函数
            
        Returns:
            预测结果列表
        """
        predictions = []
        
        for i, sample in enumerate(test_data, 1):
            prompt = sample["prompt"]
            true_label = sample["label"]
            category = sample.get("category", "未分类")
            difficulty = sample.get("difficulty", "medium")
            
            try:
                # 调用预测函数
                pred_label, confidence = predict_func(prompt)
                
                predictions.append({
                    "prompt": prompt,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "confidence": confidence,
                    "category": category,
                    "difficulty": difficulty,
                    "correct": pred_label == true_label
                })
                
            except Exception as e:
                print(f"⚠ 样本 {i} 预测失败: {e}")
                predictions.append({
                    "prompt": prompt,
                    "true_label": true_label,
                    "pred_label": 0,  # 默认SAFE
                    "confidence": 0.0,
                    "category": category,
                    "difficulty": difficulty,
                    "correct": False
                })
            
            # 打印进度
            if i % 50 == 0:
                print(f"进度: {i}/{len(test_data)}")
        
        return predictions
    
    def _evaluate_parallel(self, test_data: List[Dict], predict_func) -> List[Dict]:
        """
        多进程并行评估
        
        Args:
            test_data: 测试数据列表
            predict_func: 预测函数
            
        Returns:
            预测结果列表
        """
        total_samples = len(test_data)
        
        # 准备参数：(索引, 样本, 预测函数)
        args_list = [(i, sample, predict_func) for i, sample in enumerate(test_data)]
        
        # 计算合适的chunk大小
        chunk_size = max(1, total_samples // (self.num_workers * 4))
        
        print(f"📦 总样本数: {total_samples}, Chunk大小: {chunk_size}")
        
        # 使用进程池并行处理
        with Pool(processes=self.num_workers) as pool:
            results = []
            # 使用imap_unordered获得更好的进度显示
            for i, result in enumerate(pool.imap_unordered(_process_single_sample, args_list, chunksize=chunk_size), 1):
                results.append(result)
                # 打印进度
                if i % 50 == 0 or i == total_samples:
                    print(f"进度: {i}/{total_samples} ({i*100//total_samples}%)")
        
        # 按照原始索引排序，保持顺序一致
        results.sort(key=lambda x: x["idx"])
        
        # 移除idx字段（仅用于排序）
        predictions = []
        for result in results:
            pred = {k: v for k, v in result.items() if k not in ["idx", "error"]}
            predictions.append(pred)
        
        return predictions
    
    def _calculate_all_metrics(self, model_name: str, 
                               predictions: List[Dict]) -> Dict:
        """计算所有评估指标"""
        
        y_true = [p["true_label"] for p in predictions]
        y_pred = [p["pred_label"] for p in predictions]
        
        results = {
            "model_name": model_name,
            "overall": self._calculate_overall_metrics(y_true, y_pred),
            "by_category": self._calculate_category_metrics(predictions),
            "by_difficulty": self._calculate_difficulty_metrics(predictions),
            "error_analysis": self._analyze_errors(predictions),
            "confidence_stats": self._calculate_confidence_stats(predictions)
        }
        
        return results
    
    def _calculate_overall_metrics(self, y_true: List[int], 
                                   y_pred: List[int]) -> Dict:
        """计算总体指标"""
        
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        metrics["confusion_matrix"] = {
            "TN": int(tn),  # 正确识别为安全
            "FP": int(fp),  # 误判为不安全（假阳性）
            "FN": int(fn),  # 漏判（假阴性）- 最危险！
            "TP": int(tp),  # 正确识别为不安全
        }
        
        # 关键安全指标
        metrics["false_negative_rate"] = fn / (fn + tp) if (fn + tp) > 0 else 0
        metrics["false_positive_rate"] = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # 特异性和敏感性
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics["sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        return metrics
    
    def _calculate_category_metrics(self, predictions: List[Dict]) -> Dict:
        """按类别计算指标"""
        
        category_data = defaultdict(lambda: {"y_true": [], "y_pred": []})
        
        for pred in predictions:
            cat = pred["category"]
            category_data[cat]["y_true"].append(pred["true_label"])
            category_data[cat]["y_pred"].append(pred["pred_label"])
        
        category_metrics = {}
        for cat, data in category_data.items():
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            
            category_metrics[cat] = {
                "count": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
            }
        
        return category_metrics
    
    def _calculate_difficulty_metrics(self, predictions: List[Dict]) -> Dict:
        """按难度计算指标"""
        
        difficulty_data = defaultdict(lambda: {"y_true": [], "y_pred": []})
        
        for pred in predictions:
            diff = pred["difficulty"]
            difficulty_data[diff]["y_true"].append(pred["true_label"])
            difficulty_data[diff]["y_pred"].append(pred["pred_label"])
        
        difficulty_metrics = {}
        for diff, data in difficulty_data.items():
            y_true = data["y_true"]
            y_pred = data["y_pred"]
            
            difficulty_metrics[diff] = {
                "count": len(y_true),
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1_score": f1_score(y_true, y_pred, zero_division=0),
            }
        
        return difficulty_metrics
    
    def _analyze_errors(self, predictions: List[Dict]) -> Dict:
        """错误分析"""
        
        false_positives = []  # 误判为UNSAFE
        false_negatives = []  # 漏判（UNSAFE被判为SAFE）
        
        for pred in predictions:
            if not pred["correct"]:
                error_info = {
                    "prompt": pred["prompt"][:100] + "..." if len(pred["prompt"]) > 100 else pred["prompt"],
                    "true_label": pred["true_label"],
                    "pred_label": pred["pred_label"],
                    "category": pred["category"],
                    "difficulty": pred["difficulty"],
                    "confidence": pred["confidence"]
                }
                
                if pred["true_label"] == 0 and pred["pred_label"] == 1:
                    false_positives.append(error_info)
                elif pred["true_label"] == 1 and pred["pred_label"] == 0:
                    false_negatives.append(error_info)
        
        # 按类别统计错误
        fp_by_category = defaultdict(int)
        fn_by_category = defaultdict(int)
        
        for fp in false_positives:
            fp_by_category[fp["category"]] += 1
        for fn in false_negatives:
            fn_by_category[fn["category"]] += 1
        
        return {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "fp_count": len(false_positives),
            "fn_count": len(false_negatives),
            "fp_by_category": dict(fp_by_category),
            "fn_by_category": dict(fn_by_category),
        }
    
    def _calculate_confidence_stats(self, predictions: List[Dict]) -> Dict:
        """计算置信度统计"""
        
        confidences = [p["confidence"] for p in predictions]
        correct_confidences = [p["confidence"] for p in predictions if p["correct"]]
        incorrect_confidences = [p["confidence"] for p in predictions if not p["correct"]]
        
        return {
            "mean": np.mean(confidences) if confidences else 0,
            "std": np.std(confidences) if confidences else 0,
            "min": np.min(confidences) if confidences else 0,
            "max": np.max(confidences) if confidences else 0,
            "correct_mean": np.mean(correct_confidences) if correct_confidences else 0,
            "incorrect_mean": np.mean(incorrect_confidences) if incorrect_confidences else 0,
        }
    
    def compare_models(self, sft_results: Dict, dpo_results: Dict) -> Dict:
        """
        对比两个模型的性能
        
        Args:
            sft_results: SFT模型评估结果
            dpo_results: DPO模型评估结果
            
        Returns:
            对比结果字典
        """
        print("\n" + "=" * 70)
        print("模型对比分析: SFT vs DPO")
        print("=" * 70)
        
        comparison = {
            "overall_improvement": self._compare_overall(sft_results, dpo_results),
            "category_improvement": self._compare_by_category(sft_results, dpo_results),
            "difficulty_improvement": self._compare_by_difficulty(sft_results, dpo_results),
            "error_reduction": self._compare_errors(sft_results, dpo_results),
        }
        
        return comparison
    
    def _compare_overall(self, sft_results: Dict, dpo_results: Dict) -> Dict:
        """对比总体指标"""
        
        sft_overall = sft_results["overall"]
        dpo_overall = dpo_results["overall"]
        
        metrics_to_compare = [
            "accuracy", "precision", "recall", "f1_score",
            "false_negative_rate", "false_positive_rate"
        ]
        
        comparison = {}
        for metric in metrics_to_compare:
            sft_val = sft_overall[metric]
            dpo_val = dpo_overall[metric]
            
            # 对于rate类指标，越低越好
            if "rate" in metric:
                improvement = (sft_val - dpo_val) / sft_val * 100 if sft_val > 0 else 0
            else:
                improvement = (dpo_val - sft_val) / sft_val * 100 if sft_val > 0 else 0
            
            comparison[metric] = {
                "sft": sft_val,
                "dpo": dpo_val,
                "improvement": improvement,
                "absolute_change": dpo_val - sft_val
            }
        
        return comparison
    
    def _compare_by_category(self, sft_results: Dict, dpo_results: Dict) -> Dict:
        """按类别对比"""
        
        sft_cat = sft_results["by_category"]
        dpo_cat = dpo_results["by_category"]
        
        comparison = {}
        all_categories = set(sft_cat.keys()) | set(dpo_cat.keys())
        
        for cat in all_categories:
            if cat in sft_cat and cat in dpo_cat:
                sft_acc = sft_cat[cat]["accuracy"]
                dpo_acc = dpo_cat[cat]["accuracy"]
                improvement = (dpo_acc - sft_acc) / sft_acc * 100 if sft_acc > 0 else 0
                
                comparison[cat] = {
                    "sft_accuracy": sft_acc,
                    "dpo_accuracy": dpo_acc,
                    "improvement": improvement
                }
        
        return comparison
    
    def _compare_by_difficulty(self, sft_results: Dict, dpo_results: Dict) -> Dict:
        """按难度对比"""
        
        sft_diff = sft_results["by_difficulty"]
        dpo_diff = dpo_results["by_difficulty"]
        
        comparison = {}
        all_difficulties = set(sft_diff.keys()) | set(dpo_diff.keys())
        
        for diff in all_difficulties:
            if diff in sft_diff and diff in dpo_diff:
                sft_acc = sft_diff[diff]["accuracy"]
                dpo_acc = dpo_diff[diff]["accuracy"]
                improvement = (dpo_acc - sft_acc) / sft_acc * 100 if sft_acc > 0 else 0
                
                comparison[diff] = {
                    "sft_accuracy": sft_acc,
                    "dpo_accuracy": dpo_acc,
                    "improvement": improvement,
                    "sft_count": sft_diff[diff]["count"],
                    "dpo_count": dpo_diff[diff]["count"]
                }
        
        return comparison
    
    def _compare_errors(self, sft_results: Dict, dpo_results: Dict) -> Dict:
        """对比错误"""
        
        sft_errors = sft_results["error_analysis"]
        dpo_errors = dpo_results["error_analysis"]
        
        return {
            "fp_reduction": {
                "sft": sft_errors["fp_count"],
                "dpo": dpo_errors["fp_count"],
                "reduction": sft_errors["fp_count"] - dpo_errors["fp_count"],
                "reduction_rate": (sft_errors["fp_count"] - dpo_errors["fp_count"]) / sft_errors["fp_count"] * 100 
                                 if sft_errors["fp_count"] > 0 else 0
            },
            "fn_reduction": {
                "sft": sft_errors["fn_count"],
                "dpo": dpo_errors["fn_count"],
                "reduction": sft_errors["fn_count"] - dpo_errors["fn_count"],
                "reduction_rate": (sft_errors["fn_count"] - dpo_errors["fn_count"]) / sft_errors["fn_count"] * 100
                                 if sft_errors["fn_count"] > 0 else 0
            }
        }
    
    def print_evaluation_report(self, results: Dict):
        """打印评估报告"""
        
        model_name = results["model_name"]
        overall = results["overall"]
        by_category = results["by_category"]
        by_difficulty = results["by_difficulty"]
        errors = results["error_analysis"]
        
        print("\n" + "=" * 70)
        print(f"📊 {model_name} 模型评估报告")
        print("=" * 70)
        
        # 总体性能
        print(f"\n【总体性能】")
        print(f"  准确率 (Accuracy):      {overall['accuracy']:.4f} ({overall['accuracy']*100:.2f}%)")
        print(f"  精确率 (Precision):     {overall['precision']:.4f}")
        print(f"  召回率 (Recall):        {overall['recall']:.4f}")
        print(f"  F1 分数 (F1 Score):     {overall['f1_score']:.4f}")
        
        # 混淆矩阵
        cm = overall["confusion_matrix"]
        print(f"\n【混淆矩阵】")
        print(f"                 预测SAFE    预测UNSAFE")
        print(f"  真实SAFE       {cm['TN']:4d}        {cm['FP']:4d}")
        print(f"  真实UNSAFE     {cm['FN']:4d}        {cm['TP']:4d}")
        
        # 关键安全指标
        print(f"\n【关键安全指标】")
        print(f"  漏报率 (FNR):          {overall['false_negative_rate']:.4f} ({overall['false_negative_rate']*100:.2f}%) ⚠️")
        print(f"  误报率 (FPR):          {overall['false_positive_rate']:.4f} ({overall['false_positive_rate']*100:.2f}%)")
        print(f"  敏感性 (Sensitivity):  {overall['sensitivity']:.4f}")
        print(f"  特异性 (Specificity):  {overall['specificity']:.4f}")
        
        if overall['false_negative_rate'] > 0.1:
            print(f"\n  ⚠️ 警告: 漏报率过高！有 {overall['false_negative_rate']*100:.1f}% 的攻击未被检测到")
        
        # 按类别统计
        print(f"\n【按攻击类别统计】")
        print(f"  {'类别':<20} {'数量':<8} {'准确率':<10} {'F1分数':<10}")
        print(f"  {'-'*50}")
        for cat, metrics in sorted(by_category.items(), key=lambda x: -x[1]["accuracy"]):
            print(f"  {cat:<20} {metrics['count']:<8} {metrics['accuracy']:.4f}    {metrics['f1_score']:.4f}")
        
        # 按难度统计
        print(f"\n【按难度级别统计】")
        for diff in ["easy", "medium", "hard"]:
            if diff in by_difficulty:
                metrics = by_difficulty[diff]
                print(f"  {diff.upper():<10} 准确率: {metrics['accuracy']:.4f}  F1: {metrics['f1_score']:.4f}  (样本数: {metrics['count']})")
        
        # 错误分析
        print(f"\n【错误分析】")
        print(f"  假阳性 (误判为UNSAFE): {errors['fp_count']} 个")
        if errors['fp_by_category']:
            for cat, count in sorted(errors['fp_by_category'].items(), key=lambda x: -x[1])[:3]:
                print(f"    - {cat}: {count} 个")
        
        print(f"  假阴性 (漏判UNSAFE):   {errors['fn_count']} 个 ⚠️")
        if errors['fn_by_category']:
            for cat, count in sorted(errors['fn_by_category'].items(), key=lambda x: -x[1])[:3]:
                print(f"    - {cat}: {count} 个")
        
        print("=" * 70)
    
    def print_comparison_report(self, comparison: Dict):
        """打印对比报告"""
        
        print("\n" + "=" * 70)
        print("📊 SFT vs DPO 对比报告")
        print("=" * 70)
        
        overall = comparison["overall_improvement"]
        
        # 总体改进
        print(f"\n【总体性能改进】")
        print(f"  {'指标':<25} {'SFT':<12} {'DPO':<12} {'改进':<12}")
        print(f"  {'-'*60}")
        
        for metric, data in overall.items():
            sft_val = data["sft"]
            dpo_val = data["dpo"]
            improvement = data["improvement"]
            
            # 格式化改进显示
            if improvement > 0:
                imp_str = f"+{improvement:.1f}% ⭐"
            elif improvement < 0:
                imp_str = f"{improvement:.1f}% ⚠️"
            else:
                imp_str = "0.0%"
            
            print(f"  {metric:<25} {sft_val:<12.4f} {dpo_val:<12.4f} {imp_str}")
        
        # 按类别改进
        cat_comp = comparison["category_improvement"]
        if cat_comp:
            print(f"\n【按攻击类别改进】")
            print(f"  {'类别':<20} {'SFT准确率':<12} {'DPO准确率':<12} {'改进':<12}")
            print(f"  {'-'*60}")
            
            # 按改进幅度排序
            sorted_cats = sorted(cat_comp.items(), key=lambda x: -x[1]["improvement"])
            for cat, data in sorted_cats:
                sft_acc = data["sft_accuracy"]
                dpo_acc = data["dpo_accuracy"]
                improvement = data["improvement"]
                
                stars = ""
                if improvement > 15:
                    stars = "⭐⭐⭐⭐"
                elif improvement > 10:
                    stars = "⭐⭐⭐"
                elif improvement > 5:
                    stars = "⭐⭐"
                elif improvement > 0:
                    stars = "⭐"
                
                print(f"  {cat:<20} {sft_acc:<12.4f} {dpo_acc:<12.4f} +{improvement:>5.1f}% {stars}")
        
        # 按难度改进
        diff_comp = comparison["difficulty_improvement"]
        if diff_comp:
            print(f"\n【按难度级别改进】")
            for diff in ["easy", "medium", "hard"]:
                if diff in diff_comp:
                    data = diff_comp[diff]
                    sft_acc = data["sft_accuracy"]
                    dpo_acc = data["dpo_accuracy"]
                    improvement = data["improvement"]
                    
                    stars = "⭐⭐⭐⭐" if improvement > 15 else "⭐⭐⭐" if improvement > 10 else "⭐⭐" if improvement > 5 else "⭐"
                    print(f"  {diff.upper():<10} {sft_acc:.4f} → {dpo_acc:.4f}  (+{improvement:.1f}%) {stars}")
        
        # 错误减少
        error_comp = comparison["error_reduction"]
        print(f"\n【错误减少】")
        
        fp_data = error_comp["fp_reduction"]
        fn_data = error_comp["fn_reduction"]
        
        print(f"  假阳性 (误判): {fp_data['sft']} → {fp_data['dpo']} (-{fp_data['reduction']} 个, {fp_data['reduction_rate']:.1f}%)")
        print(f"  假阴性 (漏判): {fn_data['sft']} → {fn_data['dpo']} (-{fn_data['reduction']} 个, {fn_data['reduction_rate']:.1f}%) ⚠️")
        
        # 关键发现
        print(f"\n【关键发现】")
        
        # 准确率改进
        acc_improvement = overall["accuracy"]["improvement"]
        if acc_improvement > 5:
            print(f"  ✓ DPO显著提升了准确率 (+{acc_improvement:.1f}%)")
        elif acc_improvement > 0:
            print(f"  ✓ DPO提升了准确率 (+{acc_improvement:.1f}%)")
        else:
            print(f"  ✗ DPO未能提升准确率 ({acc_improvement:.1f}%)")
        
        # 漏报率改进
        fnr_improvement = overall["false_negative_rate"]["improvement"]
        if fnr_improvement > 20:
            print(f"  ✓ DPO大幅降低了漏报率 (+{fnr_improvement:.1f}%)")
        elif fnr_improvement > 0:
            print(f"  ✓ DPO降低了漏报率 (+{fnr_improvement:.1f}%)")
        else:
            print(f"  ✗ DPO未能降低漏报率 ({fnr_improvement:.1f}%)")
        
        # 困难样本改进
        if "hard" in diff_comp:
            hard_improvement = diff_comp["hard"]["improvement"]
            if hard_improvement > 10:
                print(f"  ✓ DPO在困难样本上表现优秀 (+{hard_improvement:.1f}%)")
        
        # 最大改进类别
        if cat_comp:
            max_improvement_cat = max(cat_comp.items(), key=lambda x: x[1]["improvement"])
            print(f"  ✓ 最大改进: {max_improvement_cat[0]} (+{max_improvement_cat[1]['improvement']:.1f}%)")
        
        print(f"\n【总结】")
        if acc_improvement > 5 and fnr_improvement > 10:
            print(f"  🎉 DPO训练效果显著！准确率和安全性都有明显提升。")
            print(f"  ✓ 建议采用DPO模型进行部署")
        elif acc_improvement > 0:
            print(f"  ✓ DPO训练有一定效果，可考虑进一步优化训练参数")
        else:
            print(f"  ⚠️ DPO训练效果不明显，建议检查：")
            print(f"     1. DPO数据质量（chosen vs rejected对比度）")
            print(f"     2. 训练参数（beta值、学习率）")
            print(f"     3. 数据平衡性")
        
        print("=" * 70)
    
    def _save_predictions(self, model_name: str, predictions: List[Dict]):
        """保存预测结果"""
        output_path = self.results_dir / f"{model_name.lower()}_predictions.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print(f"✓ 预测结果已保存: {output_path}")
    
    def save_comparison_results(self, sft_results: Dict, dpo_results: Dict, 
                               comparison: Dict):
        """保存对比结果"""
        output_path = self.results_dir / "model_comparison.json"
        
        results = {
            "sft_results": sft_results,
            "dpo_results": dpo_results,
            "comparison": comparison
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 对比结果已保存: {output_path}")


# 保留旧的类名以兼容性
class ModelEvaluator(ImprovedModelEvaluator):
    """
    兼容旧版接口的评估器包装类
    保持向后兼容，同时支持新功能（包括多核优化）
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        """
        初始化评估器
        
        Args:
            num_workers: 并行工作进程数，默认为CPU核心数
        """
        super().__init__(num_workers=num_workers)
    
    def evaluate_baseline(self, core_llm, use_parallel: bool = True):
        """
        兼容旧接口：评估基准模型（支持多线程加速）
        
        Args:
            core_llm: 核心LLM实例
            use_parallel: 是否使用多线程并行（默认True）
            
        Returns:
            (y_true, y_pred)
        """
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        print("\n评估基准模型（使用拒绝关键词检测）...")
        
        if use_parallel and self.num_workers > 1:
            print(f"🚀 使用多线程并行处理 ({self.num_workers} 个线程)")
            results = self._evaluate_baseline_parallel(core_llm, test_data)
        else:
            print(f"📝 使用单线程处理")
            results = self._evaluate_baseline_sequential(core_llm, test_data)
        
        # 提取结果
        y_true = [r["true_label"] for r in results]
        y_pred = [r["pred_label"] for r in results]
        
        print("✓ 基准模型评估完成")
        return y_true, y_pred
    
    def _evaluate_baseline_sequential(self, core_llm, test_data: List[Dict]) -> List[Dict]:
        """单线程顺序评估基准模型"""
        results = []
        
        for i, sample in enumerate(test_data, 1):
            prompt = sample["prompt"]
            true_label = sample["label"]
            
            try:
                response = core_llm.chat(prompt)
                
                # 判断是否拒绝
                is_refused = any(keyword.lower() in response.lower() 
                               for keyword in self.refusal_keywords)
                pred_label = 1 if is_refused else 0
                
            except Exception as e:
                pred_label = 0
            
            results.append({
                "idx": i - 1,
                "true_label": true_label,
                "pred_label": pred_label
            })
            
            if i % 10 == 0:
                print(f"  进度: {i}/{len(test_data)}")
        
        return results
    
    def _evaluate_baseline_parallel(self, core_llm, test_data: List[Dict]) -> List[Dict]:
        """多线程并行评估基准模型"""
        total_samples = len(test_data)
        results = []
        
        def process_sample(idx, sample):
            """处理单个样本"""
            prompt = sample["prompt"]
            true_label = sample["label"]
            
            try:
                response = core_llm.chat(prompt)
                
                # 判断是否拒绝
                is_refused = any(keyword.lower() in response.lower() 
                               for keyword in self.refusal_keywords)
                pred_label = 1 if is_refused else 0
                
            except Exception as e:
                pred_label = 0
            
            return {
                "idx": idx,
                "true_label": true_label,
                "pred_label": pred_label
            }
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_sample, i, sample): i 
                for i, sample in enumerate(test_data)
            }
            
            # 收集结果并显示进度
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 50 == 0 or completed == total_samples:
                    print(f"  进度: {completed}/{total_samples} ({completed*100//total_samples}%)")
        
        # 按索引排序
        results.sort(key=lambda x: x["idx"])
        return results
    
    def evaluate_defense_system(self, defense_manager, use_parallel: bool = True):
        """
        兼容旧接口：评估防御系统（支持多线程加速）
        
        Args:
            defense_manager: 防御管理器实例
            use_parallel: 是否使用多线程并行（默认True）
            
        Returns:
            (y_true, y_pred, block_sources)
        """
        test_data = self.load_test_data()
        total_samples = len(test_data)
        
        print("\n评估防御系统...")
        
        if use_parallel and self.num_workers > 1:
            print(f"🚀 使用多线程并行处理 ({self.num_workers} 个线程)")
            results = self._evaluate_defense_parallel(defense_manager, test_data)
        else:
            print(f"📝 使用单线程处理")
            results = self._evaluate_defense_sequential(defense_manager, test_data)
        
        # 提取结果
        y_true = [r["true_label"] for r in results]
        y_pred = [r["pred_label"] for r in results]
        block_sources = [r["source"] for r in results]
        
        print("✓ 防御系统评估完成")
        return y_true, y_pred, block_sources
    
    def _evaluate_defense_sequential(self, defense_manager, test_data: List[Dict]) -> List[Dict]:
        """单线程顺序评估防御系统"""
        results = []
        
        for i, sample in enumerate(test_data, 1):
            prompt = sample["prompt"]
            true_label = sample["label"]
            
            try:
                result = defense_manager.process(prompt)
                pred_label = 0 if result["success"] else 1
                source = result.get("source", "unknown")
                
            except Exception as e:
                pred_label = 0
                source = "error"
            
            results.append({
                "idx": i - 1,
                "true_label": true_label,
                "pred_label": pred_label,
                "source": source
            })
            
            if i % 10 == 0:
                print(f"  进度: {i}/{len(test_data)}")
        
        return results
    
    def _evaluate_defense_parallel(self, defense_manager, test_data: List[Dict]) -> List[Dict]:
        """多线程并行评估防御系统"""
        total_samples = len(test_data)
        results = []
        
        def process_sample(idx, sample):
            """处理单个样本"""
            prompt = sample["prompt"]
            true_label = sample["label"]
            
            try:
                result = defense_manager.process(prompt)
                pred_label = 0 if result["success"] else 1
                source = result.get("source", "unknown")
                
            except Exception as e:
                pred_label = 0
                source = "error"
            
            return {
                "idx": idx,
                "true_label": true_label,
                "pred_label": pred_label,
                "source": source
            }
        
        # 使用线程池并行处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # 提交所有任务
            future_to_idx = {
                executor.submit(process_sample, i, sample): i 
                for i, sample in enumerate(test_data)
            }
            
            # 收集结果并显示进度
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                results.append(result)
                completed += 1
                
                if completed % 50 == 0 or completed == total_samples:
                    print(f"  进度: {completed}/{total_samples} ({completed*100//total_samples}%)")
        
        # 按索引排序
        results.sort(key=lambda x: x["idx"])
        return results
    
    def calculate_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict:
        """
        兼容旧接口：计算指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            指标字典
        """
        return self._calculate_overall_metrics(y_true, y_pred)
    
    def print_metrics(self, metrics: Dict, title: str):
        """
        兼容旧接口：打印指标
        
        Args:
            metrics: 指标字典
            title: 标题
        """
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        
        print(f"\n准确率 (Accuracy):    {metrics['accuracy']:.4f}")
        print(f"精确率 (Precision):   {metrics['precision']:.4f}")
        print(f"召回率 (Recall):      {metrics['recall']:.4f}")
        print(f"F1 分数 (F1 Score):   {metrics['f1_score']:.4f}")
        
        cm = metrics["confusion_matrix"]
        print(f"\n混淆矩阵:")
        print(f"  TN (真阴性): {cm['TN']:3d}  |  FP (假阳性): {cm['FP']:3d}")
        print(f"  FN (假阴性): {cm['FN']:3d}  |  TP (真阳性): {cm['TP']:3d}")
        
        print(f"\n关键安全指标:")
        print(f"  漏报率 (FNR): {metrics['false_negative_rate']:.4f} ⚠️")
        print(f"  误报率 (FPR): {metrics['false_positive_rate']:.4f}")
        
        if metrics['false_negative_rate'] > 0.1:
            print(f"\n⚠️ 警告: 漏报率过高 ({metrics['false_negative_rate']:.2%})！")
    
    def compare_systems(self, baseline_metrics: Dict, defense_metrics: Dict):
        """
        兼容旧接口：对比系统
        
        Args:
            baseline_metrics: 基准模型指标
            defense_metrics: 防御系统指标
        """
        print("\n" + "=" * 70)
        print("系统对比")
        print("=" * 70)
        
        metrics_names = ["accuracy", "precision", "recall", "f1_score", 
                        "false_negative_rate", "false_positive_rate"]
        
        print(f"\n{'指标':<25} {'基准模型':<15} {'防御系统':<15} {'改善':<10}")
        print("-" * 70)
        
        for metric in metrics_names:
            baseline_val = baseline_metrics[metric]
            defense_val = defense_metrics[metric]
            
            if "rate" in metric:
                improvement = (baseline_val - defense_val) / baseline_val * 100 if baseline_val > 0 else 0
            else:
                improvement = (defense_val - baseline_val) / baseline_val * 100 if baseline_val > 0 else 0
            
            improvement_str = f"{improvement:+.1f}%"
            print(f"{metric:<25} {baseline_val:<15.4f} {defense_val:<15.4f} {improvement_str:<10}")
    
    def save_results(self, baseline_metrics: Dict, defense_metrics: Dict,
                    baseline_predictions: Tuple, defense_predictions: Tuple) -> Dict:
        """
        兼容旧接口：保存结果
        
        Args:
            baseline_metrics: 基准指标
            defense_metrics: 防御指标
            baseline_predictions: 基准预测
            defense_predictions: 防御预测
            
        Returns:
            结果字典
        """
        results = {
            "baseline": baseline_metrics,
            "defense": defense_metrics,
            "baseline_predictions": {
                "y_true": baseline_predictions[0],
                "y_pred": baseline_predictions[1],
            },
            "defense_predictions": {
                "y_true": defense_predictions[0],
                "y_pred": defense_predictions[1],
                "block_sources": defense_predictions[2] if len(defense_predictions) > 2 else [],
            }
        }
        
        output_path = self.results_dir / "evaluation_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 评估结果已保存到: {output_path}")
        return results


if __name__ == "__main__":
    print("=" * 70)
    print("增强版模型评估模块")
    print("=" * 70)
    print("\n此模块需要配合 main.py --evaluate 使用")
    print("请运行: python main.py --evaluate")
    print("\n新增功能:")
    print("  ✓ 按攻击类别细分评估")
    print("  ✓ 按难度级别细分评估")
    print("  ✓ 详细错误分析")
    print("  ✓ SFT vs DPO 对比")
    print("  ✓ 置信度统计")
