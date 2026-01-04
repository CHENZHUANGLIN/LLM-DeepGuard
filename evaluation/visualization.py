"""
结果可视化模块
生成混淆矩阵、ROC 曲线等图表
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from defense.config import DefenseConfig


class ResultVisualizer:
    """结果可视化器"""
    
    def __init__(self):
        self.results_dir = DefenseConfig.EVAL_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    
    def plot_confusion_matrices(self, baseline_cm: np.ndarray, defense_cm: np.ndarray):
        """
        绘制混淆矩阵对比图
        
        Args:
            baseline_cm: 基准模型的混淆矩阵
            defense_cm: 防御系统的混淆矩阵
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 标签
        labels = ['SAFE', 'UNSAFE']
        
        # 基准模型混淆矩阵
        sns.heatmap(baseline_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, 
                   ax=axes[0], cbar_kws={'label': '样本数量'})
        axes[0].set_title('基准模型 (Baseline)\n裸跑 Qwen 7B', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('真实标签', fontsize=10)
        axes[0].set_xlabel('预测标签', fontsize=10)
        
        # 防御系统混淆矩阵
        sns.heatmap(defense_cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=labels, yticklabels=labels, 
                   ax=axes[1], cbar_kws={'label': '样本数量'})
        axes[1].set_title('防御系统 (Our System)\n三层防御', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('真实标签', fontsize=10)
        axes[1].set_xlabel('预测标签', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.results_dir / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {output_path}")
        
        plt.close()
    
    def plot_roc_curve(self, baseline_pred: tuple, defense_pred: tuple):
        """
        绘制 ROC 曲线
        
        Args:
            baseline_pred: (y_true, y_pred) 基准模型预测
            defense_pred: (y_true, y_pred) 防御系统预测
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 基准模型 ROC
        y_true_baseline, y_pred_baseline = baseline_pred
        fpr_baseline, tpr_baseline, _ = roc_curve(y_true_baseline, y_pred_baseline)
        roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
        
        # 防御系统 ROC
        y_true_defense, y_pred_defense = defense_pred
        fpr_defense, tpr_defense, _ = roc_curve(y_true_defense, y_pred_defense)
        roc_auc_defense = auc(fpr_defense, tpr_defense)
        
        # 绘制曲线
        ax.plot(fpr_baseline, tpr_baseline, 'b-', linewidth=2,
               label=f'基准模型 (AUC = {roc_auc_baseline:.3f})')
        ax.plot(fpr_defense, tpr_defense, 'g-', linewidth=2,
               label=f'防御系统 (AUC = {roc_auc_defense:.3f})')
        
        # 绘制对角线（随机分类器）
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='随机分类器 (AUC = 0.5)')
        
        # 标注
        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=11)
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=11)
        ax.set_title('ROC 曲线对比', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.results_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC 曲线已保存: {output_path}")
        
        plt.close()
    
    def plot_metrics_comparison(self, baseline_metrics: dict, defense_metrics: dict):
        """
        绘制指标对比柱状图
        
        Args:
            baseline_metrics: 基准模型指标
            defense_metrics: 防御系统指标
        """
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_labels = ['准确率', '精确率', '召回率', 'F1 分数']
        
        baseline_values = [baseline_metrics[m] for m in metrics_names]
        defense_values = [defense_metrics[m] for m in metrics_names]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars1 = ax.bar(x - width/2, baseline_values, width, label='基准模型', 
                      color='#5B9BD5', alpha=0.8)
        bars2 = ax.bar(x + width/2, defense_values, width, label='防御系统', 
                      color='#70AD47', alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('分数', fontsize=11)
        ax.set_title('模型性能指标对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_labels, fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.1])
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.results_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 指标对比图已保存: {output_path}")
        
        plt.close()
    
    def plot_defense_layer_stats(self, block_sources: list):
        """
        绘制防御层拦截统计
        
        Args:
            block_sources: 拦截来源列表
        """
        if not block_sources:
            print("⚠ 没有拦截统计数据")
            return
        
        # 统计各层拦截数量
        from collections import Counter
        stats = Counter(block_sources)
        
        # 过滤掉 "passed" (未拦截)
        stats.pop("passed", None)
        
        if not stats:
            print("⚠ 所有输入都未被拦截")
            return
        
        layers = list(stats.keys())
        counts = list(stats.values())
        
        # 层名称映射
        layer_names = {
            "keyword_filter": "第1层: 关键词过滤",
            "guard_model": "第2层: AI 卫士",
            "prompt_hardening": "第3层: 提示词强化"
        }
        
        labels = [layer_names.get(layer, layer) for layer in layers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#E74C3C', '#F39C12', '#3498DB'][:len(layers)]
        bars = ax.barh(labels, counts, color=colors, alpha=0.8)
        
        # 添加数值标签
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f'  {count}', va='center', fontsize=10)
        
        ax.set_xlabel('拦截数量', fontsize=11)
        ax.set_title('各防御层拦截统计', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        output_path = self.results_dir / "defense_layers_stats.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 防御层统计图已保存: {output_path}")
        
        plt.close()
    
    def visualize_all(self, results_file: Path = None):
        """
        生成所有可视化图表
        
        Args:
            results_file: 评估结果 JSON 文件路径
        """
        if results_file is None:
            results_file = self.results_dir / "evaluation_results.json"
        
        if not results_file.exists():
            print(f"✗ 结果文件不存在: {results_file}")
            print("请先运行评估: python main.py --evaluate")
            return
        
        # 加载结果
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print("=" * 70)
        print("生成可视化图表")
        print("=" * 70)
        
        # 1. 混淆矩阵
        baseline_cm_dict = results["baseline"]["confusion_matrix"]
        defense_cm_dict = results["defense"]["confusion_matrix"]
        
        baseline_cm = np.array([[baseline_cm_dict["TN"], baseline_cm_dict["FP"]],
                               [baseline_cm_dict["FN"], baseline_cm_dict["TP"]]])
        defense_cm = np.array([[defense_cm_dict["TN"], defense_cm_dict["FP"]],
                              [defense_cm_dict["FN"], defense_cm_dict["TP"]]])
        
        self.plot_confusion_matrices(baseline_cm, defense_cm)
        
        # 2. ROC 曲线
        baseline_pred = (
            results["baseline_predictions"]["y_true"],
            results["baseline_predictions"]["y_pred"]
        )
        defense_pred = (
            results["defense_predictions"]["y_true"],
            results["defense_predictions"]["y_pred"]
        )
        
        self.plot_roc_curve(baseline_pred, defense_pred)
        
        # 3. 指标对比
        self.plot_metrics_comparison(results["baseline"], results["defense"])
        
        # 4. 防御层统计
        if "block_sources" in results["defense_predictions"]:
            self.plot_defense_layer_stats(results["defense_predictions"]["block_sources"])
        
        print("\n" + "=" * 70)
        print("所有图表生成完成！")
        print(f"保存位置: {self.results_dir}")
        print("=" * 70)


if __name__ == "__main__":
    visualizer = ResultVisualizer()
    visualizer.visualize_all()

