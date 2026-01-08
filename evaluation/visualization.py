"""
结果可视化模块 - 增强版
生成混淆矩阵、ROC 曲线、SFT vs DPO 对比等图表
"""

import sys
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from matplotlib.patches import Rectangle

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from defense.config import DefenseConfig


class ImprovedResultVisualizer:
    """增强版结果可视化器"""
    
    def __init__(self):
        self.results_dir = DefenseConfig.EVAL_RESULTS_DIR
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置中文字体支持和样式
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial']
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.facecolor'] = 'white'
        sns.set_style("whitegrid")
    
    def plot_confusion_matrices(self, baseline_cm: np.ndarray, defense_cm: np.ndarray):
        """
        绘制混淆矩阵对比图
        
        Args:
            baseline_cm: 基准模型的混淆矩阵
            defense_cm: 防御系统的混淆矩阵
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
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
        
        output_path = self.results_dir / "confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 混淆矩阵已保存: {output_path}")
        
        plt.close()
    
    def plot_sft_dpo_confusion_matrices(self, sft_cm: np.ndarray, dpo_cm: np.ndarray):
        """
        绘制 SFT vs DPO 混淆矩阵对比
        
        Args:
            sft_cm: SFT模型的混淆矩阵
            dpo_cm: DPO模型的混淆矩阵
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        labels = ['SAFE', 'UNSAFE']
        
        # SFT模型混淆矩阵
        sns.heatmap(sft_cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, 
                   ax=axes[0], cbar_kws={'label': '样本数量'})
        axes[0].set_title('SFT 模型', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('真实标签', fontsize=10)
        axes[0].set_xlabel('预测标签', fontsize=10)
        
        # DPO模型混淆矩阵
        sns.heatmap(dpo_cm, annot=True, fmt='d', cmap='Greens', 
                   xticklabels=labels, yticklabels=labels, 
                   ax=axes[1], cbar_kws={'label': '样本数量'})
        axes[1].set_title('SFT + DPO 模型', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('真实标签', fontsize=10)
        axes[1].set_xlabel('预测标签', fontsize=10)
        
        plt.tight_layout()
        
        output_path = self.results_dir / "sft_dpo_confusion_matrices.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ SFT vs DPO 混淆矩阵已保存: {output_path}")
        
        plt.close()
    
    def plot_roc_curve(self, baseline_pred: tuple, defense_pred: tuple):
        """
        绘制 ROC 曲线
        
        Args:
            baseline_pred: (y_true, y_score) 基准模型预测（使用置信度分数）
            defense_pred: (y_true, y_score) 防御系统预测（使用置信度分数）
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # 基准模型 ROC - 使用置信度分数
        y_true_baseline, y_score_baseline = baseline_pred
        fpr_baseline, tpr_baseline, _ = roc_curve(y_true_baseline, y_score_baseline)
        roc_auc_baseline = auc(fpr_baseline, tpr_baseline)
        
        # 防御系统 ROC - 使用置信度分数
        y_true_defense, y_score_defense = defense_pred
        fpr_defense, tpr_defense, _ = roc_curve(y_true_defense, y_score_defense)
        roc_auc_defense = auc(fpr_defense, tpr_defense)
        
        # 绘制曲线
        ax.plot(fpr_baseline, tpr_baseline, 'b-', linewidth=2,
               label=f'基准模型 (AUC = {roc_auc_baseline:.3f})')
        ax.plot(fpr_defense, tpr_defense, 'g-', linewidth=2,
               label=f'防御系统 (AUC = {roc_auc_defense:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='随机分类器 (AUC = 0.5)')
        
        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=11)
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=11)
        ax.set_title('ROC 曲线对比', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        output_path = self.results_dir / "roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC 曲线已保存: {output_path}")
        print(f"  - 基准模型: {len(set(y_score_baseline))} 个不同的置信度值")
        print(f"  - 防御系统: {len(set(y_score_defense))} 个不同的置信度值")
        
        plt.close()
    
    def plot_sft_dpo_roc_curve(self, sft_pred: tuple, dpo_pred: tuple):
        """
        绘制 SFT vs DPO ROC 曲线
        
        Args:
            sft_pred: (y_true, y_pred) SFT模型预测
            dpo_pred: (y_true, y_pred) DPO模型预测
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # SFT模型 ROC
        y_true_sft, y_pred_sft = sft_pred
        fpr_sft, tpr_sft, _ = roc_curve(y_true_sft, y_pred_sft)
        roc_auc_sft = auc(fpr_sft, tpr_sft)
        
        # DPO模型 ROC
        y_true_dpo, y_pred_dpo = dpo_pred
        fpr_dpo, tpr_dpo, _ = roc_curve(y_true_dpo, y_pred_dpo)
        roc_auc_dpo = auc(fpr_dpo, tpr_dpo)
        
        # 绘制曲线
        ax.plot(fpr_sft, tpr_sft, 'b-', linewidth=2.5,
               label=f'SFT 模型 (AUC = {roc_auc_sft:.3f})')
        ax.plot(fpr_dpo, tpr_dpo, 'g-', linewidth=2.5,
               label=f'SFT + DPO 模型 (AUC = {roc_auc_dpo:.3f})')
        ax.plot([0, 1], [0, 1], 'r--', linewidth=1.5, label='随机分类器 (AUC = 0.5)')
        
        ax.set_xlabel('假阳性率 (False Positive Rate)', fontsize=12)
        ax.set_ylabel('真阳性率 (True Positive Rate)', fontsize=12)
        ax.set_title('SFT vs DPO ROC 曲线对比', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        
        plt.tight_layout()
        
        output_path = self.results_dir / "sft_dpo_roc_curve.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ SFT vs DPO ROC 曲线已保存: {output_path}")
        
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
        
        output_path = self.results_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 指标对比图已保存: {output_path}")
        
        plt.close()
    
    def plot_sft_dpo_metrics_comparison(self, sft_results: dict, dpo_results: dict):
        """
        绘制 SFT vs DPO 详细指标对比
        
        Args:
            sft_results: SFT模型评估结果
            dpo_results: DPO模型评估结果
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('SFT vs DPO 全面性能对比', fontsize=16, fontweight='bold', y=0.995)
        
        # 1. 总体指标对比
        ax1 = axes[0, 0]
        metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
        metrics_labels = ['准确率', '精确率', '召回率', 'F1分数']
        
        sft_overall = sft_results['overall']
        dpo_overall = dpo_results['overall']
        
        sft_values = [sft_overall[m] for m in metrics_names]
        dpo_values = [dpo_overall[m] for m in metrics_names]
        
        x = np.arange(len(metrics_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, sft_values, width, label='SFT', 
                       color='#5B9BD5', alpha=0.8)
        bars2 = ax1.bar(x + width/2, dpo_values, width, label='SFT+DPO', 
                       color='#70AD47', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_ylabel('分数', fontsize=10)
        ax1.set_title('总体性能指标', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_labels, fontsize=9)
        ax1.legend(fontsize=9)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 2. 按难度级别对比
        ax2 = axes[0, 1]
        
        sft_diff = sft_results['by_difficulty']
        dpo_diff = dpo_results['by_difficulty']
        
        difficulties = ['easy', 'medium', 'hard']
        diff_labels = ['简单', '中等', '困难']
        
        sft_acc = [sft_diff.get(d, {}).get('accuracy', 0) for d in difficulties]
        dpo_acc = [dpo_diff.get(d, {}).get('accuracy', 0) for d in difficulties]
        
        x = np.arange(len(diff_labels))
        
        bars1 = ax2.bar(x - width/2, sft_acc, width, label='SFT', 
                       color='#5B9BD5', alpha=0.8)
        bars2 = ax2.bar(x + width/2, dpo_acc, width, label='SFT+DPO', 
                       color='#70AD47', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax2.set_ylabel('准确率', fontsize=10)
        ax2.set_title('按难度级别准确率', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(diff_labels, fontsize=9)
        ax2.legend(fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim([0, 1.1])
        
        # 3. 关键安全指标（漏报率和误报率）
        ax3 = axes[1, 0]
        
        safety_metrics = ['false_negative_rate', 'false_positive_rate']
        safety_labels = ['漏报率\n(危险)', '误报率\n(影响体验)']
        
        sft_safety = [sft_overall[m] for m in safety_metrics]
        dpo_safety = [dpo_overall[m] for m in safety_metrics]
        
        x = np.arange(len(safety_labels))
        
        bars1 = ax3.bar(x - width/2, sft_safety, width, label='SFT', 
                       color='#E74C3C', alpha=0.7)
        bars2 = ax3.bar(x + width/2, dpo_safety, width, label='SFT+DPO', 
                       color='#F39C12', alpha=0.7)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        ax3.set_ylabel('错误率（越低越好）', fontsize=10)
        ax3.set_title('关键安全指标', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(safety_labels, fontsize=9)
        ax3.legend(fontsize=9)
        ax3.grid(axis='y', alpha=0.3)
        ax3.set_ylim([0, max(max(sft_safety), max(dpo_safety)) * 1.2])
        
        # 4. 错误数量对比
        ax4 = axes[1, 1]
        
        sft_errors = sft_results['error_analysis']
        dpo_errors = dpo_results['error_analysis']
        
        error_types = ['假阳性', '假阴性']
        sft_error_counts = [sft_errors['fp_count'], sft_errors['fn_count']]
        dpo_error_counts = [dpo_errors['fp_count'], dpo_errors['fn_count']]
        
        x = np.arange(len(error_types))
        
        bars1 = ax4.bar(x - width/2, sft_error_counts, width, label='SFT', 
                       color='#5B9BD5', alpha=0.8)
        bars2 = ax4.bar(x + width/2, dpo_error_counts, width, label='SFT+DPO', 
                       color='#70AD47', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
        
        ax4.set_ylabel('错误数量', fontsize=10)
        ax4.set_title('错误数量对比', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(error_types, fontsize=10)
        ax4.legend(fontsize=9)
        ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.results_dir / "sft_dpo_detailed_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ SFT vs DPO 详细对比图已保存: {output_path}")
        
        plt.close()
    
    def plot_category_comparison(self, sft_results: dict, dpo_results: dict):
        """
        绘制按攻击类别的准确率对比
        
        Args:
            sft_results: SFT模型评估结果
            dpo_results: DPO模型评估结果
        """
        sft_cat = sft_results['by_category']
        dpo_cat = dpo_results['by_category']
        
        # 获取所有类别
        all_categories = set(sft_cat.keys()) | set(dpo_cat.keys())
        
        if not all_categories:
            print("⚠ 没有类别数据，跳过类别对比图")
            return
        
        categories = sorted(all_categories)
        cat_labels = [cat for cat in categories]
        
        sft_acc = [sft_cat.get(cat, {}).get('accuracy', 0) for cat in categories]
        dpo_acc = [dpo_cat.get(cat, {}).get('accuracy', 0) for cat in categories]
        improvements = [(d - s) for s, d in zip(sft_acc, dpo_acc)]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # 子图1: 准确率对比
        x = np.arange(len(cat_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, sft_acc, width, label='SFT', 
                       color='#5B9BD5', alpha=0.8)
        bars2 = ax1.bar(x + width/2, dpo_acc, width, label='SFT+DPO', 
                       color='#70AD47', alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height,
                            f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax1.set_ylabel('准确率', fontsize=11)
        ax1.set_title('各攻击类别准确率对比', fontsize=13, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(cat_labels, rotation=15, ha='right', fontsize=9)
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim([0, 1.1])
        
        # 子图2: 改进幅度
        colors = ['#70AD47' if imp > 0 else '#E74C3C' for imp in improvements]
        bars = ax2.bar(x, improvements, color=colors, alpha=0.8)
        
        for bar, imp in zip(bars, improvements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:+.3f}', ha='center', 
                    va='bottom' if height > 0 else 'top', fontsize=8)
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_ylabel('准确率改进', fontsize=11)
        ax2.set_title('DPO相对SFT的改进幅度', fontsize=13, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(cat_labels, rotation=15, ha='right', fontsize=9)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.results_dir / "category_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ 类别对比图已保存: {output_path}")
        
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
        
        from collections import Counter
        stats = Counter(block_sources)
        stats.pop("passed", None)
        
        if not stats:
            print("⚠ 所有输入都未被拦截")
            return
        
        layers = list(stats.keys())
        counts = list(stats.values())
        
        layer_names = {
            "keyword_filter": "第1层: 关键词过滤",
            "guard_model": "第2层: AI 卫士",
            "prompt_hardening": "第3层: 提示词强化"
        }
        
        labels = [layer_names.get(layer, layer) for layer in layers]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#E74C3C', '#F39C12', '#3498DB'][:len(layers)]
        bars = ax.barh(labels, counts, color=colors, alpha=0.8)
        
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(count, i, f'  {count}', va='center', fontsize=10)
        
        ax.set_xlabel('拦截数量', fontsize=11)
        ax.set_title('各防御层拦截统计', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
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
        # 检查是否有模型对比结果（SFT vs DPO）
        model_comparison_file = self.results_dir / "model_comparison.json"
        
        if model_comparison_file.exists():
            print("=" * 70)
            print("检测到 SFT vs DPO 对比数据，生成对比可视化")
            print("=" * 70)
            self.visualize_sft_dpo_comparison(model_comparison_file)
        
        # 检查旧的评估结果（Baseline vs Defense）
        if results_file is None:
            results_file = self.results_dir / "evaluation_results.json"
        
        if results_file.exists():
            print("\n=" * 70)
            print("生成基准评估可视化")
            print("=" * 70)
            self.visualize_baseline_defense(results_file)
        
        if not model_comparison_file.exists() and not results_file.exists():
            print("✗ 未找到评估结果文件")
            print(f"  SFT vs DPO: {model_comparison_file}")
            print(f"  基准评估: {results_file}")
            print("\n请先运行评估: python main.py --evaluate")
    
    def visualize_sft_dpo_comparison(self, results_file: Path):
        """生成 SFT vs DPO 对比可视化"""
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        sft_results = results['sft_results']
        dpo_results = results['dpo_results']
        
        # 1. 混淆矩阵对比
        sft_cm_dict = sft_results["overall"]["confusion_matrix"]
        dpo_cm_dict = dpo_results["overall"]["confusion_matrix"]
        
        sft_cm = np.array([[sft_cm_dict["TN"], sft_cm_dict["FP"]],
                          [sft_cm_dict["FN"], sft_cm_dict["TP"]]])
        dpo_cm = np.array([[dpo_cm_dict["TN"], dpo_cm_dict["FP"]],
                          [dpo_cm_dict["FN"], dpo_cm_dict["TP"]]])
        
        self.plot_sft_dpo_confusion_matrices(sft_cm, dpo_cm)
        
        # 2. ROC 曲线（如果有预测数据）
        # 注意：新的评估器可能没有保存完整的预测概率，这里可以跳过或使用0/1预测
        
        # 3. 详细指标对比
        self.plot_sft_dpo_metrics_comparison(sft_results, dpo_results)
        
        # 4. 按类别对比
        if 'by_category' in sft_results and 'by_category' in dpo_results:
            self.plot_category_comparison(sft_results, dpo_results)
        
        print("\n✓ SFT vs DPO 对比可视化完成")
    
    def visualize_baseline_defense(self, results_file: Path):
        """生成基准 vs 防御系统可视化"""
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        # 1. 混淆矩阵
        baseline_cm_dict = results["baseline"]["confusion_matrix"]
        defense_cm_dict = results["defense"]["confusion_matrix"]
        
        baseline_cm = np.array([[baseline_cm_dict["TN"], baseline_cm_dict["FP"]],
                               [baseline_cm_dict["FN"], baseline_cm_dict["TP"]]])
        defense_cm = np.array([[defense_cm_dict["TN"], defense_cm_dict["FP"]],
                              [defense_cm_dict["FN"], defense_cm_dict["TP"]]])
        
        self.plot_confusion_matrices(baseline_cm, defense_cm)
        
        # 2. ROC 曲线 - 使用置信度分数
        baseline_pred = (
            results["baseline_predictions"]["y_true"],
            results["baseline_predictions"].get("y_score", results["baseline_predictions"]["y_pred"])
        )
        defense_pred = (
            results["defense_predictions"]["y_true"],
            results["defense_predictions"].get("y_score", results["defense_predictions"]["y_pred"])
        )
        
        self.plot_roc_curve(baseline_pred, defense_pred)
        
        # 3. 指标对比
        self.plot_metrics_comparison(results["baseline"], results["defense"])
        
        # 4. 防御层统计
        if "block_sources" in results["defense_predictions"]:
            self.plot_defense_layer_stats(results["defense_predictions"]["block_sources"])
        
        print("\n✓ 基准评估可视化完成")


# 保留旧的类名以兼容性
ResultVisualizer = ImprovedResultVisualizer


if __name__ == "__main__":
    visualizer = ImprovedResultVisualizer()
    visualizer.visualize_all()
