"""
主程序入口
提供训练、评估和交互式对话功能
"""

import os
# ⚠️ 重要：必须在导入任何其他模块之前设置环境变量
# 设置离线模式，强制只使用本地模型文件
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import sys
import argparse
from pathlib import Path
from colorama import init, Fore, Style

# 初始化 colorama
init(autoreset=True)

from defense.config import DefenseConfig
from data.generate_data import ImprovedDataGenerator
from training.train_sft import train_sft
from training.train_dpo import train_dpo
from defense_manager import DefenseManager
from core_llm import CoreLLM
from evaluation.evaluate import ModelEvaluator
from evaluation.visualization import ImprovedResultVisualizer


def print_banner():
    """打印欢迎横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║              🛡️  Project Cerberus - AI 纵深防御系统  🛡️               ║
║                                                                      ║
║              基于 Qwen 2.5 的提示词注入防御系统                        ║
║              三层防御 + SFT + DPO + 完整评估                          ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """
    print(Fore.CYAN + banner)


def generate_data():
    """生成训练和测试数据"""
    print(Fore.YELLOW + "\n检查训练和测试数据")
    print("=" * 70)
    
    # 检查数据是否已存在
    sft_exists = DefenseConfig.SFT_DATA_PATH.exists()
    dpo_exists = DefenseConfig.DPO_DATA_PATH.exists()
    test_exists = DefenseConfig.TEST_DATA_PATH.exists()
    
    if sft_exists and dpo_exists and test_exists:
        print(Fore.GREEN + "\n✓ 检测到完整数据集已存在，直接加载：")
        print(f"  - SFT数据: {DefenseConfig.SFT_DATA_PATH} ({DefenseConfig.SFT_DATA_PATH.stat().st_size / 1024:.1f} KB)")
        print(f"  - DPO数据: {DefenseConfig.DPO_DATA_PATH} ({DefenseConfig.DPO_DATA_PATH.stat().st_size / 1024:.1f} KB)")
        print(f"  - 测试数据: {DefenseConfig.TEST_DATA_PATH} ({DefenseConfig.TEST_DATA_PATH.stat().st_size / 1024:.1f} KB)")
        print(Fore.YELLOW + "\n提示: 如需重新生成数据，请先删除现有数据文件")
        return
    
    # 显示缺失的数据文件
    print(Fore.YELLOW + "\n检测到部分数据缺失：")
    if not sft_exists:
        print(Fore.RED + "  ✗ SFT数据不存在")
    else:
        print(Fore.GREEN + f"  ✓ SFT数据存在 ({DefenseConfig.SFT_DATA_PATH.stat().st_size / 1024:.1f} KB)")
    if not dpo_exists:
        print(Fore.RED + "  ✗ DPO数据不存在")
    else:
        print(Fore.GREEN + f"  ✓ DPO数据存在 ({DefenseConfig.DPO_DATA_PATH.stat().st_size / 1024:.1f} KB)")
    if not test_exists:
        print(Fore.RED + "  ✗ 测试数据不存在")
    else:
        print(Fore.GREEN + f"  ✓ 测试数据存在 ({DefenseConfig.TEST_DATA_PATH.stat().st_size / 1024:.1f} KB)")
    
    print(Fore.CYAN + "\n开始生成缺失的数据...")
    
    generator = ImprovedDataGenerator()
    generator.generate_all()
    
    print(Fore.GREEN + "\n✓ 数据生成完成")


def train_models():
    """训练 SFT 和 DPO 模型（不包括数据生成）"""
    print(Fore.YELLOW + "\n训练模型")
    print("=" * 70)
    
    # 检查数据文件是否存在
    if not DefenseConfig.SFT_DATA_PATH.exists():
        print(Fore.RED + f"✗ SFT训练数据不存在: {DefenseConfig.SFT_DATA_PATH}")
        print(Fore.YELLOW + "请先运行: python main.py --generate-data")
        return False
    
    if not DefenseConfig.DPO_DATA_PATH.exists():
        print(Fore.RED + f"✗ DPO训练数据不存在: {DefenseConfig.DPO_DATA_PATH}")
        print(Fore.YELLOW + "请先运行: python main.py --generate-data")
        return False
    
    # 1. SFT 训练
    print(Fore.CYAN + "\n▶ 开始 SFT 训练...")
    try:
        train_sft()
        print(Fore.GREEN + "✓ SFT 训练完成")
    except Exception as e:
        print(Fore.RED + f"✗ SFT 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 2. DPO 训练
    print(Fore.CYAN + "\n▶ 开始 DPO 训练...")
    try:
        train_dpo()
        print(Fore.GREEN + "✓ DPO 训练完成")
    except Exception as e:
        print(Fore.RED + f"✗ DPO 训练失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(Fore.GREEN + "\n✓ 所有训练完成！")
    return True


def run_evaluation(num_workers: int = None):
    """运行评估流程"""
    print(Fore.YELLOW + "\n评估系统性能")
    print("=" * 70)
    
    # 初始化评估器（支持多核并行）
    evaluator = ModelEvaluator(num_workers=num_workers)
    
    # 1. 评估基准模型（裸跑 Qwen 7B）
    print(Fore.CYAN + "\n▶ 评估基准模型...")
    core_llm = CoreLLM()
    
    try:
        y_true_baseline, y_pred_baseline, y_score_baseline = evaluator.evaluate_baseline(core_llm)
        baseline_metrics = evaluator.calculate_metrics(y_true_baseline, y_pred_baseline)
        evaluator.print_metrics(baseline_metrics, "基准模型评估结果")
    except Exception as e:
        print(Fore.RED + f"✗ 基准模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 2. 评估防御系统
    print(Fore.CYAN + "\n▶ 评估防御系统...")
    
    # 初始化防御管理器
    try:
        defense_manager = DefenseManager(use_guard_model=True)
    except Exception as e:
        print(Fore.RED + f"✗ 防御系统初始化失败: {e}")
        print("尝试在不使用 AI 卫士的情况下继续...")
        defense_manager = DefenseManager(use_guard_model=False)
    
    try:
        y_true_defense, y_pred_defense, y_score_defense, block_sources = evaluator.evaluate_defense_system(defense_manager)
        defense_metrics = evaluator.calculate_metrics(y_true_defense, y_pred_defense)
        evaluator.print_metrics(defense_metrics, "防御系统评估结果")
    except Exception as e:
        print(Fore.RED + f"✗ 防御系统评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 对比两个系统
    evaluator.compare_systems(baseline_metrics, defense_metrics)
    
    # 4. 保存结果
    results = evaluator.save_results(
        baseline_metrics, defense_metrics,
        (y_true_baseline, y_pred_baseline, y_score_baseline),
        (y_true_defense, y_pred_defense, y_score_defense, block_sources)
    )
    
    # 5. 生成可视化图表
    print(Fore.CYAN + "\n▶ 生成可视化图表...")
    visualizer = ImprovedResultVisualizer()
    try:
        visualizer.visualize_all()
        print(Fore.GREEN + "\n✓ 评估完成")
    except Exception as e:
        print(Fore.RED + f"✗ 可视化生成失败: {e}")


def interactive_mode():
    """交互式对话模式"""
    print(Fore.YELLOW + "\n进入交互式对话模式")
    print("=" * 70)
    print("输入 'quit' 或 'exit' 退出")
    print("输入 'stats' 查看防御系统统计")
    print("=" * 70 + "\n")
    
    # 初始化防御系统
    try:
        defense_manager = DefenseManager(use_guard_model=True)
    except Exception as e:
        print(Fore.RED + f"⚠ AI 卫士加载失败: {e}")
        print(Fore.YELLOW + "将在不使用 AI 卫士的情况下运行（仅使用关键词过滤）")
        defense_manager = DefenseManager(use_guard_model=False)
    
    # 对话循环
    while True:
        try:
            # 获取用户输入
            user_input = input(Fore.BLUE + "用户> " + Style.RESET_ALL).strip()
            
            if not user_input:
                continue
            
            # 退出命令
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(Fore.YELLOW + "\n再见！")
                break
            
            # 统计命令
            if user_input.lower() == 'stats':
                stats = defense_manager.get_stats()
                print(Fore.CYAN + "\n防御系统统计:")
                for layer, info in stats.items():
                    print(f"  {layer}: {info}")
                print()
                continue
            
            # 处理输入
            result = defense_manager.process(user_input)
            
            if result["success"]:
                # 成功通过防御
                print(Fore.GREEN + "助手> " + Style.RESET_ALL + result["message"])
            else:
                # 被拦截
                print(Fore.RED + "🛡️  [防御系统] " + result["message"])
                print(Fore.YELLOW + f"   拦截层: {result['source']}")
                print(Fore.YELLOW + f"   原因: {result['blocked_by']}")
            
            print()  # 空行
            
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\n再见！")
            break
        except Exception as e:
            print(Fore.RED + f"错误: {e}\n")


def main():
    """主函数"""
    # 打印横幅
    print_banner()
    
    # 打印配置
    DefenseConfig.print_config()
    print()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description="Project Cerberus - AI 纵深防御系统",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--train',
        action='store_true',
        help='运行训练流程（SFT + DPO，不包括数据生成）'
    )
    
    parser.add_argument(
        '--generate-data',
        action='store_true',
        help='生成训练和测试数据'
    )
    
    parser.add_argument(
        '--full-pipeline',
        action='store_true',
        help='运行完整流程（数据生成 + 训练）'
    )
    
    parser.add_argument(
        '--train-sft',
        action='store_true',
        help='仅运行 SFT 训练'
    )
    
    parser.add_argument(
        '--train-dpo',
        action='store_true',
        help='仅运行 DPO 训练'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='运行评估模块'
    )
    
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='生成可视化图表（需要先运行评估）'
    )
    
    parser.add_argument(
        '--num-workers',
        type=int,
        default=None,
        help='评估时使用的工作进程数（默认为CPU核心数）'
    )
    
    args = parser.parse_args()
    
    # 根据参数执行相应功能
    try:
        if args.generate_data:
            generate_data()
        
        elif args.train_sft:
            print(Fore.YELLOW + "\n运行 SFT 训练...")
            train_sft()
        
        elif args.train_dpo:
            print(Fore.YELLOW + "\n运行 DPO 训练...")
            train_dpo()
        
        elif args.train:
            # 仅训练（不生成数据）
            if train_models():
                print(Fore.GREEN + "\n✓ 训练流程完成")
            else:
                print(Fore.RED + "\n✗ 训练过程中出现错误")
        
        elif args.full_pipeline:
            # 完整流程：数据生成 + 训练
            print(Fore.CYAN + "\n运行完整训练流程")
            print("=" * 70)
            generate_data()
            if train_models():
                print(Fore.GREEN + "\n✓ 完整流程全部完成")
            else:
                print(Fore.RED + "\n✗ 训练过程中出现错误")
        
        elif args.evaluate:
            run_evaluation(num_workers=args.num_workers)
        
        elif args.visualize:
            print(Fore.YELLOW + "\n生成可视化图表...")
            visualizer = ImprovedResultVisualizer()
            visualizer.visualize_all()
        
        else:
            # 默认：交互式对话模式
            interactive_mode()
    
    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n程序已终止")
    except Exception as e:
        print(Fore.RED + f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
