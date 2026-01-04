"""
DPO (Direct Preference Optimization) 训练脚本
在 SFT 模型基础上进行偏好对齐
"""

import sys
from pathlib import Path
import torch
from datasets import load_dataset
# 修改 1: 引入 DPOConfig
from trl import DPOTrainer, DPOConfig
from unsloth import FastLanguageModel, is_bfloat16_supported

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from defense.config import DefenseConfig


def format_dpo_dataset(examples):
    """
    格式化 DPO 数据集
    """
    return {
        "prompt": examples["prompt"],
        "chosen": examples["chosen"],
        "rejected": examples["rejected"],
    }


def train_dpo():
    """执行 DPO 训练"""
    
    print("=" * 70)
    print("开始 DPO 训练")
    print("=" * 70)
    
    # 1. 检查 SFT adapter 是否存在
    if not DefenseConfig.GUARD_SFT_ADAPTER_PATH.exists():
        print(f"✗ SFT Adapter 不存在: {DefenseConfig.GUARD_SFT_ADAPTER_PATH}")
        print("请先运行 training/train_sft.py 完成 SFT 训练")
        return
    
    print(f"✓ SFT Adapter 路径: {DefenseConfig.GUARD_SFT_ADAPTER_PATH}")
    
    # 2. 检查数据文件
    if not DefenseConfig.DPO_DATA_PATH.exists():
        print(f"✗ 数据文件不存在: {DefenseConfig.DPO_DATA_PATH}")
        print("请先运行 data/generate_data.py 生成训练数据")
        return
    
    print(f"✓ 数据文件: {DefenseConfig.DPO_DATA_PATH}")
    
    # 3. 加载基础模型和 SFT adapter
    print(f"\n正在加载模型: {DefenseConfig.GUARD_MODEL_ID}")
    print("并应用 SFT adapter...")
    
    # DPO 需要更长的序列长度，通常需要重新加载模型
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DefenseConfig.GUARD_MODEL_ID,
        max_seq_length=DefenseConfig.DPO_TRAINING_CONFIG["max_length"],
        dtype=None,
        load_in_4bit=True,
    )
    
    print("✓ 基础模型加载完成")
    
    # 配置 LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=DefenseConfig.SFT_TRAINING_CONFIG["lora_r"],
        target_modules=DefenseConfig.SFT_TRAINING_CONFIG["target_modules"],
        lora_alpha=DefenseConfig.SFT_TRAINING_CONFIG["lora_alpha"],
        lora_dropout=DefenseConfig.SFT_TRAINING_CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # 加载 SFT 训练好的权重
    try:
        from peft import PeftModel
        # 注意: 这种加载方式在 Unsloth 中是可行的，但要确保路径正确
        model = PeftModel.from_pretrained(model, str(DefenseConfig.GUARD_SFT_ADAPTER_PATH))
        print("✓ SFT Adapter 加载成功")
    except Exception as e:
        print(f"⚠ SFT Adapter 加载失败: {e}")
        print("停止训练，请检查 SFT Adapter 路径。")
        return # DPO 必须基于 SFT 模型，如果加载失败不建议继续
    
    # 4. 加载 DPO 数据集
    print(f"\n正在加载 DPO 数据集: {DefenseConfig.DPO_DATA_PATH}")
    
    dataset = load_dataset(
        "json",
        data_files=str(DefenseConfig.DPO_DATA_PATH),
        split="train"
    )
    
    print(f"✓ 数据集加载完成")
    print(f"  - 样本数量: {len(dataset)}")
    
    # 格式化数据
    dataset = dataset.map(
        format_dpo_dataset,
        batched=True,
    )
    
    # 5. 配置 DPO 训练参数 (核心修改部分)
    print("\n配置 DPO 训练参数...")
    
    # 修改 2: 使用 DPOConfig 替代 TrainingArguments
    training_args = DPOConfig(
        output_dir=str(DefenseConfig.GUARD_DPO_ADAPTER_PATH),
        per_device_train_batch_size=DefenseConfig.DPO_TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=DefenseConfig.DPO_TRAINING_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=DefenseConfig.DPO_TRAINING_CONFIG["num_train_epochs"],
        learning_rate=DefenseConfig.DPO_TRAINING_CONFIG["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=DefenseConfig.DPO_TRAINING_CONFIG["logging_steps"],
        optim=DefenseConfig.DPO_TRAINING_CONFIG["optim"],
        gradient_checkpointing=DefenseConfig.DPO_TRAINING_CONFIG["gradient_checkpointing"],
        save_strategy=DefenseConfig.DPO_TRAINING_CONFIG["save_strategy"],
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        
        # --- DPO 特定参数现在必须在这里定义 ---
        beta=DefenseConfig.DPO_TRAINING_CONFIG["beta"],
        max_length=DefenseConfig.DPO_TRAINING_CONFIG["max_length"],
        max_prompt_length=DefenseConfig.DPO_TRAINING_CONFIG["max_prompt_length"],
        # ----------------------------------
    )
    
    print("✓ 训练参数配置完成")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Beta: {training_args.beta}") 
    
    # 6. 创建 DPO 训练器
    print("\n创建 DPO 训练器...")
    
    # 修改 3: 初始化时移除 beta, max_length 等参数，只传 args
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None, # Unsloth 优化，不需要显式加载参考模型
        args=training_args, # 传入 DPOConfig
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    print("✓ DPO 训练器创建完成")
    
    # 7. 开始训练
    print("\n" + "=" * 70)
    print("开始 DPO 训练...")
    print("=" * 70 + "\n")
    
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_stats.name}")
        print(f"显存: {gpu_stats.total_memory / 1024**3:.2f} GB")
        print(f"当前已用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB\n")
    
    dpo_trainer.train()
    
    # 8. 保存模型
    print("\n" + "=" * 70)
    print("训练完成，正在保存模型...")
    print("=" * 70)
    
    DefenseConfig.GUARD_DPO_ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    
    model.save_pretrained(str(DefenseConfig.GUARD_DPO_ADAPTER_PATH))
    tokenizer.save_pretrained(str(DefenseConfig.GUARD_DPO_ADAPTER_PATH))
    
    print(f"✓ 模型已保存到: {DefenseConfig.GUARD_DPO_ADAPTER_PATH}")
    
    if torch.cuda.is_available():
        print(f"\n最终显存使用: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 70)
    print("DPO 训练完成！")
    print("=" * 70)
    
    return model, tokenizer


if __name__ == "__main__":
    train_dpo()