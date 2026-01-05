"""
SFT (Supervised Fine-Tuning) 训练脚本
使用 Unsloth 框架训练 Qwen 2.5 3B 模型
"""

import os
# 使用 ModelScope 作为模型源（解决 HuggingFace 连接问题）
os.environ['UNSLOTH_USE_MODELSCOPE'] = '1'

import sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import TrainingArguments
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))
from defense.config import DefenseConfig


def format_prompts(examples):
    """
    格式化训练数据为标准格式
    
    Args:
        examples: 数据集样本
        
    Returns:
        格式化后的文本列表
    """
    texts = []
    for conversations in examples["conversations"]:
        # 构造对话文本
        text = ""
        for msg in conversations:
            if msg["from"] == "human":
                text += f"### 用户输入:\n{msg['value']}\n\n"
            elif msg["from"] == "gpt":
                text += f"### 判断结果:\n{msg['value']}\n"
        texts.append(text)
    
    return {"text": texts}


def train_sft():
    """执行 SFT 训练"""
    
    print("=" * 70)
    print("开始 SFT 训练")
    print("=" * 70)
    
    # 1. 检查数据文件
    if not DefenseConfig.SFT_DATA_PATH.exists():
        print(f"✗ 数据文件不存在: {DefenseConfig.SFT_DATA_PATH}")
        print("请先生成训练数据:")
        print("  - 使用本地生成: python data/generate_data.py")
        print("  - 使用API生成:  python data/generate_data_with_api.py")
        print("  或运行: python main.py --generate-data")
        return
    
    print(f"✓ 数据文件: {DefenseConfig.SFT_DATA_PATH}")
    
    # 2. 加载模型
    print(f"\n正在加载模型: {DefenseConfig.GUARD_MODEL_ID}")
    
    # 判断是否为本地路径（包含路径分隔符 / 或 \）
    is_local_path = ('/' in DefenseConfig.GUARD_MODEL_ID or '\\' in DefenseConfig.GUARD_MODEL_ID)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=DefenseConfig.GUARD_MODEL_ID,
        max_seq_length=DefenseConfig.SFT_TRAINING_CONFIG["max_seq_length"],
        dtype=None,  # 自动选择
        load_in_4bit=DefenseConfig.SFT_TRAINING_CONFIG["load_in_4bit"],
        local_files_only=is_local_path,  # 本地路径时禁用网络检查
    )
    
    print("✓ 基础模型加载完成")
    
    # 3. 配置 LoRA
    print("\n配置 LoRA adapter...")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=DefenseConfig.SFT_TRAINING_CONFIG["lora_r"],
        target_modules=DefenseConfig.SFT_TRAINING_CONFIG["target_modules"],
        lora_alpha=DefenseConfig.SFT_TRAINING_CONFIG["lora_alpha"],
        lora_dropout=DefenseConfig.SFT_TRAINING_CONFIG["lora_dropout"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth 的优化
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✓ LoRA 配置完成")
    print(f"  - r: {DefenseConfig.SFT_TRAINING_CONFIG['lora_r']}")
    print(f"  - alpha: {DefenseConfig.SFT_TRAINING_CONFIG['lora_alpha']}")
    print(f"  - target_modules: {len(DefenseConfig.SFT_TRAINING_CONFIG['target_modules'])} 个")
    
    # 4. 加载数据集
    print(f"\n正在加载数据集: {DefenseConfig.SFT_DATA_PATH}")
    
    dataset = load_dataset(
        "json",
        data_files=str(DefenseConfig.SFT_DATA_PATH),
        split="train"
    )
    
    print(f"✓ 数据集加载完成")
    print(f"  - 样本数量: {len(dataset)}")
    
    # 格式化数据
    dataset = dataset.map(
        format_prompts,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # 5. 配置训练参数
    print("\n配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir=str(DefenseConfig.GUARD_SFT_ADAPTER_PATH),
        per_device_train_batch_size=DefenseConfig.SFT_TRAINING_CONFIG["per_device_train_batch_size"],
        gradient_accumulation_steps=DefenseConfig.SFT_TRAINING_CONFIG["gradient_accumulation_steps"],
        num_train_epochs=DefenseConfig.SFT_TRAINING_CONFIG["num_train_epochs"],
        learning_rate=DefenseConfig.SFT_TRAINING_CONFIG["learning_rate"],
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=DefenseConfig.SFT_TRAINING_CONFIG["logging_steps"],
        optim=DefenseConfig.SFT_TRAINING_CONFIG["optim"],
        weight_decay=DefenseConfig.SFT_TRAINING_CONFIG["weight_decay"],
        lr_scheduler_type=DefenseConfig.SFT_TRAINING_CONFIG["lr_scheduler_type"],
        warmup_steps=DefenseConfig.SFT_TRAINING_CONFIG["warmup_steps"],
        save_strategy=DefenseConfig.SFT_TRAINING_CONFIG["save_strategy"],
        seed=3407,
        report_to="none",  # 不使用 wandb 等
    )
    
    print("✓ 训练参数配置完成")
    print(f"  - Batch size: {training_args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {training_args.gradient_accumulation_steps}")
    print(f"  - Epochs: {training_args.num_train_epochs}")
    print(f"  - Learning rate: {training_args.learning_rate}")
    
    # 6. 创建训练器
    print("\n创建 SFT 训练器...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=DefenseConfig.SFT_TRAINING_CONFIG["max_seq_length"],
        args=training_args,
    )
    
    print("✓ 训练器创建完成")
    
    # 7. 开始训练
    print("\n" + "=" * 70)
    print("开始训练...")
    print("=" * 70 + "\n")
    
    # 显示显存使用情况
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(0)
        print(f"GPU: {gpu_stats.name}")
        print(f"显存: {gpu_stats.total_memory / 1024**3:.2f} GB")
        print(f"当前已用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB\n")
    
    trainer.train()
    
    # 8. 保存模型
    print("\n" + "=" * 70)
    print("训练完成，正在保存模型...")
    print("=" * 70)
    
    # 创建保存目录
    DefenseConfig.GUARD_SFT_ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    
    # 保存 LoRA adapter
    model.save_pretrained(str(DefenseConfig.GUARD_SFT_ADAPTER_PATH))
    tokenizer.save_pretrained(str(DefenseConfig.GUARD_SFT_ADAPTER_PATH))
    
    print(f"✓ 模型已保存到: {DefenseConfig.GUARD_SFT_ADAPTER_PATH}")
    
    # 显示最终显存使用
    if torch.cuda.is_available():
        print(f"\n最终显存使用: {torch.cuda.max_memory_allocated(0) / 1024**3:.2f} GB")
    
    print("\n" + "=" * 70)
    print("SFT 训练完成！")
    print("=" * 70)
    
    return model, tokenizer


if __name__ == "__main__":
    train_sft()

