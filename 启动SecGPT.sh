#!/bin/bash

# Project Cerberus - SecGPT 环境快速启动脚本

echo "🛡️ Project Cerberus"
echo "环境: SecGPT"
echo ""

# 设置 Ollama 模型存储目录
export OLLAMA_MODELS="/8lab/CHEN/ollama/models"
export OLLAMA_LOG_DIR="/8lab/CHEN/ollama/logs"
export PATH="/usr/local/bin:$PATH"

# 获取 conda 路径
CONDA_BASE=$(conda info --base 2>/dev/null)

if [ -z "$CONDA_BASE" ]; then
    echo "❌ 未找到 Conda"
    echo "请先安装: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 激活环境
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate SecGPT

if [ $? -ne 0 ]; then
    echo "❌ SecGPT 环境未安装"
    echo "请先运行: ./install_SecGPT.sh"
    exit 1
fi

echo "✅ SecGPT 环境已激活"
echo ""

# 显示基础信息
echo "🐍 Python: $(python --version 2>&1)"

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo "🔥 GPU: $GPU_NAME (${GPU_MEM} MB)"
fi

# Ollama 配置
echo "📂 Ollama 模型: $OLLAMA_MODELS"

# 检查并启动 Ollama
if ! pgrep -x "ollama" > /dev/null; then
    echo "⚠️  Ollama 服务未运行，正在启动..."
    mkdir -p "$OLLAMA_LOG_DIR"
    # 确保停止旧的系统服务
    sudo systemctl stop ollama 2>/dev/null || true
    nohup ollama serve > "$OLLAMA_LOG_DIR/ollama.log" 2>&1 &
    sleep 3
    if pgrep -x "ollama" > /dev/null; then
        echo "✅ Ollama 已启动 (PID: $(pgrep -x ollama))"
    else
        echo "❌ Ollama 启动失败，请检查日志: $OLLAMA_LOG_DIR/ollama.log"
    fi
else
    echo "✅ Ollama 服务运行中 (PID: $(pgrep -x ollama))"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 可用命令："
echo "  python main.py --generate-data    # 生成数据"
echo "  python main.py --train            # 完整训练"
echo "  python main.py --train-sft        # 仅 SFT"
echo "  python main.py --train-dpo        # 仅 DPO"
echo "  python main.py --evaluate         # 评估系统"
echo "  python main.py                    # 交互模式"
echo ""
echo "📖 文档："
echo "  cat README.md                     # 项目文档"
echo ""
echo "🔧 环境管理："
echo "  conda deactivate                  # 退出环境"
echo "  nvidia-smi                        # GPU 状态"
echo "  ollama list                       # 查看模型"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 启动交互式 shell
exec bash --rcfile <(echo ". ~/.bashrc; conda activate SecGPT; export OLLAMA_MODELS='/8lab/CHEN/ollama/models'; PS1='(SecGPT) \[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '")
