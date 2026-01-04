#!/bin/bash
set -e

# --- 全局配置 ---
export OLLAMA_MODELS="/8lab/CHEN/ollama/models"
export OLLAMA_LOG_DIR="/8lab/CHEN/ollama/logs"
ENV_NAME="SecGPT"
TARGET_MODEL="qwen2.5:7b"

echo "🛡️ Project Cerberus 环境检查与配置"
echo "📂 模型路径: $OLLAMA_MODELS"

# ===============================================
# [1/5] 系统与磁盘检查
# ===============================================
echo ">>> [1/5] 检查系统资源..."

# 1. 检查数据盘
if [ ! -d "/8lab" ]; then
    echo "❌ 错误: 未找到 /8lab 挂载点。"
    exit 1
fi

# 2. 检查权限
if [ ! -w "$OLLAMA_MODELS" ]; then
    echo "   - 修正目录权限..."
    sudo mkdir -p "$OLLAMA_MODELS" "$OLLAMA_LOG_DIR"
    sudo chown -R $(whoami) "/8lab/CHEN"
    sudo chmod -R 755 "/8lab/CHEN"
else
    echo "   - ✅ 权限检查通过"
fi

# ===============================================
# [2/5] Conda 环境配置 (智能跳过)
# ===============================================
echo ">>> [2/5] 检查 Conda 环境..."
source $(conda info --base)/etc/profile.d/conda.sh

if conda env list | grep -q "$ENV_NAME"; then
    echo "   - ⏭️  环境 $ENV_NAME 已存在，直接激活..."
    conda activate $ENV_NAME
else
    echo "   - 📦 创建新环境 $ENV_NAME..."
    conda create -n $ENV_NAME python=3.10 -y
    conda activate $ENV_NAME
fi

# ===============================================
# [3/5] 依赖安装 (智能跳过)
# ===============================================
echo ">>> [3/5] 检查 Python 依赖..."

# 检查所有关键依赖是否都已安装
MISSING_PACKAGES=()
for pkg in torch transformers peft unsloth trl; do
    if ! pip show $pkg &> /dev/null; then
        MISSING_PACKAGES+=($pkg)
    fi
done

if [ ${#MISSING_PACKAGES[@]} -eq 0 ]; then
    echo "   - ⏭️  所有核心依赖已安装，跳过初始安装流程。"
else
    echo "   - ⬇️  检测到缺失依赖: ${MISSING_PACKAGES[*]}"
    echo "   - 📦 开始安装依赖..."
    pip install --upgrade pip > /dev/null
    
    # PyTorch
    pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
    
    # 核心库
    pip install --no-cache-dir transformers==4.40.0 peft==0.10.0 bitsandbytes==0.43.0 \
        scikit-learn==1.3.0 datasets==2.18.0 trl==0.8.1 accelerate==0.28.0 \
        xformers==0.0.25 ollama==0.1.6 colorama==0.4.6 matplotlib==3.7.0 \
        pandas==2.0.0 numpy==1.24.0
        
    # Unsloth
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
fi

# 升级关键依赖到最新兼容版本（每次都执行以确保版本兼容）
echo "   - 🔄 升级关键依赖到兼容版本..."
pip install --upgrade torchvision
pip install --upgrade peft
pip install --upgrade unsloth

# ===============================================
# [4/5] Ollama 服务 (智能跳过)
# ===============================================
echo ">>> [4/5] 检查 Ollama 服务..."

# 写入配置到 bashrc (如果不存在)
if ! grep -q "OLLAMA_MODELS" ~/.bashrc; then
    echo "export OLLAMA_MODELS=\"$OLLAMA_MODELS\"" >> ~/.bashrc
fi

# 检查服务状态
if pgrep -x "ollama" > /dev/null; then
    echo "   - ⏭️  Ollama 服务正在运行 (PID: $(pgrep -x ollama))，跳过启动。"
else
    echo "   - 🚀 启动 Ollama 服务..."
    # 确保停止旧的系统服务
    sudo systemctl stop ollama 2>/dev/null || true
    nohup ollama serve > "$OLLAMA_LOG_DIR/ollama.log" 2>&1 &
    sleep 5
fi

# ===============================================
# [5/5] 模型状态 (智能跳过)
# ===============================================
echo ">>> [5/5] 检查模型状态..."

if ollama list | grep -q "$TARGET_MODEL"; then
    echo "   - ⏭️  模型 $TARGET_MODEL 已存在，跳过下载。"
else
    echo "   - 📥 未检测到模型，开始下载 $TARGET_MODEL..."
    ollama pull $TARGET_MODEL
fi

# ===============================================
# 最终摘要
# ===============================================
echo ""
echo "✅ ============================================"
echo "   配置检查完成"
echo "   环境: $ENV_NAME"
echo "   模型: $(ollama list | grep $TARGET_MODEL || echo '未找到')"
echo "==============================================="