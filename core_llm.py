"""
核心 LLM 接口
通过 Ollama 调用本地 Qwen 2.5 7B 模型
"""

import requests
import json
from typing import Dict, Optional
from defense.config import DefenseConfig


class CoreLLM:
    """核心对话模型（通过 Ollama API）"""
    
    def __init__(self, 
                 url: str = None,
                 model: str = None,
                 temperature: float = None,
                 max_tokens: int = None):
        """
        初始化核心 LLM
        
        Args:
            url: Ollama API 地址
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大生成 token 数
        """
        self.url = url or DefenseConfig.MAIN_LLM_URL
        self.model = model or DefenseConfig.MAIN_LLM_MODEL
        self.temperature = temperature or DefenseConfig.MAIN_LLM_TEMPERATURE
        self.max_tokens = max_tokens or DefenseConfig.MAIN_LLM_MAX_TOKENS
        
        print(f"初始化核心 LLM:")
        print(f"  - URL: {self.url}")
        print(f"  - Model: {self.model}")
        
        # 检查连接
        self._check_connection()
    
    def _check_connection(self):
        """检查 Ollama 服务是否可用"""
        try:
            # 尝试连接到 Ollama
            response = requests.get(self.url.replace("/api/chat", "/api/tags"), timeout=5)
            if response.status_code == 200:
                print("✓ Ollama 服务连接正常\n")
            else:
                print(f"⚠ Ollama 服务响应异常: {response.status_code}\n")
        except requests.exceptions.RequestException as e:
            print(f"⚠ 无法连接到 Ollama 服务: {e}")
            print("请确保 Ollama 已启动并且模型已下载")
            print(f"运行: ollama run {self.model}\n")
    
    def chat(self, 
             user_message: str, 
             system_prompt: Optional[str] = None,
             temperature: Optional[float] = None) -> str:
        """
        与 LLM 对话
        
        Args:
            user_message: 用户消息
            system_prompt: 系统提示词（可选）
            temperature: 温度参数（可选）
            
        Returns:
            LLM 的响应文本
        """
        messages = []
        
        # 添加系统提示词
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        # 添加用户消息
        messages.append({
            "role": "user",
            "content": user_message
        })
        
        # 构造请求
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                error_msg = f"API 错误: {response.status_code}"
                print(f"✗ {error_msg}")
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "请求超时"
            print(f"✗ {error_msg}")
            return error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"请求失败: {str(e)}"
            print(f"✗ {error_msg}")
            return error_msg
        except json.JSONDecodeError:
            error_msg = "响应解析失败"
            print(f"✗ {error_msg}")
            return error_msg
    
    def chat_with_history(self, messages: list) -> str:
        """
        带历史记录的对话
        
        Args:
            messages: 消息历史列表
            
        Returns:
            LLM 的响应文本
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            }
        }
        
        try:
            response = requests.post(
                self.url,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            else:
                return f"API 错误: {response.status_code}"
                
        except Exception as e:
            return f"请求失败: {str(e)}"
    
    def __repr__(self):
        return f"CoreLLM(model={self.model}, url={self.url})"


if __name__ == "__main__":
    # 测试代码
    print("=" * 70)
    print("核心 LLM 测试")
    print("=" * 70)
    
    llm = CoreLLM()
    
    # 测试对话
    test_messages = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
    ]
    
    for msg in test_messages:
        print(f"\n用户: {msg}")
        response = llm.chat(msg)
        print(f"助手: {response[:200]}...")

