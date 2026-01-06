"""
Web应用后端
提供前端界面的API服务
"""

import os
# 设置离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
from defense_manager import DefenseManager
from colorama import Fore

app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')
CORS(app)

# 全局防御管理器
defense_manager = None

# 会话历史（简单的内存存储）
session_history = []
session_stats = {
    "total_requests": 0,
    "blocked_requests": 0,
    "safe_requests": 0,
    "blocked_by_layer": {
        "keyword_filter": 0,
        "guard_model": 0,
    }
}


def init_defense_manager():
    """初始化防御管理器"""
    global defense_manager
    try:
        print(Fore.CYAN + "\n正在初始化防御系统...")
        defense_manager = DefenseManager(use_guard_model=True)
        print(Fore.GREEN + "✓ 防御系统初始化成功\n")
        return True
    except Exception as e:
        print(Fore.RED + f"✗ 防御系统初始化失败: {e}")
        print(Fore.YELLOW + "尝试在不使用AI卫士的情况下运行...")
        try:
            defense_manager = DefenseManager(use_guard_model=False)
            print(Fore.GREEN + "✓ 防御系统初始化成功（无AI卫士）\n")
            return True
        except Exception as e2:
            print(Fore.RED + f"✗ 完全初始化失败: {e2}\n")
            return False


@app.route('/')
def index():
    """主页"""
    return render_template('index.html')


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    global session_history, session_stats
    
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({
                'success': False,
                'error': '消息不能为空'
            }), 400
        
        # 处理输入
        result = defense_manager.process(user_message)
        
        # 更新统计信息
        session_stats["total_requests"] += 1
        
        if result["success"]:
            session_stats["safe_requests"] += 1
        else:
            session_stats["blocked_requests"] += 1
            if result["source"] in session_stats["blocked_by_layer"]:
                session_stats["blocked_by_layer"][result["source"]] += 1
        
        # 添加到历史
        history_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_message": user_message,
            "result": result,
        }
        session_history.append(history_entry)
        
        # 只保留最近100条
        if len(session_history) > 100:
            session_history = session_history[-100:]
        
        # 返回响应
        return jsonify({
            'success': result["success"],
            'message': result["message"],
            'source': result["source"],
            'blocked_by': result.get("blocked_by"),
            'details': result.get("details", {}),
            'timestamp': history_entry["timestamp"]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """获取系统统计信息"""
    try:
        defense_stats = defense_manager.get_stats()
        
        return jsonify({
            'success': True,
            'session_stats': session_stats,
            'defense_stats': defense_stats,
            'history_count': len(session_history)
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/history', methods=['GET'])
def get_history():
    """获取会话历史"""
    try:
        # 获取最近的N条记录
        limit = request.args.get('limit', 50, type=int)
        
        return jsonify({
            'success': True,
            'history': session_history[-limit:]
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/clear', methods=['POST'])
def clear_history():
    """清除会话历史"""
    global session_history, session_stats
    
    try:
        session_history = []
        session_stats = {
            "total_requests": 0,
            "blocked_requests": 0,
            "safe_requests": 0,
            "blocked_by_layer": {
                "keyword_filter": 0,
                "guard_model": 0,
            }
        }
        
        return jsonify({
            'success': True,
            'message': '历史记录已清除'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


def main():
    """启动Web应用"""
    print("=" * 70)
    print("🛡️  Project Cerberus - Web界面")
    print("=" * 70)
    
    # 初始化防御系统
    if not init_defense_manager():
        print(Fore.RED + "无法启动Web应用：防御系统初始化失败")
        return
    
    # 启动Flask应用
    print(Fore.CYAN + "\n启动Web服务器...")
    print(Fore.GREEN + "✓ 服务器地址: http://localhost:5000")
    print(Fore.YELLOW + "\n按 Ctrl+C 停止服务器\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
