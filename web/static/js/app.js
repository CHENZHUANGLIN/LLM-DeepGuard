// ==================== 全局变量 ====================
let isLoading = false;
let messageHistory = [];

// ==================== DOM元素 ====================
const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const statsBtn = document.getElementById('statsBtn');
const clearBtn = document.getElementById('clearBtn');
const sidebar = document.getElementById('sidebar');
const closeSidebar = document.getElementById('closeSidebar');
const loadingOverlay = document.getElementById('loadingOverlay');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');

// ==================== 初始化 ====================
document.addEventListener('DOMContentLoaded', () => {
    initializeEventListeners();
    loadStats();
    autoResizeTextarea();
});

// ==================== 事件监听器 ====================
function initializeEventListeners() {
    // 发送按钮
    sendBtn.addEventListener('click', sendMessage);
    
    // 输入框
    messageInput.addEventListener('input', () => {
        autoResizeTextarea();
        updateSendButton();
    });
    
    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // 统计按钮
    statsBtn.addEventListener('click', () => {
        sidebar.classList.add('active');
        loadStats();
    });
    
    // 关闭侧边栏
    closeSidebar.addEventListener('click', () => {
        sidebar.classList.remove('active');
    });
    
    // 清除按钮
    clearBtn.addEventListener('click', clearHistory);
}

// ==================== 发送消息 ====================
async function sendMessage() {
    const message = messageInput.value.trim();
    
    if (!message || isLoading) {
        return;
    }
    
    // 显示用户消息
    addMessage(message, 'user');
    
    // 清空输入框
    messageInput.value = '';
    autoResizeTextarea();
    updateSendButton();
    
    // 显示加载状态
    setLoading(true);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ message: message })
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 显示AI响应
            addMessage(data.message, 'ai', data);
        } else {
            // 显示拦截消息
            addBlockedMessage(message, data);
        }
        
        // 更新统计
        loadStats();
        
    } catch (error) {
        console.error('发送消息失败:', error);
        showToast('发送失败，请检查服务器连接');
        addMessage('抱歉，服务器连接失败，请稍后重试。', 'ai');
    } finally {
        setLoading(false);
    }
}

// ==================== 添加消息到聊天区域 ====================
function addMessage(content, type, data = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const textP = document.createElement('p');
    textP.textContent = content;
    contentDiv.appendChild(textP);
    
    // 添加时间戳
    if (data && data.timestamp) {
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'timestamp';
        timestampDiv.textContent = data.timestamp;
        contentDiv.appendChild(timestampDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    scrollToBottom();
}

// ==================== 添加被拦截消息 ====================
function addBlockedMessage(userMessage, data) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message blocked-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // 拦截标题
    const headerDiv = document.createElement('div');
    headerDiv.className = 'blocked-header';
    headerDiv.innerHTML = `
        <span>🛡️</span>
        <span>请求已被拦截</span>
    `;
    contentDiv.appendChild(headerDiv);
    
    // 拦截消息
    const messageP = document.createElement('p');
    messageP.textContent = data.message;
    contentDiv.appendChild(messageP);
    
    // 详细信息
    if (data.details) {
        const detailsDiv = document.createElement('div');
        detailsDiv.className = 'blocked-details';
        
        detailsDiv.innerHTML = `
            <div class="detail-item">
                <span class="detail-label">拦截层：</span>
                <span class="detail-value">${data.details.layer || '未知'}</span>
            </div>
            <div class="detail-item">
                <span class="detail-label">原因：</span>
                <span class="detail-value">${data.blocked_by || '未知'}</span>
            </div>
        `;
        
        // 显示可疑特征
        if (data.details.suspicious_features && data.details.suspicious_features.length > 0) {
            const featuresDiv = document.createElement('div');
            featuresDiv.className = 'detail-item';
            featuresDiv.innerHTML = `
                <span class="detail-label">检测特征：</span>
            `;
            
            const featuresList = document.createElement('ul');
            featuresList.className = 'features-list';
            data.details.suspicious_features.forEach(feature => {
                const li = document.createElement('li');
                li.textContent = feature;
                featuresList.appendChild(li);
            });
            
            detailsDiv.appendChild(featuresDiv);
            detailsDiv.appendChild(featuresList);
        }
        
        // 显示匹配的关键词
        if (data.details.matched_keywords && data.details.matched_keywords.length > 0) {
            const keywordsDiv = document.createElement('div');
            keywordsDiv.className = 'detail-item';
            keywordsDiv.innerHTML = `
                <span class="detail-label">匹配关键词：</span>
            `;
            
            const keywordsList = document.createElement('ul');
            keywordsList.className = 'features-list';
            data.details.matched_keywords.forEach(keyword => {
                const li = document.createElement('li');
                li.textContent = keyword;
                keywordsList.appendChild(li);
            });
            
            detailsDiv.appendChild(keywordsDiv);
            detailsDiv.appendChild(keywordsList);
        }
        
        // 显示建议
        if (data.details.suggestion) {
            const suggestionDiv = document.createElement('div');
            suggestionDiv.className = 'detail-item';
            suggestionDiv.style.marginTop = 'var(--spacing-md)';
            suggestionDiv.innerHTML = `
                <span class="detail-label">建议：</span>
                <span class="detail-value">${data.details.suggestion}</span>
            `;
            detailsDiv.appendChild(suggestionDiv);
        }
        
        contentDiv.appendChild(detailsDiv);
    }
    
    // 添加时间戳
    if (data.timestamp) {
        const timestampDiv = document.createElement('div');
        timestampDiv.className = 'timestamp';
        timestampDiv.textContent = data.timestamp;
        contentDiv.appendChild(timestampDiv);
    }
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 滚动到底部
    scrollToBottom();
}

// ==================== 加载统计信息 ====================
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            // 更新会话统计
            const stats = data.session_stats;
            document.getElementById('totalRequests').textContent = stats.total_requests || 0;
            document.getElementById('safeRequests').textContent = stats.safe_requests || 0;
            document.getElementById('blockedRequests').textContent = stats.blocked_requests || 0;
            document.getElementById('blockedByKeyword').textContent = stats.blocked_by_layer.keyword_filter || 0;
            document.getElementById('blockedByGuard').textContent = stats.blocked_by_layer.guard_model || 0;
            
            // 更新防御系统状态
            const defenseStats = data.defense_stats;
            const guardEnabled = defenseStats.guard_model.enabled;
            const guardStatusDot = document.getElementById('guardStatusDot');
            const guardStatusText = document.getElementById('guardStatusText');
            
            if (guardEnabled) {
                guardStatusDot.classList.add('active');
                guardStatusText.textContent = 'AI 卫士（已启用）';
            } else {
                guardStatusDot.classList.remove('active');
                guardStatusText.textContent = 'AI 卫士（未启用）';
            }
        }
    } catch (error) {
        console.error('加载统计信息失败:', error);
    }
}

// ==================== 清除历史 ====================
async function clearHistory() {
    if (!confirm('确定要清除所有历史记录吗？')) {
        return;
    }
    
    try {
        const response = await fetch('/api/clear', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            // 清除聊天消息（保留欢迎消息）
            const welcomeMessage = chatMessages.querySelector('.system-message');
            chatMessages.innerHTML = '';
            if (welcomeMessage) {
                chatMessages.appendChild(welcomeMessage);
            }
            
            // 重新加载统计
            loadStats();
            
            showToast('历史记录已清除');
        } else {
            showToast('清除失败：' + data.error);
        }
    } catch (error) {
        console.error('清除历史失败:', error);
        showToast('清除失败，请稍后重试');
    }
}

// ==================== 工具函数 ====================

// 自动调整textarea高度
function autoResizeTextarea() {
    messageInput.style.height = 'auto';
    messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + 'px';
}

// 更新发送按钮状态
function updateSendButton() {
    const hasText = messageInput.value.trim().length > 0;
    sendBtn.disabled = !hasText || isLoading;
}

// 设置加载状态
function setLoading(loading) {
    isLoading = loading;
    updateSendButton();
    
    if (loading) {
        loadingOverlay.classList.add('active');
    } else {
        loadingOverlay.classList.remove('active');
    }
}

// 滚动到底部
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 显示Toast通知
function showToast(message) {
    toastMessage.textContent = message;
    toast.classList.add('show');
    
    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ==================== 自动定期更新统计 ====================
setInterval(() => {
    if (sidebar.classList.contains('active')) {
        loadStats();
    }
}, 5000); // 每5秒更新一次
