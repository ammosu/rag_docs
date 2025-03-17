/**
 * 金鑰管理模組
 * 處理 BYOK (Bring Your Own Key) 相關功能
 */

// 初始化金鑰管理模組
function initKeyManagement() {
    // 恢復存儲的使用者 ID
    const savedUserId = storage.get(CONFIG.STORAGE.USER_ID);
    if (savedUserId) {
        document.getElementById('userId').value = savedUserId;
    }
    
    // 使用者 ID 輸入監聽
    document.getElementById('userId').addEventListener('input', function(e) {
        const userId = e.target.value.trim();
        storage.set(CONFIG.STORAGE.USER_ID, userId);
        updateUIState();
    });
    
    // 金鑰管理按鈕事件
    document.getElementById('generateKeyBtn').addEventListener('click', generateNewKey);
    document.getElementById('uploadKeyBtn').addEventListener('click', toggleKeyUploadForm);
    document.getElementById('registerKeyBtn').addEventListener('click', registerCustomKey);
    document.getElementById('copyKeyBtn').addEventListener('click', copyCurrentKey);
    document.getElementById('rotateKeyBtn').addEventListener('click', rotateKey);
    
    // 恢復存儲的加密金鑰
    const savedKey = storage.get(CONFIG.STORAGE.ENCRYPTION_KEY);
    if (savedKey) {
        document.getElementById('currentKey').value = savedKey;
        document.getElementById('currentKeyArea').classList.remove('hidden');
    }
    
    // 更新初始 UI 狀態
    updateUIState();
}

// 生成新的加密金鑰
async function generateNewKey() {
    const userId = document.getElementById('userId').value.trim();
    
    if (!userId) {
        showAlert('錯誤', '請先輸入用戶 ID');
        return;
    }
    
    try {
        const response = await apiRequest(CONFIG.API.KEY, 'POST', { user_id: userId });
        
        const key = response.key;
        storage.set(CONFIG.STORAGE.ENCRYPTION_KEY, key);
        
        document.getElementById('currentKey').value = key;
        document.getElementById('currentKeyArea').classList.remove('hidden');
        
        showAlert('成功', '已成功生成新的加密金鑰');
        updateUIState();
    } catch (error) {
        console.error('生成金鑰失敗:', error);
    }
}

// 切換金鑰上傳表單顯示
function toggleKeyUploadForm() {
    const form = document.getElementById('keyUploadForm');
    form.classList.toggle('hidden');
}

// 註冊自定義金鑰
async function registerCustomKey() {
    const userId = document.getElementById('userId').value.trim();
    const customKey = document.getElementById('customKey').value.trim();
    
    if (!userId) {
        showAlert('錯誤', '請先輸入用戶 ID');
        return;
    }
    
    if (!customKey) {
        showAlert('錯誤', '請輸入您的加密金鑰');
        return;
    }
    
    try {
        const response = await apiRequest(CONFIG.API.KEY, 'POST', { 
            user_id: userId,
            key: customKey 
        });
        
        const key = response.key;
        storage.set(CONFIG.STORAGE.ENCRYPTION_KEY, key);
        
        document.getElementById('currentKey').value = key;
        document.getElementById('currentKeyArea').classList.remove('hidden');
        document.getElementById('keyUploadForm').classList.add('hidden');
        
        showAlert('成功', '已成功註冊您的加密金鑰');
        updateUIState();
    } catch (error) {
        console.error('註冊金鑰失敗:', error);
    }
}

// 複製當前金鑰到剪貼板
async function copyCurrentKey() {
    const currentKey = document.getElementById('currentKey').value;
    
    try {
        await copyToClipboard(currentKey);
        
        // 視覺反饋
        const copyBtn = document.getElementById('copyKeyBtn');
        copyBtn.classList.add('copy-success');
        copyBtn.textContent = '已複製';
        
        setTimeout(() => {
            copyBtn.classList.remove('copy-success');
            copyBtn.textContent = '複製';
        }, 2000);
    } catch (error) {
        showAlert('錯誤', '複製金鑰失敗');
    }
}

// 輪替金鑰
async function rotateKey() {
    const userId = document.getElementById('userId').value.trim();
    const oldKey = document.getElementById('currentKey').value.trim();
    
    if (!userId || !oldKey) {
        showAlert('錯誤', '缺少用戶 ID 或當前金鑰');
        return;
    }
    
    if (!confirm('輪替金鑰將使用新金鑰重新加密所有文檔。此操作無法撤銷，是否繼續？')) {
        return;
    }
    
    try {
        const response = await apiRequest(CONFIG.API.ROTATE_KEY, 'POST', {
            user_id: userId,
            old_key: oldKey
            // 未提供 new_key，由服務器生成
        });
        
        const newKey = response.new_key;
        storage.set(CONFIG.STORAGE.ENCRYPTION_KEY, newKey);
        
        document.getElementById('currentKey').value = newKey;
        
        showAlert('成功', `已成功輪替金鑰。已更新 ${response.vectors_updated} 個向量。`);
    } catch (error) {
        console.error('輪替金鑰失敗:', error);
    }
}

// 檢驗金鑰有效性
function validateKey(key) {
    // 檢查是否為有效的 base64 字符串
    try {
        return btoa(atob(key)) === key;
    } catch (err) {
        return false;
    }
}

// 金鑰安全評估
function assessKeySecurityStrength(key) {
    // 基本評估金鑰強度
    if (!key) return 'weak';
    
    if (key.length < 16) return 'weak';
    if (key.length < 32) return 'medium';
    return 'strong';
}
