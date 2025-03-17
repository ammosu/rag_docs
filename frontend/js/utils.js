/**
 * 工具函數集合
 */

// 顯示提示對話框
function showAlert(title, message) {
    const modal = document.getElementById('alertModal');
    const titleEl = document.getElementById('alertTitle');
    const messageEl = document.getElementById('alertMessage');
    
    titleEl.textContent = title;
    messageEl.textContent = message;
    
    modal.classList.remove('hidden');
    modal.classList.add('fade-in');
    
    document.getElementById('alertCloseBtn').onclick = function() {
        modal.classList.add('hidden');
    };
}

// API 請求包裝函數
async function apiRequest(endpoint, method = 'GET', data = null) {
    const url = `${CONFIG.API.BASE_URL}${endpoint}`;
    
    const options = {
        method,
        headers: {
            'Accept': 'application/json'
        }
    };
    
    if (data) {
        if (data instanceof FormData) {
            options.body = data;
        } else {
            options.headers['Content-Type'] = 'application/json';
            options.body = JSON.stringify(data);
        }
    }
    
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || '請求失敗');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API 請求錯誤:', error);
        showAlert('錯誤', error.message || '發生未知錯誤');
        throw error;
    }
}

// 日期格式化
function formatDate(dateString) {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('zh-TW', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit'
    }).format(date);
}

// 複製文本到剪貼板
function copyToClipboard(text) {
    return new Promise((resolve, reject) => {
        navigator.clipboard.writeText(text)
            .then(() => resolve(true))
            .catch(err => {
                console.error('無法複製到剪貼板:', err);
                
                // 備用方法
                try {
                    const textarea = document.createElement('textarea');
                    textarea.value = text;
                    textarea.style.position = 'fixed';
                    document.body.appendChild(textarea);
                    textarea.focus();
                    textarea.select();
                    
                    const successful = document.execCommand('copy');
                    document.body.removeChild(textarea);
                    
                    if (successful) {
                        resolve(true);
                    } else {
                        reject(new Error('複製操作失敗'));
                    }
                } catch (err) {
                    reject(err);
                }
            });
    });
}

// 截斷長文本
function truncateText(text, maxLength = 100) {
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// 本地儲存管理
const storage = {
    set: (key, value) => {
        localStorage.setItem(key, value);
    },
    get: (key) => {
        return localStorage.getItem(key);
    },
    remove: (key) => {
        localStorage.removeItem(key);
    },
    clear: () => {
        localStorage.clear();
    }
};

// 檢查檔案類型是否允許
function isAllowedFileType(filename) {
    const ext = '.' + filename.split('.').pop().toLowerCase();
    return CONFIG.UPLOAD.ALLOWED_TYPES.includes(ext);
}

// 格式化檔案大小
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// 防抖函數
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        const context = this;
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(context, args), wait);
    };
}

// 檢查是否有有效的使用者 ID
function hasValidUserId() {
    const userId = storage.get(CONFIG.STORAGE.USER_ID);
    return userId && userId.trim() !== '';
}

// 檢查是否有有效的加密金鑰
function hasValidKey() {
    const dbType = document.getElementById('dbType').value;
    
    // 如果不需要金鑰，直接返回 true
    if (dbType === CONFIG.DB_TYPES.CHROMA) {
        return true;
    }
    
    const key = storage.get(CONFIG.STORAGE.ENCRYPTION_KEY);
    return key && key.trim() !== '';
}

// 更新 UI 狀態 (按鈕啟用/禁用等)
function updateUIState() {
    const hasUserId = hasValidUserId();
    const hasKey = hasValidKey();
    
    // 更新上傳按鈕狀態
    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) {
        uploadBtn.disabled = !hasUserId;
    }
    
    // 更新搜尋按鈕狀態
    const searchBtn = document.getElementById('searchBtn');
    if (searchBtn) {
        searchBtn.disabled = !hasUserId;
    }
    
    // 其他 UI 元素狀態更新...
}

// 格式化資料庫類型顯示名稱
function formatDbType(dbType) {
    if (dbType === CONFIG.DB_TYPES.CHROMA) {
        return 'Chroma DB';
    } else if (dbType === CONFIG.DB_TYPES.PINECONE_BYOK) {
        return 'Pinecone BYOK';
    }
    return dbType;
}

// 獲取當前選擇的資料庫類型
function getCurrentDbType() {
    return document.getElementById('dbType').value;
}

// 獲取搜尋資料庫類型
function getSearchDbType() {
    const selectedRadio = document.querySelector('input[name="searchDbType"]:checked');
    return selectedRadio ? selectedRadio.value : 'all';
}