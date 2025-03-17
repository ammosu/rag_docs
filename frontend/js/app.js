/**
 * 主應用程式
 * 初始化和協調所有功能模組
 */

// 當 DOM 內容加載完成時執行
document.addEventListener('DOMContentLoaded', function() {
    initApp();
});

// 初始化應用程式
function initApp() {
    // 初始化警告對話框
    initAlertModal();
    
    // 初始化各功能模組
    initKeyManagement();
    initDocumentManagement();
    initSearch();
    
    // 文件上傳輸入框監聽
    document.getElementById('fileUpload').addEventListener('change', function(e) {
        const file = e.target.files[0];
        const uploadBtn = document.getElementById('uploadBtn');
        
        if (file && hasValidUserId()) {
            uploadBtn.disabled = false;
        } else {
            uploadBtn.disabled = true;
        }
    });
    
    // 取得支援的資料庫類型
    fetchSupportedDbTypes();
    
    // 檢查瀏覽器支援
    checkBrowserSupport();
    
    console.log('RAG 向量資料庫應用程式已初始化');
}

// 初始化提示對話框
function initAlertModal() {
    const modal = document.getElementById('alertModal');
    const closeBtn = document.getElementById('alertCloseBtn');
    
    // 點擊關閉按鈕關閉對話框
    closeBtn.addEventListener('click', function() {
        modal.classList.add('hidden');
    });
    
    // 點擊對話框外部也關閉
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
    
    // ESC 鍵關閉
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
            modal.classList.add('hidden');
        }
    });
}

// 獲取支援的資料庫類型
async function fetchSupportedDbTypes() {
    try {
        const response = await apiRequest(CONFIG.API.DB_TYPES);
        
        // 填充資料庫類型選擇器
        const dbTypeSelect = document.getElementById('dbType');
        dbTypeSelect.innerHTML = '';
        
        response.db_types.forEach(type => {
            const option = document.createElement('option');
            option.value = type.id;
            option.textContent = `${type.name}${type.requires_key ? ' (需要加密金鑰)' : ''}`;
            option.setAttribute('data-requires-key', type.requires_key);
            dbTypeSelect.appendChild(option);
        });
        
        // 恢復存儲的選擇
        const savedDbType = storage.get(CONFIG.STORAGE.DB_TYPE);
        if (savedDbType) {
            dbTypeSelect.value = savedDbType;
        }
        
        // 觸發變更事件以更新 UI
        dbTypeSelect.dispatchEvent(new Event('change'));
        
    } catch (error) {
        console.error('獲取資料庫類型失敗:', error);
        // 使用硬編碼的默認選項
        const dbTypeSelect = document.getElementById('dbType');
        dbTypeSelect.innerHTML = `
            <option value="${CONFIG.DB_TYPES.CHROMA}">Chroma DB (預設，無需加密金鑰)</option>
            <option value="${CONFIG.DB_TYPES.PINECONE_BYOK}">Pinecone with BYOK (需要加密金鑰)</option>
        `;
    }
}

// 檢查瀏覽器支援
function checkBrowserSupport() {
    // 檢查 Fetch API
    if (!window.fetch) {
        showAlert('警告', '您的瀏覽器不支援部分功能，請升級到最新版瀏覽器以獲得最佳體驗。');
    }
    
    // 檢查 localStorage
    try {
        localStorage.setItem('test', 'test');
        localStorage.removeItem('test');
    } catch (e) {
        showAlert('警告', '本地儲存功能不可用，可能是您禁用了 Cookie 或瀏覽器設置的問題。應用程式將無法保存用戶設置。');
    }
    
    // 檢查檔案 API
    if (!(window.File && window.FileReader && window.FileList && window.Blob)) {
        showAlert('警告', '您的瀏覽器不完全支援檔案操作功能，部分功能可能無法正常工作。');
    }
}

// 處理全域錯誤
window.onerror = function(message, source, lineno, colno, error) {
    console.error('全域錯誤:', error);
    
    // 如果不是來自第三方腳本的錯誤，則顯示提示
    if (source.includes(window.location.origin)) {
        showAlert('錯誤', '發生意外錯誤，請刷新頁面並重試。如果問題持續存在，請聯繫支援團隊。');
    }
    
    // 允許默認錯誤處理
    return false;
};