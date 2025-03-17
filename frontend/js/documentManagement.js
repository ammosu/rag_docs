/**
 * 文件管理模組
 * 處理文件上傳、列表顯示和刪除
 */

// 初始化文件管理模組
function initDocumentManagement() {
    // 文件上傳按鈕
    document.getElementById('uploadBtn').addEventListener('click', uploadDocument);
    
    // 文件列表刷新按鈕
    document.getElementById('refreshDocumentsBtn').addEventListener('click', fetchDocuments);
    
    // 文件列表篩選按鈕
    document.getElementById('showAllDbsBtn').addEventListener('click', function() {
        filterDocumentsByDbType('all');
    });
    document.getElementById('showChromaBtn').addEventListener('click', function() {
        filterDocumentsByDbType(CONFIG.DB_TYPES.CHROMA);
    });
    document.getElementById('showPineconeBtn').addEventListener('click', function() {
        filterDocumentsByDbType(CONFIG.DB_TYPES.PINECONE_BYOK);
    });
    
    // 初始載入文件列表
    if (hasValidUserId()) {
        fetchDocuments();
    }
}

// 上傳文件
async function uploadDocument() {
    const fileInput = document.getElementById('fileUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        showAlert('錯誤', '請選擇一個檔案');
        return;
    }
    
    if (!isAllowedFileType(file.name)) {
        showAlert('錯誤', '不支援的檔案類型');
        return;
    }
    
    if (file.size > CONFIG.UPLOAD.MAX_FILE_SIZE) {
        showAlert('錯誤', `檔案大小超過限制 (${formatFileSize(CONFIG.UPLOAD.MAX_FILE_SIZE)})`);
        return;
    }
    
    const userId = document.getElementById('userId').value.trim();
    const dbType = document.getElementById('dbType').value;
    
    if (!userId) {
        showAlert('錯誤', '請先設置用戶 ID');
        return;
    }
    
    // 顯示進度條
    const progressArea = document.getElementById('uploadProgress');
    progressArea.classList.remove('hidden');
    const progressBar = document.getElementById('progressBar');
    progressBar.style.width = '0%';
    progressBar.classList.add('progress-bar-animated');
    const progressText = document.getElementById('progressText');
    progressText.textContent = '準備上傳...';
    
    // 創建表單數據
    const formData = new FormData();
    formData.append('file', file);
    formData.append('user_id', userId);
    formData.append('db_type', dbType);
    
    // 如果是 BYOK 模式，添加加密金鑰
    if (dbType === CONFIG.DB_TYPES.PINECONE_BYOK) {
        const encryptionKey = document.getElementById('currentKey').value.trim();
        if (!encryptionKey) {
            showAlert('錯誤', '使用 BYOK 模式時必須提供加密金鑰');
            progressArea.classList.add('hidden');
            return;
        }
        formData.append('encryption_key', encryptionKey);
    }
    
    try {
        // 模擬上傳進度
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress > 95) {
                clearInterval(progressInterval);
            }
            progressBar.style.width = `${progress}%`;
            progressText.textContent = `上傳中 ${progress}%...`;
        }, 300);
        
        // 發送上傳請求
        const response = await apiRequest(CONFIG.API.UPLOAD, 'POST', formData);
        
        // 清除進度條模擬
        clearInterval(progressInterval);
        progressBar.style.width = '100%';
        progressText.textContent = '處理完成';
        progressBar.classList.remove('progress-bar-animated');
        
        // 顯示結果
        showAlert('成功', `文件已成功上傳並處理，共 ${response.chunks} 個區塊`);
        
        // 重置表單
        fileInput.value = '';
        
        // 刷新文件列表
        fetchDocuments();
        
        // 隱藏進度條
        setTimeout(() => {
            progressArea.classList.add('hidden');
        }, 2000);
        
    } catch (error) {
        // 處理上傳錯誤
        progressBar.classList.remove('progress-bar-animated');
        progressText.textContent = '上傳失敗';
        console.error('文件上傳失敗:', error);
    }
}

// 獲取文件列表
async function fetchDocuments() {
    const userId = document.getElementById('userId').value.trim();
    
    if (!userId) {
        showAlert('錯誤', '請先輸入用戶 ID');
        return;
    }
    
    try {
        // 獲取 Chroma 文件
        const chromaResponse = await apiRequest(`${CONFIG.API.DOCUMENTS}?user_id=${encodeURIComponent(userId)}&db_type=${CONFIG.DB_TYPES.CHROMA}`);
        
        // 如果有 BYOK 金鑰，也獲取 Pinecone 文件
        let pineconeDocuments = [];
        const hasKey = document.getElementById('currentKey').value.trim() !== '';
        
        if (hasKey) {
            try {
                const pineconeResponse = await apiRequest(`${CONFIG.API.DOCUMENTS}?user_id=${encodeURIComponent(userId)}&db_type=${CONFIG.DB_TYPES.PINECONE_BYOK}`);
                pineconeDocuments = pineconeResponse.documents || [];
            } catch (error) {
                console.error('獲取 Pinecone 文件失敗:', error);
            }
        }
        
        // 合併文件列表
        const allDocuments = [
            ...(chromaResponse.documents || []),
            ...pineconeDocuments
        ];
        
        // 更新 UI
        updateDocumentsList(allDocuments);
        
    } catch (error) {
        console.error('獲取文件列表失敗:', error);
    }
}

// 更新文件列表 UI
function updateDocumentsList(documents) {
    const documentsList = document.getElementById('documentsList');
    const noDocuments = document.getElementById('noDocuments');
    
    // 清空現有文件列表
    documentsList.innerHTML = '';
    
    if (documents && documents.length > 0) {
        // 有文件時顯示列表
        documentsList.parentElement.classList.remove('hidden');
        noDocuments.classList.add('hidden');
        
        // 添加每個文件到列表
        documents.forEach(doc => {
            const row = document.createElement('tr');
            row.className = 'document-item';
            row.setAttribute('data-db-type', doc.db_type || CONFIG.DB_TYPES.CHROMA);
            
            row.innerHTML = `
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${doc.document_id}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    ${doc.filename}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <span class="px-2 py-1 text-xs font-medium rounded-full ${doc.db_type === CONFIG.DB_TYPES.PINECONE_BYOK ? 'bg-purple-100 text-purple-800' : 'bg-green-100 text-green-800'}">
                        ${formatDbType(doc.db_type || CONFIG.DB_TYPES.CHROMA)}
                    </span>
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    ${formatDate(doc.timestamp)}
                </td>
                <td class="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <button class="delete-doc-btn text-red-600 hover:text-red-900 mr-3" 
                            data-id="${doc.document_id}"
                            data-db-type="${doc.db_type || CONFIG.DB_TYPES.CHROMA}">
                        刪除
                    </button>
                </td>
            `;
            documentsList.appendChild(row);
        });
        
        // 添加刪除按鈕事件
        document.querySelectorAll('.delete-doc-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const docId = this.getAttribute('data-id');
                const dbType = this.getAttribute('data-db-type');
                if (confirm('確定要刪除這個文件嗎？此操作無法撤銷。')) {
                    deleteDocument(docId, dbType);
                }
            });
        });
    } else {
        // 無文件時顯示提示
        documentsList.parentElement.classList.add('hidden');
        noDocuments.classList.remove('hidden');
    }
}

// 根據資料庫類型篩選文件列表
function filterDocumentsByDbType(dbType) {
    // 更新篩選按鈕樣式
    document.getElementById('showAllDbsBtn').className = dbType === 'all' ? 
        'px-3 py-1 bg-blue-100 text-blue-800 rounded-md text-sm font-medium' : 
        'px-3 py-1 text-gray-600 rounded-md text-sm font-medium ml-1';
    
    document.getElementById('showChromaBtn').className = dbType === CONFIG.DB_TYPES.CHROMA ? 
        'px-3 py-1 bg-blue-100 text-blue-800 rounded-md text-sm font-medium ml-1' : 
        'px-3 py-1 text-gray-600 rounded-md text-sm font-medium ml-1';
    
    document.getElementById('showPineconeBtn').className = dbType === CONFIG.DB_TYPES.PINECONE_BYOK ? 
        'px-3 py-1 bg-blue-100 text-blue-800 rounded-md text-sm font-medium ml-1' : 
        'px-3 py-1 text-gray-600 rounded-md text-sm font-medium ml-1';
    
    // 篩選文件列表
    const documentItems = document.querySelectorAll('.document-item');
    documentItems.forEach(item => {
        const itemDbType = item.getAttribute('data-db-type');
        if (dbType === 'all' || itemDbType === dbType) {
            item.classList.remove('hidden');
        } else {
            item.classList.add('hidden');
        }
    });
    
    // 檢查是否有可見的文件
    const hasVisibleDocuments = Array.from(documentItems).some(item => 
        !item.classList.contains('hidden')
    );
    
    // 顯示或隱藏無文件提示
    const noDocuments = document.getElementById('noDocuments');
    const documentsTable = document.getElementById('documentsList').parentElement;
    
    if (hasVisibleDocuments) {
        documentsTable.classList.remove('hidden');
        noDocuments.classList.add('hidden');
    } else {
        documentsTable.classList.add('hidden');
        noDocuments.classList.remove('hidden');
        noDocuments.textContent = `沒有${dbType === 'all' ? '' : formatDbType(dbType) + ' '}文件`;
    }
}

// 刪除文件
async function deleteDocument(documentId, dbType) {
    const userId = document.getElementById('userId').value.trim();
    
    if (!userId) {
        showAlert('錯誤', '請先輸入用戶 ID');
        return;
    }
    
    try {
        await apiRequest(`${CONFIG.API.DOCUMENTS}/${documentId}?user_id=${encodeURIComponent(userId)}&db_type=${dbType}`, 'DELETE');
        
        showAlert('成功', '文件已成功刪除');
        
        // 刷新文件列表
        fetchDocuments();
    } catch (error) {
        console.error('刪除文件失敗:', error);
    }
}