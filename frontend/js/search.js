/**
 * 搜尋模組
 * 處理向量搜尋功能
 */

// 初始化搜尋模組
function initSearch() {
    document.getElementById('searchBtn').addEventListener('click', performSearch);
    
    // 按下 Enter 鍵也觸發搜尋
    document.getElementById('searchQuery').addEventListener('keyup', function(event) {
        if (event.key === 'Enter') {
            performSearch();
        }
    });
}

// 執行向量搜尋
async function performSearch() {
    const query = document.getElementById('searchQuery').value.trim();
    
    if (!query) {
        showAlert('提示', '請輸入搜尋查詢');
        return;
    }
    
    const userId = document.getElementById('userId').value.trim();
    if (!userId) {
        showAlert('錯誤', '請先設置用戶 ID');
        return;
    }
    
    // 獲取搜尋範圍
    const searchDbType = getSearchDbType();
    
    // 顯示載入狀態
    const searchBtn = document.getElementById('searchBtn');
    const originalBtnText = searchBtn.textContent;
    searchBtn.textContent = '搜尋中...';
    searchBtn.disabled = true;
    
    try {
        // 初始化結果列表
        let allResults = [];
        
        // 根據選擇的資料庫類型執行搜尋
        if (searchDbType === 'all' || searchDbType === CONFIG.DB_TYPES.CHROMA) {
            // 搜尋 Chroma DB
            try {
                const chromaResponse = await apiRequest(CONFIG.API.SEARCH, 'POST', {
                    query: query,
                    user_id: userId,
                    db_type: CONFIG.DB_TYPES.CHROMA
                });
                
                if (chromaResponse.results && chromaResponse.results.length > 0) {
                    // 添加資料庫類型標記
                    const chromaResults = chromaResponse.results.map(result => ({
                        ...result,
                        db_type: CONFIG.DB_TYPES.CHROMA
                    }));
                    allResults = [...allResults, ...chromaResults];
                }
            } catch (error) {
                console.error('Chroma 搜尋失敗:', error);
            }
        }
        
        if (searchDbType === 'all' || searchDbType === CONFIG.DB_TYPES.PINECONE_BYOK) {
            // 檢查是否有 BYOK 金鑰
            const encryptionKey = document.getElementById('currentKey').value.trim();
            
            if (encryptionKey) {
                try {
                    // 搜尋 Pinecone BYOK
                    const pineconeResponse = await apiRequest(CONFIG.API.SEARCH, 'POST', {
                        query: query,
                        user_id: userId,
                        db_type: CONFIG.DB_TYPES.PINECONE_BYOK,
                        encryption_key: encryptionKey
                    });
                    
                    if (pineconeResponse.results && pineconeResponse.results.length > 0) {
                        // 添加資料庫類型標記
                        const pineconeResults = pineconeResponse.results.map(result => ({
                            ...result,
                            db_type: CONFIG.DB_TYPES.PINECONE_BYOK
                        }));
                        allResults = [...allResults, ...pineconeResults];
                    }
                } catch (error) {
                    console.error('Pinecone 搜尋失敗:', error);
                }
            } else if (searchDbType === CONFIG.DB_TYPES.PINECONE_BYOK) {
                showAlert('提示', '搜尋 Pinecone BYOK 需要加密金鑰');
            }
        }
        
        // 根據分數排序結果（高到低）
        allResults.sort((a, b) => b.score - a.score);
        
        // 限制結果數量
        if (allResults.length > CONFIG.SEARCH.MAX_RESULTS) {
            allResults = allResults.slice(0, CONFIG.SEARCH.MAX_RESULTS);
        }
        
        // 顯示搜尋結果
        displaySearchResults(allResults, query);
        
    } catch (error) {
        console.error('搜尋失敗:', error);
    } finally {
        // 恢復按鈕狀態
        searchBtn.textContent = originalBtnText;
        searchBtn.disabled = false;
    }
}

// 顯示搜尋結果
function displaySearchResults(results, query) {
    const searchResultsArea = document.getElementById('searchResults');
    const resultsList = document.getElementById('resultsList');
    
    // 清空現有結果
    resultsList.innerHTML = '';
    
    if (results && results.length > 0) {
        // 顯示結果區域
        searchResultsArea.classList.remove('hidden');
        
        // 添加結果項目
        results.forEach(result => {
            const resultItem = document.createElement('li');
            resultItem.className = 'p-4 bg-white rounded-lg shadow search-result';
            
            // 計算相似度百分比
            const similarityPercent = Math.round(result.score * 100);
            
            // 高亮顯示查詢詞
            let highlightedText = highlightQuery(result.text, query);
            
            // 資料庫類型標籤樣式
            const dbTypeClass = result.db_type === CONFIG.DB_TYPES.PINECONE_BYOK 
                ? 'bg-purple-100 text-purple-800' 
                : 'bg-green-100 text-green-800';
            
            resultItem.innerHTML = `
                <div class="flex justify-between items-start mb-2">
                    <h4 class="text-md font-medium">${result.filename}</h4>
                    <div class="flex space-x-2">
                        <span class="text-sm ${dbTypeClass} px-2 py-1 rounded">
                            ${formatDbType(result.db_type)}
                        </span>
                        <span class="text-sm bg-blue-100 text-blue-800 px-2 py-1 rounded">
                            相關度: ${similarityPercent}%
                        </span>
                    </div>
                </div>
                <p class="text-gray-600 mb-2">${highlightedText}</p>
                <div class="text-xs text-gray-500">
                    文件 ID: ${result.document_id}
                </div>
            `;
            
            resultsList.appendChild(resultItem);
        });
    } else {
        // 無結果時顯示提示
        const noResultItem = document.createElement('li');
        noResultItem.className = 'p-4 bg-white rounded-lg shadow';
        noResultItem.innerHTML = `
            <p class="text-gray-600 text-center">未找到與「${query}」相關的結果</p>
        `;
        
        resultsList.appendChild(noResultItem);
        searchResultsArea.classList.remove('hidden');
    }
    
    // 滾動到結果區域
    searchResultsArea.scrollIntoView({ behavior: 'smooth' });
}

// 高亮顯示查詢詞
function highlightQuery(text, query) {
    if (!query || !text) return text;
    
    // 簡單的詞語匹配高亮
    const words = query.split(/\s+/).filter(word => word.length > 1);
    let highlightedText = text;
    
    words.forEach(word => {
        const regex = new RegExp(word, 'gi');
        highlightedText = highlightedText.replace(regex, match => `<span class="bg-yellow-200">${match}</span>`);
    });
    
    return highlightedText;
}

// 提取上下文
function extractContext(text, query, contextSize = 200) {
    if (!query || !text) return text;
    
    // 查找查詢詞在文本中的位置
    const lowerText = text.toLowerCase();
    const lowerQuery = query.toLowerCase();
    const position = lowerText.indexOf(lowerQuery);
    
    if (position === -1) return text;
    
    // 提取上下文
    const start = Math.max(0, position - contextSize);
    const end = Math.min(text.length, position + query.length + contextSize);
    
    let snippet = text.substring(start, end);
    
    // 如果上下文被裁剪，添加省略號
    if (start > 0) snippet = '...' + snippet;
    if (end < text.length) snippet = snippet + '...';
    
    return snippet;
}