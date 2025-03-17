/**
 * 應用程式配置
 */
const CONFIG = {
    // API 端點
    API: {
        BASE_URL: 'http://localhost:5000/api',
        KEY: '/key',
        UPLOAD: '/upload',
        SEARCH: '/search',
        DOCUMENTS: '/documents',
        ROTATE_KEY: '/rotate-key',
        DB_TYPES: '/db-types'
    },
    
    // 本地存儲鍵名
    STORAGE: {
        USER_ID: 'rag_user_id',
        ENCRYPTION_KEY: 'rag_encryption_key',
        DB_TYPE: 'rag_db_type'
    },
    
    // 上傳文件配置
    UPLOAD: {
        MAX_FILE_SIZE: 16 * 1024 * 1024, // 16MB
        ALLOWED_TYPES: ['.pdf', '.docx', '.txt'],
        CHUNK_SIZE: 1000,
        CHUNK_OVERLAP: 200
    },
    
    // UI 配置
    UI: {
        ANIMATION_DURATION: 300, // 毫秒
        ALERT_DURATION: 3000, // 毫秒
        TABLE_PAGE_SIZE: 10
    },
    
    // 搜尋配置
    SEARCH: {
        MAX_RESULTS: 5,
        MIN_SCORE: 0.5
    },
    
    // 資料庫類型
    DB_TYPES: {
        CHROMA: 'chroma',
        PINECONE_BYOK: 'pinecone_byok'
    }
};