# RAG 向量資料庫系統 (支援多種資料庫選項)

這是一個具有彈性的 RAG 向量資料庫系統，支援 Chroma DB 作為預設資料庫選項，以及具有 BYOK (Bring Your Own Key) 功能的 Pinecone 選項。使用者可以上傳自己的文件，並通過向量搜尋功能進行高效檢索。

## 功能特點

- **多資料庫支援**：
  - **Chroma DB (預設)**：開源、無需加密金鑰，快速上手
  - **Pinecone BYOK**：使用者可以使用自己的加密金鑰加密文件內容，提供更高安全性

- **多格式文件支援**：支援 PDF、DOCX、TXT 等多種文件格式
- **文件管理**：上傳、查看和刪除文件
- **混合搜尋**：可同時搜尋不同資料庫的內容，或選擇特定資料庫
- **金鑰管理**：生成、上傳和輪替加密金鑰 (僅 BYOK 模式)

## 系統架構

- **前端**：HTML, CSS, JavaScript
- **後端**：Python Flask API
- **向量資料庫**：
  - Chroma DB (開源、本地部署)
  - Pinecone (雲端服務，支援 BYOK)
- **部署**：Docker 容器化

## 目錄結構

```
rag-system/
├── backend/                   # 後端 API
│   ├── app.py                 # 主應用程式
│   ├── requirements.txt       # 依賴套件
│   └── Dockerfile             # 後端容器配置
├── data/                      # 資料存儲目錄
│   ├── uploads/               # 上傳的文件
│   ├── temp/                  # 臨時文件
│   ├── vectors/               # 向量資料
│   ├── keys/                  # 加密金鑰
│   ├── chroma_db/             # Chroma 資料庫文件
│   └── models/                # 模型相關資料
├── frontend/                  # 前端 Web 界面
│   ├── index.html             # 主 HTML 頁面
│   ├── css/                   # 樣式文件
│   └── js/                    # JavaScript 模組
│       ├── config.js          # 配置文件
│       ├── utils.js           # 工具函數
│       ├── keyManagement.js   # 金鑰管理
│       ├── documentManagement.js # 文件管理
│       ├── search.js          # 搜尋功能
│       └── app.js             # 主應用程式
├── docker-compose.yml         # Docker Compose 配置
├── nginx.conf                 # Nginx 配置
└── README.md                  # 專案說明文件
```

## 安裝與部署

### 前置需求

- Docker 和 Docker Compose
- 對於 Pinecone BYOK 選項，需要 Pinecone 帳號和 API 金鑰
- 對於嵌入生成功能，需要 OpenAI API 金鑰（或其他嵌入服務提供商）

### 設置步驟

1. 複製專案

```bash
git clone https://github.com/yourusername/rag-system.git
cd rag-system
```

2. 執行設置腳本

執行提供的設置腳本，它會自動創建必要的目錄結構和配置文件：

```bash
./setup.sh
```

這個腳本會：
- 創建 `data` 目錄及其子目錄
- 複製 `.env.example` 到 `.env`（如果尚未存在）
- 設置適當的目錄權限

3. 配置環境變數（可選）

如果需要自定義配置，編輯 `backend/.env` 文件。預設配置使用 Ollama 作為嵌入提供者，如果需要使用其他提供者（如 OpenAI）或 Pinecone BYOK，請取消註釋相關配置並填入您的 API 金鑰。

4. 構建和啟動容器

```bash
docker-compose up -d
```

4. 訪問系統

打開瀏覽器，訪問 `http://localhost`

## 使用指南

### 1. 設置用戶 ID 和選擇資料庫類型

- 輸入唯一的用戶 ID
- 選擇資料庫類型：
  - **Chroma DB**：預設選項，無需金鑰
  - **Pinecone BYOK**：需要加密金鑰

### 2. 設置加密金鑰 (僅 Pinecone BYOK)

- 選擇「生成新金鑰」或「上傳自有金鑰」
- 安全保存您的加密金鑰，金鑰丟失將無法解密數據

### 3. 上傳文件

- 選擇要上傳的文件（支援 PDF、DOCX、TXT）
- 點擊「上傳文件」按鈕
- 等待文件處理完成

### 4. 管理文件

- 查看已上傳的文件列表
- 可以按資料庫類型篩選文件
- 可以刪除不需要的文件

### 5. 搜尋內容

- 在搜尋框輸入查詢內容
- 選擇搜尋範圍（全部資料庫、僅 Chroma、僅 Pinecone）
- 查看語意匹配的相關結果
- 系統會顯示相關度分數和上下文

## 區別性特點

### Chroma DB (預設)

- **優點**：
  - 無需註冊外部服務
  - 開源免費
  - 無需管理加密金鑰
  - 本地部署，數據不離開服務器
- **適合場景**：
  - 初學者或快速原型開發
  - 非敏感文件管理
  - 低預算項目

### Pinecone BYOK

- **優點**：
  - 高級安全性，用戶控制加密金鑰
  - 客戶端加密，服務提供商無法讀取內容
  - 支援金鑰輪替和管理
  - 雲端托管，擴展性好
- **適合場景**：
  - 處理敏感或機密文件
  - 有嚴格合規要求的環境
  - 企業級應用

## 安全考量

- 使用加密金鑰的 BYOK 模式確保敏感數據安全
- 所有文件內容在客戶端加密後再存儲到向量數據庫
- 建議定期輪替加密金鑰
- 系統不會在服務器端保存用戶加密金鑰

## 局限性

- 當前實現是概念驗證，未完全針對生產環境優化
- 大文件處理可能需要較長時間
- 嵌入實現目前使用模擬數據，實際部署需連接 OpenAI 或其他嵌入服務

## 開發注意事項

- 專案使用 `.gitignore` 配置，防止將運行時生成的資料目錄和環境變數文件提交到版本控制系統
- 所有資料文件（上傳的文件、向量資料、資料庫文件等）都存儲在 `data` 目錄中，該目錄不會被 Git 追蹤
- 開發時請確保創建並正確配置 `.env` 文件，可以參考 `.env.example` 模板

## 許可證

MIT

## 免責聲明

此系統僅供學習和研究使用，請遵守相關法律法規。
