# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import uuid
import hashlib
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from cryptography.fernet import Fernet
import base64
import numpy as np
from dotenv import load_dotenv
import pinecone
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import json

# 載入環境變數
load_dotenv()

app = Flask(__name__)
CORS(app)

# 配置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 檔案上傳配置
DATA_DIR = './data'  # 相對於 backend 目錄的路徑
UPLOAD_FOLDER = os.path.join(DATA_DIR, 'uploads')
TEMP_FOLDER = os.path.join(DATA_DIR, 'temp')
VECTOR_FOLDER = os.path.join(DATA_DIR, 'vectors')
KEYS_FOLDER = os.path.join(DATA_DIR, 'keys')
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt'}

# 確保資料目錄存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)
os.makedirs(VECTOR_FOLDER, exist_ok=True)
os.makedirs(KEYS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 上傳限制

# 向量嵌入維度
DIMENSION = 1024  # 使用 jeffh/intfloat-multilingual-e5-large:f16 模型的向量維度

# 向量資料庫抽象類
class VectorDB:
    def store_vectors(self, vectors, metadata):
        pass
    
    def search(self, query_vector, filter_dict=None, top_k=5):
        pass
    
    def delete(self, ids=None, filter_dict=None):
        pass
    
    def get_documents(self, filter_dict=None):
        pass

# Pinecone BYOK 向量資料庫
class PineconeBYOKVectorDB(VectorDB):
    def __init__(self):
        # 初始化 Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        environment = os.getenv("PINECONE_ENVIRONMENT")
        
        if not api_key or not environment:
            raise ValueError("Pinecone API 金鑰和環境變數必須設置")
        
        pinecone.init(api_key=api_key, environment=environment)
        
        # 創建或獲取索引
        index_name = "documents-index"
        
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=DIMENSION,
                metric="cosine"
            )
        
        self.index = pinecone.Index(index_name)
    
    def store_vectors(self, vectors, metadata):
        # 將向量和元數據上傳到 Pinecone
        return self.index.upsert(vectors=vectors)
    
    def search(self, query_vector, filter_dict=None, top_k=5):
        # 在 Pinecone 中搜索向量
        return self.index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            filter=filter_dict
        )
    
    def delete(self, ids=None, filter_dict=None):
        # 從 Pinecone 刪除向量
        if ids:
            return self.index.delete(ids=ids)
        else:
            # Pinecone 需要 ID 列表來刪除，所以需要先搜尋匹配的 ID
            dummy_vector = [0.0] * DIMENSION
            results = self.index.query(
                vector=dummy_vector,
                top_k=10000,  # 嘗試獲取足夠多的匹配項
                include_metadata=False,
                filter=filter_dict
            )
            ids_to_delete = [match['id'] for match in results['matches']]
            
            if ids_to_delete:
                return self.index.delete(ids=ids_to_delete)
            return {"deleted": 0}
    
    def get_documents(self, filter_dict=None):
        # 獲取匹配篩選條件的文檔
        dummy_vector = [0.0] * DIMENSION
        return self.index.query(
            vector=dummy_vector,
            top_k=1000,
            include_metadata=True,
            filter=filter_dict
        )

# Chroma DB 向量資料庫 (預設，無需加密金鑰)
class ChromaVectorDB(VectorDB):
    def __init__(self):
        # 設置 Chroma 持久化目錄
        persist_directory = os.path.join(DATA_DIR, "chroma_db")
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化 Chroma 客戶端
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # 創建通用嵌入函數基類
        class BaseEmbeddingFunction(embedding_functions.EmbeddingFunction):
            def __init__(self):
                pass
            
            def __call__(self, texts):
                raise NotImplementedError("子類必須實現此方法")
            
            def get_fallback_embedding(self):
                """返回後備嵌入向量（當API調用失敗時使用）"""
                return np.random.rand(DIMENSION).tolist()
        
        # Ollama 嵌入函數實現
        class OllamaEmbeddingFunction(BaseEmbeddingFunction):
            def __init__(self, model_name=None, ollama_url=None):
                super().__init__()
                self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
                self.model = model_name or os.getenv("OLLAMA_MODEL", "jeffh/intfloat-multilingual-e5-large:f16")
            
            def __call__(self, texts):
                import requests
                
                embeddings = []
                for text in texts:
                    try:
                        response = requests.post(
                            f"{self.ollama_url}/api/embeddings",
                            json={"model": self.model, "prompt": text}
                        )
                        
                        if response.status_code == 200:
                            embedding = response.json().get('embedding', [])
                            embeddings.append(embedding)
                        else:
                            logger.error(f"Ollama API 錯誤: {response.status_code} - {response.text}")
                            embeddings.append(self.get_fallback_embedding())
                    except Exception as e:
                        logger.error(f"生成嵌入時出錯: {str(e)}")
                        embeddings.append(self.get_fallback_embedding())
                
                return embeddings
        
        # 獲取嵌入函數（可以根據環境變數選擇不同的嵌入提供者）
        embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
        
        if embedding_provider == "ollama":
            self.embedding_function = OllamaEmbeddingFunction()
        else:
            # 默認使用 Ollama
            logger.warning(f"未知的嵌入提供者: {embedding_provider}，使用 Ollama 作為默認")
            self.embedding_function = OllamaEmbeddingFunction()
        
        try:
            self.collection = self.client.get_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
        except:
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedding_function
            )
    
    def store_vectors(self, vectors, metadata):
        # 將向量和元數據上傳到 Chroma
        ids = [m['id'] for m in metadata]
        texts = [m.get('text', '') for m in metadata]
        
        self.collection.add(
            ids=ids,
            embeddings=vectors,
            metadatas=metadata,
            documents=texts
        )
        
        return {"upserted_count": len(ids)}
    
    def search(self, query_vector, filter_dict=None, top_k=5):
        # 在 Chroma 中搜索向量
        try:
            # Chroma 需要特定格式的 where 條件
            chroma_where = {}
            
            if filter_dict:
                # 處理 user_id
                if 'user_id' in filter_dict:
                    chroma_where["user_id"] = {"$eq": filter_dict["user_id"]}
                
                # 處理 document_id
                if 'document_id' in filter_dict:
                    chroma_where["document_id"] = {"$eq": filter_dict["document_id"]}
                
                # 處理 is_encrypted
                if 'is_encrypted' in filter_dict:
                    chroma_where["is_encrypted"] = {"$eq": filter_dict["is_encrypted"]}
            
            # 執行查詢
            results = self.collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=chroma_where if chroma_where else None
            )
            
            # 格式化結果以匹配 Pinecone 的輸出格式
            matches = []
            if results and 'ids' in results and len(results['ids']) > 0:
                for i in range(len(results['ids'][0])):
                    match = {
                        'id': results['ids'][0][i],
                        'score': results['distances'][0][i] if 'distances' in results else 0.0,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
                    }
                    matches.append(match)
            
            return {"matches": matches}
            
        except Exception as e:
            logger.error(f"Chroma 搜索時出錯: {str(e)}")
            # 返回空結果
            return {"matches": []}
    
    def delete(self, ids=None, filter_dict=None):
        # 從 Chroma 刪除向量
        if ids:
            try:
                logger.info(f"通過 ID 列表刪除: {ids}")
                self.collection.delete(ids=ids)
                return {"deleted": len(ids)}
            except Exception as e:
                logger.error(f"通過 ID 列表刪除失敗: {str(e)}")
                return {"deleted": 0, "error": str(e)}
        
        elif filter_dict:
            logger.info(f"嘗試通過過濾條件刪除: {filter_dict}")
            deleted_count = 0
            
            try:
                # 方法 1: 使用 where 條件獲取 ID 後刪除
                chroma_where = {}
                
                # 處理 user_id
                if 'user_id' in filter_dict:
                    chroma_where["user_id"] = {"$eq": filter_dict["user_id"]}
                
                # 處理 document_id
                if 'document_id' in filter_dict:
                    chroma_where["document_id"] = {"$eq": filter_dict["document_id"]}
                
                logger.info(f"使用 Chroma where 條件: {chroma_where}")
                
                # 獲取匹配的文檔
                results = self.collection.get(where=chroma_where)
                
                if results and 'ids' in results and results['ids']:
                    ids_to_delete = results['ids']
                    logger.info(f"找到要刪除的 ID: {ids_to_delete}")
                    
                    if ids_to_delete:
                        self.collection.delete(ids=ids_to_delete)
                        deleted_count = len(ids_to_delete)
                        logger.info(f"成功刪除 {deleted_count} 個向量")
                        return {"deleted": deleted_count}
                else:
                    logger.warning("未找到匹配的文檔")
            
            except Exception as e:
                logger.error(f"方法 1 刪除失敗: {str(e)}")
            
            # 如果方法 1 失敗，嘗試方法 2
            if deleted_count == 0:
                try:
                    logger.info("嘗試方法 2: 使用查詢獲取所有文檔")
                    # 使用查詢方法獲取所有文檔
                    dummy_vector = [0.0] * DIMENSION
                    all_results = self.collection.query(
                        query_embeddings=[dummy_vector],
                        n_results=10000,
                        include_metadata=True
                    )
                    
                    if all_results and 'ids' in all_results and len(all_results['ids']) > 0:
                        # 過濾匹配條件的 ID
                        ids_to_delete = []
                        
                        for i, metadata in enumerate(all_results['metadatas'][0]):
                            matches_all = True
                            for key, value in filter_dict.items():
                                if key not in metadata or metadata[key] != value:
                                    matches_all = False
                                    break
                            
                            if matches_all:
                                ids_to_delete.append(all_results['ids'][0][i])
                        
                        logger.info(f"方法 2 找到要刪除的 ID: {ids_to_delete}")
                        
                        if ids_to_delete:
                            self.collection.delete(ids=ids_to_delete)
                            deleted_count = len(ids_to_delete)
                            logger.info(f"方法 2 成功刪除 {deleted_count} 個向量")
                            return {"deleted": deleted_count}
                    else:
                        logger.warning("方法 2 未找到匹配的文檔")
                
                except Exception as inner_e:
                    logger.error(f"方法 2 刪除失敗: {str(inner_e)}")
            
            # 如果方法 2 也失敗，嘗試方法 3
            if deleted_count == 0:
                try:
                    logger.info("嘗試方法 3: 直接使用 where 參數刪除")
                    # 直接使用 where 參數刪除
                    self.collection.delete(where=chroma_where)
                    logger.info("方法 3 刪除操作已執行")
                    # 由於無法確定刪除了多少項，返回一個估計值
                    return {"deleted": 1, "method": "direct_where"}
                except Exception as e3:
                    logger.error(f"方法 3 刪除失敗: {str(e3)}")
        
        logger.warning("所有刪除方法都失敗")
        return {"deleted": 0}
    
    def get_documents(self, filter_dict=None):
        # 獲取匹配篩選條件的文檔
        try:
            # Chroma 需要特定格式的 where 條件
            chroma_where = {}
            
            if filter_dict:
                # 處理 user_id
                if 'user_id' in filter_dict:
                    chroma_where["user_id"] = {"$eq": filter_dict["user_id"]}
                
                # 處理 document_id
                if 'document_id' in filter_dict:
                    chroma_where["document_id"] = {"$eq": filter_dict["document_id"]}
                
                # 處理 is_encrypted
                if 'is_encrypted' in filter_dict:
                    chroma_where["is_encrypted"] = {"$eq": filter_dict["is_encrypted"]}
            
            # 獲取匹配的文檔
            results = self.collection.get(where=chroma_where if chroma_where else None)
            
            # 格式化結果以匹配 Pinecone 的輸出格式
            matches = []
            if results and 'ids' in results:
                for i in range(len(results['ids'])):
                    match = {
                        'id': results['ids'][i],
                        'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                    }
                    matches.append(match)
            
            return {"matches": matches}
            
        except Exception as e:
            logger.error(f"Chroma 獲取文檔時出錯: {str(e)}")
            # 嘗試使用備用方法
            try:
                # 使用查詢方法獲取所有文檔
                dummy_vector = [0.0] * DIMENSION
                all_results = self.collection.query(
                    query_embeddings=[dummy_vector],
                    n_results=10000,
                    include_metadata=True
                )
                
                matches = []
                if all_results and 'ids' in all_results and len(all_results['ids']) > 0:
                    for i, metadata in enumerate(all_results['metadatas'][0]):
                        # 檢查是否匹配過濾條件
                        if filter_dict:
                            matches_all = True
                            for key, value in filter_dict.items():
                                if key not in metadata or metadata[key] != value:
                                    matches_all = False
                                    break
                            
                            if not matches_all:
                                continue
                        
                        match = {
                            'id': all_results['ids'][0][i],
                            'metadata': metadata
                        }
                        matches.append(match)
                
                return {"matches": matches}
            except Exception as inner_e:
                logger.error(f"備用獲取文檔方法失敗: {str(inner_e)}")
                return {"matches": []}

# 向量資料庫工廠
class VectorDBFactory:
    @staticmethod
    def get_db(db_type="chroma"):
        if db_type == "chroma":
            return ChromaVectorDB()
        elif db_type == "pinecone_byok":
            return PineconeBYOKVectorDB()
        else:
            raise ValueError(f"不支援的資料庫類型: {db_type}")

# 用戶金鑰存儲（實際應用中應使用更安全的存儲方式，如 HSM 或 KMS）
user_keys = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_text(file_path):
    ext = file_path.rsplit('.', 1)[1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(file_path)
    elif ext == 'docx':
        return extract_text_from_docx(file_path)
    elif ext == 'txt':
        return extract_text_from_txt(file_path)
    return ""

def chunk_text(text, chunk_size=1000, overlap=200):
    """將文字分成重疊的塊"""
    chunks = []
    
    # 如果文本太短，直接作為一個塊返回
    if len(text) < chunk_size // 2:
        return [text] if text.strip() else []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if len(chunk) >= chunk_size // 2:  # 避免太小的塊
            chunks.append(chunk)
    
    # 確保至少返回一個塊
    if not chunks and text.strip():
        chunks.append(text)
    
    return chunks

# 修改 app.py 中的 get_embeddings 函數

def get_embeddings(text_chunks):
    """獲取文本嵌入向量，支援多種嵌入提供者"""
    import requests
    import numpy as np
    
    # 獲取嵌入提供者配置
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "ollama").lower()
    
    # 創建嵌入函數
    if embedding_provider == "ollama":
        # Ollama 嵌入
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        model_name = os.getenv("OLLAMA_MODEL", "jeffh/intfloat-multilingual-e5-large:f16")
        
        embeddings = []
        for chunk in text_chunks:
            try:
                response = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={
                        "model": model_name,
                        "prompt": chunk
                    }
                )
                
                if response.status_code == 200:
                    embedding = response.json().get('embedding', [])
                    embeddings.append(embedding)
                else:
                    logger.error(f"Ollama API 錯誤: {response.status_code} - {response.text}")
                    embeddings.append(np.random.rand(DIMENSION).tolist())
            except Exception as e:
                logger.error(f"生成嵌入時出錯: {str(e)}")
                embeddings.append(np.random.rand(DIMENSION).tolist())
    
    # 可以在這裡添加其他嵌入提供者的實現
    # elif embedding_provider == "openai":
    #     # OpenAI 嵌入實現
    #     ...
    
    else:
        # 默認使用 Ollama
        logger.warning(f"未知的嵌入提供者: {embedding_provider}，使用 Ollama 作為默認")
        return get_embeddings(text_chunks)  # 遞歸調用，使用默認提供者
    
    return embeddings

def encrypt_text(text, key):
    """使用客戶金鑰加密文本"""
    fernet = Fernet(key)
    return fernet.encrypt(text.encode()).decode()

def decrypt_text(encrypted_text, key):
    """使用客戶金鑰解密文本"""
    fernet = Fernet(key)
    return fernet.decrypt(encrypted_text.encode()).decode()

@app.route('/api/key', methods=['POST'])
def generate_key():
    """生成新的加密金鑰或註冊現有金鑰"""
    data = request.json
    user_id = data.get('user_id')
    provided_key = data.get('key')
    
    if not user_id:
        return jsonify({"error": "必須提供用戶 ID"}), 400
    
    if provided_key:
        # 用戶提供了自己的金鑰
        try:
            # 檢查金鑰格式是否有效
            key = provided_key.encode() if isinstance(provided_key, str) else provided_key
            Fernet(key)
            user_keys[user_id] = key
            return jsonify({"message": "使用者金鑰已註冊", "key": provided_key})
        except Exception as e:
            return jsonify({"error": f"無效的金鑰格式: {str(e)}"}), 400
    else:
        # 生成新金鑰
        key = Fernet.generate_key()
        user_keys[user_id] = key
        return jsonify({"message": "新金鑰已生成", "key": key.decode()})

@app.route('/api/upload', methods=['POST'])
def upload_file():
    # 檢查是否有檔案
    if 'file' not in request.files:
        return jsonify({"error": "未找到檔案"}), 400
    
    file = request.files['file']
    user_id = request.form.get('user_id')
    db_type = request.form.get('db_type', 'chroma')  # 預設使用 Chroma DB
    encryption_key = request.form.get('encryption_key')
    
    # 檢查必要參數
    if not user_id:
        return jsonify({"error": "必須提供用戶 ID"}), 400
    
    # 只有在使用 BYOK 時才需要加密金鑰
    if db_type == 'pinecone_byok':
        if not encryption_key and user_id not in user_keys:
            return jsonify({"error": "使用 BYOK 模式時必須提供加密金鑰或先註冊金鑰"}), 400
    
    if not file or file.filename == '':
        return jsonify({"error": "未選擇檔案"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "不支援的檔案類型"}), 400
    
    # 獲取加密金鑰（如果使用 BYOK）
    key = None
    if db_type == 'pinecone_byok':
        if encryption_key:
            try:
                key = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
                Fernet(key)  # 驗證金鑰格式
            except Exception as e:
                return jsonify({"error": f"無效的加密金鑰: {str(e)}"}), 400
        else:
            key = user_keys[user_id]
    
    try:
        # 獲取向量資料庫實例
        vector_db = VectorDBFactory.get_db(db_type)
        
        # 保存檔案
        original_filename = file.filename
        safe_filename = secure_filename(file.filename)
        document_id = str(uuid.uuid4())
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{document_id}_{safe_filename}")
        file.save(file_path)
        
        # 提取文字
        text = extract_text(file_path)
        
        # 分塊
        chunks = chunk_text(text)
        
        # 獲取嵌入向量
        embeddings = get_embeddings(chunks)
        
        # 準備向量和元數據
        vectors_to_upsert = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}_{i}"
            
            # 元數據
            metadata = {
                'id': chunk_id,
                'document_id': document_id,
                'chunk_index': i,
                'filename': original_filename,
                'safe_filename': safe_filename,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }
            
            # 如果使用 BYOK，加密文本
            if db_type == 'pinecone_byok':
                encrypted_chunk = encrypt_text(chunk, key)
                metadata['encrypted_text'] = encrypted_chunk
                metadata['is_encrypted'] = True
            else:
                # Chroma DB 需要原始文本
                metadata['text'] = chunk
                metadata['is_encrypted'] = False
            
            vectors_to_upsert.append({
                'id': chunk_id,
                'values': embedding,
                'metadata': metadata
            })
        
        # 存儲向量
        result = vector_db.store_vectors(
            vectors=[v['values'] for v in vectors_to_upsert],
            metadata=[v['metadata'] for v in vectors_to_upsert]
        )
        
        return jsonify({
            "message": "檔案已成功處理並存儲到向量資料庫",
            "document_id": document_id,
            "chunks": len(chunks),
            "db_type": db_type
        })
        
    except Exception as e:
        logger.error(f"處理檔案時出錯: {str(e)}")
        return jsonify({"error": f"處理檔案時出錯: {str(e)}"}), 500

@app.route('/api/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    user_id = data.get('user_id')
    db_type = data.get('db_type', 'chroma')  # 預設使用 Chroma DB
    encryption_key = data.get('encryption_key')
    
    # 檢查必要參數
    if not query:
        return jsonify({"error": "必須提供查詢文本"}), 400
    
    if not user_id:
        return jsonify({"error": "必須提供用戶 ID"}), 400
    
    # 只有在使用 BYOK 時才需要加密金鑰
    if db_type == 'pinecone_byok':
        if not encryption_key and user_id not in user_keys:
            return jsonify({"error": "使用 BYOK 模式時必須提供解密金鑰或先註冊金鑰"}), 400
    
    # 獲取解密金鑰（如果使用 BYOK）
    key = None
    if db_type == 'pinecone_byok':
        if encryption_key:
            try:
                key = encryption_key.encode() if isinstance(encryption_key, str) else encryption_key
                Fernet(key)  # 驗證金鑰格式
            except Exception as e:
                return jsonify({"error": f"無效的解密金鑰: {str(e)}"}), 400
        else:
            key = user_keys[user_id]
    
    try:
        # 獲取向量資料庫實例
        vector_db = VectorDBFactory.get_db(db_type)
        
        # 生成查詢向量
        query_embedding = get_embeddings([query])[0]
        
        # 搜索向量資料庫
        filter_dict = {"user_id": user_id}
        search_results = vector_db.search(
            query_vector=query_embedding,
            filter_dict=filter_dict,
            top_k=5
        )
        
        # 處理結果
        results = []
        for match in search_results['matches']:
            try:
                metadata = match['metadata']
                
                # 根據是否加密處理文本
                if metadata.get('is_encrypted', False):
                    # 解密文本
                    encrypted_text = metadata.get('encrypted_text', '')
                    text = decrypt_text(encrypted_text, key)
                else:
                    # 未加密，直接使用文本
                    text = metadata.get('text', '')
                
                # 添加到結果中
                result = {
                    'id': match['id'],
                    'score': match.get('score', 0),
                    'document_id': metadata.get('document_id', ''),
                    'filename': metadata.get('filename', ''),
                    'text': text
                }
                results.append(result)
            except Exception as e:
                logger.error(f"處理搜尋結果時出錯: {str(e)}")
                # 跳過無法處理的結果
        
        return jsonify({
            "results": results,
            "db_type": db_type
        })
        
    except Exception as e:
        logger.error(f"搜索時出錯: {str(e)}")
        return jsonify({"error": f"搜索時出錯: {str(e)}"}), 500

@app.route('/api/documents', methods=['GET'])
def get_documents():
    user_id = request.args.get('user_id')
    db_type = request.args.get('db_type', 'chroma')  # 預設使用 Chroma DB
    
    if not user_id:
        return jsonify({"error": "必須提供用戶 ID"}), 400
    
    logger.info(f"獲取文檔列表: user_id={user_id}, db_type={db_type}")
    
    try:
        # 獲取向量資料庫實例
        vector_db = VectorDBFactory.get_db(db_type)
        
        # 搜索匹配用戶 ID 的文檔
        filter_dict = {"user_id": user_id}
        
        # 對於 Chroma，使用特定的查詢方式
        if db_type == 'chroma':
            try:
                # 使用特定格式的 where 條件
                chroma_where = {"user_id": {"$eq": user_id}}
                
                # 直接使用 ChromaVectorDB 實例以獲取更多控制
                chroma_db = vector_db
                results = chroma_db.collection.get(where=chroma_where)
                
                # 格式化結果
                matches = []
                if results and 'ids' in results:
                    for i in range(len(results['ids'])):
                        match = {
                            'id': results['ids'][i],
                            'metadata': results['metadatas'][i] if 'metadatas' in results else {}
                        }
                        matches.append(match)
                
                results = {"matches": matches}
                logger.info(f"Chroma 直接查詢找到 {len(matches)} 個向量")
            except Exception as e:
                logger.error(f"Chroma 直接查詢失敗: {str(e)}")
                # 如果直接查詢失敗，使用標準方法
                results = vector_db.get_documents(filter_dict)
        else:
            # 對於其他資料庫類型，使用標準方法
            results = vector_db.get_documents(filter_dict)
        
        # 提取唯一文檔
        documents = {}
        for match in results['matches']:
            metadata = match.get('metadata', {})
            doc_id = metadata.get('document_id')
            
            if doc_id and doc_id not in documents:
                documents[doc_id] = {
                    'document_id': doc_id,
                    'filename': metadata.get('filename', '未知檔案'),
                    'timestamp': metadata.get('timestamp', datetime.now().isoformat()),
                    'db_type': db_type
                }
        
        logger.info(f"找到 {len(documents)} 個唯一文檔")
        
        # 強制 Chroma 持久化更改
        if db_type == 'chroma':
            try:
                # 嘗試獲取 Chroma 客戶端並持久化
                chroma_db = vector_db
                # 某些版本的 Chroma 可能需要顯式持久化
                if hasattr(chroma_db.client, 'persist'):
                    chroma_db.client.persist()
                    logger.info("已強制 Chroma 持久化更改")
            except Exception as e:
                logger.error(f"Chroma 持久化失敗: {str(e)}")
        
        return jsonify({
            "documents": list(documents.values())
        })
        
    except Exception as e:
        logger.error(f"獲取文檔列表時出錯: {str(e)}")
        return jsonify({"error": f"獲取文檔列表時出錯: {str(e)}"}), 500

@app.route('/api/documents/<document_id>', methods=['DELETE'])
def delete_document(document_id):
    user_id = request.args.get('user_id')
    db_type = request.args.get('db_type', 'chroma')  # 預設使用 Chroma DB
    
    if not user_id:
        return jsonify({"error": "必須提供用戶 ID"}), 400
    
    try:
        # 獲取向量資料庫實例
        vector_db = VectorDBFactory.get_db(db_type)
        
        # 記錄刪除操作開始
        logger.info(f"開始刪除文檔: document_id={document_id}, user_id={user_id}, db_type={db_type}")
        
        # 刪除匹配的文檔
        filter_dict = {
            "user_id": user_id,
            "document_id": document_id
        }
        
        # 首先嘗試獲取要刪除的文檔，確認它存在
        if db_type == 'chroma':
            # 對於 Chroma，使用特定格式的 where 條件
            chroma_where = {
                "user_id": {"$eq": user_id},
                "document_id": {"$eq": document_id}
            }
            
            try:
                # 直接使用 ChromaVectorDB 實例以獲取更多控制
                chroma_db = vector_db
                results = chroma_db.collection.get(where=chroma_where)
                
                if results and 'ids' in results and results['ids']:
                    logger.info(f"找到要刪除的文檔: {len(results['ids'])} 個向量")
                    
                    # 直接使用 ID 列表刪除
                    ids_to_delete = results['ids']
                    chroma_db.collection.delete(ids=ids_to_delete)
                    logger.info(f"已刪除 {len(ids_to_delete)} 個向量")
                    delete_result = {"deleted": len(ids_to_delete)}
                else:
                    logger.warning(f"未找到要刪除的文檔: document_id={document_id}")
                    delete_result = {"deleted": 0}
            except Exception as e:
                logger.error(f"Chroma 刪除操作失敗: {str(e)}")
                # 嘗試使用標準方法
                delete_result = vector_db.delete(filter_dict=filter_dict)
        else:
            # 對於其他資料庫類型，使用標準方法
            delete_result = vector_db.delete(filter_dict=filter_dict)
        
        # 也刪除上傳的文件（如果存在）
        delete_count = 0
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            if filename.startswith(f"{document_id}_"):
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                os.remove(file_path)
                logger.info(f"已刪除文件: {file_path}")
                delete_count += 1
        
        # 強制 Chroma 持久化更改
        if db_type == 'chroma':
            try:
                # 嘗試獲取 Chroma 客戶端並持久化
                chroma_db = vector_db
                # 某些版本的 Chroma 可能需要顯式持久化
                if hasattr(chroma_db.client, 'persist'):
                    chroma_db.client.persist()
                    logger.info("已強制 Chroma 持久化更改")
            except Exception as e:
                logger.error(f"Chroma 持久化失敗: {str(e)}")
        
        # 記錄刪除操作完成
        logger.info(f"文檔刪除完成: document_id={document_id}, 刪除了 {delete_result.get('deleted', 0)} 個向量和 {delete_count} 個文件")
        
        return jsonify({
            "message": "文檔已成功刪除",
            "document_id": document_id,
            "files_deleted": delete_count,
            "vectors_deleted": delete_result.get("deleted", 0)
        })
        
    except Exception as e:
        logger.error(f"刪除文檔時出錯: {str(e)}")
        return jsonify({"error": f"刪除文檔時出錯: {str(e)}"}), 500

@app.route('/api/rotate-key', methods=['POST'])
def rotate_key():
    data = request.json
    user_id = data.get('user_id')
    old_key = data.get('old_key')
    new_key = data.get('new_key')
    db_type = data.get('db_type', 'pinecone_byok')  # 金鑰輪替僅適用於 BYOK 模式
    
    if db_type != 'pinecone_byok':
        return jsonify({"error": "金鑰輪替僅適用於 BYOK 模式"}), 400
    
    if not user_id or not old_key:
        return jsonify({"error": "必須提供用戶 ID 和舊金鑰"}), 400
    
    try:
        # 獲取向量資料庫實例
        vector_db = VectorDBFactory.get_db(db_type)
        
        # 將字符串轉換為二進制金鑰（如果需要）
        old_key_binary = old_key.encode() if isinstance(old_key, str) else old_key
        
        # 如果未提供新金鑰，則生成一個
        if not new_key:
            new_key_binary = Fernet.generate_key()
        else:
            new_key_binary = new_key.encode() if isinstance(new_key, str) else new_key
            Fernet(new_key_binary)  # 驗證金鑰格式
        
        # 驗證舊金鑰
        if user_id in user_keys and user_keys[user_id] != old_key_binary:
            return jsonify({"error": "提供的舊金鑰不正確"}), 401
        
        # 獲取用戶的所有文檔
        filter_dict = {"user_id": user_id, "is_encrypted": True}
        results = vector_db.get_documents(filter_dict)
        
        # 為每個向量重新加密
        old_fernet = Fernet(old_key_binary)
        new_fernet = Fernet(new_key_binary)
        
        vectors_to_update = []
        update_count = 0
        
        for match in results['matches']:
            try:
                metadata = match.get('metadata', {})
                
                # 只處理加密的文檔
                if not metadata.get('is_encrypted', False):
                    continue
                
                # 解密文本
                encrypted_text = metadata.get('encrypted_text', '')
                decrypted_text = decrypt_text(encrypted_text, old_key_binary)
                
                # 使用新金鑰重新加密
                new_encrypted_text = encrypt_text(decrypted_text, new_key_binary)
                
                # 更新元數據
                metadata['encrypted_text'] = new_encrypted_text
                
                # 準備更新
                vectors_to_update.append({
                    'id': match['id'],
                    'values': match.get('values', [0] * DIMENSION),
                    'metadata': metadata
                })
                
                update_count += 1
            except Exception as e:
                logger.error(f"處理向量時出錯: {str(e)}")
        
        # 批量更新
        if vectors_to_update:
            vector_db.store_vectors(
                vectors=[v.get('values') for v in vectors_to_update],
                metadata=[v.get('metadata') for v in vectors_to_update]
            )
        
        # 更新存儲的金鑰
        user_keys[user_id] = new_key_binary
        
        return jsonify({
            "message": "金鑰已成功輪替",
            "new_key": new_key_binary.decode(),
            "vectors_updated": update_count
        })
        
    except Exception as e:
        logger.error(f"輪替金鑰時出錯: {str(e)}")
        return jsonify({"error": f"輪替金鑰時出錯: {str(e)}"}), 500

@app.route('/api/db-types', methods=['GET'])
def get_db_types():
    """獲取支援的向量資料庫類型"""
    return jsonify({
        "db_types": [
            {
                "id": "chroma",
                "name": "Chroma DB (預設)",
                "description": "開源向量資料庫，無需加密金鑰",
                "requires_key": False
            },
            {
                "id": "pinecone_byok",
                "name": "Pinecone with BYOK",
                "description": "使用您自己的金鑰 (BYOK) 進行加密的 Pinecone 向量資料庫",
                "requires_key": True
            }
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
