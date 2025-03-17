FROM python:3.9-slim

WORKDIR /app

# 安裝基本依賴
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製需求檔案並安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製程式碼
COPY . .

# 創建必要的目錄
RUN mkdir -p uploads temp vectors keys

# 設置環境變數
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# 暴露端口
EXPOSE 5000

# 啟動程式
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
