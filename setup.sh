#!/bin/bash

# 設置顏色
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}開始設置 RAG 向量資料庫系統...${NC}"

# 創建資料目錄結構
echo -e "${YELLOW}創建資料目錄結構...${NC}"
mkdir -p data/{uploads,temp,vectors,keys,chroma_db,models}
echo -e "${GREEN}資料目錄已創建${NC}"

# 檢查並創建環境變數文件
if [ ! -f backend/.env ]; then
    echo -e "${YELLOW}創建環境變數文件...${NC}"
    cp backend/.env.example backend/.env
    echo -e "${GREEN}環境變數文件已創建，請根據需要修改 backend/.env${NC}"
else
    echo -e "${GREEN}環境變數文件已存在${NC}"
fi

# 設置權限
echo -e "${YELLOW}設置目錄權限...${NC}"
chmod -R 755 data
echo -e "${GREEN}權限已設置${NC}"

echo -e "${GREEN}設置完成！${NC}"
echo -e "${YELLOW}接下來請執行以下命令啟動系統：${NC}"
echo -e "docker-compose up -d"
