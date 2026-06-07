# companion-chat

癌症病患與家屬的語音陪伴系統，包含：

- OpenAI Realtime 語音對話
- 情緒分類與傾聽／建議模式
- 醫療 QA Vector Store 查詢
- 即時 Web Search
- Hindsight 跨場次長期記憶
- 對模型回答留下 feedback，並更新到既有 OpenAI Vector Store
- SQLite 對話索引，以及逐場錄音、影格、逐字稿存檔

## 服務架構

| 服務 | Port | 網址 | 說明 |
| --- | ---: | --- | --- |
| Realtime | `9050` | `http://localhost:9050/realtime` | 主對話系統，包含 Hindsight 記憶 |
| Feedback | `9051` | `http://localhost:9051/realtime` | 回饋版，將有效 feedback 更新到 Vector Store |
| Hindsight API | `8888` | `http://localhost:8888/health` | 長期記憶 API |
| Hindsight UI | `9999` | `http://localhost:9999` | 記憶管理介面 |

Realtime 與 Feedback 預設共用 `data/app_data.sqlite3`，但檔案分別存到：

```text
uploads/realtime/
uploads/feedback/
```

Hindsight 只整合在 Realtime，不會在 Feedback 服務中執行 Recall 或 Retain。

## 前置需求

- Linux
- Python 3.10 或更新版本
- Git
- Docker Engine
- Docker Compose v2
- 可使用 Realtime API 的 OpenAI API key
- 選用：已建立的 OpenAI Vector Store ID

確認版本：

```bash
python3 --version
git --version
sudo docker version
sudo docker compose version
```

若目前使用者已加入 `docker` 群組，以下 Docker 指令可以移除 `sudo`。

## Step 1：下載專案

```bash
cd ~/companion
git clone https://github.com/r13922134/companion-chat.git
cd companion-chat
```

如果專案已存在：

```bash
cd ~/companion/companion-chat
git pull --ff-only
```

## Step 2：建立 Python venv

不要共用其他專案的 `.venv`：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

之後每次開新 terminal，都要先執行：

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
```

## Step 3：設定環境變數

建立本機 `.env`：

```bash
touch .env
nano .env
```

至少設定：

```dotenv
OPENAI_API_KEY=sk-your-openai-api-key
```

若要啟用醫療 QA 與 Feedback Vector Store 更新，再設定：

```dotenv
OPENAI_MED_QA_VECTOR_STORE_ID=vs_your_vector_store_id
```

`.env` 已被 Git 忽略，不可提交 API key。`OPENAI_MED_QA_VECTOR_STORE_ID` 未設定時，語音對話仍可使用，但 `medical_qa` 與 feedback 上傳 Vector Store 不會啟用。

## Step 4：啟動 Hindsight Docker

Compose 會讀取專案根目錄的 `.env`，並將 `OPENAI_API_KEY` 傳入 Hindsight。
語音延遲優先的預設設定使用 Hindsight 的 `rrf` reranker，保留記憶檢索並避免本機
cross-encoder 阻塞每一輪。`HINDSIGHT_RECALL_TIMEOUT_SECONDS` 是 Realtime 服務等待
Hindsight recall 的時間上限，預設 10 秒。若需要調整 recall 或改用本機 reranker，
可在 `.env` 覆寫：

```dotenv
HINDSIGHT_RERANKER_PROVIDER=rrf
HINDSIGHT_RERANKER_MAX_CANDIDATES=50
HINDSIGHT_RECALL_TIMEOUT_SECONDS=10
```

```bash
sudo docker compose -f docker-compose.hindsight.yml pull
sudo docker compose -f docker-compose.hindsight.yml up -d
```

第一次啟動會下載 Hindsight image 與 embedding model，可能需要數分鐘。預設 `rrf`
不需要下載本機 reranker；只有把 `HINDSIGHT_RERANKER_PROVIDER` 改成本機 reranker
時，才會再下載對應的 reranker model。

檢查狀態：

```bash
sudo docker compose -f docker-compose.hindsight.yml ps
sudo docker compose -f docker-compose.hindsight.yml logs -f hindsight
```

看到 API 完成啟動後，另開 terminal 測試：

```bash
curl http://127.0.0.1:8888/health
```

Hindsight UI：

```text
http://localhost:9999
```

Hindsight 資料保存在 Docker named volume。一般重啟或 `docker compose down` 不會刪除記憶；不要執行 `docker compose down -v`，除非確定要刪除全部 Hindsight 記憶。

## Step 5：啟動 Realtime 服務

開啟第一個 terminal：

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_realtime
```

開啟：

```text
http://localhost:9050/realtime
```

健康檢查：

```bash
curl http://127.0.0.1:9050/health
```

確認回傳內容中的：

- `status` 是 `ok`
- `openai_client_ready` 是 `true`
- `hindsight_enabled` 是 `true`
- `hindsight_base_url` 是 `http://127.0.0.1:8888`

Hindsight 暫時無法連線時，Realtime 仍會繼續回答，只會跳過該輪長期記憶。

## Step 6：啟動 Feedback 服務

開啟第二個 terminal：

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_feedback
```

開啟：

```text
http://localhost:9051/realtime
```

健康檢查：

```bash
curl http://127.0.0.1:9051/health
```

Feedback 不需要 Hindsight。當 assistant feedback 非空時，系統會將「前一句 user utterance + feedback」建立成 QA pair，更新到 `OPENAI_MED_QA_VECTOR_STORE_ID` 指定的同一個 Vector Store。

## Step 7：建立測試使用者

Realtime 首頁會從 SQLite 顯示 Google Form 使用者。若清單是空的，可以先建立一筆測試資料：

```bash
curl -X POST http://127.0.0.1:9050/google_form \
  -H 'Content-Type: application/json' \
  -d '{
    "form_title": "PHQ-8情緒量表",
    "submitted_at": "2026-06-06T12:00:00+08:00",
    "fields": {
      "姓名": "測試使用者",
      "年齡": "40",
      "日期": "2026-06-06"
    }
  }'
```

重新整理 `http://localhost:9050/realtime` 後即可選擇該使用者。

## 完整啟動順序

日常啟動只需要：

```bash
# Terminal 1：Hindsight
cd ~/companion/companion-chat
sudo docker compose -f docker-compose.hindsight.yml up -d

# Terminal 2：Realtime
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_realtime

# Terminal 3：Feedback
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_feedback
```
