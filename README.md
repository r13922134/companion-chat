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
cp .env.example .env
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

```bash
sudo docker compose -f docker-compose.hindsight.yml pull
sudo docker compose -f docker-compose.hindsight.yml up -d
```

第一次啟動會下載 Hindsight image、embedding model 與 reranker，可能需要數分鐘。

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

## 停止服務

Realtime 與 Feedback terminal 分別按 `Ctrl+C`。

停止 Hindsight：

```bash
cd ~/companion/companion-chat
sudo docker compose -f docker-compose.hindsight.yml down
```

這不會刪除 Hindsight volume。

## 測試

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m unittest discover -s tests -v
python -m py_compile app/server_realtime.py app/server_feedback.py app/storage.py
```

## 遠端連線注意事項

瀏覽器的麥克風與攝影機通常只允許安全來源：

- 同一台主機使用 `localhost` 可以直接測試。
- 從其他電腦連線時，正式環境應使用 HTTPS reverse proxy。
- Hindsight 的 `8888/9999` 只綁定 `127.0.0.1`，不會直接暴露到區域網路。

需要從另一台電腦查看 Hindsight UI 時，可建立 SSH tunnel：

```bash
ssh \
  -L 8888:127.0.0.1:8888 \
  -L 9999:127.0.0.1:9999 \
  penguin37@charm-master
```

然後在本機開啟：

```text
http://localhost:9999
```

## 常見問題

### `http://localhost:9999` 無法連線

```bash
sudo docker compose -f docker-compose.hindsight.yml ps -a
sudo docker compose -f docker-compose.hindsight.yml logs --tail=200 hindsight
curl http://127.0.0.1:8888/health
```

若 log 顯示 embedded PostgreSQL `Permission denied`，確認 Compose 使用 named volume，然後重新建立容器：

```bash
sudo docker compose -f docker-compose.hindsight.yml down
sudo docker compose -f docker-compose.hindsight.yml up -d --force-recreate
```

### 出現 `[HINDSIGHT] Recall failed ... timed out`

目前 Recall timeout 預設為 5 秒。該輪會跳過記憶，但 Realtime 回答不會中斷。先確認 Hindsight health 與 logs；第一次啟動時也要等待本機模型載入完成。

### `openai_client_ready` 是 `false`

確認 `.env` 位於專案根目錄，且包含有效的：

```dotenv
OPENAI_API_KEY=...
```

修改 `.env` 後，要重新啟動兩個 Python 服務與 Hindsight container。

### Realtime 沒有使用到舊記憶

每位使用者的 Google Form `form_hash` 對應一個 Hindsight memory bank。完成一場對話並正常按下結束後，逐字稿才會非同步送入 Hindsight Retain；下一場對話才會透過 Recall 使用長期記憶。

## 資料位置

```text
data/app_data.sqlite3                         # 共用 SQLite
uploads/realtime/<session_hash>/<run_id>/    # 主 Realtime 多場對話
uploads/feedback/                            # Feedback 對話與意見
Docker named volume: hindsight-data           # Hindsight PostgreSQL/記憶
```

以上本機資料都不會提交到 Git。
