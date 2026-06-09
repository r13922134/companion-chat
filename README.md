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
| Depression Worker | 無 | SQLite queue | 單 GPU 常駐模型與 PHQ-8 預測 |
| Hindsight API | `8888` | `http://localhost:8888/health` | 長期記憶 API |
| Hindsight UI | `9999` | `http://localhost:9999` | 記憶管理介面 |

Realtime 與 Feedback 預設共用 `data/app_data.sqlite3`，但檔案分別存到：

```text
uploads/realtime/
uploads/feedback/
```

Hindsight 只整合在 Realtime，不會在 Feedback 服務中執行 Recall 或 Retain。

### Realtime 論文資料契約

每次 Realtime 對話會存到
`uploads/realtime/<session_hash>/<run_id>/`，並由 SQLite
`realtime_session_runs` 索引。主要檔案如下：

| 檔案 | 用途 |
| --- | --- |
| `metadata.json` | 使用者、PHQ-8 ground truth、錄製格式與場次資訊 |
| `transcript.json` / `transcript.txt` | 含 speaker 與時間戳的完整逐字稿 |
| `user_audio.wav` | 使用者音訊；模型只從 participant speech intervals 取特徵 |
| `assistant_audio.wav` | 稽核用 assistant 音訊，不送入憂鬱預測 |
| `video_frames.zip` | 依時間排序的 JPEG 影格，預設每 250 ms 一張 |
| `archive_manifest.json` | schema、artifact SHA-256、模態狀態與 ground truth |
| `depression_result.json` | 八個 aspect 的 query、retrieval、推理與分數 |
| `depression_aspect_predictions.csv` | 便於論文分析的 aspect-level 表格 |

音訊或影像缺失時仍會執行預測，對應特徵以零向量補齊，並在結果
`metadata.hard_warnings` 留下原因。逐字稿中的 assistant 內容可供場次稽核，
但 retrieval 與 participant transcript 僅使用 user utterances；Feedback 服務不會排程憂鬱預測。

Realtime 上傳 API 會先回傳 queued 狀態，前端再輪詢預測結果，因此不會讓上傳 request
長時間卡住。評估期間顯示 loading modal；完成後顯示 personal aspect query、各 aspect
推理、retrieved participant transcript、預測分數與 PHQ-8 ground truth。

憂鬱預測工作透過 SQLite `depression_jobs` 佇列交給獨立 GPU worker。Flask process
不載入 checkpoint，也不執行 GPU inference；Web process 重啟不會遺失 queued job。
Worker 使用 lease 與 heartbeat，異常中止的 running job 會在 lease 到期後重新排隊。
同一 GPU ID 使用 filesystem lock，避免誤啟動兩份模型。

應用程式資料庫與 `uploads/` 是 host filesystem 資料；Docker Compose 只執行
Hindsight，記憶資料獨立存放在 named volume `companion-chat_hindsight-data`。兩者沒有共用資料庫
或 volume，`docker compose down -v` 只會刪除 Hindsight volume，不會刪除上述應用程式檔案。

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

本專案固定使用 PyTorch `2.6.0+cu124`，可搭配目前 NVIDIA driver 535，且符合
Transformers 載入 HuBERT `.bin` checkpoint 的 `torch.load` 安全版本要求。不可直接安裝
CUDA 13 build；Transformers 固定為 `4.53.2`。Worker 預設拒絕 silent CPU fallback。
安裝後確認：

```bash
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

應顯示 `2.6.0+cu124 12.4 True`。

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

先啟動單張 RTX 3090 的常駐 prediction worker：

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.depression_worker --gpu-id 0
```

Worker 啟動時會載入並 warm up DynRAG、base-Qwen retrieval、HuBERT 與 ResNet。
Retrieval 預設共用 detector 的 Qwen 4B 權重，並在 retrieval 時停用 LoRA，因此不會
在 3090 上重複載入第二份 Qwen 4B。訓練用 gradient checkpointing 會關閉，KV cache
會重新啟用。

可用環境變數調整 worker：

```dotenv
DEPRESSION_GPU_ID=0
DEPRESSION_WORKER_POLL_SECONDS=2
DEPRESSION_WORKER_LEASE_SECONDS=300
DEPRESSION_JOB_MAX_ATTEMPTS=3
DEPRESSION_ASPECT_RETRIEVAL_SHARE_LLM=1
```

`/health` 的 `depression_workers` 應顯示 `ready`，`depression_queue` 會顯示各狀態數量。
不要為同一張 GPU 啟動多個 worker；第二個 process 會因 GPU lock 直接失敗。
若 CUDA 不可用，worker 會立即結束並顯示版本資訊；`--allow-cpu` 僅供診斷，不應用於
實際預測。

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

Realtime 預設不啟用 Flask debugger/reloader，避免 production process 重啟後重新載入
Web 程式。開發時若需要自動重載，可暫時設定 `FLASK_DEBUG=1`。GPU worker 是獨立
process，不會跟著 Flask reloader 重啟。

若要讓 worker 登入後自動啟動，可安裝 user-level systemd service：

```bash
mkdir -p ~/.config/systemd/user
cp deploy/companion-depression-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now companion-depression-worker
systemctl --user status companion-depression-worker
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

# Terminal 2：Depression GPU worker
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.depression_worker --gpu-id 0

# Terminal 3：Realtime
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_realtime

# Terminal 4：Feedback
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_feedback
```
