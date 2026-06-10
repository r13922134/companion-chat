# companion-chat

語音陪伴系統，包含 Realtime 對話、PHQ-8 depression prediction、Realtime analytics、Feedback、Hindsight 長期記憶。

## 服務

| 服務 | Port | URL |
| --- | ---: | --- |
| Realtime | `9050` | `http://localhost:9050/realtime` |
| Realtime Analytics | `9050` | `http://localhost:9050/realtime-analytics` |
| Feedback | `9051` | `http://localhost:9051/realtime` |
| Hindsight API | `8888` | `http://localhost:8888/health` |
| Hindsight UI | `9999` | `http://localhost:9999` |

資料位置：

```text
data/app_data.sqlite3
uploads/realtime/
uploads/feedback/
```

## 安裝

```bash
git clone https://github.com/r13922134/companion-chat.git
cd companion-chat

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

建立 `.env`：

```dotenv
OPENAI_API_KEY=sk-your-openai-api-key
OPENAI_MED_QA_VECTOR_STORE_ID=vs_your_vector_store_id
```

`OPENAI_MED_QA_VECTOR_STORE_ID` 可省略；省略時醫療 QA / feedback vector store 更新不啟用。

## Google Form 規則

Google Form 必須開啟收集 email。Apps Script payload 必須有：

```json
{
  "respondent_email": "user@example.com",
  "fields": {
    "姓名": "測試使用者",
    "年齡": "40"
  }
}
```

規則：

- 同一個 email 會是同一個使用者。
- 使用者選擇頁只顯示該 email 最新一份 PHQ-8。
- 每份表單都有自己的 `form_hash`，不會覆蓋舊表單。
- 新 run 會使用當下最新 PHQ-8 當 GT。
- Analytics 歷史卡片使用該 run 當時保存的 GT snapshot，不會被後來的新表單覆蓋。

測試送一筆表單：

```bash
curl -X POST http://127.0.0.1:9050/google_form \
  -H 'Content-Type: application/json' \
  -d '{
    "form_title": "PHQ-8情緒量表",
    "respondent_email": "test@example.com",
    "submitted_at": "2026-06-06T12:00:00+08:00",
    "fields": {
      "姓名": "測試使用者",
      "年齡": "40",
      "日期": "2026-06-06"
    }
  }'
```

## 啟動

Terminal 1：Hindsight

```bash
cd ~/companion/companion-chat
sudo docker compose -f docker-compose.hindsight.yml up -d
```

Terminal 2：Depression worker

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.depression_worker --gpu-id 0
```

Terminal 3：Realtime

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_realtime
```

Terminal 4：Feedback

```bash
cd ~/companion/companion-chat
source .venv/bin/activate
python -m app.server_feedback
```

## 檢查

```bash
curl http://127.0.0.1:9050/health
curl http://127.0.0.1:9051/health
curl http://127.0.0.1:8888/health
```

開頁面：

```text
http://localhost:9050/realtime
http://localhost:9050/realtime-analytics
http://localhost:9051/realtime
http://localhost:9999
```
