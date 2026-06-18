# Lab 16 — Reflexion Agent

## Tổng quan

Bài lab giúp bạn hiểu và triển khai **Reflexion Agent** — một kiến trúc agent có khả năng tự phản chiếu (self-reflection) để cải thiện câu trả lời qua nhiều lần thử.

Repo cung cấp một scaffold hoàn chỉnh với mock data. Nhiệm vụ của bạn là **thay thế mock bằng LLM thật** và chạy benchmark trên dữ liệu thật.

## Cách hoạt động của Scaffold

Repo sử dụng **Mock Runtime** (`mock_runtime.py`) để giả lập phản hồi LLM:
- `actor_answer()` → trả lời câu hỏi (giả lập)
- `evaluator()` → chấm điểm đúng/sai (giả lập)
- `reflector()` → phân tích lỗi và đề xuất chiến thuật mới (giả lập)

Kết quả mock hoàn toàn deterministic — giúp bạn hiểu flow trước khi tốn chi phí API.

### Chạy thử với mock
```bash
# Cài đặt môi trường
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Chạy benchmark với mock data
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run

# Chạy benchmark đủ ngưỡng autograde (104 mẫu mock mở rộng)
python run_benchmark.py --dataset data/hotpot_mock_104.json --out-dir outputs/mock_104

# Chạy bản benchmark chi tiết để nộp/xem báo cáo đầy đủ
python run_benchmark.py --dataset data/hotpot_mock_104.json --out-dir outputs/detailed_benchmark

# Tự kiểm tra thêm trên bộ 120 câu hỏi multi-hop đa dạng
python run_benchmark.py --dataset data/hotpot_diverse_120.json --out-dir outputs/diverse_120

# Chạy chấm điểm tự động
python autograde.py --report-path outputs/mock_104/report.json
```

### Chạy với LLM thật
Mặc định repo vẫn dùng mock runtime để chạy offline và phục vụ autograde. Để thay bằng LLM thật, đặt `REFLEXION_RUNTIME` trước khi chạy benchmark:

```bash
# Ollama local
export REFLEXION_RUNTIME=ollama
export OLLAMA_MODEL=llama3.1
python run_benchmark.py --dataset data/hotpot_mock_104.json --out-dir outputs/ollama_run

# OpenAI hoặc endpoint OpenAI-compatible
export REFLEXION_RUNTIME=openai
export OPENAI_API_KEY=...
export OPENAI_MODEL=gpt-4o-mini
# Tuỳ chọn nếu dùng endpoint compatible khác:
# export REFLEXION_OPENAI_BASE_URL=https://your-endpoint.example/v1
python run_benchmark.py --dataset data/hotpot_mock_104.json --out-dir outputs/openai_run
```

Bạn cũng có thể tạo file `.env` để khỏi export mỗi lần:

```bash
cp .env.example .env
# sửa OPENAI_API_KEY trong .env
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/openai_smoke
```

Để test chi phí thấp, dùng `OPENAI_MODEL=gpt-4o-mini` và chạy trước trên `data/hotpot_mini.json`. Khi smoke test ổn mới chạy Golden hoặc dataset lớn.

Smoke test OpenAI tiết kiệm nhất:

```bash
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/openai_smoke --limit 1
```

Nếu bị timeout SSL tới `api.openai.com`, kiểm tra VPN/proxy/firewall hoặc dùng endpoint OpenAI-compatible mà mạng của bạn truy cập được:

```bash
export REFLEXION_OPENAI_BASE_URL=https://your-endpoint.example/v1
export REFLEXION_LLM_TIMEOUT=30
export REFLEXION_LLM_RETRIES=2
```

Report sẽ tự ước tính tiền token trong `report.md` và `report.json`. Mặc định dùng giá tham khảo thấp cho `gpt-4o-mini`:

```bash
export REFLEXION_INPUT_PRICE_PER_1K_TOKENS=0.00015
export REFLEXION_OUTPUT_PRICE_PER_1K_TOKENS=0.0006
export REFLEXION_USD_TO_VND=25000
```

### Golden Test Set cuối ngày
Khi giảng viên phát file Golden Test Set, lưu file vào `data/` theo đúng format `QAExample` hoặc HotpotQA gốc, ví dụ `data/golden_test.json`, rồi chạy:

```bash
python run_golden.py
python autograde.py --report-path outputs/golden_test/report.json
```

Nếu file Golden có tên khác:

```bash
python run_golden.py --dataset data/<ten_file_golden>.json --out-dir outputs/golden_test
```

Các file cần nộp thường nằm trong `outputs/golden_test/`:
- `report.json`
- `report.md`
- `react_runs.jsonl`
- `reflexion_runs.jsonl`

`report.md` tự sinh các phần cần trình bày:
- Bảng so sánh ReAct Agent và Reflexion Agent
- Bảng ước tính cost, token và running time
- Summary benchmark, failure modes, extensions và discussion
- Breakdown chi tiết theo agent, difficulty, correct/incorrect và sample trace/reflection

## Nhiệm vụ của Học viên

### Bước 1: Hiểu flow (đọc code)
Đọc và hiểu luồng hoạt động trong các file sau:
- `src/reflexion_lab/agents.py` — Vòng lặp chính của ReAct và Reflexion Agent
- `src/reflexion_lab/mock_runtime.py` — Logic giả lập (để biết cần thay thế gì)
- `src/reflexion_lab/schemas.py` — Cấu trúc dữ liệu (có TODO cần hoàn thiện)
- `src/reflexion_lab/prompts.py` — System prompts (có TODO cần viết)

### Bước 2: Hoàn thiện TODO trong scaffold
1. **`schemas.py`**: Định nghĩa các trường cho `JudgeResult` và `ReflectionEntry` (hiện tại là `pass`)
2. **`agents.py`** (dòng 31-35): Triển khai logic Reflexion loop — gọi `reflector()`, cập nhật `reflection_memory`
3. **`prompts.py`**: Viết System Prompt cho Actor, Evaluator, và Reflector

### Bước 3: Thay thế Mock bằng LLM thật
Thay thế 3 hàm trong `mock_runtime.py` bằng LLM call thật:

| Hàm mock | Thay bằng |
|---|---|
| `actor_answer()` | Gửi `ACTOR_SYSTEM` + question + context → LLM → parse câu trả lời |
| `evaluator()` | Gửi `EVALUATOR_SYSTEM` + question + gold_answer + predicted → LLM → parse `JudgeResult` |
| `reflector()` | Gửi `REFLECTOR_SYSTEM` + question + wrong answer + lý do sai → LLM → parse `ReflectionEntry` |

Có thể sử dụng: Ollama, vLLM, OpenAI API, Gemini API, hoặc bất kỳ LLM nào.

### Bước 4: Tạo dữ liệu test và chạy Benchmark

> **Quan trọng:** File `data/hotpot_mini.json` chỉ có 8 câu hỏi và được thiết kế cho mock runtime. Bạn **cần tự tạo thêm dữ liệu test** để kiểm tra implementation của mình.

**Cách tạo dữ liệu test:**
- Tải từ [HotpotQA dataset](https://hotpotqa.github.io/) hoặc từ https://drive.google.com/file/d/1382R9RhGUFZZpuRsfi8BMKuv3yorOB9H/view?usp=sharing và chuyển đổi sang format `QAExample`:
  ```json
  {
    "qid": "my_q1",
    "difficulty": "medium",
    "question": "Câu hỏi multi-hop...",
    "gold_answer": "Đáp án đúng",
    "context": [
      {"title": "Nguồn 1", "text": "Thông tin liên quan..."},
      {"title": "Nguồn 2", "text": "Thông tin liên quan..."}
    ]
  }
  ```
- Hoặc tự viết câu hỏi multi-hop của riêng bạn
- Lưu vào `data/` và chạy: `python run_benchmark.py --dataset data/my_test_set.json`

**Yêu cầu tối thiểu:** Chạy benchmark trên ít nhất **100 mẫu** để đạt điểm đầy đủ cho phần Experiment (`autograde.py` kiểm tra `num_records >= 100`).

### Bước 5: Tính toán Token thực tế
Thay thế `token_estimate` và `latency_ms` hardcoded trong `agents.py` bằng giá trị thật từ LLM response.

## Tiêu chí chấm điểm (Rubric)

| Phần | Điểm | Yêu cầu |
|---|---:|---|
| **Core Flow** | **80** | |
| Schema completeness | 30 | Report có đủ các key: `meta`, `summary`, `failure_modes`, `examples`, `extensions`, `discussion` |
| Experiment completeness | 30 | Có cả ReAct + Reflexion, ≥100 records, ≥20 examples chi tiết |
| Analysis depth | 20 | ≥3 failure modes được phân tích, discussion ≥250 ký tự |
| **Bonus** | **20** | Triển khai ≥1 extension (mỗi extension = 10đ, tối đa 20đ) |

**Bonus extensions:** `structured_evaluator`, `reflection_memory`, `adaptive_max_attempts`, `memory_compression`, `mini_lats_branching`, `plan_then_execute`, `benchmark_report_json`, `mock_mode_for_autograding`

## ⏰ Golden Test Set (Bonus cuối ngày)

> Trong **15 phút cuối** của buổi lab, giảng viên sẽ phát một **Golden Test Set** — bộ dữ liệu test mà học viên chưa từng thấy trước đó.
>
> Bạn sẽ chạy agent của mình trên bộ dữ liệu này và nộp kết quả. Điểm từ Golden Test Set sẽ được dùng để **xếp hạng và tính điểm bonus** giữa các nhóm.
>
> **Lưu ý:** Đây là lý do bạn cần đảm bảo agent hoạt động tốt trên **nhiều loại câu hỏi khác nhau**, không chỉ trên `hotpot_mini.json`. Hãy tự tạo dữ liệu test đa dạng để kiểm tra trước!

## Thành phần mã nguồn

| File | Mô tả |
|---|---|
| `src/reflexion_lab/schemas.py` | Kiểu dữ liệu: `QAExample`, `RunRecord`, `JudgeResult`, `ReflectionEntry`, ... |
| `src/reflexion_lab/prompts.py` | Template prompt cho Actor, Evaluator, Reflector **(TODO)** |
| `src/reflexion_lab/mock_runtime.py` | Logic giả lập LLM **(cần thay thế)** |
| `src/reflexion_lab/agents.py` | Vòng lặp chính ReAct + Reflexion Agent **(có TODO)** |
| `src/reflexion_lab/reporting.py` | Xuất báo cáo benchmark |
| `src/reflexion_lab/utils.py` | Helpers: `load_dataset`, `normalize_answer`, `save_jsonl` |
| `run_benchmark.py` | Script chạy đánh giá |
| `autograde.py` | Chấm điểm tự động từ `report.json` |
| `data/hotpot_mini.json` | 8 câu hỏi multi-hop mẫu (dùng cho mock) |
