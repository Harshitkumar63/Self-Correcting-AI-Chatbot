# 🧠 Self-Improving LLM Pipeline

> Automated Evaluation & LoRA Fine-Tuning Feedback Loop

An end-to-end system where LLM outputs are automatically evaluated using semantic similarity, low-quality responses are collected, and the model is iteratively improved through PEFT/LoRA fine-tuning.

## Architecture

```
User Query → LLM Response → Evaluation (Semantic Similarity)
                                ↓
                        Score ≥ 0.70? → ✅ Passed
                        Score < 0.70? → ⚠ Flagged → Feedback Dataset
                                                        ↓
                                                 50 samples collected
                                                        ↓
                                                 LoRA Fine-Tuning
                                                        ↓
                                                 Improved Model
```

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 14 (App Router), TypeScript, TailwindCSS, Shadcn/ui |
| **Backend** | FastAPI, SQLAlchemy 2.0, aiosqlite, Pydantic v2 |
| **ML - Inference** | Qwen/Qwen2.5-0.5B-Instruct (HuggingFace Transformers) |
| **ML - Evaluation** | sentence-transformers/all-MiniLM-L6-v2 |
| **ML - Training** | PEFT, LoRA, trl (SFTTrainer) |
| **Storage** | SQLite |

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip

### 1. Backend

```bash
cd backend
pip install -r requirements.txt
python run.py
```

Backend starts at `http://localhost:8000`.

> **Note:** First run will download the ML models (~1GB for Qwen + ~90MB for MiniLM).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend starts at `http://localhost:3000`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/chat` | Send query → get evaluated response |
| `GET` | `/api/logs` | Paginated interaction logs |
| `GET` | `/api/logs/stats` | Dashboard statistics |
| `POST` | `/api/trigger-tuning` | Manually trigger LoRA fine-tuning |
| `GET` | `/api/tuning-status` | Check training progress |
| `GET` | `/api/health` | Pipeline health check |

## Project Structure

```
├── frontend/           # Next.js 14 dashboard
│   └── src/
│       ├── app/        # Pages and layout
│       ├── components/ # Chat, Monitor, Training panel
│       └── lib/        # API client
│
├── backend/            # FastAPI server
│   ├── app/
│   │   ├── routers/    # API endpoints
│   │   ├── services/   # ML pipeline bridge
│   │   ├── models.py   # SQLAlchemy ORM
│   │   └── schemas.py  # Pydantic schemas
│   └── run.py          # Uvicorn entry point
│
├── ml_pipeline/        # Core ML modules
│   ├── inference.py    # Qwen2.5 text generation
│   ├── evaluation.py   # Semantic similarity scoring
│   ├── feedback.py     # Flagged response collector
│   ├── trainer.py      # PEFT/LoRA training script
│   └── ground_truth.json # Reference Q&A dataset
```

## How It Works

1. **Chat**: User sends a query through the dashboard.
2. **Inference**: Qwen2.5-0.5B generates a response.
3. **Evaluation**: The response is compared against ground-truth answers using cosine similarity (via all-MiniLM-L6-v2).
4. **Feedback**: If similarity score < 0.70, the response is flagged and the correct answer is saved to `fine_tune_dataset.jsonl`.
5. **Training**: When 50+ flagged samples accumulate, LoRA fine-tuning can be triggered (manually or automatically) to improve the model.

## Configuration

Edit `ml_pipeline/config.py` to customize:

- `similarity_threshold` — Score cutoff for pass/fail (default: 0.70)
- `fine_tune_trigger_count` — Samples before auto-training (default: 50)
- `lora_r`, `lora_alpha` — LoRA hyperparameters
- `inference_model_id` — Base model to use
