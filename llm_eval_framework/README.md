# LLM Evaluation & Red-Teaming Framework

**100% free stack** — Ollama (local LLM) + Detoxify + sentence-transformers + MLflow

Automatically tests LLM outputs for hallucination, toxicity, bias, and prompt injection.
Generates adversarial prompts, scores responses, logs to MLflow, and produces an HTML report.

---

## Quick Start (5 minutes)

### 1. Install Ollama (local LLM — free, runs offline)
```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Pull a model (free)
ollama pull llama3.2        # ~2GB, fast
# or: ollama pull mistral   # alternative
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full eval suite
```bash
python main.py --model llama3.2 --suite all
```

### 4. View your report
Open `reports/eval_report.html` in your browser.

### 5. View MLflow dashboard (optional)
```bash
mlflow ui
# Open http://localhost:5000
```

---

## Eval Suites

| Suite | What it tests |
|---|---|
| `hallucination` | Faithfulness vs ground truth (semantic sim + token overlap) |
| `toxicity` | 6-dim toxicity scoring via Detoxify (offline) |
| `redteam` | Jailbreaks, prompt injections, bias probes |
| `rag` | RAG faithfulness, answer relevancy, context recall/precision |
| `all` | All of the above |

### Run a specific suite
```bash
python main.py --suite toxicity --model llama3.2
python main.py --suite redteam --model mistral
python main.py --suite rag     --model llama3.2
```

---

## Project Structure

```
llm_eval_framework/
├── main.py                        # Entry point
├── runners/
│   ├── model_runner.py            # Ollama + Groq unified interface
│   └── batch_eval.py              # Runs all suites, logs to MLflow
├── red_team/
│   ├── prompt_generator.py        # Jailbreaks + prompt injections
│   └── bias_probes.py             # Gender, political, racial bias prompts
├── metrics/
│   ├── hallucination.py           # Faithfulness scoring
│   ├── toxicity.py                # Detoxify (offline)
│   ├── relevancy.py               # Answer relevancy
│   └── ragas_metrics.py           # RAG-specific metrics
├── reporting/
│   └── report_gen.py              # HTML report generator
├── .github/workflows/
│   └── eval_ci.yml                # CI/CD — runs eval on every PR
├── Dockerfile
└── requirements.txt
```

---

## Free Stack — Zero Cost Breakdown

| Component | Tool | Cost |
|---|---|---|
| LLM inference | Ollama (local) | Free forever |
| Toxicity scoring | Detoxify | Free, offline |
| Embeddings | sentence-transformers | Free, offline |
| Experiment tracking | MLflow (local) | Free forever |
| CI/CD | GitHub Actions | Free (2000 min/month) |
| Hosting | HuggingFace Spaces | Free tier |

### Optional: Groq free API (faster inference)
```bash
# Sign up free at https://groq.com (no credit card)
export GROQ_API_KEY=your_key_here
python main.py --model llama-3.3-70b-versatile  # uses Groq backend
```

---

## Resume Bullets (copy-paste)

- Built an automated LLM red-teaming framework generating adversarial prompts (jailbreaks,
  bias probes, prompt injections) and scoring model outputs on hallucination, toxicity
  (Detoxify), and faithfulness (sentence-transformers) — fully local, zero API cost.

- Evaluated Llama-3 and Mistral across 200+ adversarial test cases; logged all results
  with MLflow and auto-generated HTML pass/fail eval reports per safety category.

- Integrated GitHub Actions CI/CD to trigger full eval suite on every model update,
  enforcing quality gates with faithfulness threshold of 0.7 before deployment.
