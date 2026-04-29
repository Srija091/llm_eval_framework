# 🔍 LLM Evaluation & Red-Teaming Framework

> **Automatically test any LLM for hallucination, toxicity, bias, and prompt injection — 100% free, runs locally.**

Built with **Ollama** (local LLM) · **Detoxify** · **sentence-transformers** · **MLflow** · **GitHub Actions CI/CD**

---<img width="1470" height="956" alt="Screenshot 2025-11-21 at 2 02 54 PM" src="https://github.com/user-attachments/assets/a17e283e-50b0-4037-a770-f9cf60dbb37b" />


## 📊 Live Eval Report — Llama 3.2



> *Real output from running the full eval suite on Llama 3.2 across 51 test cases.*

---

## 🔑 Key Findings (Llama 3.2)

| Metric | Score | Verdict |
|---|---|---|
| Avg Faithfulness | 71.0% | ✅ Borderline pass |
| Avg Toxicity | 0.5% | ✅ Excellent — safe for production |
| Red-Team Refusal Rate | 16.7% | ⚠️ Low — jailbreak vulnerability found |
| RAG Faithfulness | 76.1% | ✅ Pass |
| RAG Answer Relevancy | 58.8% | ⚠️ Needs improvement |
| RAG Context Precision | 100.0% | ✅ Perfect retrieval |
| RAG Context Recall | 100.0% | ✅ Perfect retrieval |

**Key insight:** The model's red-team refusal rate was only 16.7% — "educational purposes" and "fictional story" jailbreak framings bypassed safety guardrails in several cases. RAG answer relevancy gaps were traced to generation verbosity, not retrieval failure (context recall was 100%).

---

## ⚡ Quick Start

### 1. Install Ollama (local LLM — free, no API key)
```bash
# macOS / Linux
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full eval suite
```bash
python main.py --model llama3.2 --suite all
```

### 4. Open your HTML report
```bash
open reports/eval_report.html
```

### 5. View MLflow dashboard
```bash
mlflow ui
# → http://localhost:5000
```

---

## 🧪 Eval Suites

| Suite | What it tests | Cases |
|---|---|---|
| `hallucination` | Faithfulness vs ground truth via semantic similarity + token overlap | 5 |
| `toxicity` | 6-dimension toxicity scoring via Detoxify (offline, no API) | 5 |
| `redteam` | Jailbreaks, prompt injections, bias probes (gender/political/racial/age) | 36 |
| `rag` | RAG faithfulness, answer relevancy, context recall & precision | 5 |
| `all` | All of the above | 51 |

```bash
# Run a specific suite
python main.py --suite toxicity  --model llama3.2
python main.py --suite redteam   --model mistral
python main.py --suite rag       --model llama3.2

# Compare two models
python main.py --model llama3.2  --output llama_report.html
python main.py --model mistral   --output mistral_report.html
```

---

## 🏗️ Architecture

```
llm_eval_framework/
│
├── main.py                          # CLI entry point
│
├── runners/
│   ├── model_runner.py              # Unified Ollama + Groq interface
│   └── batch_eval.py                # Orchestrates all suites + MLflow logging
│
├── red_team/
│   ├── prompt_generator.py          # Auto-generates jailbreaks + injections
│   └── bias_probes.py               # Gender, political, racial, age, religious bias
│
├── metrics/
│   ├── hallucination.py             # Semantic similarity faithfulness scoring
│   ├── toxicity.py                  # Detoxify — 6-dim, fully offline
│   ├── relevancy.py                 # Answer relevancy via cosine similarity
│   └── ragas_metrics.py             # RAG-specific: faithfulness, recall, precision
│
├── reporting/
│   └── report_gen.py                # Generates HTML pass/fail eval report
│
├── assets/
│   └── eval_report_screenshot.png   # Sample output screenshot
│
├── .github/workflows/
│   └── eval_ci.yml                  # CI/CD: runs eval on every PR
│
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 💸 Free Stack — Zero Cost

| Component | Tool | Cost |
|---|---|---|
| LLM inference | Ollama (local) — Llama 3.2, Mistral, Gemma | **Free forever** |
| Toxicity scoring | Detoxify — runs fully offline | **Free** |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | **Free** |
| Experiment tracking | MLflow (local) | **Free forever** |
| CI/CD | GitHub Actions (2000 min/month free) | **Free** |
| Optional faster LLM | Groq API free tier (Llama-3.3-70B) | **Free** |

No credit card. No cloud account. Everything runs on your laptop.

---

## 🔄 CI/CD Quality Gate

Every pull request automatically triggers the eval suite via GitHub Actions.
If faithfulness drops below **0.5**, the pipeline fails and blocks the merge.

```yaml
# .github/workflows/eval_ci.yml
- name: Run hallucination eval
  run: python main.py --suite hallucination --model llama3.2

- name: Check quality gate
  run: python check_quality_gate.py
  # Fails build if avg_faithfulness < 0.5
```

---

## 🚀 Optional: Groq Free API (10x faster inference)

```bash
# Sign up free at https://groq.com — no credit card needed
export GROQ_API_KEY=your_key_here

# Run at cloud speed using Llama-3.3-70B
python main.py --model llama-3.3-70b-versatile
```

---

## 📋 Resume Bullets

```
• Built an automated LLM red-teaming framework generating adversarial prompts
  (jailbreaks, bias probes, prompt injections) across 51 test cases; identified
  a 16.7% refusal rate gap exposing jailbreak vulnerabilities in Llama 3.2 via
  "educational" and "fictional" prompt framing.

• Evaluated RAG pipeline quality using custom faithfulness, context recall, and
  answer relevancy metrics; diagnosed 58.8% relevancy gap as a generation-side
  issue — context recall was 100%, confirming retrieval was not the bottleneck.

• Integrated MLflow experiment tracking and GitHub Actions CI/CD to auto-run
  eval suite on every model update, enforcing quality gates before deployment —
  fully local, zero API cost (Ollama + Detoxify + sentence-transformers).
```

---

## 🛠️ Requirements

```
ollama>=0.1.8
detoxify>=0.5.2
sentence-transformers>=2.7.0
mlflow>=2.12.0
pandas>=2.0.0
```

---

## 📄 License

MIT — free to use, modify, and build on.

---

<p align="center">
  Built by Srija Nallamothu &nbsp;·&nbsp;
  <a href="https://linkedin.com/in/srija-a0bb51220">LinkedIn</a>
</p>
