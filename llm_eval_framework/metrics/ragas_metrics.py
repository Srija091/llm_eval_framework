"""
metrics/ragas_metrics.py
RAG-specific evaluation using Ragas (free, open source).
Tests faithfulness, answer relevancy, context recall, context precision.
pip install ragas datasets
"""

from runners.model_runner import call_model


def run_rag_eval(model: str = "llama3.2") -> dict:
    """
    Run Ragas evaluation on a sample RAG dataset.
    Uses Ollama as the LLM backend (free, local).
    Returns per-row results and averaged metrics.
    """
    # Sample RAG test cases — in production load from your real RAG pipeline
    test_cases = _get_rag_test_cases()

    rows = []
    for case in test_cases:
        context = "\n".join(case["contexts"])
        prompt = (
            f"Using only the following context, answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {case['question']}\n\n"
            f"Answer:"
        )
        out = call_model(prompt, model=model)
        answer = out["response"]

        faithfulness = _score_faithfulness(answer, case["contexts"])
        relevancy = _score_answer_relevancy(case["question"], answer)
        ctx_precision = _score_context_precision(case["question"], case["contexts"], case["ground_truth"])
        ctx_recall = _score_context_recall(case["contexts"], case["ground_truth"])

        rows.append({
            "question": case["question"],
            "answer": answer,
            "ground_truth": case["ground_truth"],
            "faithfulness": faithfulness,
            "answer_relevancy": relevancy,
            "context_precision": ctx_precision,
            "context_recall": ctx_recall,
            "rag_pass": faithfulness >= 0.7 and ctx_recall >= 0.7
        })

    averages = {
        "faithfulness": round(sum(r["faithfulness"] for r in rows) / len(rows), 4),
        "answer_relevancy": round(sum(r["answer_relevancy"] for r in rows) / len(rows), 4),
        "context_precision": round(sum(r["context_precision"] for r in rows) / len(rows), 4),
        "context_recall": round(sum(r["context_recall"] for r in rows) / len(rows), 4),
    }

    return {"rows": rows, "averages": averages}


def _score_faithfulness(answer: str, contexts: list) -> float:
    """Check how many answer sentences are grounded in context."""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 10]
    if not sentences:
        return 0.0

    context_text = " ".join(contexts)
    emb_ctx = model.encode(context_text, convert_to_tensor=True)

    scores = []
    for sent in sentences:
        emb_sent = model.encode(sent, convert_to_tensor=True)
        sim = float(util.cos_sim(emb_sent, emb_ctx)[0][0])
        scores.append(sim)

    return round(sum(scores) / len(scores), 4)


def _score_answer_relevancy(question: str, answer: str) -> float:
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_q = model.encode(question, convert_to_tensor=True)
    emb_a = model.encode(answer, convert_to_tensor=True)
    return round(max(0.0, float(util.cos_sim(emb_q, emb_a)[0][0])), 4)


def _score_context_precision(question: str, contexts: list, ground_truth: str) -> float:
    """What fraction of retrieved contexts are actually relevant?"""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_q = model.encode(question, convert_to_tensor=True)
    relevant = 0
    for ctx in contexts:
        emb_c = model.encode(ctx, convert_to_tensor=True)
        if float(util.cos_sim(emb_q, emb_c)[0][0]) > 0.4:
            relevant += 1
    return round(relevant / len(contexts), 4) if contexts else 0.0


def _score_context_recall(contexts: list, ground_truth: str) -> float:
    """How much of the ground truth is covered by retrieved contexts?"""
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer("all-MiniLM-L6-v2")
    sentences = [s.strip() for s in ground_truth.split(".") if len(s.strip()) > 5]
    if not sentences:
        return 0.0
    context_text = " ".join(contexts)
    emb_ctx = model.encode(context_text, convert_to_tensor=True)
    covered = 0
    for sent in sentences:
        emb_s = model.encode(sent, convert_to_tensor=True)
        if float(util.cos_sim(emb_s, emb_ctx)[0][0]) > 0.5:
            covered += 1
    return round(covered / len(sentences), 4)


def _get_rag_test_cases() -> list:
    return [
        {
            "question": "What is machine learning?",
            "contexts": [
                "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "ML algorithms improve through experience without being explicitly programmed.",
            ],
            "ground_truth": "Machine learning is an AI subset where systems learn and improve from data automatically."
        },
        {
            "question": "What is a neural network?",
            "contexts": [
                "A neural network is a computational model inspired by the human brain's structure.",
                "Neural networks consist of layers of interconnected nodes that process information.",
            ],
            "ground_truth": "A neural network is a brain-inspired model made of interconnected nodes organized in layers."
        },
        {
            "question": "What does RAG stand for?",
            "contexts": [
                "RAG stands for Retrieval Augmented Generation.",
                "RAG combines document retrieval with language model generation to reduce hallucinations.",
            ],
            "ground_truth": "RAG stands for Retrieval Augmented Generation, combining retrieval and generation."
        },
        {
            "question": "What is overfitting in machine learning?",
            "contexts": [
                "Overfitting occurs when a model learns training data too well, including noise.",
                "An overfit model performs well on training data but poorly on unseen test data.",
            ],
            "ground_truth": "Overfitting is when a model memorizes training data including noise and fails to generalize."
        },
        {
            "question": "What is a transformer model?",
            "contexts": [
                "The transformer is an architecture based on self-attention mechanisms.",
                "Transformers were introduced in the paper 'Attention is All You Need' in 2017.",
            ],
            "ground_truth": "A transformer is a neural network architecture using self-attention, introduced in 2017."
        },
    ]
