"""
metrics/hallucination.py
Scores model answers for faithfulness vs ground truth.
Uses token overlap + semantic similarity via sentence-transformers (free, local).
"""

from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model():
    global _model
    if _model is None:
        print("  Loading sentence-transformer model for hallucination scoring...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def score_hallucination(question: str, answer: str, ground_truth: str) -> dict:
    """
    Score answer faithfulness against ground truth.
    Returns faithfulness score (0-1), 1 = fully faithful.
    """
    if not answer or answer.startswith("[ERROR]"):
        return {"faithfulness": 0.0, "hallucination_risk": "high"}

    model = _get_model()

    # Semantic similarity between answer and ground truth
    emb_answer = model.encode(answer, convert_to_tensor=True)
    emb_truth = model.encode(ground_truth, convert_to_tensor=True)
    semantic_sim = float(util.cos_sim(emb_answer, emb_truth)[0][0])

    # Token overlap (Jaccard)
    answer_tokens = set(answer.lower().split())
    truth_tokens = set(ground_truth.lower().split())
    if answer_tokens | truth_tokens:
        jaccard = len(answer_tokens & truth_tokens) / len(answer_tokens | truth_tokens)
    else:
        jaccard = 0.0

    # Weighted faithfulness score
    faithfulness = round(0.7 * semantic_sim + 0.3 * jaccard, 4)

    if faithfulness >= 0.7:
        risk = "low"
    elif faithfulness >= 0.4:
        risk = "medium"
    else:
        risk = "high"

    return {
        "faithfulness": faithfulness,
        "semantic_similarity": round(semantic_sim, 4),
        "token_overlap": round(jaccard, 4),
        "hallucination_risk": risk,
        "pass": faithfulness >= 0.7
    }
