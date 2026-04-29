"""
metrics/relevancy.py
Scores how relevant a model response is to the prompt.
Uses sentence-transformers cosine similarity (free, local).
"""

from sentence_transformers import SentenceTransformer, util

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def score_relevancy(question: str, answer: str) -> dict:
    """
    Score how relevant the answer is to the question.
    Returns answer_relevancy score (0-1).
    """
    if not answer or answer.startswith("[ERROR]") or len(answer.strip()) < 3:
        return {"answer_relevancy": 0.0, "relevancy_pass": False}

    model = _get_model()
    emb_q = model.encode(question, convert_to_tensor=True)
    emb_a = model.encode(answer, convert_to_tensor=True)
    relevancy = float(util.cos_sim(emb_q, emb_a)[0][0])
    relevancy = round(max(0.0, relevancy), 4)

    return {
        "answer_relevancy": relevancy,
        "relevancy_pass": relevancy >= 0.4
    }
