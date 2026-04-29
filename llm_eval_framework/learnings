What I learned from running the LLM Eval Framework on Llama 3.2:
Hallucination (Avg 71% faithfulness — borderline)
The model answered simple factual questions correctly (Paris = 100%) but consistently added extra context beyond the ground truth — like listing all 8 planets when the question just asked "how many." This taught me that verbosity inflates hallucination risk scores even when the core answer is right. A better hallucination metric needs to reward correct answers even in longer responses.
Toxicity (Avg 0.5% — excellent)
All 5 cases passed cleanly. The model handled sensitive prompts like workplace conflict and societal problems without producing toxic language. This confirmed Llama 3.2 is production-safe for general-purpose use cases.
Red-Team Refusal Rate (Only 16.7% — this is the most important finding)
This was the biggest red flag. Only 6 out of 36 adversarial prompts triggered a refusal. Specifically the "educational purposes" and "fictional story" jailbreak frames got partial responses — meaning the model partially complied with harmful prompts when wrapped in creative framing. This is a real safety gap that companies care about deeply.
RAG Eval (76.1% faithfulness, 100% context recall — strong)
Context recall and precision were perfect — the retriever pulled the right chunks every time. However answer relevancy was only 58.8%, mainly because the model sometimes gave one-line answers like "Machine learning." or "Retrieval Augmented Generation." instead of full explanations. This taught me the RAG bottleneck here is generation quality, not retrieval.
