"""
LLM Evaluation & Red-Teaming Framework
=======================================
100% free stack: Ollama (local LLM) + Ragas + Detoxify + MLflow
Run: python main.py
"""

import argparse
from runners.batch_eval import run_full_evaluation
from reporting.report_gen import generate_html_report


def main():
    parser = argparse.ArgumentParser(description="LLM Eval & Red-Teaming Framework")
    parser.add_argument("--model", default="llama3.2", help="Ollama model name")
    parser.add_argument("--suite", default="all",
                        choices=["all", "hallucination", "toxicity", "redteam", "rag"],
                        help="Which eval suite to run")
    parser.add_argument("--output", default="eval_report.html", help="Output report filename")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  LLM Eval & Red-Teaming Framework")
    print(f"  Model : {args.model}")
    print(f"  Suite : {args.suite}")
    print(f"{'='*60}\n")

    results = run_full_evaluation(model=args.model, suite=args.suite)
    generate_html_report(results, output_path=f"reports/{args.output}")
    print(f"\nReport saved to reports/{args.output}")


if __name__ == "__main__":
    main()
