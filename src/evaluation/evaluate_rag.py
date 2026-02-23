"""
RAG Evaluation Framework
=========================
Evaluates the RAG pipeline on retrieval quality and answer accuracy.

Metrics:
- Precision@K: Fraction of top-K retrieved docs that are relevant
- Recall@K: Fraction of relevant docs found in top-K
- Mean Reciprocal Rank (MRR): Average of 1/rank of first relevant result
- NDCG@K: Normalized Discounted Cumulative Gain
- Answer Faithfulness: Whether the answer is grounded in retrieved context
- Latency: End-to-end response time

Usage:
    python -m src.evaluation.evaluate_rag --test-set data/eval/test_questions.json
"""

import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

from loguru import logger


@dataclass
class EvalQuestion:
    """A single evaluation question with ground truth."""
    question: str
    expected_answer: str
    relevant_tickers: list[str]
    relevant_sections: list[str]
    relevant_keywords: list[str]
    difficulty: str = "medium"  # easy, medium, hard


@dataclass
class EvalResult:
    """Evaluation result for a single question."""
    question: str
    precision_at_k: dict  # {k: score}
    recall_at_k: dict
    mrr: float
    ndcg_at_k: dict
    faithfulness: float
    latency: float
    num_sources: int
    answer_length: int


class RAGEvaluator:
    """
    Comprehensive RAG evaluation framework.
    """
    
    def __init__(self, rag_chain=None, k_values: list[int] = None):
        self.rag_chain = rag_chain
        self.k_values = k_values or [1, 3, 5, 10]
    
    def evaluate(
        self,
        test_questions: list[EvalQuestion],
        verbose: bool = True
    ) -> dict:
        """
        Run full evaluation over a set of test questions.
        
        Returns:
            Dictionary with aggregated metrics
        """
        results = []
        
        for i, q in enumerate(test_questions):
            if verbose:
                logger.info(f"Evaluating [{i+1}/{len(test_questions)}]: {q.question[:60]}...")
            
            result = self._evaluate_single(q)
            results.append(result)
        
        # Aggregate metrics
        aggregated = self._aggregate_results(results)
        
        if verbose:
            self._print_report(aggregated, results)
        
        return aggregated
    
    def _evaluate_single(self, question: EvalQuestion) -> EvalResult:
        """Evaluate a single question."""
        start_time = time.time()
        
        if self.rag_chain:
            response = self.rag_chain.query(question.question, top_k=max(self.k_values))
            sources = response.sources
            answer = response.answer
        else:
            sources = []
            answer = ""
        
        latency = time.time() - start_time
        
        # Check relevance of each retrieved document
        relevance_labels = []
        for source in sources:
            is_relevant = self._check_relevance(source, question)
            relevance_labels.append(is_relevant)
        
        # Calculate metrics
        precision_at_k = {}
        recall_at_k = {}
        ndcg_at_k = {}
        
        total_relevant = max(sum(relevance_labels), 1)
        
        for k in self.k_values:
            top_k_labels = relevance_labels[:k]
            
            # Precision@K
            if top_k_labels:
                precision_at_k[k] = sum(top_k_labels) / len(top_k_labels)
            else:
                precision_at_k[k] = 0.0
            
            # Recall@K
            recall_at_k[k] = sum(top_k_labels) / total_relevant if total_relevant > 0 else 0.0
            
            # NDCG@K
            ndcg_at_k[k] = self._calculate_ndcg(top_k_labels, k)
        
        # MRR
        mrr = self._calculate_mrr(relevance_labels)
        
        # Faithfulness (simplified: check if answer mentions relevant companies/terms)
        faithfulness = self._check_faithfulness(answer, question, sources)
        
        return EvalResult(
            question=question.question,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            faithfulness=faithfulness,
            latency=latency,
            num_sources=len(sources),
            answer_length=len(answer.split())
        )
    
    def _check_relevance(self, source: dict, question: EvalQuestion) -> bool:
        """
        Check if a retrieved source is relevant to the question.
        
        A source is relevant if:
        1. It's from a relevant company (ticker match), OR
        2. It's from a relevant section, AND
        3. It contains at least one relevant keyword
        """
        ticker = source.get("ticker", "")
        section = source.get("section_name", "")
        content = source.get("content_preview", "").lower()
        
        # Check ticker relevance
        ticker_relevant = (
            not question.relevant_tickers or
            ticker in question.relevant_tickers
        )
        
        # Check section relevance
        section_relevant = (
            not question.relevant_sections or
            any(s.lower() in section.lower() for s in question.relevant_sections)
        )
        
        # Check keyword relevance
        keyword_relevant = (
            not question.relevant_keywords or
            any(kw.lower() in content for kw in question.relevant_keywords)
        )
        
        return ticker_relevant and (section_relevant or keyword_relevant)
    
    def _calculate_mrr(self, relevance_labels: list[bool]) -> float:
        """Calculate Mean Reciprocal Rank."""
        for i, is_relevant in enumerate(relevance_labels):
            if is_relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    def _calculate_ndcg(self, relevance_labels: list[bool], k: int) -> float:
        """Calculate NDCG@K."""
        dcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(relevance_labels[:k])
        )
        
        # Ideal DCG (all relevant documents at the top)
        ideal_labels = sorted(relevance_labels[:k], reverse=True)
        idcg = sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(ideal_labels)
        )
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def _check_faithfulness(
        self,
        answer: str,
        question: EvalQuestion,
        sources: list[dict]
    ) -> float:
        """
        Simple faithfulness check: does the answer reference
        content that exists in the retrieved sources?
        
        Returns a score between 0 and 1.
        """
        if not answer or not sources:
            return 0.0
        
        answer_lower = answer.lower()
        
        # Check if answer mentions expected companies
        company_mentioned = any(
            t.lower() in answer_lower for t in question.relevant_tickers
        ) if question.relevant_tickers else True
        
        # Check if answer contains keywords from sources
        source_texts = " ".join(s.get("content_preview", "") for s in sources).lower()
        
        # Extract key numbers/facts from answer
        answer_facts = set()
        import re
        numbers = re.findall(r'\$[\d,.]+\s*(?:billion|million|trillion|B|M|T)?', answer)
        answer_facts.update(numbers)
        
        # Check if those facts appear in sources
        facts_grounded = sum(
            1 for fact in answer_facts if fact.lower() in source_texts
        )
        
        fact_score = facts_grounded / len(answer_facts) if answer_facts else 0.5
        
        # Combine scores
        faithfulness = 0.5 * float(company_mentioned) + 0.5 * fact_score
        return min(faithfulness, 1.0)
    
    def _aggregate_results(self, results: list[EvalResult]) -> dict:
        """Aggregate evaluation results across all questions."""
        n = len(results)
        if n == 0:
            return {}
        
        aggregated = {
            "num_questions": n,
            "metrics": {},
            "latency": {
                "mean": np.mean([r.latency for r in results]),
                "p50": np.median([r.latency for r in results]),
                "p95": np.percentile([r.latency for r in results], 95),
                "p99": np.percentile([r.latency for r in results], 99),
            }
        }
        
        # Aggregate per-K metrics
        for k in self.k_values:
            aggregated["metrics"][f"precision@{k}"] = np.mean(
                [r.precision_at_k.get(k, 0) for r in results]
            )
            aggregated["metrics"][f"recall@{k}"] = np.mean(
                [r.recall_at_k.get(k, 0) for r in results]
            )
            aggregated["metrics"][f"ndcg@{k}"] = np.mean(
                [r.ndcg_at_k.get(k, 0) for r in results]
            )
        
        aggregated["metrics"]["mrr"] = np.mean([r.mrr for r in results])
        aggregated["metrics"]["faithfulness"] = np.mean([r.faithfulness for r in results])
        aggregated["metrics"]["avg_sources"] = np.mean([r.num_sources for r in results])
        aggregated["metrics"]["avg_answer_length"] = np.mean(
            [r.answer_length for r in results]
        )
        
        return aggregated
    
    def _print_report(self, aggregated: dict, results: list[EvalResult]):
        """Print a formatted evaluation report."""
        logger.info("\n" + "=" * 70)
        logger.info("RAG EVALUATION REPORT")
        logger.info("=" * 70)
        logger.info(f"Questions Evaluated: {aggregated['num_questions']}")
        logger.info("")
        
        logger.info("RETRIEVAL METRICS:")
        logger.info("-" * 40)
        for key, value in aggregated.get("metrics", {}).items():
            if key.startswith("precision") or key.startswith("recall") or key.startswith("ndcg"):
                logger.info(f"  {key:>15}: {value:.4f}")
        
        logger.info(f"\n  {'MRR':>15}: {aggregated['metrics'].get('mrr', 0):.4f}")
        logger.info(f"  {'Faithfulness':>15}: {aggregated['metrics'].get('faithfulness', 0):.4f}")
        
        logger.info("\nLATENCY:")
        logger.info("-" * 40)
        latency = aggregated.get("latency", {})
        logger.info(f"  {'Mean':>15}: {latency.get('mean', 0):.3f}s")
        logger.info(f"  {'P50':>15}: {latency.get('p50', 0):.3f}s")
        logger.info(f"  {'P95':>15}: {latency.get('p95', 0):.3f}s")
        
        logger.info("=" * 70)


# Pre-built evaluation dataset
EVAL_QUESTIONS = [
    {
        "question": "What was Apple's total net revenue in fiscal year 2023?",
        "expected_answer": "Apple's total net revenue was approximately $383.3 billion in FY2023.",
        "relevant_tickers": ["AAPL"],
        "relevant_sections": ["Financial Statements", "MD&A"],
        "relevant_keywords": ["revenue", "net sales", "total net sales"],
        "difficulty": "easy"
    },
    {
        "question": "What are the key risk factors mentioned by NVIDIA in their latest 10-K?",
        "expected_answer": "NVIDIA's key risks include supply chain concentration, export controls, and customer concentration.",
        "relevant_tickers": ["NVDA"],
        "relevant_sections": ["Risk Factors"],
        "relevant_keywords": ["risk", "uncertainty", "supply", "export", "regulation"],
        "difficulty": "easy"
    },
    {
        "question": "Compare Microsoft and Alphabet's research and development expenses.",
        "expected_answer": "Microsoft spent approximately $27.2B on R&D while Alphabet spent $39.5B.",
        "relevant_tickers": ["MSFT", "GOOGL"],
        "relevant_sections": ["MD&A", "Financial Statements"],
        "relevant_keywords": ["research", "development", "R&D", "expense"],
        "difficulty": "medium"
    },
    {
        "question": "What is JPMorgan Chase's total assets value?",
        "expected_answer": "JPMorgan Chase reported total assets of approximately $3.87 trillion.",
        "relevant_tickers": ["JPM"],
        "relevant_sections": ["Financial Statements"],
        "relevant_keywords": ["total assets", "assets", "balance sheet"],
        "difficulty": "easy"
    },
    {
        "question": "Describe Amazon's main business segments and their revenue contribution.",
        "expected_answer": "Amazon operates through North America, International, and AWS segments.",
        "relevant_tickers": ["AMZN"],
        "relevant_sections": ["Business", "MD&A"],
        "relevant_keywords": ["segment", "AWS", "North America", "International", "revenue"],
        "difficulty": "medium"
    },
    {
        "question": "What legal proceedings is Meta currently involved in?",
        "expected_answer": "Meta faces various legal proceedings including antitrust, privacy, and content moderation cases.",
        "relevant_tickers": ["META"],
        "relevant_sections": ["Risk Factors", "Business"],
        "relevant_keywords": ["legal", "litigation", "proceedings", "lawsuit", "regulatory"],
        "difficulty": "medium"
    },
    {
        "question": "How has Pfizer's revenue changed post-pandemic compared to 2020?",
        "expected_answer": "Pfizer's revenue surged due to COVID vaccine and treatment sales, then declined.",
        "relevant_tickers": ["PFE"],
        "relevant_sections": ["MD&A", "Financial Statements"],
        "relevant_keywords": ["revenue", "vaccine", "Comirnaty", "Paxlovid", "growth", "decline"],
        "difficulty": "hard"
    },
    {
        "question": "What are ExxonMobil's environmental and climate-related disclosures?",
        "expected_answer": "ExxonMobil discusses climate risk, emissions reduction targets, and energy transition.",
        "relevant_tickers": ["XOM"],
        "relevant_sections": ["Risk Factors", "Business"],
        "relevant_keywords": ["climate", "environmental", "emission", "sustainability", "carbon"],
        "difficulty": "hard"
    },
    {
        "question": "What is Berkshire Hathaway's insurance float and investment portfolio composition?",
        "expected_answer": "Berkshire's insurance float and investment holdings are substantial components of its business.",
        "relevant_tickers": ["BRK-B"],
        "relevant_sections": ["Business", "MD&A", "Financial Statements"],
        "relevant_keywords": ["insurance", "float", "investment", "equity", "portfolio"],
        "difficulty": "hard"
    },
    {
        "question": "Compare the operating margins of Walmart vs Costco.",
        "expected_answer": "Walmart and Costco have different operating margin profiles due to business model differences.",
        "relevant_tickers": ["WMT"],
        "relevant_sections": ["MD&A", "Financial Statements"],
        "relevant_keywords": ["operating", "margin", "income", "profit", "expenses"],
        "difficulty": "medium"
    }
]


def load_eval_questions(path: Optional[str] = None) -> list[EvalQuestion]:
    """Load evaluation questions from file or use built-in set."""
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        return [EvalQuestion(**q) for q in data]
    
    return [EvalQuestion(**q) for q in EVAL_QUESTIONS]


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    parser.add_argument("--test-set", type=str, default=None)
    parser.add_argument("--output", type=str, default="data/eval/results.json")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10])
    
    args = parser.parse_args()
    
    # Load test questions
    questions = load_eval_questions(args.test_set)
    logger.info(f"Loaded {len(questions)} evaluation questions")
    
    # Try to load RAG system
    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.app.rag_chain import RAGChain
        
        retriever = HybridRetriever(
            vector_store_dir="data/vectors",
            bm25_path="data/processed/bm25_corpus.pkl",
            chunks_path="data/processed/chunks.jsonl"
        )
        rag_chain = RAGChain(retriever=retriever)
    except Exception as e:
        logger.warning(f"Could not load RAG system: {e}")
        rag_chain = None
    
    # Run evaluation
    evaluator = RAGEvaluator(rag_chain=rag_chain, k_values=args.k_values)
    results = evaluator.evaluate(questions)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
