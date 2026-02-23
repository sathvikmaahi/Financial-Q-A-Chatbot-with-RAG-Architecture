"""
Hybrid Retrieval Engine
========================
Combines BM25 sparse retrieval with FAISS dense retrieval using
Reciprocal Rank Fusion (RRF) for optimal document retrieval.

Features:
- BM25 keyword-based retrieval for exact term matching
- FAISS dense vector retrieval for semantic similarity
- Reciprocal Rank Fusion (RRF) for combining ranked lists
- Optional cross-encoder reranking for precision
- Metadata filtering (by company, section, date range)
"""

import re
import pickle
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class RetrievalResult:
    """A single retrieval result with content, metadata, and scores."""
    content: str
    metadata: dict
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fused_score: float = 0.0
    rerank_score: float = 0.0
    rank: int = 0


class HybridRetriever:
    """
    Hybrid retriever combining BM25 sparse and FAISS dense retrieval
    with Reciprocal Rank Fusion (RRF).
    
    Architecture:
        Query → [BM25 Retriever] → Sparse Results ─┐
                                                     ├→ RRF Fusion → [Reranker] → Final Results
        Query → [FAISS Retriever] → Dense Results ──┘
    """
    
    def __init__(
        self,
        vector_store_dir: str,
        bm25_path: str,
        chunks_path: str,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        bm25_weight: float = 0.3,
        dense_weight: float = 0.7,
        rrf_k: int = 60,
        top_k: int = 10,
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.rrf_k = rrf_k
        self.top_k = top_k
        self.use_reranker = use_reranker
        self.reranker_model_name = reranker_model
        
        # Load components
        self._load_vector_store(vector_store_dir)
        self._load_bm25(bm25_path)
        self._load_chunks(chunks_path)
        self._load_embedder(embedding_model)
        
        if use_reranker:
            self._load_reranker()
        
        logger.info(
            f"HybridRetriever initialized: "
            f"BM25 weight={bm25_weight}, Dense weight={dense_weight}, "
            f"RRF k={rrf_k}, Top-K={top_k}"
        )
    
    def _load_vector_store(self, vector_store_dir: str):
        """Load FAISS vector store."""
        from src.pipeline.embedding_generator import VectorStoreBuilder
        self.vector_store = VectorStoreBuilder.load(vector_store_dir)
        logger.info(f"Loaded FAISS index: {self.vector_store.index.ntotal} vectors")
    
    def _load_bm25(self, bm25_path: str):
        """Load BM25 index."""
        try:
            with open(bm25_path, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.bm25_corpus = data["corpus"]
            self.bm25_chunk_ids = data.get("chunk_ids", [])
            logger.info(f"Loaded BM25 index: {len(self.bm25_corpus)} documents")
        except (FileNotFoundError, ImportError):
            logger.warning("BM25 index not found. Sparse retrieval disabled.")
            self.bm25 = None
            self.bm25_corpus = []
            self.bm25_chunk_ids = []
    
    def _load_chunks(self, chunks_path: str):
        """Load chunk data for text retrieval."""
        import json
        self.chunks = []
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                self.chunks.append(json.loads(line.strip()))
        logger.info(f"Loaded {len(self.chunks)} chunks")
    
    def _load_embedder(self, model_name: str):
        """Load embedding model for query encoding."""
        from src.pipeline.embedding_generator import EmbeddingGenerator
        self.embedder = EmbeddingGenerator(model_name=model_name)
    
    def _load_reranker(self):
        """Load cross-encoder reranker model."""
        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.reranker_model_name)
            logger.info(f"Loaded reranker: {self.reranker_model_name}")
        except ImportError:
            logger.warning("CrossEncoder not available. Reranking disabled.")
            self.reranker = None
            self.use_reranker = False
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[dict] = None
    ) -> list[RetrievalResult]:
        """
        Perform hybrid retrieval with RRF fusion.
        
        Args:
            query: User's question
            top_k: Number of results to return (overrides default)
            filter_metadata: Optional filters (e.g., {"ticker": "AAPL"})
            
        Returns:
            List of RetrievalResult objects, ranked by fused score
        """
        k = top_k or self.top_k
        
        # Retrieve from both sources (get more candidates for fusion)
        candidate_k = k * 3
        
        # 1. Dense retrieval via FAISS
        dense_results = self._dense_retrieve(query, candidate_k, filter_metadata)
        
        # 2. Sparse retrieval via BM25
        sparse_results = self._sparse_retrieve(query, candidate_k, filter_metadata)
        
        # 3. Reciprocal Rank Fusion
        fused_results = self._reciprocal_rank_fusion(
            dense_results, sparse_results, k=k * 2
        )
        
        # 4. Optional reranking
        if self.use_reranker and self.reranker:
            fused_results = self._rerank(query, fused_results, k)
        else:
            fused_results = fused_results[:k]
        
        # Assign final ranks
        for i, result in enumerate(fused_results):
            result.rank = i + 1
        
        return fused_results
    
    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[dict] = None
    ) -> list[RetrievalResult]:
        """Retrieve using FAISS dense embeddings."""
        query_embedding = self.embedder.embed_query(query)
        
        raw_results = self.vector_store.search(
            query_embedding, top_k=top_k, filter_metadata=filter_metadata
        )
        
        results = []
        for r in raw_results:
            results.append(RetrievalResult(
                content=r["content"],
                metadata=r["metadata"],
                dense_score=r["score"]
            ))
        
        return results
    
    def _sparse_retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: Optional[dict] = None
    ) -> list[RetrievalResult]:
        """Retrieve using BM25 sparse matching."""
        if self.bm25 is None:
            return []
        
        # Tokenize query
        query_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', query.lower())
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        
        results = []
        for idx in top_indices:
            if idx >= len(self.chunks):
                continue
            
            chunk = self.chunks[idx]
            metadata = chunk.get("metadata", {})
            
            # Apply metadata filter
            if filter_metadata:
                match = all(
                    metadata.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            results.append(RetrievalResult(
                content=chunk.get("content", ""),
                metadata=metadata,
                sparse_score=float(scores[idx])
            ))
            
            if len(results) >= top_k:
                break
        
        return results
    
    def _reciprocal_rank_fusion(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        k: int
    ) -> list[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = sum over lists of: 1 / (rrf_k + rank_in_list)
        
        This approach is robust to score scale differences between
        BM25 and dense retrieval.
        """
        # Build a map: content_hash -> RetrievalResult
        result_map = {}
        
        # Process dense results
        for rank, result in enumerate(dense_results):
            content_key = hash(result.content[:200])
            if content_key not in result_map:
                result_map[content_key] = result
            result_map[content_key].dense_score = result.dense_score
            rrf_score = self.dense_weight / (self.rrf_k + rank + 1)
            result_map[content_key].fused_score += rrf_score
        
        # Process sparse results
        for rank, result in enumerate(sparse_results):
            content_key = hash(result.content[:200])
            if content_key not in result_map:
                result_map[content_key] = result
            result_map[content_key].sparse_score = result.sparse_score
            rrf_score = self.bm25_weight / (self.rrf_k + rank + 1)
            result_map[content_key].fused_score += rrf_score
        
        # Sort by fused score
        fused_results = sorted(
            result_map.values(),
            key=lambda x: x.fused_score,
            reverse=True
        )
        
        return fused_results[:k]
    
    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int
    ) -> list[RetrievalResult]:
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:top_k]
        
        # Create query-document pairs
        pairs = [(query, r.content) for r in results]
        
        # Score pairs
        scores = self.reranker.predict(pairs)
        
        # Assign rerank scores
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
        
        # Sort by rerank score
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        
        return results[:top_k]


class QueryProcessor:
    """
    Process and enhance user queries for better retrieval.
    
    Features:
    - Query expansion with financial synonyms
    - Sub-query decomposition for complex questions
    - Hypothetical Document Embedding (HyDE)
    """
    
    # Financial term synonyms for query expansion
    FINANCIAL_SYNONYMS = {
        "revenue": ["sales", "net revenue", "total revenue", "top line"],
        "profit": ["net income", "earnings", "bottom line", "net profit"],
        "loss": ["net loss", "deficit", "negative earnings"],
        "expenses": ["costs", "expenditures", "spending", "operating expenses"],
        "debt": ["liabilities", "borrowings", "obligations", "indebtedness"],
        "assets": ["holdings", "resources", "property", "total assets"],
        "growth": ["increase", "expansion", "improvement", "year-over-year"],
        "risk": ["risk factors", "uncertainties", "threats", "headwinds"],
        "r&d": ["research and development", "research & development", "R&D expenses"],
        "capex": ["capital expenditure", "capital spending", "capital investment"],
        "dividend": ["dividend payment", "shareholder distribution", "payout"],
        "margin": ["profit margin", "gross margin", "operating margin"],
        "guidance": ["outlook", "forecast", "projection", "forward-looking"],
        "acquisition": ["merger", "purchase", "takeover", "M&A"],
        "segment": ["business segment", "operating segment", "division"],
    }
    
    def __init__(self, enable_expansion: bool = True):
        self.enable_expansion = enable_expansion
    
    def process_query(self, query: str) -> dict:
        """
        Process a user query into retrieval-ready format.
        
        Returns:
            dict with:
            - original_query: Original user input
            - expanded_query: Query with synonym expansion
            - sub_queries: Decomposed sub-queries (if complex)
            - detected_entities: Extracted companies, dates, metrics
        """
        result = {
            "original_query": query,
            "expanded_query": query,
            "sub_queries": [query],
            "detected_entities": self._extract_entities(query)
        }
        
        if self.enable_expansion:
            result["expanded_query"] = self._expand_query(query)
        
        # Decompose complex queries
        if self._is_complex_query(query):
            result["sub_queries"] = self._decompose_query(query)
        
        return result
    
    def _expand_query(self, query: str) -> str:
        """Expand query with financial synonyms."""
        expanded_terms = []
        query_lower = query.lower()
        
        for term, synonyms in self.FINANCIAL_SYNONYMS.items():
            if term in query_lower:
                expanded_terms.extend(synonyms[:2])
        
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms)}"
        return query
    
    def _extract_entities(self, query: str) -> dict:
        """Extract companies, dates, and financial metrics from query."""
        entities = {
            "tickers": [],
            "years": [],
            "metrics": []
        }
        
        # Extract potential tickers (uppercase 1-5 letter words)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', query)
        common_words = {"THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
                       "HER", "WAS", "ONE", "OUR", "OUT", "HAS", "HIS", "HOW",
                       "DID", "GET", "HAS", "HIM", "HAD", "CAN", "ITS", "MAY",
                       "NEW", "NOW", "OLD", "SEE", "WAY", "WHO", "BOY", "DID",
                       "WHAT", "WHEN", "WHERE", "WHICH", "WITH", "FROM", "THAT",
                       "THAN", "WILL", "BEEN", "HAVE", "MANY", "SOME", "THEM",
                       "MOST", "EACH", "MAKE", "LIKE", "OVER", "SUCH", "INTO"}
        entities["tickers"] = [t for t in tickers if t not in common_words]
        
        # Extract years
        years = re.findall(r'\b(20[12][0-9])\b', query)
        entities["years"] = years
        
        # Extract financial metrics
        metric_patterns = [
            "revenue", "profit", "income", "loss", "expenses", "debt",
            "assets", "margin", "growth", "dividend", "eps", "ebitda",
            "cash flow", "capex", "r&d"
        ]
        for metric in metric_patterns:
            if metric in query.lower():
                entities["metrics"].append(metric)
        
        return entities
    
    def _is_complex_query(self, query: str) -> bool:
        """Detect if a query requires decomposition."""
        complex_indicators = [
            r'\b(compare|comparison|versus|vs\.?|differ)\b',
            r'\b(and also|additionally|furthermore|as well as)\b',
            r'\b(between .+ and .+)\b',
            r'\b(how .+ and .+ (relate|compare|differ))\b'
        ]
        
        for pattern in complex_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _decompose_query(self, query: str) -> list[str]:
        """Decompose a complex query into sub-queries."""
        sub_queries = [query]  # Always include original
        
        # Handle "compare X vs Y" patterns
        compare_match = re.search(
            r'compare\s+(.+?)\s+(?:vs\.?|versus|and|with)\s+(.+?)(?:\s+in|\s+for|\?|$)',
            query, re.IGNORECASE
        )
        
        if compare_match:
            entity1 = compare_match.group(1).strip()
            entity2 = compare_match.group(2).strip()
            
            # Extract the metric being compared
            metric_match = re.search(
                r'(revenue|profit|income|expenses|growth|debt|margin|r&d|spending)',
                query, re.IGNORECASE
            )
            metric = metric_match.group(1) if metric_match else ""
            
            sub_queries.extend([
                f"{entity1} {metric}".strip(),
                f"{entity2} {metric}".strip()
            ])
        
        return sub_queries
