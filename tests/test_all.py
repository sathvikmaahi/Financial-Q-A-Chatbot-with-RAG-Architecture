"""
Test Suite for Financial RAG Chatbot
=====================================
Unit and integration tests for all major components.

Run with: pytest tests/ -v
"""

import json
import os
import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================
# Test: SEC EDGAR Scraper
# ============================================================

class TestSECEdgarScraper:
    """Tests for the SEC EDGAR scraping module."""
    
    def test_scraper_initialization(self):
        """Test scraper initializes with correct defaults."""
        from src.scraper.sec_edgar_scraper import SECEdgarScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = SECEdgarScraper(
                output_dir=tmp_dir,
                email="test@example.com"
            )
            assert scraper.headers["User-Agent"].endswith("test@example.com")
            assert scraper.rate_limit > 0
    
    def test_html_parsing(self):
        """Test HTML to text conversion."""
        from src.scraper.sec_edgar_scraper import SECEdgarScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = SECEdgarScraper(output_dir=tmp_dir)
            
            html = """
            <html><body>
                <h1>Item 1. Business</h1>
                <p>Apple Inc. designs, manufactures, and markets smartphones.</p>
                <script>alert('test')</script>
            </body></html>
            """
            text = scraper.parse_filing_html(html)
            assert "Apple Inc." in text
            assert "alert" not in text
            assert "Business" in text
    
    def test_section_extraction(self):
        """Test 10-K section extraction."""
        from src.scraper.sec_edgar_scraper import SECEdgarScraper
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            scraper = SECEdgarScraper(output_dir=tmp_dir)
            
            text = """
            Item 1. Business
            
            We are a technology company that designs and sells consumer electronics.
            Our products include smartphones, tablets, and personal computers.
            We also provide software and services to our customers worldwide.
            Revenue from product sales represents the majority of our total revenue.
            
            Item 1A. Risk Factors
            
            Our business faces significant risks including supply chain disruptions,
            regulatory changes, and intense competition in the technology sector.
            Currency fluctuations may impact our international operations.
            We depend on key suppliers for critical components.
            
            Item 2. Properties
            """
            
            sections = scraper.extract_sections(text)
            assert len(sections) >= 1
            
            section_names = [s.section_name for s in sections]
            assert "Business" in section_names or "Risk Factors" in section_names


# ============================================================
# Test: Chunking Engine
# ============================================================

class TestChunkingEngine:
    """Tests for the document chunking module."""
    
    def test_recursive_chunking(self):
        """Test recursive character splitting."""
        from src.pipeline.chunking_engine import ChunkingEngine
        
        engine = ChunkingEngine(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=50,
            strategy="recursive"
        )
        
        # Create a sample filing
        filing_data = {
            "metadata": {
                "ticker": "TEST",
                "company_name": "Test Corp",
                "filing_date": "2024-01-01"
            },
            "sections": [],
            "raw_text": "This is a test document. " * 100  # ~600 words
        }
        
        chunks = engine.chunk_filing(filing_data)
        assert len(chunks) > 1
        
        for chunk in chunks:
            assert chunk.word_count > 0
            assert chunk.metadata["ticker"] == "TEST"
    
    def test_section_aware_chunking(self):
        """Test section-aware chunking respects boundaries."""
        from src.pipeline.chunking_engine import ChunkingEngine
        
        engine = ChunkingEngine(chunk_size=300, strategy="section_aware")
        
        filing_data = {
            "metadata": {"ticker": "AAPL", "filing_date": "2024-01-01"},
            "sections": [
                {
                    "section_name": "Business",
                    "section_number": "1",
                    "content": "Apple designs and sells consumer electronics. " * 50,
                    "word_count": 350
                },
                {
                    "section_name": "Risk Factors",
                    "section_number": "1A",
                    "content": "The company faces significant competition. " * 50,
                    "word_count": 300
                }
            ],
            "raw_text": ""
        }
        
        chunks = engine.chunk_filing(filing_data)
        assert len(chunks) >= 2
        
        # Verify chunks preserve section info
        section_names = set(c.metadata.get("section_name") for c in chunks)
        assert "Business" in section_names
        assert "Risk Factors" in section_names
    
    def test_chunk_id_deterministic(self):
        """Test that chunk IDs are deterministic."""
        from src.pipeline.chunking_engine import ChunkingEngine
        
        id1 = ChunkingEngine._generate_chunk_id("test content", {"ticker": "AAPL"})
        id2 = ChunkingEngine._generate_chunk_id("test content", {"ticker": "AAPL"})
        id3 = ChunkingEngine._generate_chunk_id("different", {"ticker": "AAPL"})
        
        assert id1 == id2
        assert id1 != id3
    
    def test_minimum_chunk_size(self):
        """Test that chunks below minimum size are filtered."""
        from src.pipeline.chunking_engine import ChunkingEngine
        
        engine = ChunkingEngine(chunk_size=200, min_chunk_size=50, strategy="recursive")
        
        filing_data = {
            "metadata": {"ticker": "TEST", "filing_date": "2024-01-01"},
            "sections": [],
            "raw_text": "Short text. " * 5  # Very short
        }
        
        chunks = engine.chunk_filing(filing_data)
        for chunk in chunks:
            assert chunk.char_count >= 50 or len(chunks) <= 1


# ============================================================
# Test: Query Processor
# ============================================================

class TestQueryProcessor:
    """Tests for query processing and expansion."""
    
    def test_entity_extraction(self):
        """Test extraction of tickers, years, and metrics."""
        from src.retrieval.hybrid_retriever import QueryProcessor
        
        processor = QueryProcessor()
        result = processor.process_query(
            "What was AAPL revenue in 2023?"
        )
        
        entities = result["detected_entities"]
        assert "AAPL" in entities["tickers"]
        assert "2023" in entities["years"]
        assert "revenue" in entities["metrics"]
    
    def test_query_expansion(self):
        """Test financial synonym expansion."""
        from src.retrieval.hybrid_retriever import QueryProcessor
        
        processor = QueryProcessor(enable_expansion=True)
        result = processor.process_query("What is the company revenue?")
        
        # Should expand "revenue" with synonyms
        expanded = result["expanded_query"]
        assert len(expanded) > len("What is the company revenue?")
    
    def test_complex_query_detection(self):
        """Test detection of comparison queries."""
        from src.retrieval.hybrid_retriever import QueryProcessor
        
        processor = QueryProcessor()
        
        result = processor.process_query(
            "Compare AAPL vs MSFT revenue growth"
        )
        assert len(result["sub_queries"]) > 1
    
    def test_simple_query_no_decomposition(self):
        """Test that simple queries are not decomposed."""
        from src.retrieval.hybrid_retriever import QueryProcessor
        
        processor = QueryProcessor()
        result = processor.process_query("What is Apple's revenue?")
        assert len(result["sub_queries"]) == 1


# ============================================================
# Test: Vector Store
# ============================================================

class TestVectorStore:
    """Tests for FAISS vector store operations."""
    
    def test_build_and_search(self):
        """Test building index and searching."""
        import numpy as np
        from src.pipeline.embedding_generator import VectorStoreBuilder
        
        dim = 384
        n_vectors = 100
        
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"This is test chunk number {i}",
                "metadata": {"ticker": "AAPL" if i % 2 == 0 else "MSFT"}
            }
            for i in range(n_vectors)
        ]
        
        store = VectorStoreBuilder(embedding_dim=dim)
        store.build_index(embeddings, chunks)
        
        assert store.index.ntotal == n_vectors
        
        # Search
        query = np.random.randn(dim).astype(np.float32)
        results = store.search(query, top_k=5)
        assert len(results) == 5
        assert all("content" in r for r in results)
    
    def test_metadata_filtering(self):
        """Test search with metadata filters."""
        import numpy as np
        from src.pipeline.embedding_generator import VectorStoreBuilder
        
        dim = 384
        n_vectors = 50
        
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
        chunks = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"Chunk {i}",
                "metadata": {"ticker": "AAPL" if i < 25 else "MSFT"}
            }
            for i in range(n_vectors)
        ]
        
        store = VectorStoreBuilder(embedding_dim=dim)
        store.build_index(embeddings, chunks)
        
        query = np.random.randn(dim).astype(np.float32)
        results = store.search(query, top_k=10, filter_metadata={"ticker": "AAPL"})
        
        for r in results:
            assert r["metadata"]["ticker"] == "AAPL"
    
    def test_save_and_load(self):
        """Test saving and loading vector store."""
        import numpy as np
        from src.pipeline.embedding_generator import VectorStoreBuilder
        
        dim = 64
        n_vectors = 20
        
        embeddings = np.random.randn(n_vectors, dim).astype(np.float32)
        chunks = [
            {"chunk_id": f"chunk_{i}", "content": f"Content {i}", "metadata": {}}
            for i in range(n_vectors)
        ]
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            store = VectorStoreBuilder(embedding_dim=dim)
            store.build_index(embeddings, chunks)
            store.save(tmp_dir)
            
            loaded_store = VectorStoreBuilder.load(tmp_dir)
            assert loaded_store.index.ntotal == n_vectors
            assert len(loaded_store.chunk_texts) == n_vectors


# ============================================================
# Test: RAG Chain
# ============================================================

class TestRAGChain:
    """Tests for the RAG orchestration chain."""
    
    def test_template_answer_generation(self):
        """Test fallback template-based answer generation."""
        from src.app.rag_chain import RAGChain
        
        chain = RAGChain(retriever=None, llm_provider="none")
        
        answer = chain._template_answer(
            "What is Apple's revenue?",
            "[AAPL, 2023, MD&A] Apple reported total net revenue of $383 billion in fiscal 2023."
        )
        assert len(answer) > 0
    
    def test_confidence_calculation(self):
        """Test confidence scoring."""
        from src.app.rag_chain import RAGChain
        from src.retrieval.hybrid_retriever import RetrievalResult
        
        chain = RAGChain(retriever=None)
        
        # High confidence: multiple relevant results
        results = [
            RetrievalResult(
                content="Test", metadata={"ticker": "AAPL"},
                fused_score=0.05
            )
            for _ in range(5)
        ]
        
        confidence = chain._calculate_confidence(results, "A detailed answer with data.")
        assert 0 < confidence <= 1
        
        # Low confidence: no results
        confidence = chain._calculate_confidence([], "")
        assert confidence == 0.0


# ============================================================
# Test: Evaluation Framework
# ============================================================

class TestEvaluator:
    """Tests for the RAG evaluation framework."""
    
    def test_mrr_calculation(self):
        """Test Mean Reciprocal Rank calculation."""
        from src.evaluation.evaluate_rag import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # First result is relevant -> MRR = 1.0
        assert evaluator._calculate_mrr([True, False, False]) == 1.0
        
        # Second result is relevant -> MRR = 0.5
        assert evaluator._calculate_mrr([False, True, False]) == 0.5
        
        # No relevant results -> MRR = 0
        assert evaluator._calculate_mrr([False, False, False]) == 0.0
    
    def test_ndcg_calculation(self):
        """Test NDCG@K calculation."""
        from src.evaluation.evaluate_rag import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Perfect ranking
        ndcg = evaluator._calculate_ndcg([True, True, True], k=3)
        assert ndcg == 1.0
        
        # No relevant results
        ndcg = evaluator._calculate_ndcg([False, False, False], k=3)
        assert ndcg == 0.0
    
    def test_eval_question_loading(self):
        """Test loading built-in evaluation questions."""
        from src.evaluation.evaluate_rag import load_eval_questions
        
        questions = load_eval_questions()
        assert len(questions) > 0
        assert questions[0].question != ""
        assert len(questions[0].relevant_tickers) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
