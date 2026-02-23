"""
End-to-End Ingestion Pipeline
==============================
Orchestrates the full data pipeline:
1. Load scraped filings from disk
2. Chunk documents using configurable strategies
3. Generate embeddings via HuggingFace
4. Build FAISS vector store
5. Create BM25 index for hybrid retrieval

Usage:
    python -m src.pipeline.ingestion_pipeline --input data/raw --output data/processed
"""

import json
import pickle
import argparse
from pathlib import Path

import yaml
from loguru import logger

from src.pipeline.chunking_engine import ChunkingEngine
from src.pipeline.embedding_generator import (
    EmbeddingGenerator,
    VectorStoreBuilder,
    build_vector_store_from_chunks
)


class IngestionPipeline:
    """
    Full ingestion pipeline from raw filings to searchable vector store.
    """
    
    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Initialize chunking engine
        chunking_config = self.config.get("chunking", {}).get("recursive", {})
        self.chunker = ChunkingEngine(
            chunk_size=chunking_config.get("chunk_size", 1000),
            chunk_overlap=chunking_config.get("chunk_overlap", 200),
            strategy="section_aware"
        )
        
        # Embedding config
        embedding_config = self.config.get("embedding", {})
        self.embedding_model = embedding_config.get(
            "model_name", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.batch_size = embedding_config.get("batch_size", 64)
    
    def run(
        self,
        input_dir: str = "data/raw",
        output_dir: str = "data/processed",
        vector_dir: str = "data/vectors"
    ) -> dict:
        """
        Run the full ingestion pipeline.
        
        Args:
            input_dir: Directory with scraped filing JSONs
            output_dir: Directory for processed chunks
            vector_dir: Directory for FAISS vector store
            
        Returns:
            Pipeline statistics dictionary
        """
        logger.info("=" * 60)
        logger.info("INGESTION PIPELINE STARTING")
        logger.info("=" * 60)
        
        # Step 1: Load and chunk filings
        logger.info("\n📄 Step 1: Chunking filings...")
        chunks_path = Path(output_dir) / "chunks.jsonl"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        all_chunks = self.chunker.process_all_filings(
            input_dir=input_dir,
            output_path=str(chunks_path)
        )
        
        logger.info(f"Generated {len(all_chunks)} chunks")
        
        # Step 2: Build BM25 index for sparse retrieval
        logger.info("\n🔍 Step 2: Building BM25 index...")
        bm25_path = Path(output_dir) / "bm25_corpus.pkl"
        self._build_bm25_index(all_chunks, str(bm25_path))
        
        # Step 3: Generate embeddings and build FAISS index
        logger.info("\n🧮 Step 3: Generating embeddings & building FAISS index...")
        store = build_vector_store_from_chunks(
            chunks_path=str(chunks_path),
            output_dir=vector_dir,
            model_name=self.embedding_model,
            batch_size=self.batch_size
        )
        
        # Summary
        stats = {
            "total_chunks": len(all_chunks),
            "vector_store_size": store.index.ntotal if store.index else 0,
            "embedding_model": self.embedding_model,
            "chunks_path": str(chunks_path),
            "vector_store_path": vector_dir,
            "bm25_path": str(bm25_path)
        }
        
        # Save stats
        with open(Path(output_dir) / "pipeline_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION PIPELINE COMPLETE")
        logger.info(f"  Chunks: {stats['total_chunks']}")
        logger.info(f"  Vectors: {stats['vector_store_size']}")
        logger.info("=" * 60)
        
        return stats
    
    def _build_bm25_index(self, chunks: list[dict], output_path: str):
        """Build BM25 corpus for sparse retrieval."""
        try:
            from rank_bm25 import BM25Okapi
            import re
            
            # Tokenize chunks for BM25
            corpus = []
            for chunk in chunks:
                text = chunk.get("content", "")
                # Simple tokenization: lowercase, split, remove short tokens
                tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
                corpus.append(tokens)
            
            bm25 = BM25Okapi(corpus)
            
            with open(output_path, "wb") as f:
                pickle.dump({
                    "bm25": bm25,
                    "corpus": corpus,
                    "chunk_ids": [c.get("chunk_id", str(i)) for i, c in enumerate(chunks)]
                }, f)
            
            logger.info(f"BM25 index built with {len(corpus)} documents")
            
        except ImportError:
            logger.warning("rank_bm25 not installed. Skipping BM25 index.")


def main():
    parser = argparse.ArgumentParser(description="Run ingestion pipeline")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--vectors", type=str, default="data/vectors")
    parser.add_argument("--config", type=str, default="config/model_config.yaml")
    
    args = parser.parse_args()
    
    pipeline = IngestionPipeline(config_path=args.config)
    stats = pipeline.run(
        input_dir=args.input,
        output_dir=args.output,
        vector_dir=args.vectors
    )
    
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
