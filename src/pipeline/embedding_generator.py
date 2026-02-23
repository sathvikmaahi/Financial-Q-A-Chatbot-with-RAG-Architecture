"""
Embedding Generator & Vector Store Builder
==========================================
Generates embeddings using HuggingFace sentence-transformers and builds
a FAISS vector index with metadata storage for hybrid retrieval.

Features:
- Batch embedding generation with GPU support
- FAISS index with IVFFlat for scalable search
- Metadata store alongside vectors for filtering
- Incremental index updates
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import faiss
from loguru import logger
from tqdm import tqdm


class EmbeddingGenerator:
    """
    Generate embeddings using HuggingFace sentence-transformers.
    
    Uses all-MiniLM-L6-v2 by default (384 dimensions, fast inference).
    Supports batched generation with progress tracking.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = "auto",
        normalize: bool = True
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize
        self.model = None
        self.device = device
        self.embedding_dim = None
        
        logger.info(f"EmbeddingGenerator initialized with model: {model_name}")
    
    def load_model(self):
        """Lazy load the embedding model."""
        if self.model is not None:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
            logger.info(f"Loading embedding model on {self.device}...")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
            
        except ImportError:
            logger.warning("sentence-transformers not available. Using mock embeddings.")
            self.embedding_dim = 384
            self.model = None
    
    def embed_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        self.load_model()
        
        if self.model is not None:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True
            )
            return embeddings
        else:
            # Mock embeddings for development without GPU
            logger.warning("Using random mock embeddings (no model loaded)")
            embeddings = np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
            if self.normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / norms
            return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query], show_progress=False)[0]


class VectorStoreBuilder:
    """
    Build and manage a FAISS vector store with metadata.
    
    Supports:
    - Flat index (exact search, good for < 100K vectors)
    - IVFFlat index (approximate search, good for > 100K vectors)
    - Metadata storage for filtering (ticker, section, date)
    - Save/load to disk
    """
    
    def __init__(
        self,
        embedding_dim: int = 384,
        index_type: str = "Flat",
        nlist: int = 100,
        metric: str = "cosine"
    ):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.nlist = nlist
        self.metric = metric
        self.index = None
        self.metadata_store = []  # Parallel list of metadata dicts
        self.chunk_texts = []     # Parallel list of chunk texts
        self.id_map = {}          # chunk_id -> index position
        
    def build_index(
        self,
        embeddings: np.ndarray,
        chunks: list[dict],
        texts: Optional[list[str]] = None
    ):
        """
        Build FAISS index from embeddings and metadata.
        
        Args:
            embeddings: numpy array of shape (n, embedding_dim)
            chunks: List of chunk dictionaries with metadata
            texts: Optional list of chunk texts (extracted from chunks if not provided)
        """
        n_vectors = embeddings.shape[0]
        
        logger.info(f"Building FAISS {self.index_type} index with {n_vectors} vectors...")
        
        # Ensure float32
        embeddings = embeddings.astype(np.float32)
        
        if self.metric == "cosine":
            # Normalize for cosine similarity (use inner product after normalization)
            faiss.normalize_L2(embeddings)
            metric_type = faiss.METRIC_INNER_PRODUCT
        else:
            metric_type = faiss.METRIC_L2
        
        if self.index_type == "IVFFlat" and n_vectors > self.nlist * 40:
            # Use IVFFlat for large datasets
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, self.nlist, metric_type
            )
            self.index.train(embeddings)
            self.index.add(embeddings)
            self.index.nprobe = min(10, self.nlist)
        else:
            # Use Flat index for smaller datasets (exact search)
            if self.metric == "cosine":
                self.index = faiss.IndexFlatIP(self.embedding_dim)
            else:
                self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.index.add(embeddings)
        
        # Store metadata and texts
        self.metadata_store = [c.get("metadata", {}) for c in chunks]
        self.chunk_texts = texts if texts else [c.get("content", "") for c in chunks]
        self.id_map = {
            c.get("chunk_id", str(i)): i for i, c in enumerate(chunks)
        }
        
        logger.info(f"FAISS index built: {self.index.ntotal} vectors indexed")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_metadata: Optional[dict] = None
    ) -> list[dict]:
        """
        Search the vector store.
        
        Args:
            query_embedding: Query vector (1D array)
            top_k: Number of results to return
            filter_metadata: Optional dict of metadata filters (e.g., {"ticker": "AAPL"})
            
        Returns:
            List of result dicts with content, metadata, and score
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
        
        # Reshape query
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        if self.metric == "cosine":
            faiss.normalize_L2(query)
        
        # Search more than needed if filtering
        search_k = top_k * 5 if filter_metadata else top_k
        
        scores, indices = self.index.search(query, min(search_k, self.index.ntotal))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            
            metadata = self.metadata_store[idx]
            
            # Apply metadata filters
            if filter_metadata:
                match = all(
                    metadata.get(k) == v for k, v in filter_metadata.items()
                )
                if not match:
                    continue
            
            results.append({
                "content": self.chunk_texts[idx],
                "metadata": metadata,
                "score": float(score),
                "index": int(idx)
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, output_dir: str):
        """Save index and metadata to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(output_path / "faiss_index.bin"))
        
        # Save metadata and texts
        with open(output_path / "metadata_store.pkl", "wb") as f:
            pickle.dump({
                "metadata_store": self.metadata_store,
                "chunk_texts": self.chunk_texts,
                "id_map": self.id_map,
                "config": {
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type,
                    "metric": self.metric
                }
            }, f)
        
        logger.info(f"Vector store saved to {output_dir}")
    
    @classmethod
    def load(cls, input_dir: str) -> "VectorStoreBuilder":
        """Load index and metadata from disk."""
        input_path = Path(input_dir)
        
        # Load metadata
        with open(input_path / "metadata_store.pkl", "rb") as f:
            data = pickle.load(f)
        
        config = data["config"]
        store = cls(
            embedding_dim=config["embedding_dim"],
            index_type=config["index_type"],
            metric=config["metric"]
        )
        
        # Load FAISS index
        store.index = faiss.read_index(str(input_path / "faiss_index.bin"))
        store.metadata_store = data["metadata_store"]
        store.chunk_texts = data["chunk_texts"]
        store.id_map = data["id_map"]
        
        logger.info(f"Vector store loaded from {input_dir}: {store.index.ntotal} vectors")
        return store


def build_vector_store_from_chunks(
    chunks_path: str,
    output_dir: str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64
):
    """
    End-to-end vector store building from chunks JSONL file.
    
    Args:
        chunks_path: Path to JSONL file with chunks
        output_dir: Output directory for vector store
        model_name: HuggingFace embedding model name
        batch_size: Embedding batch size
    """
    # Load chunks
    logger.info(f"Loading chunks from {chunks_path}...")
    chunks = []
    texts = []
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            chunk = json.loads(line.strip())
            chunks.append(chunk)
            texts.append(chunk["content"])
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Generate embeddings
    embedder = EmbeddingGenerator(model_name=model_name, batch_size=batch_size)
    embeddings = embedder.embed_texts(texts)
    
    logger.info(f"Generated embeddings: shape {embeddings.shape}")
    
    # Build and save index
    store = VectorStoreBuilder(
        embedding_dim=embeddings.shape[1],
        index_type="IVFFlat" if len(chunks) > 5000 else "Flat"
    )
    store.build_index(embeddings, chunks, texts)
    store.save(output_dir)
    
    return store
