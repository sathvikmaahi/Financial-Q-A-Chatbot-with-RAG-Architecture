# Financial RAG Chatbot - Developer Guide

> **Purpose**: Comprehensive reference for architecture, concepts, and development tasks. Use this for interviews, onboarding, and troubleshooting.

---

## Table of Contents
1. [System Architecture](#1-system-architecture)
2. [Core Concepts Explained](#2-core-concepts-explained)
3. [Data Flow (Step-by-Step)](#3-data-flow-step-by-step)
4. [File Structure & Responsibilities](#4-file-structure--responsibilities)
5. [Key Classes & Functions](#5-key-classes--functions)
6. [Configuration Reference](#6-configuration-reference)
7. [Common Development Tasks](#7-common-development-tasks)
8. [Troubleshooting Guide](#8-troubleshooting-guide)
9. [Interview Q&A](#9-interview-qa)

---

## 1. System Architecture

### High-Level Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User Query    │────▶│   Streamlit UI   │────▶│   RAG Chain     │
│  (Natural Lang) │     │  (streamlit_app) │     │  (rag_chain.py) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                    ┌──────────────────────────────────────┼──────────────────────────────────────┐
                    │                                      │                                      │
                    ▼                                      ▼                                      ▼
           ┌─────────────────┐                  ┌─────────────────┐                  ┌─────────────────┐
           │ Query Processor │                  │ Hybrid Retriever│                  │   LLM (OpenAI)  │
           │  (Enhance query)│                  │ (BM25 + FAISS)  │                  │ (Generate answer│
           └─────────────────┘                  └────────┬────────┘                  └─────────────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                              ▼                        ▼                        ▼
                     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                     │  BM25 Index     │    │  FAISS Vector   │    │ Cross-Encoder   │
                     │ (Keyword Search)│    │  Store (Semantic)│   │  (Reranker)     │
                     └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Component Interaction

```
User asks: "What is Apple's revenue?"
         │
         ▼
┌────────────────────┐
│ 1. Query Processor │ ──▶ Extracts "AAPL" ticker, expands "revenue" → "sales, earnings"
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 2. Hybrid Retrieval│ ──▶ BM25 finds docs with "revenue", FAISS finds conceptually similar docs
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 3. Reranking       │ ──▶ Cross-encoder re-scores top 10 results for precision
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 4. Context Assembly│ ──▶ Formats top 5 chunks with source citations
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 5. LLM Generation  │ ──▶ GPT-3.5 synthesizes answer from context
└────────────────────┘
         │
         ▼
┌────────────────────┐
│ 6. Response        │ ──▶ Answer + sources + confidence score
└────────────────────┘
```

---

## 2. Core Concepts Explained

### 2.1 RAG (Retrieval-Augmented Generation)

**What it is**: A technique where an LLM answers questions using retrieved documents rather than its training data.

**Why it matters**:
- **Prevents hallucination**: LLM only uses provided facts
- **Up-to-date**: Can reference documents newer than training cutoff
- **Citable**: Sources are explicit and verifiable
- **Domain-specific**: Works on private/proprietary data

**Analogy**: Like an open-book exam — the LLM "looks up" answers in provided documents instead of relying on memory.

### 2.2 Embeddings & Vector Search

**What are embeddings?**
- Text converted to numerical vectors (e.g., 384 dimensions)
- Similar texts have similar vectors (cosine similarity)
- Example: "Apple revenue" and "iPhone sales" will have close vectors

**How FAISS works**:
- Stores millions of vectors efficiently
- Uses clustering (IVFFlat) to search only relevant partitions
- Returns top-K most similar vectors in milliseconds

**Code reference**:
```python
# In embedding_generator.py
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("Apple's revenue grew 5%")  # Returns 384-dim vector
```

### 2.3 Hybrid Retrieval (BM25 + Dense)

**BM25 (Sparse Retrieval)**:
- Classic keyword matching algorithm
- Good for exact terms: "revenue", "iPhone", "risk factors"
- Fast, interpretable, but misses semantic similarity

**Dense Retrieval (FAISS)**:
- Semantic similarity via embeddings
- Catches: "earnings" ≈ "profit" ≈ "net income"
- Can miss exact keyword matches

**Reciprocal Rank Fusion (RRF)**:
- Combines both scores: `score = 0.3*BM25 + 0.7*Dense`
- RRF formula: `score = Σ 1/(k + rank)` where k=60
- Best of both worlds: exact matching + semantic understanding

### 2.4 Chunking Strategies

**Why chunk?**
- 10-K documents are 50,000+ words
- LLMs have context limits (4K-128K tokens)
- Smaller chunks = more precise retrieval

**Recursive Character Splitting** (default):
```
Chunk 1: Items 1-2 (Business, Risk Factors) ──overlap──▶ Chunk 2: Items 2-3 (Risk Factors, MD&A)
                ↑ 200 chars shared ↑
```
- Tries to split at paragraph → sentence → word boundaries
- Overlap prevents losing context at boundaries

**Semantic Chunking** (alternative):
- Groups sentences by embedding similarity
- Keeps related ideas together
- Better quality but slower

### 2.5 Cross-Encoder Reranking

**Problem**: Initial retrieval (BM25+FAISS) is fast but approximate

**Solution**: Use a more accurate model on just the top 10 results

**How it works**:
```python
# Cross-encoder takes query + document together
score = cross_encoder("What is Apple's revenue?", document_text)
# Produces more accurate relevance score than separate embeddings
```

**Trade-off**: Slower (processes 10 docs) but much more accurate

### 2.6 Query Processing

**Entity Extraction**:
- Detects company tickers: "Apple" → "AAPL"
- Detects years: "2023" → filter by filing date
- Uses regex patterns and company name mappings

**Query Expansion**:
- "revenue" → ["revenue", "sales", "earnings", "income"]
- Improves recall by searching synonyms

**HyDE (Hypothetical Document Embeddings)**:
- Generates a fake "ideal answer" to the question
- Embeds that and searches for similar real documents
- Helps when query is vague

---

## 3. Data Flow (Step-by-Step)

### Ingestion Pipeline (One-time setup)

```
SEC EDGAR API
      │
      ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ 1. Scraper      │───▶│ 2. Text Cleaner │───▶│ 3. Chunker      │
│ (sec_edgar_     │    │ (Remove HTML,   │    │ (Split into     │
│  scraper.py)    │    │  normalize)     │    │  1000-char      │
│                 │    │                 │    │  chunks)        │
└─────────────────┘    └─────────────────┘    └────────┬────────┘
                                                       │
                              ┌────────────────────────┼────────────────────────┐
                              │                        │                        │
                              ▼                        ▼                        ▼
                     ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
                     │ 4a. BM25 Index  │    │ 4b. FAISS Index │    │ 4c. Metadata    │
                     │ (tokenize,      │    │ (embeddings     │    │ (company, year, │
                     │  build inverted │    │  stored as      │    │  section info)  │
                     │  index)         │    │  vectors)       │    │                 │
                     └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Query Pipeline (Runtime)

```
User: "What risks does Tesla face?"
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 1: Query Processing (QueryProcessor.process_query)                     │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Input:  "What risks does Tesla face?"                                       │
│ Output: {                                                                   │
│   "original_query": "What risks does Tesla face?",                          │
│   "expanded_query": "What risks does Tesla face risk factors challenges",   │
│   "detected_entities": {                                                    │
│     "tickers": ["TSLA"],                                                    │
│     "years": []                                                              │
│   }                                                                          │
│ }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 2: Hybrid Retrieval (HybridRetriever.retrieve)                           │
│ ─────────────────────────────────────────────────────────────────────────── │
│ BM25 Search:  Finds docs with "risks", "Tesla", "risk factors"                │
│ FAISS Search: Finds semantically similar docs (embeddings)                   │
│ RRF Fusion:   Combines rankings with weights (0.3 BM25 + 0.7 FAISS)           │
│ Output: Top 10 chunks with fused scores                                       │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 3: Reranking (CrossEncoderReranker.rerank)                             │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Input:  Top 10 chunks from hybrid retrieval                                   │
│ Process: Cross-encoder scores each (query, chunk) pair                       │
│ Output: Re-ordered top 5 chunks (more accurate)                               │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 4: Context Assembly (RAGChain._assemble_context)                       │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Deduplicates chunks (by content hash)                                        │
│ Estimates token count (len/4)                                                │
│ Truncates to fit context window (4096 - 1024 - 500 = ~2500 tokens)            │
│ Formats with source headers:                                                 │
│   --- Source: [TSLA, 2023, Item 1A - Risk Factors] ---                     │
│   [chunk text...]                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 5: LLM Generation (RAGChain._generate_answer)                          │
│ ─────────────────────────────────────────────────────────────────────────── │
│ Prompt Template:                                                             │
│   SYSTEM: You are a financial analyst...                                     │
│   CONTEXT: [assembled chunks with sources]                                   │
│   QUESTION: What risks does Tesla face?                                      │
│   ANSWER: [GPT-3.5 generates here]                                           │
│                                                                              │
│ Output: Natural language answer with citations                               │
└─────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ STEP 6: Response Packaging (RAGResponse)                                    │
│ ─────────────────────────────────────────────────────────────────────────── │
│ {                                                                            │
│   "answer": "Tesla faces several risks including...",                        │
│   "sources": [{ticker, company_name, section, relevance_score, preview}], │
│   "confidence": 0.85,                                                        │
│   "query_info": {...},                                                       │
│   "retrieval_scores": [0.92, 0.88, 0.85, ...]                               │
│ }                                                                            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. File Structure & Responsibilities

```
financial-rag-chatbot/
│
├── config/                          # Configuration files
│   ├── model_config.yaml           # Model & retrieval settings
│   ├── target_companies.json       # Companies to scrape
│   └── aws_deployment.yaml         # AWS deployment config
│
├── src/                            # Source code
│   ├── app/                        # Application layer
│   │   ├── streamlit_app.py      # Web UI (Streamlit)
│   │   └── rag_chain.py          # Main RAG orchestration
│   │
│   ├── pipeline/                   # Data processing
│   │   ├── ingestion_pipeline.py  # End-to-end data pipeline
│   │   ├── chunking.py            # Document chunking strategies
│   │   └── embedding_generator.py # FAISS vector store management
│   │
│   ├── retrieval/                  # Search & retrieval
│   │   ├── hybrid_retriever.py    # BM25 + FAISS + Reranking
│   │   └── query_processor.py     # Query enhancement
│   │
│   ├── scraper/                    # Data acquisition
│   │   └── sec_edgar_scraper.py   # SEC EDGAR API client
│   │
│   └── utils/                      # Utilities
│       └── text_cleaner.py        # HTML/text normalization
│
├── data/                           # Data storage (gitignored)
│   ├── raw/                        # Scraped 10-K JSON files
│   ├── processed/                  # Chunks, BM25 corpus
│   └── vectors/                    # FAISS index, metadata
│
├── scripts/                        # Deployment scripts
│   └── setup_aws.sh               # AWS EC2 setup
│
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container definition
└── DEVELOPER_GUIDE.md             # This file
```

### File Responsibilities

| File | What It Does | Key Output |
|------|--------------|------------|
| `sec_edgar_scraper.py` | Downloads 10-K filings from SEC API | `data/raw/{TICKER}_{DATE}_10K.json` |
| `text_cleaner.py` | Removes HTML, normalizes text | Clean plain text |
| `chunking.py` | Splits documents into chunks | List of text chunks |
| `embedding_generator.py` | Creates embeddings, manages FAISS | `faiss_index.bin`, `metadata_store.pkl` |
| `hybrid_retriever.py` | Searches BM25 + FAISS, reranks | List of `RetrievalResult` objects |
| `query_processor.py` | Enhances queries, extracts entities | `QueryInfo` with expansions |
| `rag_chain.py` | Orchestrates full RAG pipeline | `RAGResponse` with answer |
| `streamlit_app.py` | Web UI, handles user interaction | Streamlit interface |

---

## 5. Key Classes & Functions

### 5.1 RAGChain (`src/app/rag_chain.py`)

**Purpose**: Main orchestrator for the RAG pipeline

```python
class RAGChain:
    def __init__(
        self,
        retriever=None,                    # HybridRetriever instance
        llm_provider: str = "openai",      # "openai", "huggingface", "template"
        model_name: str = "gpt-3.5-turbo", # Model to use
        temperature: float = 0.1,          # Creativity (0=deterministic)
        max_tokens: int = 1024,            # Max answer length
        context_window: int = 4096         # Total tokens (context + answer)
    )
    
    def query(
        self,
        question: str,                     # User question
        filter_company: Optional[str] = None,  # Ticker filter (e.g., "AAPL")
        filter_year: Optional[str] = None, # Year filter
        top_k: int = 5                     # Number of chunks to retrieve
    ) -> RAGResponse                      # Answer + sources + metadata
```

**When to use**: This is the main entry point. Call `rag_chain.query()` with any question.

### 5.2 HybridRetriever (`src/retrieval/hybrid_retriever.py`)

**Purpose**: Combines BM25 and FAISS for document retrieval

```python
class HybridRetriever:
    def __init__(
        self,
        vector_store_dir: str,             # Path to FAISS index
        bm25_path: str,                    # Path to BM25 pickle
        chunks_path: str,                   # Path to chunks.jsonl
        bm25_weight: float = 0.3,          # Weight for BM25 scores
        dense_weight: float = 0.7,         # Weight for FAISS scores
        top_k: int = 10,                   # Initial retrieval count
        use_reranker: bool = True,         # Enable cross-encoder reranking
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    
    def retrieve(
        self,
        query: str,                        # Search query
        top_k: int = 10,                   # Number of results
        filter_metadata: Optional[dict] = None  # Filter by ticker/year
    ) -> List[RetrievalResult]            # Ranked list of results
```

**When to use**: When you need to search the document collection programmatically.

### 5.3 QueryProcessor (`src/retrieval/hybrid_retriever.py`)

**Purpose**: Enhances raw user queries for better retrieval

```python
class QueryProcessor:
    def process_query(self, query: str) -> QueryInfo:
        """
        Input:  "What is Apple's revenue in 2023?"
        Output: {
            "original_query": "What is Apple's revenue in 2023?",
            "expanded_query": "What is Apple AAPL revenue sales earnings 2023",
            "detected_entities": {
                "tickers": ["AAPL"],
                "years": ["2023"],
                "companies": ["Apple Inc."]
            },
            "sub_queries": ["Apple revenue 2023", "AAPL financial performance"]
        }
        """
```

**When to use**: Automatically called by RAGChain, but can be used standalone for query analysis.

### 5.4 EmbeddingGenerator (`src/pipeline/embedding_generator.py`)

**Purpose**: Creates embeddings and manages FAISS vector store

```python
class EmbeddingGenerator:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: str = "auto"               # "cuda", "cpu", or "auto"
    )
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts."""
        
    def build_vector_store(
        self,
        chunks: List[Dict],
        output_dir: str
    ) -> None:
        """Build and save FAISS index from chunks."""
        
    def load(self, vector_store_dir: str) -> None:
        """Load existing FAISS index."""
```

**When to use**: During ingestion to build the vector store, or to embed new queries.

### 5.5 RAGResponse (`src/app/rag_chain.py`)

**Purpose**: Structured output from the RAG pipeline

```python
@dataclass
class RAGResponse:
    answer: str                        # Generated answer text
    sources: List[Dict]                # Source documents with metadata
    confidence: float                  # 0-1 confidence score
    query_info: Dict                   # Query processing details
    retrieval_scores: List[float]    # Relevance scores for each source
```

---

## 6. Configuration Reference

### 6.1 Model Config (`config/model_config.yaml`)

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `embedding` | `model_name` | `all-MiniLM-L6-v2` | Sentence transformer for embeddings |
| `embedding` | `batch_size` | 64 | Texts processed per batch |
| `embedding` | `max_seq_length` | 256 | Max tokens per embedding |
| `chunking` | `chunk_size` | 1000 | Characters per chunk |
| `chunking` | `chunk_overlap` | 200 | Overlap between chunks |
| `vector_store` | `index_type` | `IVFFlat` | FAISS index type (IVFFlat for speed) |
| `vector_store` | `nprobe` | 10 | Cells to search (speed vs accuracy) |
| `retrieval` | `bm25_weight` | 0.3 | Weight for keyword search |
| `retrieval` | `dense_weight` | 0.7 | Weight for semantic search |
| `retrieval` | `reranker_model` | `ms-marco-MiniLM-L-6-v2` | Cross-encoder for reranking |
| `llm` | `provider` | `huggingface` | LLM backend |
| `llm` | `model_name` | `Mistral-7B` | Model identifier |
| `llm` | `temperature` | 0.1 | Randomness (0=deterministic) |
| `llm` | `max_tokens` | 1024 | Max output length |

### 6.2 Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENAI_API_KEY` | OpenAI API access | `sk-...` |
| `LLM_PROVIDER` | Override LLM backend | `openai`, `huggingface`, `template` |
| `LLM_MODEL` | Override model name | `gpt-4`, `gpt-3.5-turbo` |
| `VECTOR_STORE_PATH` | Custom vector store location | `/path/to/vectors` |
| `TOKENIZERS_PARALLELISM` | Prevent macOS segfaults | `false` |
| `OMP_NUM_THREADS` | Limit OpenMP threads | `1` |

---

## 7. Common Development Tasks

### 7.1 Add a New Company

1. Edit `config/target_companies.json`:
```json
{
  "ticker": "NFLX",
  "name": "Netflix Inc.",
  "cik": "0001065280"
}
```

2. Scrape the new company:
```bash
python -m src.scraper.sec_edgar_scraper
```

3. Rebuild indexes:
```bash
python -m src.pipeline.ingestion_pipeline
```

4. Restart Streamlit app

### 7.2 Change LLM Model

**Option A: Environment variable (temporary)**
```bash
export LLM_MODEL="gpt-4"
streamlit run src/app/streamlit_app.py
```

**Option B: Modify code (permanent)**
Edit `src/app/streamlit_app.py`:
```python
model_name = os.environ.get("LLM_MODEL", "gpt-4")  # Change default
```

### 7.3 Adjust Retrieval Settings

Edit `config/model_config.yaml`:
```yaml
retrieval:
  bm25_weight: 0.5    # Increase for more keyword matching
  dense_weight: 0.5   # Decrease for less semantic matching
  top_k: 15           # Retrieve more documents
  use_reranker: false # Disable for faster retrieval
```

### 7.4 Debug Retrieval Quality

Add to `src/app/rag_chain.py` in `query()` method:
```python
# After retrieval
print(f"Retrieved {len(retrieval_results)} documents")
for i, r in enumerate(retrieval_results[:3]):
    print(f"{i+1}. Score: {r.fused_score:.3f}")
    print(f"   Source: {r.metadata.get('ticker')}, {r.metadata.get('section_name')}")
    print(f"   Preview: {r.content[:100]}...")
    print()
```

### 7.5 Add a New Section to Scrape

Edit `src/scraper/sec_edgar_scraper.py`:
```python
SECTIONS_TO_EXTRACT = [
    "1",    # Business
    "1A",   # Risk Factors
    "7",    # MD&A
    "7A",   # Market Risk
    "8",    # Financial Statements
    "9",    # New section
]
```

### 7.6 Run Evaluation

```python
from src.evaluation.evaluator import RAGEvaluator

evaluator = RAGEvaluator(rag_chain)
results = evaluator.evaluate_on_dataset("data/evaluation/questions.jsonl")
print(f"Average Precision@5: {results['precision_at_5']:.3f}")
```

---

## 8. Troubleshooting Guide

### 8.1 "RAG system not loaded" Error

**Symptoms**: Streamlit shows warning, no answers generated

**Causes & Fixes**:
1. **Missing data files**
   ```bash
   # Check if files exist
   ls data/vectors/faiss_index.bin
   ls data/processed/chunks.jsonl
   
   # If missing, run pipeline
   python -m src.scraper.sec_edgar_scraper
   python -m src.pipeline.ingestion_pipeline
   ```

2. **Module not found errors**
   - Already fixed in current code (sys.path modification)
   - If persists: `export PYTHONPATH="${PYTHONPATH}:$(pwd)"`

3. **Stale Streamlit cache**
   ```bash
   pkill -f "streamlit run"
   # Wait 2 seconds, then restart
   streamlit run src/app/streamlit_app.py
   ```

### 8.2 Answers Are Irrelevant or Generic

**Diagnosis Steps**:
1. Check retrieval scores in logs (should be >0.5 for good matches)
2. Verify query entity detection (is "Apple" being recognized as "AAPL"?)
3. Inspect retrieved chunks (are they from the right company/section?)

**Fixes**:
- Increase `top_k` to retrieve more documents
- Adjust BM25/dense weights in config
- Add query expansion synonyms
- Check if company is in `target_companies.json`

### 8.3 "No module named 'src'" Error

**Cause**: Python can't find the `src` package

**Fix**: Already implemented in `streamlit_app.py`:
```python
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

If still failing, run from project root:
```bash
cd /Users/sathviksanka/Downloads/financial-rag-chatbot
python -m src.app.streamlit_app
```

### 8.4 Segfault During Embedding Generation

**Symptoms**: Python crashes with exit code 139 during `ingestion_pipeline`

**Cause**: Tokenizers multiprocessing conflict on macOS

**Fix**: Already implemented — sets `TOKENIZERS_PARALLELISM=false` and `OMP_NUM_THREADS=1`

If still crashing:
```bash
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
python -m src.pipeline.ingestion_pipeline
```

### 8.5 OpenAI API Errors

**"Authentication Error"**: Invalid API key
- Check key is set: `echo $OPENAI_API_KEY`
- Verify key format: should start with `sk-`

**"Rate Limit Exceeded"**: Too many requests
- Add rate limiting: `time.sleep(1)` between calls
- Upgrade OpenAI plan

**"Context Length Exceeded"**: Prompt too long
- Reduce `top_k` to retrieve fewer chunks
- Decrease `chunk_size` in config
- Increase `context_window` if using GPT-4 (8K or 32K)

### 8.6 Slow Response Times

**Bottlenecks & Solutions**:

| Bottleneck | Symptom | Fix |
|------------|---------|-----|
| Embedding generation | Slow first query | Pre-load model at startup |
| FAISS search | High latency on large index | Use `IVFFlat` instead of `Flat` |
| LLM API call | 2-3s per query | Use faster model (GPT-3.5 vs GPT-4) |
| Reranking | Slow with many results | Reduce `rerank_top_k` or disable |

### 8.7 High Memory Usage

**Causes**:
- Loading full FAISS index into RAM
- Keeping all chunks in memory
- Large LLM model (if using local)

**Fixes**:
- Use `IVFFlat` index (smaller than `Flat`)
- Lazy-load chunks (load from disk on demand)
- Use OpenAI API instead of local LLM
- Reduce `batch_size` in embedding generator

---

## 9. Interview Q&A

### Q1: What is RAG and why use it?

**Answer**: RAG (Retrieval-Augmented Generation) is a technique where an LLM answers questions using retrieved documents rather than its training data. Benefits:
- **Reduces hallucination**: LLM is grounded in provided facts
- **Current information**: Can reference documents newer than training cutoff
- **Verifiable**: Sources are explicit and checkable
- **Domain adaptation**: Works on private/proprietary data without fine-tuning

**Example**: In this project, we retrieve relevant 10-K filing sections, feed them to GPT-3.5 with the question, and get a cited answer.

### Q2: Why hybrid retrieval (BM25 + FAISS) instead of just one?

**Answer**: They complement each other:
- **BM25 (sparse)**: Excels at exact keyword matching — finds "revenue" when user asks about revenue
- **FAISS (dense)**: Excels at semantic similarity — finds "earnings" when user asks about "profit"
- **RRF fusion**: Combines both rankings, getting best of both worlds

**Result**: Higher recall (finds more relevant docs) and better precision (top results are actually relevant).

### Q3: How do you prevent the LLM from hallucinating?

**Answer**: Multiple safeguards:
1. **System prompt**: Explicitly instruct "ONLY use information from provided context"
2. **Context isolation**: Only send retrieved chunks, no external knowledge
3. **Source citations**: Require LLM to cite sources, making verification possible
4. **Confidence scoring**: Calculate confidence based on retrieval quality
5. **Template fallback**: If LLM fails, extract sentences algorithmically from retrieved docs

### Q4: What is chunking and why does it matter?

**Answer**: Chunking splits large documents into smaller pieces because:
- LLMs have context limits (4K-128K tokens)
- Smaller chunks = more precise retrieval (don't retrieve entire 10-K)
- Overlap prevents losing context at boundaries

**Strategies**:
- **Recursive**: Split at paragraph → sentence → word boundaries
- **Semantic**: Group related sentences by embedding similarity
- **Fixed-size**: Simple but may cut mid-sentence

**Trade-off**: Smaller chunks = precise retrieval but less context per chunk.

### Q5: How does the cross-encoder reranker improve results?

**Answer**: Initial retrieval (BM25+FAISS) uses **bi-encoders** — query and document are embedded separately, then similarity is computed. This is fast but less accurate.

**Cross-encoder** takes (query, document) together and outputs a relevance score directly. It's more accurate because it can see both texts simultaneously.

**Trade-off**: Cross-encoders are slower (O(n) per document), so we only run them on the top 10 results from initial retrieval.

### Q6: How would you scale this to 100,000 documents?

**Answer**: Several optimizations:
1. **FAISS index**: Use `IVFPQ` (Product Quantization) instead of `IVFFlat` — 10x smaller, faster search
2. **Sharding**: Split index by company or year, search only relevant shards
3. **Caching**: Cache embedding model, BM25 index in memory
4. **Async**: Process queries asynchronously with queue
5. **Distributed**: Multiple retrieval workers, single LLM API
6. **Approximate**: Reduce `nprobe` for faster (but less accurate) search

### Q7: How do you evaluate RAG system quality?

**Answer**: Two levels:

**Retrieval metrics**:
- **Precision@K**: % of top-K results that are relevant
- **Recall@K**: % of all relevant docs found in top-K
- **MRR**: Mean Reciprocal Rank (how high is first relevant doc)
- **NDCG**: Normalized Discounted Cumulative Gain (ranking quality)

**Generation metrics**:
- **Faithfulness**: Does answer match retrieved context? (LLM-judged)
- **Answer relevance**: Does answer address the question? (LLM-judged)
- **Citation accuracy**: Are sources correctly cited?

**Human evaluation**: Ask financial experts to rate answer accuracy.

### Q8: What are the limitations of this system?

**Answer**:
1. **Retrieval dependency**: If retrieval fails, answer is wrong
2. **Context window**: Limited by LLM's token limit (4K-32K)
3. **Citation granularity**: Cites entire chunk, not specific sentence
4. **Temporal**: 10-Ks are annual, miss recent quarterly updates
5. **Single-hop**: Can't answer questions requiring multi-document reasoning
6. **Cost**: OpenAI API calls cost money per query

**Improvements**: Add agent-based multi-hop reasoning, use larger context models (Claude 100K), add real-time 10-Q filings.

### Q9: Explain the difference between embedding model and LLM

**Answer**:

| Aspect | Embedding Model (all-MiniLM) | LLM (GPT-3.5) |
|--------|------------------------------|---------------|
| **Purpose** | Convert text to vectors | Generate natural language |
| **Input** | Single text | Prompt (instruction + context) |
| **Output** | 384-dim vector | Generated text |
| **Size** | ~100MB | ~100GB (for local) or API |
| **Speed** | Fast (milliseconds) | Slower (seconds) |
| **Use in RAG** | Encode chunks for FAISS, encode queries | Synthesize answer from retrieved chunks |

**Analogy**: Embedding model is like a librarian who finds relevant books. LLM is like an expert who reads the books and answers your question.

### Q10: How would you add multi-hop reasoning?

**Answer**: Current system is single-hop: query → retrieve → answer.

For multi-hop (e.g., "Compare Apple's 2023 revenue to their 2022 revenue"):

1. **Query decomposition**: Break into sub-queries:
   - "Apple revenue 2023"
   - "Apple revenue 2022"

2. **Iterative retrieval**: For each sub-query:
   - Retrieve relevant chunks
   - Extract answer
   - Use as context for next hop

3. **Synthesis**: Combine intermediate answers into final response

**Implementation**: Use LangChain's `MultiQueryRetriever` or build agent with ReAct pattern.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────┐
│                    STARTING THE SYSTEM                          │
├─────────────────────────────────────────────────────────────────┤
│ 1. Scrape:  python -m src.scraper.sec_edgar_scraper             │
│ 2. Process: python -m src.pipeline.ingestion_pipeline          │
│ 3. Run:      streamlit run src/app/streamlit_app.py             │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    KEY CONFIGURATIONS                           │
├─────────────────────────────────────────────────────────────────┤
│ LLM Provider:  export LLM_PROVIDER=openai                       │
│ API Key:       export OPENAI_API_KEY=sk-...                     │
│ Model:         export LLM_MODEL=gpt-3.5-turbo                   │
│ Chunk Size:    config/model_config.yaml → chunking.chunk_size   │
│ Retrieval:     config/model_config.yaml → retrieval.top_k       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    DEBUGGING CHECKLIST                            │
├─────────────────────────────────────────────────────────────────┤
│ □ Check data files exist: ls data/vectors/                        │
│ □ Check API key: echo $OPENAI_API_KEY                            │
│ □ Clear Streamlit cache: pkill -f "streamlit run"               │
│ □ Check logs for retrieval scores                               │
│ □ Verify company in target_companies.json                       │
└─────────────────────────────────────────────────────────────────┘
```

---

**Last Updated**: 2026-02-23
**Version**: 1.0
