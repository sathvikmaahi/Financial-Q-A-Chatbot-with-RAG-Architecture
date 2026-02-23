# Financial Q&A Chatbot with RAG

A chatbot that answers questions about SEC 10-K filings using Retrieval-Augmented Generation (RAG). Built with Python, LangChain, FAISS, and Streamlit.

## What it does

This project scrapes 10-K annual reports from the SEC EDGAR database, processes them into searchable chunks, and lets you ask questions like:
- "What was Apple's revenue in 2023?"
- "What risks does Tesla face?"
- "Compare Microsoft and Google's R&D spending"

The system retrieves relevant document sections and uses OpenAI's GPT-3.5 to generate answers with citations.

## Architecture

```
User Query → Query Processing → Hybrid Retrieval (BM25 + FAISS) → 
Reranking → Context Assembly → LLM Answer → Response with Sources
```

**Key components:**
- **SEC Scraper**: Downloads 10-K filings from SEC EDGAR API
- **Chunking**: Splits documents into ~1000 character chunks with overlap
- **Embeddings**: Uses `all-MiniLM-L6-v2` for semantic search
- **Hybrid Retrieval**: Combines BM25 (keyword) + FAISS (semantic) with RRF fusion
- **Reranker**: Cross-encoder re-scores top results for better precision
- **RAG Chain**: LangChain pipeline for query processing and answer generation
- **Streamlit UI**: Simple web interface for asking questions

## Setup

### 1. Clone and install

```bash
git clone https://github.com/sathvikmaahi/Financial-Q-A-Chatbot-with-RAG-Architecture.git
cd Financial-Q-A-Chatbot-with-RAG-Architecture

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Set up OpenAI API key (optional but recommended)

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_key_here
```

Without an OpenAI key, the system falls back to template-based answers (less natural but still functional).

### 3. Scrape SEC filings

```bash
python -m src.scraper.sec_edgar_scraper
```

This downloads 10-K filings for companies listed in `config/target_companies.json` (20 major companies by default).

### 4. Process and build indexes

```bash
python -m src.pipeline.ingestion_pipeline
```

This creates:
- `data/processed/chunks.jsonl` - Document chunks
- `data/processed/bm25_corpus.pkl` - BM25 keyword index
- `data/vectors/faiss_index.bin` - FAISS vector index

### 5. Run the app

```bash
streamlit run src/app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Project Structure

```
├── src/
│   ├── app/
│   │   ├── streamlit_app.py      # Web UI
│   │   └── rag_chain.py         # RAG orchestration
│   ├── pipeline/
│   │   ├── ingestion_pipeline.py # Main data pipeline
│   │   ├── chunking.py          # Document chunking
│   │   └── embedding_generator.py # FAISS vector store
│   ├── retrieval/
│   │   ├── hybrid_retriever.py  # BM25 + FAISS retrieval
│   │   └── query_processor.py   # Query enhancement
│   └── scraper/
│       └── sec_edgar_scraper.py # SEC EDGAR API client
├── config/
│   ├── target_companies.json    # Companies to scrape
│   └── model_config.yaml        # Model settings
├── data/                        # Data directory (gitignored)
├── requirements.txt
└── README.md
```

## How it works

1. **Query Processing**: Extracts company tickers, expands synonyms ("revenue" → "sales, earnings")
2. **Hybrid Retrieval**: 
   - BM25 finds documents with matching keywords
   - FAISS finds semantically similar documents
   - RRF combines both rankings
3. **Reranking**: Cross-encoder re-scores top 10 results
4. **Context Assembly**: Formats top chunks with source citations
5. **Answer Generation**: GPT-3.5 synthesizes answer from retrieved context

## Configuration

Edit `config/model_config.yaml` to adjust:

- **Chunk size**: `chunking.recursive.chunk_size` (default: 1000)
- **Retrieval weights**: `retrieval.bm25_weight` / `dense_weight` (default: 0.3/0.7)
- **Top-K results**: `retrieval.top_k` (default: 10)
- **LLM settings**: `llm.temperature`, `llm.max_tokens`

## Adding Companies

Edit `config/target_companies.json`:

```json
{
  "ticker": "NFLX",
  "name": "Netflix Inc.",
  "cik": "0001065280"
}
```

Then re-run scraper and ingestion pipeline.

## Tech Stack

- **LangChain**: RAG pipeline orchestration
- **FAISS**: Vector similarity search
- **Sentence-Transformers**: Embeddings (all-MiniLM-L6-v2)
- **Streamlit**: Web UI
- **OpenAI API**: Answer generation (GPT-3.5)
- **SEC EDGAR API**: Free 10-K filing access

## Notes

- SEC EDGAR API is free but has rate limits (10 requests/second)
- First query may be slow (embedding model loads on first use)
- Without OpenAI API key, answers are extracted using keyword matching (less natural)
- Data files are gitignored - each user needs to scrape their own filings

## License

MIT
