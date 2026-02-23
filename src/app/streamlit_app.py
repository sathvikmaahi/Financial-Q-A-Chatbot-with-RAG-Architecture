"""
Financial Q&A Chatbot - Streamlit Application
===============================================
Production-ready Streamlit interface for the Financial RAG Chatbot.

Features:
- Real-time chat interface with message history
- Company and section filtering
- Source attribution panel with expandable citations
- Confidence indicators
- Financial metrics dashboard
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

# Add project root to Python path so `from src.xxx` imports work
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Prevent macOS segfaults with tokenizers/multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Financial Q&A Chatbot | SEC 10-K Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .source-card {
        background: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
    }
    .confidence-high { color: #155724; font-weight: 600; }
    .confidence-medium { color: #856404; font-weight: 600; }
    .confidence-low { color: #721c24; font-weight: 600; }
    .metric-container {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "messages": [],
        "rag_chain": None,
        "companies": [],
        "query_count": 0,
        "total_latency": 0.0,
        "system_loaded": False,
        "last_sources": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def load_company_list() -> list[dict]:
    """Load company list from config."""
    config_path = Path("config/target_companies.json")
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f).get("companies", [])
    return []


@st.cache_resource
def load_rag_system():
    """Load and cache the RAG pipeline components."""
    # Resolve paths relative to the project root
    project_root = Path(__file__).resolve().parent.parent.parent
    vector_dir = os.environ.get("VECTOR_STORE_PATH", str(project_root / "data" / "vectors"))
    chunks_path = str(project_root / "data" / "processed" / "chunks.jsonl")
    bm25_path = str(project_root / "data" / "processed" / "bm25_corpus.pkl")
    
    # Check if data exists
    missing = [p for p in [vector_dir, chunks_path] if not Path(p).exists()]
    if missing:
        st.warning(f"Missing data files: {missing}")
        return None
    
    try:
        from src.retrieval.hybrid_retriever import HybridRetriever
        from src.app.rag_chain import RAGChain
        
        retriever = HybridRetriever(
            vector_store_dir=vector_dir,
            bm25_path=bm25_path,
            chunks_path=chunks_path,
            bm25_weight=0.3,
            dense_weight=0.7,
            top_k=10
        )
        
        # Use OpenAI if API key is available, otherwise fall back to template
        if os.environ.get("OPENAI_API_KEY"):
            llm_provider = "openai"
            model_name = os.environ.get("LLM_MODEL", "gpt-3.5-turbo")
        else:
            llm_provider = os.environ.get("LLM_PROVIDER", "template")
            model_name = os.environ.get("LLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
        
        rag_chain = RAGChain(
            retriever=retriever,
            llm_provider=llm_provider,
            model_name=model_name
        )
        
        return rag_chain
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}\n\n{traceback.format_exc()}")
        return None


def render_sidebar():
    """Render the sidebar with filters and settings."""
    with st.sidebar:
        st.markdown("## 📊 Financial Q&A Chatbot")
        st.markdown("*Powered by RAG over SEC 10-K Filings*")
        st.divider()
        
        # Company Filter
        st.markdown("### 🏢 Filter by Company")
        companies = load_company_list()
        company_options = ["All Companies"] + [
            f"{c['ticker']} - {c['name']}" for c in companies
        ]
        selected_company = st.selectbox("Select Company", company_options)
        
        # Extract ticker from selection
        filter_ticker = None
        if selected_company != "All Companies":
            filter_ticker = selected_company.split(" - ")[0]
        
        # Section Filter
        st.markdown("### 📑 Filter by Section")
        section_options = [
            "All Sections",
            "Item 1 - Business",
            "Item 1A - Risk Factors",
            "Item 7 - MD&A",
            "Item 7A - Market Risk",
            "Item 8 - Financial Statements"
        ]
        selected_section = st.selectbox("Select Section", section_options)
        
        # Settings
        st.divider()
        st.markdown("### ⚙️ Settings")
        top_k = st.slider("Number of retrieved documents", 3, 15, 5)
        show_sources = st.toggle("Show source documents", value=True)
        show_scores = st.toggle("Show relevance scores", value=False)
        
        # System Stats
        st.divider()
        st.markdown("### 📈 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", st.session_state.query_count)
        with col2:
            avg_lat = (
                st.session_state.total_latency / st.session_state.query_count
                if st.session_state.query_count > 0 else 0
            )
            st.metric("Avg Latency", f"{avg_lat:.1f}s")
        
        # Example queries
        st.divider()
        st.markdown("### 💡 Example Questions")
        example_queries = [
            "What was Apple's total revenue in FY2023?",
            "What are NVIDIA's key risk factors?",
            "Compare Microsoft and Google R&D spending",
            "What is JPMorgan's total asset value?",
            "Describe Amazon's business segments",
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": query})
                st.rerun()
        
        return {
            "filter_ticker": filter_ticker,
            "selected_section": selected_section,
            "top_k": top_k,
            "show_sources": show_sources,
            "show_scores": show_scores,
        }


def render_confidence_badge(confidence: float) -> str:
    """Render a confidence level indicator."""
    if confidence >= 0.7:
        return f'🟢 High Confidence ({confidence:.0%})'
    elif confidence >= 0.4:
        return f'🟡 Medium Confidence ({confidence:.0%})'
    else:
        return f'🔴 Low Confidence ({confidence:.0%})'


def render_sources(sources: list[dict], show_scores: bool = False):
    """Render source attribution cards."""
    if not sources:
        return
    
    st.markdown("**📚 Sources:**")
    
    for i, source in enumerate(sources):
        ticker = source.get("ticker", "Unknown")
        company = source.get("company_name", "Unknown Company")
        date = source.get("filing_date", "N/A")
        section = source.get("section_name", "General")
        preview = source.get("content_preview", "")
        score = source.get("relevance_score", 0)
        
        with st.expander(
            f"📄 {ticker} | {date[:4] if date else 'N/A'} | {section}"
            + (f" | Score: {score:.3f}" if show_scores else ""),
            expanded=(i == 0)
        ):
            st.markdown(f"**Company:** {company}")
            st.markdown(f"**Filing Date:** {date}")
            st.markdown(f"**Section:** Item {source.get('section_number', 'N/A')} - {section}")
            if show_scores:
                st.markdown(f"**Relevance Score:** {score:.4f}")
            st.markdown("---")
            st.markdown(f"*{preview}*")


def process_query(question: str, settings: dict) -> dict:
    """Process a user query through the RAG pipeline."""
    rag_chain = st.session_state.get("rag_chain")
    
    if rag_chain is None:
        return {
            "answer": (
                "⚠️ **RAG system not loaded.** Please run the ingestion pipeline first:\n\n"
                "```bash\n"
                "# Step 1: Scrape SEC filings\n"
                "python -m src.scraper.sec_edgar_scraper\n\n"
                "# Step 2: Process and build vector store\n"
                "python -m src.pipeline.ingestion_pipeline\n\n"
                "# Step 3: Restart this app\n"
                "streamlit run src/app/streamlit_app.py\n"
                "```"
            ),
            "sources": [],
            "confidence": 0.0,
            "latency": 0.0,
        }
    
    start_time = time.time()
    
    response = rag_chain.query(
        question=question,
        filter_company=settings.get("filter_ticker"),
        top_k=settings.get("top_k", 5)
    )
    
    latency = time.time() - start_time
    
    return {
        "answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "latency": latency,
    }


def main():
    """Main application entry point."""
    init_session_state()
    
    # Load RAG system
    if not st.session_state.system_loaded:
        with st.spinner("Loading RAG system..."):
            rag_chain = load_rag_system()
            st.session_state.rag_chain = rag_chain
            st.session_state.system_loaded = True
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Main content area
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0;">
            <h1>📊 Financial Q&A Chatbot</h1>
            <p style="color: #6c757d; font-size: 1.1rem;">
                Ask questions about SEC 10-K filings from 20 major public companies (2020-2024)
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # System status
    if st.session_state.rag_chain is None:
        st.warning(
            "⚠️ RAG system not loaded. Run the ingestion pipeline to get started. "
            "See the sidebar for example questions once the system is ready."
        )
    else:
        st.success("✅ RAG system loaded and ready!")
    
    st.divider()
    
    # Chat interface
    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                if settings["show_sources"] and message["sources"]:
                    render_sources(message["sources"], settings["show_scores"])
                if "confidence" in message:
                    st.caption(render_confidence_badge(message["confidence"]))
                if "latency" in message:
                    st.caption(f"⏱️ Response time: {message['latency']:.2f}s")
    
    # Chat input
    if prompt := st.chat_input("Ask about any company's 10-K filing..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching financial documents..."):
                result = process_query(prompt, settings)
            
            st.markdown(result["answer"])
            
            if settings["show_sources"] and result["sources"]:
                render_sources(result["sources"], settings["show_scores"])
            
            if result["confidence"] > 0:
                st.caption(render_confidence_badge(result["confidence"]))
            
            if result["latency"] > 0:
                st.caption(f"⏱️ Response time: {result['latency']:.2f}s")
        
        # Save assistant message with metadata
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "latency": result["latency"],
        })
        
        # Update stats
        st.session_state.query_count += 1
        st.session_state.total_latency += result["latency"]


if __name__ == "__main__":
    main()
