"""
RAG Chain Orchestration
========================
Implements the full Retrieval-Augmented Generation chain using LangChain.

Pipeline:
1. Query Processing → Entity extraction, expansion
2. Hybrid Retrieval → BM25 + FAISS with RRF
3. Context Assembly → Deduplicate, order, truncate to context window
4. LLM Generation → Answer with citations
5. Post-processing → Source attribution, confidence scoring

Supports both HuggingFace (local) and OpenAI API backends.
"""

import os
import re
from typing import Optional
from dataclasses import dataclass

from loguru import logger


@dataclass
class RAGResponse:
    """Structured response from the RAG pipeline."""
    answer: str
    sources: list[dict]
    confidence: float
    query_info: dict
    retrieval_scores: list[float]


# System prompt for the financial Q&A assistant
SYSTEM_PROMPT = """You are an expert financial analyst assistant. Your role is to answer questions
about publicly-traded companies based on their SEC 10-K annual filing documents.

IMPORTANT RULES:
1. ONLY use information from the provided context (retrieved from 10-K filings).
2. Always cite which company and filing year your information comes from.
3. If the context doesn't contain enough information to answer, say so clearly.
4. Use specific numbers and data points when available.
5. For comparative questions, present data side-by-side clearly.
6. Never make up financial data. If you're unsure, say "Based on the available filings..."

When referencing sources, use format: [Company Ticker, FY Year, Section Name]
Example: [AAPL, FY2023, Item 7 - MD&A]
"""

ANSWER_PROMPT_TEMPLATE = """Based on the following context from SEC 10-K filings, answer the user's question.

CONTEXT:
{context}

USER QUESTION: {question}

Provide a detailed, accurate answer with specific citations to the source filings.
If the context is insufficient, clearly state what information is missing.

ANSWER:"""


class RAGChain:
    """
    Complete RAG pipeline for financial Q&A.
    
    Integrates:
    - HybridRetriever for document retrieval
    - QueryProcessor for query enhancement
    - LLM for answer generation (HuggingFace or OpenAI)
    """
    
    def __init__(
        self,
        retriever=None,
        llm_provider: str = "huggingface",
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        context_window: int = 4096
    ):
        self.retriever = retriever
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_window = context_window
        self.llm = None
        
        # Initialize query processor
        from src.retrieval.hybrid_retriever import QueryProcessor
        self.query_processor = QueryProcessor()
        
        logger.info(f"RAGChain initialized: provider={llm_provider}, model={model_name}")
    
    def _init_llm(self):
        """Lazy-initialize the LLM."""
        if self.llm is not None:
            return
        
        if self.llm_provider == "openai":
            self._init_openai_llm()
        elif self.llm_provider == "huggingface":
            self._init_hf_llm()
        else:
            logger.info("No LLM provider configured. Using template-based responses.")
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM via LangChain."""
        try:
            from langchain_community.chat_models import ChatOpenAI
            
            self.llm = ChatOpenAI(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            logger.info(f"OpenAI LLM initialized: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI LLM: {e}")
            self.llm = None
    
    def _init_hf_llm(self):
        """Initialize HuggingFace LLM via LangChain."""
        try:
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline
            import torch
            
            device = 0 if torch.cuda.is_available() else -1
            
            pipe = pipeline(
                "text-generation",
                model=self.model_name,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                device=device,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.llm = HuggingFacePipeline(pipeline=pipe)
            logger.info(f"HuggingFace LLM initialized: {self.model_name}")
        except Exception as e:
            logger.warning(f"Failed to initialize HuggingFace LLM: {e}")
            self.llm = None
    
    def query(
        self,
        question: str,
        filter_company: Optional[str] = None,
        filter_year: Optional[str] = None,
        top_k: int = 5
    ) -> RAGResponse:
        """
        Process a user question through the full RAG pipeline.
        
        Args:
            question: User's natural language question
            filter_company: Optional ticker to filter results
            filter_year: Optional year to filter results
            top_k: Number of chunks to retrieve
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Step 1: Process query
        query_info = self.query_processor.process_query(question)
        
        # Build metadata filter
        filter_metadata = {}
        if filter_company:
            filter_metadata["ticker"] = filter_company
        
        # Auto-detect company from query entities
        if not filter_company and query_info["detected_entities"]["tickers"]:
            detected_ticker = query_info["detected_entities"]["tickers"][0]
            # Don't auto-filter, but note it
            query_info["detected_company"] = detected_ticker
        
        # Step 2: Retrieve relevant documents
        if self.retriever:
            retrieval_results = self.retriever.retrieve(
                query=query_info.get("expanded_query", question),
                top_k=top_k,
                filter_metadata=filter_metadata if filter_metadata else None
            )
        else:
            # Mock results for development
            retrieval_results = []
        
        # Step 3: Assemble context
        context, sources = self._assemble_context(retrieval_results)
        
        # Step 4: Generate answer
        answer = self._generate_answer(question, context)
        
        # Step 5: Calculate confidence
        confidence = self._calculate_confidence(retrieval_results, answer)
        
        return RAGResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            query_info=query_info,
            retrieval_scores=[r.fused_score for r in retrieval_results]
        )
    
    def _assemble_context(self, retrieval_results: list) -> tuple[str, list[dict]]:
        """
        Assemble context string from retrieval results.
        Deduplicates, orders by relevance, and truncates to context window.
        """
        if not retrieval_results:
            return "No relevant documents found.", []
        
        context_parts = []
        sources = []
        seen_content = set()
        total_tokens = 0
        max_context_tokens = self.context_window - self.max_tokens - 500  # Reserve for prompt
        
        for result in retrieval_results:
            # Deduplicate
            content_hash = hash(result.content[:200])
            if content_hash in seen_content:
                continue
            seen_content.add(content_hash)
            
            # Estimate tokens (rough: 1 token ≈ 4 chars)
            est_tokens = len(result.content) // 4
            if total_tokens + est_tokens > max_context_tokens:
                break
            
            # Build source reference
            metadata = result.metadata
            source_ref = (
                f"[{metadata.get('ticker', 'Unknown')}, "
                f"{metadata.get('filing_date', 'N/A')[:4]}, "
                f"{metadata.get('section_name', 'General')}]"
            )
            
            context_parts.append(f"--- Source: {source_ref} ---\n{result.content}")
            
            sources.append({
                "ticker": metadata.get("ticker", ""),
                "company_name": metadata.get("company_name", ""),
                "filing_date": metadata.get("filing_date", ""),
                "section_name": metadata.get("section_name", ""),
                "section_number": metadata.get("section_number", ""),
                "relevance_score": result.fused_score,
                "content_preview": result.content[:200] + "..."
            })
            
            total_tokens += est_tokens
        
        context = "\n\n".join(context_parts)
        return context, sources
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM or template-based fallback."""
        self._init_llm()
        
        prompt = ANSWER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        if self.llm:
            try:
                if hasattr(self.llm, 'invoke'):
                    response = self.llm.invoke(prompt)
                    if hasattr(response, 'content'):
                        return response.content
                    return str(response)
                else:
                    return self.llm(prompt)
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return self._template_answer(question, context)
        else:
            return self._template_answer(question, context)
    
    def _template_answer(self, question: str, context: str) -> str:
        """
        Template-based answer generation (fallback when no LLM available).
        Extracts key sentences from context that are relevant to the question.
        """
        if context == "No relevant documents found.":
            return (
                "I couldn't find relevant information in the available 10-K filings "
                "to answer your question. Please try rephrasing or specifying a company."
            )
        
        # Extract source references
        source_refs = re.findall(r'\[([^\]]+)\]', context)
        
        # Find sentences containing key terms from the question
        question_terms = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
        stop_words = {"what", "when", "where", "which", "how", "does", "the",
                      "and", "for", "that", "this", "with", "from", "are", "was",
                      "been", "being", "have", "has", "had", "will", "would",
                      "could", "should", "their", "about", "into", "over",
                      "also", "such", "than", "these", "those", "each",
                      "other", "some", "most", "more", "any", "its", "our",
                      "may", "can", "including", "related", "certain"}
        question_terms -= stop_words
        
        # Split context into source blocks and sentences
        blocks = context.split("--- Source:")
        scored_sentences = []
        
        for block in blocks:
            if not block.strip():
                continue
            sentences = re.split(r'(?<=[.!?])\s+', block)
            for sentence in sentences:
                clean = sentence.strip()
                # Skip source headers, very short, or very long sentences
                if not clean or len(clean) < 20 or clean.startswith("---"):
                    continue
                sentence_lower = clean.lower()
                # Score by number of matching question terms
                matching = sum(1 for term in question_terms if term in sentence_lower)
                # Bonus for sentences with numbers (financial data)
                has_numbers = bool(re.search(r'\$[\d,.]+|\d+[.,]\d+\s*(%|billion|million|thousand)', sentence_lower))
                score = matching + (0.5 if has_numbers else 0)
                if score > 0:
                    scored_sentences.append((score, clean))
        
        # Sort by relevance score (descending)
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        
        # Take the best sentences
        top_sentences = [s for _, s in scored_sentences[:8]]
        
        if top_sentences:
            answer = "**Based on the available 10-K filings:**\n\n"
            # Deduplicate while preserving order
            seen = set()
            unique = []
            for s in top_sentences:
                key = s[:80].lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            answer += "\n\n".join(f"• {s}" for s in unique[:5])
            if source_refs:
                unique_sources = list(dict.fromkeys(source_refs[:3]))
                answer += f"\n\n**Sources:** {', '.join(unique_sources)}"
            return answer
        else:
            # Even if no keyword matches, show the most relevant retrieved content
            # Split all context into paragraphs and show the first few
            paragraphs = [p.strip() for p in context.split("\n\n") if p.strip() and not p.strip().startswith("--- Source:")]
            if paragraphs:
                answer = "**From the retrieved 10-K filing documents:**\n\n"
                answer += "\n\n".join(paragraphs[:3])
                if source_refs:
                    unique_sources = list(dict.fromkeys(source_refs[:3]))
                    answer += f"\n\n**Sources:** {', '.join(unique_sources)}"
                return answer
            return (
                "I found related documents but couldn't extract a specific answer. "
                "Try asking a more specific question about a company's revenue, risk factors, "
                "or business operations.\n\n"
                f"Documents retrieved from: {', '.join(set(source_refs[:3]))}"
            )
    
    def _calculate_confidence(self, results: list, answer: str) -> float:
        """
        Calculate confidence score based on retrieval quality.
        
        Factors:
        - Average retrieval score
        - Number of unique sources
        - Answer length (very short = low confidence)
        """
        if not results:
            return 0.0
        
        # Retrieval score component (0-0.5)
        avg_score = sum(r.fused_score for r in results) / len(results)
        retrieval_confidence = min(avg_score * 50, 0.5)
        
        # Source diversity component (0-0.3)
        unique_tickers = len(set(r.metadata.get("ticker", "") for r in results))
        diversity_confidence = min(unique_tickers / 5, 0.3)
        
        # Answer quality component (0-0.2)
        answer_words = len(answer.split())
        if answer_words > 50:
            answer_confidence = 0.2
        elif answer_words > 20:
            answer_confidence = 0.1
        else:
            answer_confidence = 0.05
        
        return min(retrieval_confidence + diversity_confidence + answer_confidence, 1.0)
