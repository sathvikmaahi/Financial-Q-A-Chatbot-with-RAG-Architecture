"""
Chunking Engine
===============
Implements multiple chunking strategies for financial documents:
1. Recursive Character Splitting - preserves document structure
2. Semantic Chunking - groups sentences by semantic similarity
3. Section-Aware Chunking - respects 10-K section boundaries

Each chunk includes rich metadata for filtering and attribution.
"""

import re
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

import numpy as np
from loguru import logger


@dataclass
class DocumentChunk:
    """A single chunk from a financial document with metadata."""
    chunk_id: str
    content: str
    metadata: dict
    word_count: int
    char_count: int
    
    def to_dict(self):
        return asdict(self)


class ChunkingEngine:
    """
    Multi-strategy document chunking engine optimized for financial filings.
    
    Strategies:
    1. recursive: LangChain-style recursive character splitting
    2. semantic: Groups sentences by embedding similarity
    3. section_aware: Respects 10-K section boundaries, then applies recursive splitting
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
        max_chunk_size: int = 2000,
        strategy: str = "section_aware"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.strategy = strategy
        
        self.separators = ["\n\n", "\n", ". ", "; ", ", ", " "]
        
        logger.info(
            f"ChunkingEngine initialized: strategy={strategy}, "
            f"chunk_size={chunk_size}, overlap={chunk_overlap}"
        )
    
    def chunk_filing(self, filing_data: dict) -> list[DocumentChunk]:
        """
        Chunk a single filing based on the selected strategy.
        
        Args:
            filing_data: Dictionary with metadata, sections, and raw_text
            
        Returns:
            List of DocumentChunk objects with metadata
        """
        metadata = filing_data.get("metadata", {})
        sections = filing_data.get("sections", [])
        raw_text = filing_data.get("raw_text", "")
        
        if self.strategy == "section_aware" and sections:
            chunks = self._section_aware_chunk(sections, metadata)
        elif self.strategy == "semantic":
            chunks = self._semantic_chunk(raw_text, metadata)
        else:
            chunks = self._recursive_chunk(raw_text, metadata)
        
        logger.info(
            f"Chunked {metadata.get('ticker', 'Unknown')} "
            f"{metadata.get('filing_date', '')}: {len(chunks)} chunks"
        )
        
        return chunks
    
    def _section_aware_chunk(
        self,
        sections: list[dict],
        base_metadata: dict
    ) -> list[DocumentChunk]:
        """
        Chunk respecting 10-K section boundaries.
        Each section is individually chunked with recursive splitting.
        """
        all_chunks = []
        
        for section in sections:
            section_name = section.get("section_name", "Unknown")
            section_num = section.get("section_number", "")
            content = section.get("content", "")
            
            if not content or len(content.split()) < 20:
                continue
            
            # Build section-specific metadata
            section_metadata = {
                **base_metadata,
                "section_name": section_name,
                "section_number": section_num,
                "source_type": "10-K_section"
            }
            
            # Apply recursive splitting within each section
            text_chunks = self._recursive_split(content)
            
            for i, text in enumerate(text_chunks):
                chunk_metadata = {
                    **section_metadata,
                    "chunk_index": i,
                    "total_section_chunks": len(text_chunks)
                }
                
                chunk_id = self._generate_chunk_id(text, chunk_metadata)
                
                all_chunks.append(DocumentChunk(
                    chunk_id=chunk_id,
                    content=text,
                    metadata=chunk_metadata,
                    word_count=len(text.split()),
                    char_count=len(text)
                ))
        
        return all_chunks
    
    def _recursive_chunk(
        self,
        text: str,
        base_metadata: dict
    ) -> list[DocumentChunk]:
        """
        Recursive character splitting (LangChain-style).
        """
        text_chunks = self._recursive_split(text)
        
        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "source_type": "10-K_full"
            }
            
            chunk_id = self._generate_chunk_id(chunk_text, metadata)
            
            chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=metadata,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))
        
        return chunks
    
    def _recursive_split(self, text: str) -> list[str]:
        """
        Recursively split text using hierarchical separators.
        Mimics LangChain's RecursiveCharacterTextSplitter.
        """
        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_chunk_size else []
        
        return self._split_with_separators(text, self.separators)
    
    def _split_with_separators(self, text: str, separators: list[str]) -> list[str]:
        """Split text using the first applicable separator."""
        if not separators:
            # No more separators — hard split by character count
            return self._hard_split(text)
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        splits = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for split in splits:
            candidate = (current_chunk + separator + split).strip() if current_chunk else split.strip()
            
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
            else:
                if current_chunk:
                    if len(current_chunk) >= self.min_chunk_size:
                        chunks.append(current_chunk)
                    
                    # Start new chunk with overlap
                    overlap_text = self._get_overlap(current_chunk)
                    current_chunk = (overlap_text + " " + split).strip() if overlap_text else split.strip()
                else:
                    # Single split is too large — recurse with next separator
                    sub_chunks = self._split_with_separators(split, remaining_separators)
                    chunks.extend(sub_chunks)
                    current_chunk = ""
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        return chunks
    
    def _hard_split(self, text: str) -> list[str]:
        """Last resort: split by character count."""
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if len(chunk) >= self.min_chunk_size:
                chunks.append(chunk)
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """Get the last `chunk_overlap` characters for context continuity."""
        if len(text) <= self.chunk_overlap:
            return text
        return text[-self.chunk_overlap:]
    
    def _semantic_chunk(
        self,
        text: str,
        base_metadata: dict,
        buffer_size: int = 3,
        breakpoint_percentile: float = 85
    ) -> list[DocumentChunk]:
        """
        Semantic chunking: group sentences by semantic similarity.
        Splits at points where consecutive sentence similarity drops below threshold.
        
        Note: Requires embedding model. Falls back to recursive if not available.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        if len(sentences) < buffer_size * 2:
            return self._recursive_chunk(text, base_metadata)
        
        # Group sentences into initial buffers
        buffers = []
        for i in range(len(sentences) - buffer_size + 1):
            buffer = " ".join(sentences[i:i + buffer_size])
            buffers.append(buffer)
        
        # Without an embedding model, use a simpler heuristic:
        # Split at paragraph boundaries and large gaps
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            candidate = (current_chunk + " " + sentence).strip() if current_chunk else sentence
            
            if len(candidate) > self.max_chunk_size:
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = sentence
            elif len(candidate) > self.chunk_size and self._is_break_point(sentence):
                if current_chunk and len(current_chunk) >= self.min_chunk_size:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk = candidate
        
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append(current_chunk)
        
        # Convert to DocumentChunks
        result = []
        for i, chunk_text in enumerate(chunks):
            metadata = {
                **base_metadata,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source_type": "10-K_semantic",
                "chunking_strategy": "semantic"
            }
            
            chunk_id = self._generate_chunk_id(chunk_text, metadata)
            result.append(DocumentChunk(
                chunk_id=chunk_id,
                content=chunk_text,
                metadata=metadata,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text)
            ))
        
        return result
    
    def _is_break_point(self, sentence: str) -> bool:
        """Heuristic: detect natural breakpoints in financial text."""
        break_indicators = [
            r"^(Item|ITEM)\s+\d",
            r"^(Note|NOTE)\s+\d",
            r"^(Table|TABLE)\s+",
            r"^(Part|PART)\s+[IVX]",
            r"^\d+\.\s+[A-Z]",
            r"^(Overview|Summary|Conclusion|Results)",
            r"^(Revenue|Income|Expense|Asset|Liabilit)",
        ]
        
        for pattern in break_indicators:
            if re.match(pattern, sentence.strip()):
                return True
        return False
    
    @staticmethod
    def _generate_chunk_id(content: str, metadata: dict) -> str:
        """Generate a deterministic unique ID for a chunk."""
        id_string = f"{metadata.get('ticker', '')}-{metadata.get('filing_date', '')}-{content[:100]}"
        return hashlib.md5(id_string.encode()).hexdigest()
    
    def process_all_filings(
        self,
        input_dir: str,
        output_path: str
    ) -> list[dict]:
        """
        Process all scraped filings in a directory.
        
        Args:
            input_dir: Directory containing scraped filing JSON files
            output_path: Output JSONL file path for chunks
            
        Returns:
            List of chunk dictionaries
        """
        input_path = Path(input_dir)
        filing_files = list(input_path.glob("*_10K.json"))
        
        logger.info(f"Processing {len(filing_files)} filing files from {input_dir}")
        
        all_chunks = []
        
        for filing_file in filing_files:
            with open(filing_file, "r", encoding="utf-8") as f:
                filing_data = json.load(f)
            
            chunks = self.chunk_filing(filing_data)
            all_chunks.extend(chunks)
        
        # Save chunks as JSONL
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w", encoding="utf-8") as f:
            for chunk in all_chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
        
        logger.info(f"Total chunks generated: {len(all_chunks)}")
        logger.info(f"Chunks saved to: {output_path}")
        
        # Print summary statistics
        word_counts = [c.word_count for c in all_chunks]
        if word_counts:
            logger.info(f"Chunk stats - Mean: {np.mean(word_counts):.0f} words, "
                       f"Median: {np.median(word_counts):.0f}, "
                       f"Min: {min(word_counts)}, Max: {max(word_counts)}")
        
        return [c.to_dict() for c in all_chunks]
