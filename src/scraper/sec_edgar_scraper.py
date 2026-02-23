"""
SEC EDGAR 10-K Filing Scraper
=============================
Scrapes 10-K annual filings from SEC EDGAR using the free EDGAR API.
No API key required - uses the official SEC EDGAR full-text search and
submissions endpoints.

Features:
- Downloads 10-K filings for configurable list of companies
- Extracts key sections (Business, Risk Factors, MD&A, Financial Statements)
- Handles rate limiting per SEC fair access policy (10 requests/sec)
- Saves structured JSON output with metadata for RAG pipeline

Usage:
    python -m src.scraper.sec_edgar_scraper --companies config/target_companies.json
    python -m src.scraper.sec_edgar_scraper --tickers AAPL MSFT GOOGL --years 2020-2024
"""

import os
import re
import json
import time
import argparse
import requests
from pathlib import Path
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from bs4 import BeautifulSoup
import html2text
from loguru import logger
from tqdm import tqdm

# SEC EDGAR rate limit: 10 requests per second
SEC_RATE_LIMIT = 0.12  # seconds between requests
SEC_BASE_URL = "https://efts.sec.gov/LATEST/search-index"
EDGAR_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
EDGAR_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}"


@dataclass
class FilingMetadata:
    """Metadata for a single SEC filing."""
    ticker: str
    company_name: str
    cik: str
    filing_type: str
    filing_date: str
    period_of_report: str
    accession_number: str
    primary_document: str
    filing_url: str
    sector: str = ""


@dataclass
class FilingSection:
    """A parsed section from a 10-K filing."""
    section_name: str
    section_number: str
    content: str
    word_count: int


@dataclass
class ProcessedFiling:
    """Complete processed filing with metadata and extracted sections."""
    metadata: FilingMetadata
    sections: list
    raw_text: str
    total_word_count: int
    extraction_date: str


class SECEdgarScraper:
    """
    Scrapes and parses 10-K filings from SEC EDGAR.
    
    Uses the free SEC EDGAR API endpoints:
    1. Submissions API: Get filing history for a company by CIK
    2. Archives: Download the actual filing documents
    
    Complies with SEC fair access policy:
    - Declares User-Agent with company name and email
    - Rate limits to 10 requests/second
    """
    
    # 10-K section patterns for extraction
    SECTION_PATTERNS = {
        "1": {
            "name": "Business",
            "start_patterns": [
                r"(?i)item\s*1[\.\s]*[-–—]?\s*business",
                r"(?i)item\s*1\b(?!\d)"
            ],
            "end_patterns": [
                r"(?i)item\s*1a",
                r"(?i)item\s*2\b"
            ]
        },
        "1A": {
            "name": "Risk Factors",
            "start_patterns": [
                r"(?i)item\s*1a[\.\s]*[-–—]?\s*risk\s*factors",
                r"(?i)item\s*1a\b"
            ],
            "end_patterns": [
                r"(?i)item\s*1b",
                r"(?i)item\s*2\b"
            ]
        },
        "7": {
            "name": "Management Discussion and Analysis (MD&A)",
            "start_patterns": [
                r"(?i)item\s*7[\.\s]*[-–—]?\s*management",
                r"(?i)item\s*7\b(?!a)"
            ],
            "end_patterns": [
                r"(?i)item\s*7a",
                r"(?i)item\s*8\b"
            ]
        },
        "7A": {
            "name": "Quantitative and Qualitative Disclosures About Market Risk",
            "start_patterns": [
                r"(?i)item\s*7a[\.\s]*[-–—]?\s*quantitative",
                r"(?i)item\s*7a\b"
            ],
            "end_patterns": [
                r"(?i)item\s*8\b"
            ]
        },
        "8": {
            "name": "Financial Statements and Supplementary Data",
            "start_patterns": [
                r"(?i)item\s*8[\.\s]*[-–—]?\s*financial\s*statements",
                r"(?i)item\s*8\b"
            ],
            "end_patterns": [
                r"(?i)item\s*9\b"
            ]
        }
    }
    
    def __init__(
        self,
        company_name: str = "Financial RAG Research",
        email: str = "research@example.com",
        output_dir: str = "data/raw",
        rate_limit: float = SEC_RATE_LIMIT
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # SEC requires User-Agent declaration
        self.headers = {
            "User-Agent": f"{company_name} {email}",
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/json"
        }
        
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self.html_converter.ignore_emphasis = False
        self.html_converter.body_width = 0  # Don't wrap lines
        
        logger.info(f"SEC EDGAR Scraper initialized. Output: {self.output_dir}")
    
    def _rate_limit_wait(self):
        """Enforce SEC rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()
    
    def _make_request(self, url: str, accept_html: bool = False) -> Optional[requests.Response]:
        """Make a rate-limited request to SEC EDGAR."""
        self._rate_limit_wait()
        headers = self.headers.copy()
        if accept_html:
            headers["Accept"] = "text/html,application/xhtml+xml"
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None
    
    def get_company_filings(self, cik: str, filing_type: str = "10-K") -> list[dict]:
        """
        Get filing metadata from SEC EDGAR Submissions API.
        
        Args:
            cik: Central Index Key (with leading zeros to 10 digits)
            filing_type: Type of filing (e.g., "10-K")
            
        Returns:
            List of filing metadata dictionaries
        """
        # Pad CIK to 10 digits
        cik_padded = cik.lstrip("0").zfill(10)
        url = EDGAR_SUBMISSIONS_URL.format(cik=cik_padded)
        
        response = self._make_request(url)
        if not response:
            return []
        
        data = response.json()
        filings = []
        
        # Parse recent filings
        recent = data.get("filings", {}).get("recent", {})
        if not recent:
            return []
        
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        periods = recent.get("reportDate", [])
        
        for i, form in enumerate(forms):
            if form == filing_type:
                filings.append({
                    "form": form,
                    "filing_date": dates[i] if i < len(dates) else "",
                    "accession_number": accessions[i] if i < len(accessions) else "",
                    "primary_document": primary_docs[i] if i < len(primary_docs) else "",
                    "period_of_report": periods[i] if i < len(periods) else ""
                })
        
        # Also check additional filing files if they exist
        filing_files = data.get("filings", {}).get("files", [])
        for file_ref in filing_files:
            file_name = file_ref.get("name", "")
            if file_name:
                file_url = f"https://data.sec.gov/submissions/{file_name}"
                file_response = self._make_request(file_url)
                if file_response:
                    file_data = file_response.json()
                    file_forms = file_data.get("form", [])
                    file_dates = file_data.get("filingDate", [])
                    file_accessions = file_data.get("accessionNumber", [])
                    file_primary_docs = file_data.get("primaryDocument", [])
                    file_periods = file_data.get("reportDate", [])
                    
                    for i, form in enumerate(file_forms):
                        if form == filing_type:
                            filings.append({
                                "form": form,
                                "filing_date": file_dates[i] if i < len(file_dates) else "",
                                "accession_number": file_accessions[i] if i < len(file_accessions) else "",
                                "primary_document": file_primary_docs[i] if i < len(file_primary_docs) else "",
                                "period_of_report": file_periods[i] if i < len(file_periods) else ""
                            })
        
        logger.info(f"Found {len(filings)} {filing_type} filings for CIK {cik}")
        return filings
    
    def download_filing(self, cik: str, accession: str, primary_doc: str) -> Optional[str]:
        """
        Download the HTML content of a specific filing.
        
        Args:
            cik: Company CIK number
            accession: Accession number (formatted with dashes)
            primary_doc: Primary document filename
            
        Returns:
            HTML content of the filing
        """
        cik_clean = cik.lstrip("0")
        accession_clean = accession.replace("-", "")
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_doc}"
        
        response = self._make_request(url, accept_html=True)
        if response:
            return response.text
        return None
    
    def parse_filing_html(self, html_content: str) -> str:
        """
        Convert HTML filing to clean text.
        
        Uses BeautifulSoup for cleaning and html2text for conversion.
        Preserves section structure while removing boilerplate.
        """
        soup = BeautifulSoup(html_content, "lxml")
        
        # Remove script, style, and hidden elements
        for element in soup.find_all(["script", "style", "meta", "link"]):
            element.decompose()
        
        # Remove XBRL tags
        for tag in soup.find_all(re.compile(r"^(ix:|xbrli:)")):
            tag.unwrap()
        
        # Convert to text
        text = self.html_converter.handle(str(soup))
        
        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = text.strip()
        
        return text
    
    def extract_sections(self, text: str) -> list[FilingSection]:
        """
        Extract key sections from a 10-K filing text.
        
        Identifies and extracts Item 1 (Business), Item 1A (Risk Factors),
        Item 7 (MD&A), Item 7A (Market Risk), and Item 8 (Financial Statements).
        """
        sections = []
        
        for section_num, section_info in self.SECTION_PATTERNS.items():
            section_text = self._extract_section(
                text,
                section_info["start_patterns"],
                section_info["end_patterns"]
            )
            
            if section_text and len(section_text.split()) > 50:
                sections.append(FilingSection(
                    section_name=section_info["name"],
                    section_number=section_num,
                    content=section_text,
                    word_count=len(section_text.split())
                ))
        
        return sections
    
    def _extract_section(
        self,
        text: str,
        start_patterns: list[str],
        end_patterns: list[str]
    ) -> Optional[str]:
        """Extract text between start and end patterns."""
        start_match = None
        for pattern in start_patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                # Use the last match (often the actual content, not TOC)
                start_match = matches[-1]
                break
        
        if not start_match:
            return None
        
        end_match = None
        search_start = start_match.end()
        for pattern in end_patterns:
            match = re.search(pattern, text[search_start:])
            if match:
                if end_match is None or match.start() < end_match.start():
                    end_match = match
                break
        
        if end_match:
            section_text = text[start_match.start():search_start + end_match.start()]
        else:
            # Take up to 50,000 chars if no end pattern found
            section_text = text[start_match.start():start_match.start() + 50000]
        
        return section_text.strip()
    
    def scrape_company(
        self,
        ticker: str,
        company_name: str,
        cik: str,
        sector: str = "",
        filing_type: str = "10-K",
        year_start: int = 2020,
        year_end: int = 2024,
        max_filings: int = 5
    ) -> list[ProcessedFiling]:
        """
        Scrape all 10-K filings for a single company.
        
        Args:
            ticker: Stock ticker symbol
            company_name: Full company name
            cik: SEC CIK number
            sector: Industry sector
            filing_type: Filing type to scrape
            year_start: Start year for filtering
            year_end: End year for filtering
            max_filings: Maximum number of filings to download
            
        Returns:
            List of ProcessedFiling objects
        """
        logger.info(f"Scraping {filing_type} filings for {ticker} ({company_name})")
        
        # Get filing list
        filings_meta = self.get_company_filings(cik, filing_type)
        
        # Filter by year range
        filtered_filings = []
        for filing in filings_meta:
            filing_date = filing.get("filing_date", "")
            if filing_date:
                year = int(filing_date[:4])
                if year_start <= year <= year_end:
                    filtered_filings.append(filing)
        
        # Limit number of filings
        filtered_filings = filtered_filings[:max_filings]
        
        logger.info(f"Processing {len(filtered_filings)} filings for {ticker}")
        
        processed_filings = []
        for filing in filtered_filings:
            try:
                # Build metadata
                metadata = FilingMetadata(
                    ticker=ticker,
                    company_name=company_name,
                    cik=cik,
                    filing_type=filing_type,
                    filing_date=filing["filing_date"],
                    period_of_report=filing.get("period_of_report", ""),
                    accession_number=filing["accession_number"],
                    primary_document=filing["primary_document"],
                    filing_url=f"https://www.sec.gov/Archives/edgar/data/"
                               f"{cik.lstrip('0')}/{filing['accession_number'].replace('-', '')}/"
                               f"{filing['primary_document']}",
                    sector=sector
                )
                
                # Download filing
                html_content = self.download_filing(
                    cik, filing["accession_number"], filing["primary_document"]
                )
                
                if not html_content:
                    logger.warning(f"Failed to download filing: {metadata.accession_number}")
                    continue
                
                # Parse HTML to text
                raw_text = self.parse_filing_html(html_content)
                
                if len(raw_text.split()) < 100:
                    logger.warning(f"Filing too short after parsing: {metadata.accession_number}")
                    continue
                
                # Extract sections
                sections = self.extract_sections(raw_text)
                
                processed = ProcessedFiling(
                    metadata=metadata,
                    sections=[asdict(s) for s in sections],
                    raw_text=raw_text,
                    total_word_count=len(raw_text.split()),
                    extraction_date=datetime.now().isoformat()
                )
                
                processed_filings.append(processed)
                
                # Save individual filing
                self._save_filing(processed)
                
                logger.info(
                    f"  ✓ {ticker} {filing['filing_date']}: "
                    f"{len(sections)} sections, "
                    f"{len(raw_text.split()):,} words"
                )
                
            except Exception as e:
                logger.error(f"Error processing filing for {ticker}: {e}")
                continue
        
        return processed_filings
    
    def _save_filing(self, filing: ProcessedFiling):
        """Save processed filing to JSON."""
        ticker = filing.metadata.ticker
        date = filing.metadata.filing_date
        
        output_file = self.output_dir / f"{ticker}_{date}_10K.json"
        
        filing_dict = {
            "metadata": asdict(filing.metadata),
            "sections": filing.sections,
            "raw_text": filing.raw_text,
            "total_word_count": filing.total_word_count,
            "extraction_date": filing.extraction_date
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filing_dict, f, indent=2, ensure_ascii=False)
    
    def scrape_all(
        self,
        companies: list[dict],
        filing_type: str = "10-K",
        year_start: int = 2020,
        year_end: int = 2024,
        max_filings_per_company: int = 5
    ) -> dict:
        """
        Scrape filings for all companies in the list.
        
        Args:
            companies: List of company dicts with ticker, name, cik, sector
            filing_type: Filing type to scrape
            year_start: Start year
            year_end: End year
            max_filings_per_company: Max filings per company
            
        Returns:
            Summary statistics dictionary
        """
        total_filings = 0
        total_sections = 0
        total_words = 0
        failed_companies = []
        
        logger.info(f"Starting scrape of {len(companies)} companies ({year_start}-{year_end})")
        
        for company in tqdm(companies, desc="Scraping companies"):
            try:
                processed = self.scrape_company(
                    ticker=company["ticker"],
                    company_name=company["name"],
                    cik=company["cik"],
                    sector=company.get("sector", ""),
                    filing_type=filing_type,
                    year_start=year_start,
                    year_end=year_end,
                    max_filings=max_filings_per_company
                )
                
                total_filings += len(processed)
                for filing in processed:
                    total_sections += len(filing.sections)
                    total_words += filing.total_word_count
                    
            except Exception as e:
                logger.error(f"Failed to scrape {company['ticker']}: {e}")
                failed_companies.append(company["ticker"])
        
        summary = {
            "total_companies": len(companies),
            "total_filings_downloaded": total_filings,
            "total_sections_extracted": total_sections,
            "total_words": total_words,
            "failed_companies": failed_companies,
            "output_directory": str(self.output_dir),
            "scrape_date": datetime.now().isoformat()
        }
        
        # Save summary
        with open(self.output_dir / "scrape_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Scraping Complete!")
        logger.info(f"  Companies: {len(companies)} ({len(failed_companies)} failed)")
        logger.info(f"  Filings: {total_filings}")
        logger.info(f"  Sections: {total_sections}")
        logger.info(f"  Total Words: {total_words:,}")
        logger.info(f"{'='*60}")
        
        return summary


def main():
    parser = argparse.ArgumentParser(description="SEC EDGAR 10-K Filing Scraper")
    parser.add_argument(
        "--companies", type=str, default="config/target_companies.json",
        help="Path to companies JSON config"
    )
    parser.add_argument(
        "--tickers", nargs="+", type=str, default=None,
        help="Specific tickers to scrape (overrides config)"
    )
    parser.add_argument(
        "--years", type=str, default="2020-2024",
        help="Year range (e.g., 2020-2024)"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory for scraped filings"
    )
    parser.add_argument(
        "--max-filings", type=int, default=5,
        help="Max filings per company"
    )
    parser.add_argument(
        "--email", type=str, default="research@example.com",
        help="Email for SEC User-Agent"
    )
    
    args = parser.parse_args()
    
    # Parse year range
    year_start, year_end = map(int, args.years.split("-"))
    
    # Load companies
    with open(args.companies, "r") as f:
        config = json.load(f)
    
    companies = config["companies"]
    
    # Filter by tickers if specified
    if args.tickers:
        companies = [c for c in companies if c["ticker"] in args.tickers]
    
    # Initialize scraper
    scraper = SECEdgarScraper(
        email=args.email,
        output_dir=args.output
    )
    
    # Run scraping
    summary = scraper.scrape_all(
        companies=companies,
        year_start=year_start,
        year_end=year_end,
        max_filings_per_company=args.max_filings
    )
    
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
