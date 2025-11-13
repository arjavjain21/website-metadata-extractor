#!/usr/bin/env python3
"""
Consolidated Domain Meta Extractor - Single File Streamlit Application
A powerful tool for extracting meta titles and descriptions from domain names.
"""

import streamlit as st
import pandas as pd
import asyncio
import aiohttp
import time
from datetime import datetime
import tempfile
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, List
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin
from lxml import html, etree
from bs4 import BeautifulSoup
import re
import json
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Domain Meta Extractor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .upload-container {
        border: 2px dashed #ccc;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .progress-container {
        margin: 1rem 0;
    }
    .stats-box {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@dataclass
class ExtractionResult:
    """Result of a meta extraction attempt."""
    domain: str
    title: Optional[str] = None
    description: Optional[str] = None
    method: str = "unknown"
    status_code: Optional[int] = None
    extraction_time: Optional[float] = None
    error_message: Optional[str] = None
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV output."""
        return {
            'domain': self.domain,
            'meta_title': self.title or '',
            'meta_description': self.description or '',
            'extraction_method': self.method,
            'status_code': self.status_code or 0,
            'extraction_time': self.extraction_time or 0,
            'error_message': self.error_message or ''
        }


class DomainUtils:
    """Utility class for domain operations."""

    @staticmethod
    def normalize_domain(domain: str) -> Optional[str]:
        """Normalize a domain name."""
        if not domain:
            return None

        try:
            domain = domain.strip()
            if domain.startswith(('http://', 'https://')):
                parsed = urlparse(domain)
                domain = parsed.netloc
            else:
                parsed = urlparse(f'http://{domain}')
                domain = parsed.netloc

            domain = domain.lower()
            domain = domain.split(':')[0]
            domain = domain.rstrip('.')

            if DomainUtils.is_valid_domain(domain):
                return domain
            else:
                return None

        except Exception:
            return None

    @staticmethod
    def is_valid_domain(domain: str) -> bool:
        """Validate if a domain name is properly formatted."""
        if not domain:
            return False

        domain_pattern = re.compile(
            r'^(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)*[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?$'
        )

        if not domain_pattern.match(domain):
            return False

        parts = domain.split('.')
        if len(parts) < 2 or len(parts[-1]) < 2:
            return False

        if len(domain) > 253:
            return False

        for part in parts:
            if len(part) > 63 or part.startswith('-') or part.endswith('-'):
                return False

        return True


class BaseExtractor:
    """Base class for all meta extractors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger

        self.min_title_length = config.get('extraction', {}).get('min_title_length', 3)
        self.max_title_length = config.get('extraction', {}).get('max_title_length', 200)
        self.min_description_length = config.get('extraction', {}).get('min_description_length', 10)
        self.max_description_length = config.get('extraction', {}).get('max_description_length', 500)
        self.max_content_length = config.get('extraction', {}).get('max_content_length', 1048576)

    def validate_title(self, title: str) -> bool:
        """Validate if a title meets quality criteria."""
        if not title:
            return False

        title = title.strip()
        length = len(title)

        return (self.min_title_length <= length <= self.max_title_length and
                not title.isdigit() and
                not title.isspace())

    def validate_description(self, description: str) -> bool:
        """Validate if a description meets quality criteria."""
        if not description:
            return False

        description = description.strip()
        length = len(description)

        return (self.min_description_length <= length <= self.max_description_length and
                not description.isdigit() and
                not description.isspace())

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""

        text = ' '.join(text.split())
        unwanted_patterns = ['\n', '\r', '\t']
        for pattern in unwanted_patterns:
            text = text.replace(pattern, ' ')

        while '  ' in text:
            text = text.replace('  ', ' ')

        return text.strip()

    def create_success_result(self, domain: str, title: str, description: str,
                            method: str, extraction_time: float,
                            status_code: int = 200) -> ExtractionResult:
        """Create a successful extraction result."""
        return ExtractionResult(
            domain=domain,
            title=self.clean_text(title) if title else None,
            description=self.clean_text(description) if description else None,
            method=method,
            status_code=status_code,
            extraction_time=extraction_time,
            success=True
        )

    def create_error_result(self, domain: str, error_message: str,
                          method: str, extraction_time: float,
                          status_code: Optional[int] = None) -> ExtractionResult:
        """Create an error extraction result."""
        return ExtractionResult(
            domain=domain,
            method=method,
            status_code=status_code,
            extraction_time=extraction_time,
            error_message=error_message,
            success=False
        )


class HTMLExtractor(BaseExtractor):
    """Fast HTML-based meta extractor using lxml."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('performance', {}).get('timeout', 30)
        self.follow_redirects = config.get('advanced', {}).get('follow_redirects', True)
        self.max_redirects = config.get('advanced', {}).get('max_redirects', 10)
        self.verify_ssl = config.get('advanced', {}).get('verify_ssl', True)

    async def extract(self, domain: str, **kwargs) -> ExtractionResult:
        """Extract meta information using HTML parsing."""
        start_time = asyncio.get_event_loop().time()
        session = kwargs.get('session')

        try:
            url = f'https://{domain.strip()}'

            async with session.get(
                url,
                allow_redirects=self.follow_redirects,
                max_redirects=self.max_redirects
            ) as response:
                status_code = response.status
                content_type = response.headers.get('content-type', '').lower()

                if not self._is_valid_response(response.status, content_type):
                    return self.create_error_result(
                        domain=domain,
                        error_message=f"Invalid response: {response.status} {content_type}",
                        method="html_extractor",
                        extraction_time=asyncio.get_event_loop().time() - start_time,
                        status_code=response.status
                    )

                content = await self._read_content_safely(response)
                if not content:
                    return self.create_error_result(
                        domain=domain,
                        error_message="No content received",
                        method="html_extractor",
                        extraction_time=asyncio.get_event_loop().time() - start_time,
                        status_code=response.status
                    )

                title, description = self._extract_from_html(content, url)
                extraction_time = asyncio.get_event_loop().time() - start_time

                if title or description:
                    return self.create_success_result(
                        domain=domain,
                        title=title,
                        description=description,
                        method="html_extractor",
                        extraction_time=extraction_time,
                        status_code=status_code
                    )
                else:
                    return self.create_error_result(
                        domain=domain,
                        error_message="No meta information found in HTML",
                        method="html_extractor",
                        extraction_time=extraction_time,
                        status_code=status_code
                    )

        except Exception as e:
            return self.create_error_result(
                domain=domain,
                error_message=f"HTML extraction error: {str(e)}",
                method="html_extractor",
                extraction_time=asyncio.get_event_loop().time() - start_time
            )

    def _is_valid_response(self, status_code: int, content_type: str) -> bool:
        """Check if the response is valid for HTML parsing."""
        if status_code >= 400:
            return False

        html_types = ['text/html', 'text/xhtml', 'application/xhtml+xml']
        return any(html_type in content_type for html_type in html_types)

    async def _read_content_safely(self, response: aiohttp.ClientResponse) -> Optional[str]:
        """Safely read response content with size limits."""
        try:
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.max_content_length:
                return None

            content = await response.text()
            if len(content.encode('utf-8')) > self.max_content_length:
                return None

            return content

        except Exception:
            return None

    def _extract_from_html(self, content: str, base_url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract title and description from HTML content."""
        try:
            tree = html.fromstring(content)
            title = self._extract_title(tree)
            description = self._extract_description(tree)
            return title, description

        except etree.ParserError:
            return self._extract_from_partial_html(content)
        except Exception:
            return None, None

    def _extract_title(self, tree) -> Optional[str]:
        """Extract title from HTML tree with multiple fallbacks."""
        # Primary: title tag
        title_elem = tree.find('.//title')
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            if self.validate_title(title):
                return title

        # Secondary: h1 tag
        h1_elem = tree.find('.//h1')
        if h1_elem is not None and h1_elem.text:
            title = h1_elem.text.strip()
            if self.validate_title(title):
                return title

        # Tertiary: meta property og:title
        og_title = tree.xpath('.//meta[@property="og:title"]/@content')
        if og_title and og_title[0]:
            title = og_title[0].strip()
            if self.validate_title(title):
                return title

        return None

    def _extract_description(self, tree) -> Optional[str]:
        """Extract description from HTML tree with multiple fallbacks."""
        # Primary: meta name description
        meta_desc = tree.xpath('.//meta[@name="description"]/@content')
        if meta_desc and meta_desc[0]:
            desc = meta_desc[0].strip()
            if self.validate_description(desc):
                return desc

        # Secondary: meta property og:description
        og_desc = tree.xpath('.//meta[@property="og:description"]/@content')
        if og_desc and og_desc[0]:
            desc = og_desc[0].strip()
            if self.validate_description(desc):
                return desc

        return None

    def _extract_from_partial_html(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract from potentially malformed HTML using regex fallbacks."""
        title = None
        description = None

        # Extract title using regex
        title_match = re.search(r'<title[^>]*>(.*?)</title>', content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = self.clean_text(title_match.group(1))
            if not self.validate_title(title):
                title = None

        # Extract description using regex
        desc_patterns = [
            r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']',
            r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*name=["\']description["\']',
            r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
        ]

        for pattern in desc_patterns:
            desc_match = re.search(pattern, content, re.IGNORECASE)
            if desc_match:
                desc = self.clean_text(desc_match.group(1))
                if self.validate_description(desc):
                    description = desc
                    break

        return title, description


class MetaExtractor(BaseExtractor):
    """Specialized meta tag extractor for OpenGraph, Twitter Cards, and structured data."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('performance', {}).get('timeout', 30)
        self.follow_redirects = config.get('advanced', {}).get('follow_redirects', True)
        self.max_redirects = config.get('advanced', {}).get('max_redirects', 10)

    async def extract(self, domain: str, **kwargs) -> ExtractionResult:
        """Extract meta information focusing on structured meta tags."""
        start_time = asyncio.get_event_loop().time()
        session = kwargs.get('session')

        try:
            url = f'https://{domain.strip()}'

            async with session.get(
                url,
                allow_redirects=self.follow_redirects,
                max_redirects=self.max_redirects
            ) as response:
                status_code = response.status
                content_type = response.headers.get('content-type', '').lower()

                if not self._is_valid_response(response.status, content_type):
                    return self.create_error_result(
                        domain=domain,
                        error_message=f"Invalid response: {response.status} {content_type}",
                        method="meta_extractor",
                        extraction_time=asyncio.get_event_loop().time() - start_time,
                        status_code=response.status
                    )

                content = await self._read_content_safely(response)
                if not content:
                    return self.create_error_result(
                        domain=domain,
                        error_message="No content received",
                        method="meta_extractor",
                        extraction_time=asyncio.get_event_loop().time() - start_time,
                        status_code=response.status
                    )

                title, description = self._extract_structured_meta(content)
                extraction_time = asyncio.get_event_loop().time() - start_time

                if title or description:
                    return self.create_success_result(
                        domain=domain,
                        title=title,
                        description=description,
                        method="meta_extractor",
                        extraction_time=extraction_time,
                        status_code=status_code
                    )
                else:
                    return self.create_error_result(
                        domain=domain,
                        error_message="No structured meta information found",
                        method="meta_extractor",
                        extraction_time=extraction_time,
                        status_code=status_code
                    )

        except Exception as e:
            return self.create_error_result(
                domain=domain,
                error_message=f"Meta extraction error: {str(e)}",
                method="meta_extractor",
                extraction_time=asyncio.get_event_loop().time() - start_time
            )

    def _is_valid_response(self, status_code: int, content_type: str) -> bool:
        """Check if the response is valid for HTML parsing."""
        if status_code >= 400:
            return False

        html_types = ['text/html', 'text/xhtml', 'application/xhtml+xml']
        return any(html_type in content_type for html_type in html_types)

    async def _read_content_safely(self, response: aiohttp.ClientResponse) -> Optional[str]:
        """Safely read response content with size limits."""
        try:
            content = await response.text()
            if len(content.encode('utf-8')) > self.max_content_length:
                return None
            return content
        except Exception:
            return None

    def _extract_structured_meta(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract meta information focusing on structured data."""
        try:
            tree = html.fromstring(content)
            title = self._extract_structured_title(tree)
            description = self._extract_structured_description(tree)

            if not title:
                title = self._extract_fallback_title(tree)
            if not description:
                description = self._extract_fallback_description(tree)

            if not title or not description:
                json_title, json_desc = self._extract_json_ld(content)
                if not title:
                    title = json_title
                if not description:
                    description = json_desc

            return title, description

        except Exception:
            return self._extract_with_regex(content)

    def _extract_structured_title(self, tree) -> Optional[str]:
        """Extract title from structured meta tags."""
        # OpenGraph title
        og_title = tree.xpath('.//meta[@property="og:title"]/@content')
        if og_title and og_title[0]:
            title = og_title[0].strip()
            if self.validate_title(title):
                return title

        # Twitter title
        twitter_title = tree.xpath('.//meta[@name="twitter:title"]/@content')
        if twitter_title and twitter_title[0]:
            title = twitter_title[0].strip()
            if self.validate_title(title):
                return title

        return None

    def _extract_structured_description(self, tree) -> Optional[str]:
        """Extract description from structured meta tags."""
        # OpenGraph description
        og_desc = tree.xpath('.//meta[@property="og:description"]/@content')
        if og_desc and og_desc[0]:
            desc = og_desc[0].strip()
            if self.validate_description(desc):
                return desc

        # Twitter description
        twitter_desc = tree.xpath('.//meta[@name="twitter:description"]/@content')
        if twitter_desc and twitter_desc[0]:
            desc = twitter_desc[0].strip()
            if self.validate_description(desc):
                return desc

        return None

    def _extract_fallback_title(self, tree) -> Optional[str]:
        """Extract title using standard fallbacks."""
        title_elem = tree.find('.//title')
        if title_elem is not None and title_elem.text:
            title = title_elem.text.strip()
            if self.validate_title(title):
                return title

        h1_elem = tree.find('.//h1')
        if h1_elem is not None and h1_elem.text:
            title = h1_elem.text.strip()
            if self.validate_title(title):
                return title

        return None

    def _extract_fallback_description(self, tree) -> Optional[str]:
        """Extract description using standard fallbacks."""
        meta_desc = tree.xpath('.//meta[@name="description"]/@content')
        if meta_desc and meta_desc[0]:
            desc = meta_desc[0].strip()
            if self.validate_description(desc):
                return desc

        return None

    def _extract_json_ld(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract information from JSON-LD structured data."""
        try:
            json_pattern = r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>'
            matches = re.findall(json_pattern, content, re.IGNORECASE | re.DOTALL)

            title = None
            description = None

            for match in matches:
                try:
                    data = json.loads(match.strip())
                    if isinstance(data, list):
                        items = data
                    else:
                        items = [data]

                    for item in items:
                        if isinstance(item, dict):
                            if not title:
                                possible_titles = [
                                    item.get('name'),
                                    item.get('headline'),
                                    item.get('title')
                                ]
                                for possible_title in possible_titles:
                                    if possible_title and self.validate_title(str(possible_title)):
                                        title = str(possible_title)
                                        break

                            if not description:
                                possible_descriptions = [
                                    item.get('description'),
                                    item.get('about'),
                                    item.get('abstract')
                                ]
                                for possible_desc in possible_descriptions:
                                    if possible_desc and self.validate_description(str(possible_desc)):
                                        description = str(possible_desc)
                                        break

                except json.JSONDecodeError:
                    continue

            return title, description

        except Exception:
            return None, None

    def _extract_with_regex(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract using regex as final fallback."""
        title = None
        description = None

        try:
            # Extract title using regex
            title_patterns = [
                r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
                r'<meta[^>]*name=["\']twitter:title["\'][^>]*content=["\']([^"\']+)["\']',
                r'<title[^>]*>(.*?)</title>'
            ]

            for pattern in title_patterns:
                match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
                if match:
                    title_text = self.clean_text(match.group(1))
                    if self.validate_title(title_text):
                        title = title_text
                        break

            # Extract description using regex
            desc_patterns = [
                r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']+)["\']',
                r'<meta[^>]*name=["\']twitter:description["\'][^>]*content=["\']([^"\']+)["\']',
                r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']+)["\']'
            ]

            for pattern in desc_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    desc_text = self.clean_text(match.group(1))
                    if self.validate_description(desc_text):
                        description = desc_text
                        break

        except Exception:
            pass

        return title, description


class FallbackExtractor(BaseExtractor):
    """Fallback extractor using alternative methods and content analysis."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.timeout = config.get('performance', {}).get('timeout', 30)
        self.follow_redirects = config.get('advanced', {}).get('follow_redirects', True)

    async def extract(self, domain: str, **kwargs) -> ExtractionResult:
        """Extract meta information using fallback methods."""
        start_time = asyncio.get_event_loop().time()
        session = kwargs.get('session')

        try:
            title, description, status_code = await self._try_multiple_approaches(session, domain)
            extraction_time = asyncio.get_event_loop().time() - start_time

            if title or description:
                return self.create_success_result(
                    domain=domain,
                    title=title,
                    description=description,
                    method="fallback_extractor",
                    extraction_time=extraction_time,
                    status_code=status_code
                )
            else:
                return self.create_error_result(
                    domain=domain,
                    error_message="All fallback methods failed",
                    method="fallback_extractor",
                    extraction_time=extraction_time,
                    status_code=status_code
                )

        except Exception as e:
            return self.create_error_result(
                domain=domain,
                error_message=f"Fallback error: {str(e)}",
                method="fallback_extractor",
                extraction_time=asyncio.get_event_loop().time() - start_time
            )

    async def _try_multiple_approaches(self, session: aiohttp.ClientSession, domain: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Try multiple fallback approaches to extract information."""

        # Generate all domain variants to try
        domain_variants = self._generate_domain_variants(domain)

        for variant_name, variant_domain in domain_variants:
            try:
                # Try with BeautifulSoup (most robust)
                title, desc, status = await self._try_beautifulsoup(session, variant_domain)
                if title or desc:
                    if title or desc:
                        return title, desc, status

                # Try with simple request (different headers)
                title, desc, status = await self._try_simple_request(session, variant_domain)
                if title or desc:
                    return title, desc, status

                # Try with alternative approach
                title, desc, status = await self._try_alternative_parsing(session, variant_domain)
                if title or desc:
                    return title, desc, status

            except Exception:
                continue  # Try next variant

        # Final fallback: Smart domain inference
        inferred_title = self._infer_smart_title_from_domain(domain)
        inferred_desc = self._infer_smart_description_from_domain(domain)
        return inferred_title, inferred_desc, None

    def _generate_domain_variants(self, domain: str) -> List[Tuple[str, str]]:
        """Generate multiple domain variants to try."""
        variants = []

        # Clean domain
        clean_domain = domain.strip()
        if clean_domain.startswith(('http://', 'https://')):
            clean_domain = clean_domain.split('://', 1)[1]
        clean_domain = clean_domain.rstrip('/')

        # Original variants
        variants.append(("original", f"https://{clean_domain}"))
        variants.append(("www", f"https://www.{clean_domain}"))
        variants.append(("http", f"http://{clean_domain}"))
        variants.append(("http_www", f"http://www.{clean_domain}"))

        # Common path variants
        common_paths = ["", "/home", "/index.html", "/about", "/contact"]
        for path in common_paths:
            if path:
                variants.append((f"https_{path}", f"https://{clean_domain}{path}"))
                variants.append((f"http_{path}", f"http://{clean_domain}{path}"))

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for name, url in variants:
            if url not in seen:
                seen.add(url)
                unique_variants.append((name, url))

        return unique_variants[:10]  # Limit to 10 variants to avoid too many requests

    async def _try_beautifulsoup(self, session: aiohttp.ClientSession, domain: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Try extraction with BeautifulSoup."""
        try:
            url = f'https://{domain.strip()}'

            async with session.get(url, allow_redirects=True) as response:
                status_code = response.status
                content = await self._read_content_safely(response)

                if not content:
                    return None, None, status_code

                soup = BeautifulSoup(content, 'html.parser')
                title = self._extract_title_with_soup(soup)
                description = self._extract_description_with_soup(soup)

                if title and not self.validate_title(title):
                    title = None
                if description and not self.validate_description(description):
                    description = None

                return title, description, status_code

        except Exception:
            return None, None, None

    async def _try_alternative_parsing(self, session: aiohttp.ClientSession, domain: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Try alternative parsing methods with different approaches."""
        try:
            # Use different user agents that might bypass blocking
            alternative_headers = [
                {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.5',
                    'Accept-Encoding': 'gzip, deflate'
                },
                {
                    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9'
                },
                {
                    'User-Agent': 'curl/7.68.0',
                    'Accept': '*/*'
                }
            ]

            for headers in alternative_headers:
                try:
                    async with session.get(domain, headers=headers, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=15)) as response:
                        if response.status == 200:
                            content = await self._read_content_safely(response)
                            if content:
                                # Try comprehensive extraction
                                title, description = self._extract_comprehensive(content, domain)
                                if title or description:
                                    return title, description, response.status
                except Exception:
                    continue

        except Exception:
            pass

        return None, None, None

    async def _try_simple_request(self, session: aiohttp.ClientSession, domain: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Try a simple HTTP request with minimal parsing."""
        try:
            simple_headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; MetaBot/1.0)',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
            }

            async with session.get(domain, headers=simple_headers, allow_redirects=True, timeout=aiohttp.ClientTimeout(total=15)) as response:
                status_code = response.status
                content = await self._read_content_safely(response)

                if not content:
                    return None, None, status_code

                title, description = self._extract_with_advanced_regex(content, domain)
                return title, description, status_code

        except Exception:
            return None, None, None

    async def _read_content_safely(self, response: aiohttp.ClientResponse) -> Optional[str]:
        """Safely read response content."""
        try:
            content = await response.text()
            if len(content.encode('utf-8')) > self.max_content_length:
                return None
            return content
        except Exception:
            return None

    def _extract_title_with_soup(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract title using BeautifulSoup."""
        title_tag = soup.find('title')
        if title_tag and title_tag.string:
            title = title_tag.string.strip()
            if self.validate_title(title):
                return title

        h1_tag = soup.find('h1')
        if h1_tag and h1_tag.string:
            title = h1_tag.string.strip()
            if self.validate_title(title):
                return title

        for meta_type in ['og:title', 'twitter:title']:
            meta_tag = soup.find('meta', property=meta_type) or soup.find('meta', attrs={'name': meta_type})
            if meta_tag and meta_tag.get('content'):
                title = meta_tag['content'].strip()
                if self.validate_title(title):
                    return title

        return None

    def _extract_description_with_soup(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract description using BeautifulSoup."""
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            desc = meta_desc['content'].strip()
            if self.validate_description(desc):
                return desc

        og_desc = soup.find('meta', property='og:description')
        if og_desc and og_desc.get('content'):
            desc = og_desc['content'].strip()
            if self.validate_description(desc):
                return desc

        return None

    def _extract_with_advanced_regex(self, content: str, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract using advanced regex patterns."""
        title = None
        description = None

        try:
            title_patterns = [
                r'<title[^>]*>([^{<}]*)</title>',
                r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']{3,200})["\']',
                r'<meta[^>]*name=["\']twitter:title["\'][^>]*content=["\']([^"\']{3,200})["\']',
            ]

            for pattern in title_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    title_text = self.clean_text(match)
                    if self.validate_title(title_text):
                        title = title_text
                        break
                if title:
                    break

            desc_patterns = [
                r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']{10,500})["\']',
                r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']{10,500})["\']',
                r'<meta[^>]*name=["\']twitter:description["\'][^>]*content=["\']([^"\']{10,500})["\']',
            ]

            for pattern in desc_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    desc_text = self.clean_text(match)
                    if self.validate_description(desc_text):
                        description = desc_text
                        break
                if description:
                    break

        except Exception:
            pass

        return title, description

    def _extract_comprehensive(self, content: str, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Comprehensive extraction using multiple strategies."""
        title = None
        description = None

        try:
            # Try lxml parsing first (fastest)
            try:
                tree = html.fromstring(content)
                title = self._extract_all_title_variants(tree)
                description = self._extract_all_description_variants(tree)
                if title or description:
                    return title, description
            except Exception:
                pass

            # Try BeautifulSoup (more lenient)
            try:
                soup = BeautifulSoup(content, 'html.parser')
                title = self._extract_title_with_soup(soup)
                description = self._extract_description_with_soup(soup)
                if title or description:
                    return title, description
            except Exception:
                pass

            # Try smart text extraction
            title, description = self._extract_smart_from_text(content)
            if title or description:
                return title, description

            # Try advanced regex as last resort
            return self._extract_with_advanced_regex(content, url)

        except Exception:
            return None, None

    def _extract_all_title_variants(self, tree) -> Optional[str]:
        """Extract title using all possible methods."""
        title_sources = [
            # Standard HTML title
            lambda: tree.find('.//title').text.strip() if tree.find('.//title') is not None and tree.find('.//title').text else None,

            # OpenGraph title
            lambda: tree.xpath('.//meta[@property="og:title"]/@content')[0].strip() if tree.xpath('.//meta[@property="og:title"]/@content') else None,

            # Twitter title
            lambda: tree.xpath('.//meta[@name="twitter:title"]/@content')[0].strip() if tree.xpath('.//meta[@name="twitter:title"]/@content') else None,

            # Schema.org title
            lambda: tree.xpath('.//meta[@itemprop="name"]/@content')[0].strip() if tree.xpath('.//meta[@itemprop="name"]/@content') else None,

            # h1-h6 tags
            lambda: tree.find('.//h1').text.strip() if tree.find('.//h1') is not None and tree.find('.//h1').text else None,
            lambda: tree.find('.//h2').text.strip() if tree.find('.//h2') is not None and tree.find('.//h2').text else None,
            lambda: tree.find('.//h3').text.strip() if tree.find('.//h3') is not None and tree.find('.//h3').text else None,

            # Meta title
            lambda: tree.xpath('.//meta[@name="title"]/@content')[0].strip() if tree.xpath('.//meta[@name="title"]/@content') else None,

            # First strong/bold text
            lambda: tree.find('.//strong').text.strip() if tree.find('.//strong') is not None and tree.find('.//strong').text else None,
            lambda: tree.find('.//b').text.strip() if tree.find('.//b') is not None and tree.find('.//b').text else None,
        ]

        for source in title_sources:
            try:
                candidate = source()
                if candidate and self.validate_title(candidate):
                    return candidate
            except Exception:
                continue

        return None

    def _extract_all_description_variants(self, tree) -> Optional[str]:
        """Extract description using all possible methods."""
        desc_sources = [
            # Standard meta description
            lambda: tree.xpath('.//meta[@name="description"]/@content')[0].strip() if tree.xpath('.//meta[@name="description"]/@content') else None,

            # OpenGraph description
            lambda: tree.xpath('.//meta[@property="og:description"]/@content')[0].strip() if tree.xpath('.//meta[@property="og:description"]/@content') else None,

            # Twitter description
            lambda: tree.xpath('.//meta[@name="twitter:description"]/@content')[0].strip() if tree.xpath('.//meta[@name="twitter:description"]/@content') else None,

            # Schema.org description
            lambda: tree.xpath('.//meta[@itemprop="description"]/@content')[0].strip() if tree.xpath('.//meta[@itemprop="description"]/@content') else None,

            # First meaningful paragraph
            lambda: self._extract_first_meaningful_paragraph(tree),

            # First few paragraphs combined
            lambda: self._extract_multiple_paragraphs(tree),

            # Text from about section
            lambda: self._extract_about_section(tree),
        ]

        for source in desc_sources:
            try:
                candidate = source()
                if candidate and self.validate_description(candidate):
                    return candidate
            except Exception:
                continue

        return None

    def _extract_first_meaningful_paragraph(self, tree) -> Optional[str]:
        """Extract the first meaningful paragraph."""
        paragraphs = tree.xpath('.//p//text()')
        for p_text in paragraphs[:5]:  # Check first 5 paragraphs
            text = p_text.strip()
            if text and len(text) > 20 and not text.isdigit() and not text.isspace():
                # Filter out common boilerplate text
                if not any(boilerplate in text.lower() for boilerplate in
                          ['copyright', 'all rights reserved', 'privacy policy', 'terms of service', 'cookie']):
                    return text[:500]  # Limit to 500 chars

        return None

    def _extract_multiple_paragraphs(self, tree) -> Optional[str]:
        """Extract and combine multiple paragraphs."""
        paragraphs = tree.xpath('.//p//text()')
        meaningful_paragraphs = []

        for p_text in paragraphs[:3]:  # Take first 3 paragraphs
            text = p_text.strip()
            if text and len(text) > 15 and not text.isdigit():
                meaningful_paragraphs.append(text)

        if meaningful_paragraphs:
            combined = ' '.join(meaningful_paragraphs)
            if len(combined) > 30:
                return combined[:500]

        return None

    def _extract_about_section(self, tree) -> Optional[str]:
        """Extract text from about sections."""
        about_selectors = [
            './/div[contains(@class, "about")]//text()',
            './/section[contains(@class, "about")]//text()',
            './/footer//text()',
            './/header//p//text()'
        ]

        for selector in about_selectors:
            try:
                texts = tree.xpath(selector)
                meaningful_text = ' '.join([t.strip() for t in texts if t.strip() and len(t.strip()) > 10])
                if meaningful_text and len(meaningful_text) > 30:
                    return meaningful_text[:500]
            except Exception:
                continue

        return None

    def _extract_smart_from_text(self, content: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract information from raw text using smart analysis."""
        try:
            # Extract title using enhanced regex
            title_patterns = [
                r'<title[^>]*>([^<]{3,200})</title>',
                r'<h1[^>]*>([^<]{3,200})</h1>',
                r'<h2[^>]*>([^<]{3,200})</h2>',
                r'<meta[^>]*property=["\']og:title["\'][^>]*content=["\']([^"\']{3,200})["\']',
                r'<meta[^>]*name=["\']twitter:title["\'][^>]*content=["\']([^"\']{3,200})["\']',
                r'<meta[^>]*name=["\']title["\'][^>]*content=["\']([^"\']{3,200})["\']',
            ]

            title = None
            for pattern in title_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    clean_title = self.clean_text(match)
                    if self.validate_title(clean_title):
                        title = clean_title
                        break
                if title:
                    break

            # Extract description using enhanced regex
            desc_patterns = [
                r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']{10,500})["\']',
                r'<meta[^>]*property=["\']og:description["\'][^>]*content=["\']([^"\']{10,500})["\']',
                r'<meta[^>]*name=["\']twitter:description["\'][^>]*content=["\']([^"\']{10,500})["\']',
                r'<meta[^>]*itemprop=["\']description["\'][^>]*content=["\']([^"\']{10,500})["\']',
                r'<p[^>]*>([^<]{20,500})</p>',
                r'<div[^>]*class=["\'][^"\']*summary[^"\']*["\'][^>]*>([^<]{20,500})</div>',
            ]

            description = None
            for pattern in desc_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    clean_desc = self.clean_text(match)
                    if self.validate_description(clean_desc):
                        description = clean_desc
                        break
                if description:
                    break

            return title, description

        except Exception:
            return None, None

    def _infer_smart_title_from_domain(self, domain: str) -> Optional[str]:
        """Smart title inference using multiple strategies."""
        try:
            parsed = urlparse(f'https://{domain}')
            domain_name = parsed.netloc.replace('www.', '')
            parts = domain_name.split('.')

            if len(parts) >= 2:
                main_part = parts[-2]
            else:
                main_part = parts[0] if parts else domain_name

            # Clean and format the domain name
            title = main_part.replace('-', ' ').replace('_', ' ').title()

            # Apply common corrections
            title = self._apply_common_corrections(title)

            # Try to get more specific title from common patterns
            enhanced_title = self._enhance_title_from_patterns(domain_name, title)

            final_title = enhanced_title if enhanced_title else title

            if self.validate_title(final_title):
                return final_title

        except Exception:
            pass

        return None

    def _apply_common_corrections(self, title: str) -> str:
        """Apply common corrections to domain-derived titles."""
        corrections = {
            'Github': 'GitHub',
            'Linkedin': 'LinkedIn',
            'Youtube': 'YouTube',
            'Facebook': 'Facebook',
            'Twitter': 'Twitter',
            'Instagram': 'Instagram',
            'Whatsapp': 'WhatsApp',
            'Google': 'Google',
            'Amazon': 'Amazon',
            'Microsoft': 'Microsoft',
            'Apple': 'Apple',
            'Netflix': 'Netflix',
            'Spotify': 'Spotify',
            'Reddit': 'Reddit',
            'Wikipedia': 'Wikipedia',
            'Stackoverflow': 'Stack Overflow',
            'Medium': 'Medium',
            'Tumblr': 'Tumblr',
            'Pinterest': 'Pinterest',
            'Flickr': 'Flickr',
            'Vimeo': 'Vimeo',
            'Dribbble': 'Dribbble',
            'Behance': 'Behance',
        }

        return corrections.get(title, title)

    def _enhance_title_from_patterns(self, domain: str, base_title: str) -> Optional[str]:
        """Enhance title based on common domain patterns."""
        domain_lower = domain.lower()

        # Common patterns and their enhanced titles
        patterns = {
            # Tech companies
            r'(.*\.?tech?)$': lambda m: f"{m.group(1).title()} Technology",
            r'(.*\.?io)$': lambda m: f"{m.group(1).title()} - Tech Platform",
            r'(.*\.?dev)$': lambda m: f"{m.group(1).title()} - Development",

            # Service providers
            r'(.*\.?services?)$': lambda m: f"{m.group(1).title()} Services",
            r'(.*\.?solutions?)$': lambda m: f"{m.group(1).title()} Solutions",
            r'(.*\.?consulting?)$': lambda m: f"{m.group(1).title()} Consulting",

            # E-commerce
            r'(.*\.?shop)$': lambda m: f"{m.group(1).title()} Shop",
            r'(.*\.?store)$': lambda m: f"{m.group(1).title()} Store",
            r'(.*\.?market)$': lambda m: f"{m.group(1).title()} Marketplace",

            # Media/Content
            r'(.*\.?blog)$': lambda m: f"{m.group(1).title()} Blog",
            r'(.*\.?news)$': lambda m: f"{m.group(1).title()} News",
            r'(.*\.?media)$': lambda m: f"{m.group(1).title()} Media",

            # Organizations
            r'(.*\.?org)$': lambda m: f"{m.group(1).title()} Organization",
            r'(.*\.?foundation)$': lambda m: f"{m.group(1).title()} Foundation",
            r'(.*\.?institute)$': lambda m: f"{m.group(1).title()} Institute",
        }

        for pattern, enhancer in patterns.items():
            match = re.match(pattern, domain_lower)
            if match:
                try:
                    enhanced = enhancer(match)
                    if enhanced and enhanced != base_title:
                        return enhanced
                except Exception:
                    continue

        return base_title

    def _infer_smart_description_from_domain(self, domain: str) -> Optional[str]:
        """Smart description inference based on domain patterns."""
        try:
            title = self._infer_smart_title_from_domain(domain)
            if not title:
                return None

            domain_lower = domain.lower()

            # Generate contextual descriptions based on domain patterns
            if 'tech' in domain_lower or 'technology' in domain_lower:
                description = f"{title} - Technology solutions and services for modern businesses."
            elif 'shop' in domain_lower or 'store' in domain_lower or 'market' in domain_lower:
                description = f"{title} - Online shopping destination with quality products and great deals."
            elif 'blog' in domain_lower or 'news' in domain_lower or 'media' in domain_lower:
                description = f"{title} - Latest news, insights, and updates in our area of expertise."
            elif 'service' in domain_lower or 'solution' in domain_lower:
                description = f"{title} - Professional services and solutions tailored to your needs."
            elif 'edu' in domain_lower or 'learn' in domain_lower or 'academy' in domain_lower:
                description = f"{title} - Educational platform offering courses and learning resources."
            elif 'health' in domain_lower or 'medical' in domain_lower or 'care' in domain_lower:
                description = f"{title} - Healthcare services and medical information for better health."
            elif 'finance' in domain_lower or 'money' in domain_lower or 'bank' in domain_lower:
                description = f"{title} - Financial services and solutions for individuals and businesses."
            elif 'travel' in domain_lower or 'tour' in domain_lower or 'hotel' in domain_lower:
                description = f"{title} - Travel services, accommodations, and destination guides."
            elif 'food' in domain_lower or 'restaurant' in domain_lower or 'dining' in domain_lower:
                description = f"{title} - Food and dining experiences with quality cuisine."
            elif 'sport' in domain_lower or 'fitness' in domain_lower or 'gym' in domain_lower:
                description = f"{title} - Sports, fitness, and wellness programs for active lifestyles."
            elif 'art' in domain_lower or 'design' in domain_lower or 'creative' in domain_lower:
                description = f"{title} - Creative services, design solutions, and artistic expressions."
            else:
                # Generic but contextual description
                description = f"{title} - Professional services and solutions in our field of expertise."

            if self.validate_description(description):
                return description

        except Exception:
            pass

        return None


class StreamlitProgress:
    """Custom progress tracker for Streamlit."""

    def __init__(self, total):
        self.total = total
        self.processed = 0
        self.successful = 0
        self.failed = 0
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.stats_text = st.empty()

    def update(self, success: bool, domain: str = "", method: str = ""):
        self.processed += 1
        if success:
            self.successful += 1
        else:
            self.failed += 1

        progress = self.processed / self.total
        self.progress_bar.progress(progress)

        if success:
            status = f"✅ Successfully processed: {domain}"
        else:
            status = f"❌ Failed to process: {domain}"

        self.status_text.text(status)

        success_rate = (self.successful / self.processed) * 100 if self.processed > 0 else 0
        stats_text = f"""
        **Progress:** {self.processed}/{self.total} ({progress:.1%})

        **Success Rate:** {success_rate:.1f}% ({self.successful} successful, {self.failed} failed)
        """
        self.stats_text.markdown(stats_text)

    def finish(self):
        self.progress_bar.progress(1.0)
        self.status_text.text("🎉 Processing complete!")


class ConsolidatedExtractor:
    """Consolidated extractor for Streamlit app."""

    def __init__(self, concurrency: Optional[int] = None):
        self.config = self.get_default_config()

        if concurrency is not None:
            # Ensure the configured concurrency is always at least 1
            safe_concurrency = max(1, int(concurrency))
            self.config.setdefault('performance', {})['concurrency'] = safe_concurrency

        self.concurrency = self.config.get('performance', {}).get('concurrency', 10)

        self.html_extractor = HTMLExtractor(self.config)
        self.meta_extractor = MetaExtractor(self.config)
        self.fallback_extractor = FallbackExtractor(self.config)

    def get_default_config(self):
        """Get default configuration for Streamlit."""
        return {
            'performance': {
                'timeout': 30,
                'read_timeout': 45,
                'max_retries': 2,
                'concurrency': 10
            },
            'extraction': {
                'enable_js_fallback': False,
                'max_content_length': 1048576,
                'min_title_length': 3,
                'max_title_length': 200,
                'min_description_length': 10,
                'max_description_length': 500
            },
            'advanced': {
                'follow_redirects': True,
                'max_redirects': 5,
                'verify_ssl': True,
                'enable_compression': True
            },
            'fallback': {
                'enable_domain_variants': True,
                'max_variants_per_domain': 10,
                'enable_smart_inference': True,
                'enable_alternative_parsing': True,
                'multiple_user_agents': True
            }
        }

    async def extract_domain(self, domain: str, session, progress_tracker=None) -> dict:
        """Extract meta information from a single domain."""
        start_time = time.time()

        try:
            normalized_domain = DomainUtils.normalize_domain(domain)
            if not normalized_domain:
                return {
                    'domain': domain,
                    'meta_title': '',
                    'meta_description': '',
                    'extraction_method': 'invalid_domain',
                    'status_code': 0,
                    'extraction_time': 0,
                    'error_message': 'Invalid domain format'
                }

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br'
            }

            extractors = [
                ('html_extractor', self.html_extractor),
                ('meta_extractor', self.meta_extractor),
                ('fallback_extractor', self.fallback_extractor)
            ]

            for extractor_name, extractor in extractors:
                try:
                    result = await extractor.extract(normalized_domain, session=session, headers=headers)

                    if result.success:
                        extraction_time = time.time() - start_time
                        if progress_tracker:
                            progress_tracker.update(True, normalized_domain, extractor_name)

                        return {
                            'domain': domain,
                            'meta_title': result.title or '',
                            'meta_description': result.description or '',
                            'extraction_method': result.method,
                            'status_code': result.status_code or 200,
                            'extraction_time': round(extraction_time, 2),
                            'error_message': ''
                        }

                except Exception:
                    continue

            extraction_time = time.time() - start_time
            if progress_tracker:
                progress_tracker.update(False, normalized_domain, 'none')

            return {
                'domain': domain,
                'meta_title': '',
                'meta_description': '',
                'extraction_method': 'none',
                'status_code': 0,
                'extraction_time': round(extraction_time, 2),
                'error_message': 'All extraction methods failed'
            }

        except Exception as e:
            extraction_time = time.time() - start_time
            if progress_tracker:
                progress_tracker.update(False, domain, 'exception')

            return {
                'domain': domain,
                'meta_title': '',
                'meta_description': '',
                'extraction_method': 'exception',
                'status_code': 0,
                'extraction_time': round(extraction_time, 2),
                'error_message': str(e)
            }

    async def process_domains(self, domains: list, progress_tracker=None):
        """Process a list of domains."""
        performance_config = self.config.get('performance', {})
        concurrency = max(1, int(performance_config.get('concurrency', self.concurrency or 1)))

        connector = aiohttp.TCPConnector(
            limit=concurrency,
            limit_per_host=concurrency,
            ttl_dns_cache=300,
            use_dns_cache=True
        )

        timeout = aiohttp.ClientTimeout(
            total=performance_config.get('timeout', 30),
            sock_read=performance_config.get('read_timeout', None)
        )

        semaphore = asyncio.Semaphore(concurrency)

        async with aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={'User-Agent': 'DomainMetaExtractor/1.0'}
        ) as session:

            async def bounded_extract(domain: str):
                async with semaphore:
                    return await self.extract_domain(domain, session, progress_tracker)

            tasks = [bounded_extract(domain) for domain in domains]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        'domain': domains[i],
                        'meta_title': '',
                        'meta_description': '',
                        'extraction_method': 'exception',
                        'status_code': 0,
                        'extraction_time': 0,
                        'error_message': str(result)
                    })
                else:
                    processed_results.append(result)

            return processed_results


def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🔍 Domain Meta Extractor</h1>', unsafe_allow_html=True)
    st.markdown("""
    Extract meta titles and descriptions from domain names with intelligent fallback strategies.
    Upload a CSV file with domains and get enhanced data with meta information.
    """)

    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")

    # Processing options
    st.sidebar.subheader("Processing Options")
    max_domains = st.sidebar.slider(
        "Maximum domains to process",
        10,
        100000,
        1000,
        help=(
            "Caps how many domains are processed in a single run. Higher values can "
            "significantly increase processing time and network load, so consider "
            "keeping this under 10,000 unless you are prepared for longer waits."
        ),
    )
    concurrency = st.sidebar.slider("Concurrency level", 1, 20, 10)

    # File upload section
    st.header("📁 Upload CSV File")

    upload_container = st.container()
    with upload_container:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose a CSV file with domains",
            type=['csv'],
            help="CSV should have a column named 'domain' with domain names"
        )

        st.markdown('</div>', unsafe_allow_html=True)

    # Sample data section
    with st.expander("📋 View Sample CSV Format"):
        sample_data = pd.DataFrame({
            'domain': ['google.com', 'github.com', 'stackoverflow.com']
        })
        st.write(sample_data)
        st.code("""
domain
google.com
github.com
stackoverflow.com
        """)

    # Processing section
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ File loaded successfully! Found {len(df)} rows.")

            # Show data preview
            with st.expander("👀 Preview uploaded data"):
                st.dataframe(df.head(10))

            # Check for domain column
            if 'domain' not in df.columns:
                st.error("❌ CSV must contain a 'domain' column!")
                return

            # Extract domains
            domains = df['domain'].dropna().tolist()

            # Sanity check to prevent overwhelming the app with massive uploads
            hard_limit = 500_000
            if len(domains) > hard_limit:
                st.error(
                    "❌ The uploaded file contains"
                    f" {len(domains):,} domains, which exceeds the supported limit of"
                    f" {hard_limit:,}. Please split the file into smaller batches"
                    " before processing."
                )
                return

            # Limit domains if necessary
            if len(domains) > max_domains:
                st.warning(f"⚠️ Limiting to first {max_domains} domains for demo purposes.")
                domains = domains[:max_domains]

            # Process button
            st.header("🚀 Start Processing")

            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("🔍 Extract Meta Information", type="primary", use_container_width=True):
                    # Show processing status
                    status_container = st.container()
                    with status_container:
                        st.markdown("### Processing Status")

                        # Initialize progress tracker
                        progress_tracker = StreamlitProgress(len(domains))

                        # Process domains
                        with st.spinner("Processing domains..."):
                            extractor = ConsolidatedExtractor(concurrency=concurrency)

                            # Run async processing
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)

                            try:
                                results = loop.run_until_complete(
                                    extractor.process_domains(domains, progress_tracker)
                                )
                            finally:
                                loop.close()

                        # Complete progress
                        progress_tracker.finish()

                    # Create results DataFrame
                    results_df = pd.DataFrame(results)

                    # Show results summary
                    st.header("📊 Results Summary")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Processed", len(results))
                    with col2:
                        successful = len([r for r in results if r['meta_title'] or r['meta_description']])
                        st.metric("Successful", successful)
                    with col3:
                        success_rate = (successful / len(results)) * 100 if results else 0
                        st.metric("Success Rate", f"{success_rate:.1f}%")

                    # Show results table
                    st.header("📋 Results")

                    # Add download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name=f"domain_extraction_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Display results
                    st.dataframe(results_df, use_container_width=True)

                    # Method breakdown
                    if results:
                        st.header("🔧 Extraction Method Breakdown")
                        method_counts = results_df['extraction_method'].value_counts()
                        st.bar_chart(method_counts)

            with col2:
                st.markdown("---")
                st.markdown("### ℹ️ Information")
                st.info("""
                **What will be extracted:**
                - Meta titles from HTML head
                - Meta descriptions
                - OpenGraph data
                - Fallback content

                **Processing time varies** based on:
                - Number of domains
                - Site responsiveness
                - Network conditions
                """)

        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.error("Please ensure your CSV file has a 'domain' column with valid domain names.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; margin-top: 2rem;'>
        Made with ❤️ | Domain Meta Extractor v2.0 (Consolidated)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()