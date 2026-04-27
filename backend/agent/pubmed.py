"""
PubMed E-utilities client (free, không cần API key).
Sử dụng efetch + esearch theo chuẩn NCBI.
"""
from __future__ import annotations

import logging
import re

import httpx

from .schemas import PubMedArticle

logger = logging.getLogger(__name__)

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


async def search_pubmed(query: str, max_results: int = 3, timeout: float = 8.0) -> list[PubMedArticle]:
    """
    Tìm tối đa `max_results` bài PubMed liên quan tới `query`.
    Trả về list rỗng nếu lỗi (không raise) — agent vẫn chạy tiếp được.
    """
    if not query or not query.strip():
        return []

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            search = await client.get(
                f"{BASE}/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmode": "json",
                    "retmax": max_results,
                    "sort": "relevance",
                },
            )
            search.raise_for_status()
            ids = search.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return []

            summary = await client.get(
                f"{BASE}/esummary.fcgi",
                params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"},
            )
            summary.raise_for_status()
            result = summary.json().get("result", {})

            out: list[PubMedArticle] = []
            for pmid in ids:
                item = result.get(pmid)
                if not item:
                    continue
                title = re.sub(r"\s+", " ", item.get("title", "")).strip()
                out.append(
                    PubMedArticle(
                        pmid=str(pmid),
                        title=title or "(no title)",
                        journal=item.get("fulljournalname"),
                        pub_date=item.get("pubdate"),
                        url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    )
                )
            return out
    except Exception as e:
        logger.warning("PubMed search failed: %s", e)
        return []
