from __future__ import annotations

from pathlib import Path
import re
from typing import List

import pandas as pd
import streamlit as st

from backend.generation.answerer import GroundedAnswerer
from backend.generation.llm_client import LLMClient
from backend.ingestion.pdf_loader import load_multiple_pdfs
from backend.processing.chunker import build_pdf_chunks
from backend.retrieval.embedder import Embedder
from backend.retrieval.faiss_store import FAISSStore
from backend.retrieval.retriever import Retriever


APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "index"

for d in [DATA_DIR, UPLOADS_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def get_embedder() -> Embedder:
    return Embedder(model_name="all-MiniLM-L6-v2")


@st.cache_resource
def get_store() -> FAISSStore:
    return FAISSStore(storage_dir=INDEX_DIR)


def get_answerer(
    *,
    enable_llm: bool = False,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o-mini",
    llm_base_url: str | None = None,
) -> GroundedAnswerer:
    embedder = get_embedder()
    store = get_store()
    retriever = Retriever(embedder=embedder, store=store)
    llm_client = LLMClient(
        model=llm_model,
        provider=llm_provider,
        base_url=llm_base_url,
    )
    return GroundedAnswerer(retriever=retriever, llm_client=llm_client, enable_llm=enable_llm)


def save_uploaded_files(uploaded_files, suffix: str) -> List[Path]:
    paths: List[Path] = []
    for file in uploaded_files:
        safe_name = file.name.replace("/", "_").replace("\\", "_")
        dest = UPLOADS_DIR / f"{suffix}_{safe_name}"
        with open(dest, "wb") as f:
            f.write(file.getbuffer())
        paths.append(dest)
    return paths


def _extract_pdf_financial_viz(pages) -> dict:
    strong_keywords = [
        "revenue",
        "profit",
        "loss",
        "ebitda",
        "cash flow",
        "balance sheet",
        "assets",
        "liabilities",
        "expenses",
        "net income",
        "operating income",
        "dividend",
        "earnings",
    ]
    weak_keywords = [
        "income",
        "margin",
        "growth",
        "forecast",
        "guidance",
    ]

    money_or_percent = re.compile(
        r"(?i)(?:₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:million|billion|thousand|crore|lakh|bn)\b"
    )
    currency_only_pattern = re.compile(
        r"(?i)(?:₹|\$|€|£)\s?\d[\d,]*(?:\.\d+)?"
    )

    month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_index = {m: i for i, m in enumerate(month_order)}
    monthly_row_pattern = re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\b\s+"
        r"\$?([\d,]+(?:\.\d+)?)\s+"
        r"\$?([\d,]+(?:\.\d+)?)\s+"
        r"\$?([\d,]+(?:\.\d+)?)\s+"
        r"\$?([\d,]+(?:\.\d+)?)\s+"
        r"\$?([\d,]+(?:\.\d+)?)"
    )

    page_rows = []
    keyword_counts = {k: 0 for k in (strong_keywords + weak_keywords)}
    monthly_records = []
    total_strong_keyword_hits = 0
    total_currency_hits = 0

    for page in pages:
        text = page.text or ""
        lower = text.lower()
        keyword_hits = 0

        strong_hits_on_page = 0
        for k in strong_keywords:
            count = lower.count(k)
            keyword_counts[k] += count
            keyword_hits += count
            strong_hits_on_page += count

        for k in weak_keywords:
            count = lower.count(k)
            keyword_counts[k] += count
            keyword_hits += count

        numeric_hits = len(money_or_percent.findall(text))
        currency_hits = len(currency_only_pattern.findall(text))

        total_strong_keyword_hits += strong_hits_on_page
        total_currency_hits += currency_hits

        page_rows.append(
            {
                "file": page.file_name,
                "page": page.page_number,
                "keyword_hits": keyword_hits,
                "strong_keyword_hits": strong_hits_on_page,
                "numeric_hits": numeric_hits,
                "currency_hits": currency_hits,
                "total_financial_signals": keyword_hits + numeric_hits,
            }
        )

        for match in monthly_row_pattern.finditer(text):
            month = match.group(1)
            values = [
                float(match.group(2).replace(",", "")),
                float(match.group(3).replace(",", "")),
                float(match.group(4).replace(",", "")),
                float(match.group(5).replace(",", "")),
                float(match.group(6).replace(",", "")),
            ]
            monthly_records.append(
                {
                    "Month": month,
                    "Housing": values[0],
                    "Bills & Utilities": values[1],
                    "Food & Dining": values[2],
                    "Personal": values[3],
                    "Auto & Transport": values[4],
                }
            )

    page_df = pd.DataFrame(page_rows)
    keyword_df = pd.DataFrame(
        [{"keyword": k, "count": v} for k, v in keyword_counts.items() if v > 0]
    ).sort_values("count", ascending=False) if any(v > 0 for v in keyword_counts.values()) else pd.DataFrame(columns=["keyword", "count"])

    total_signals = int(page_df["total_financial_signals"].sum()) if not page_df.empty else 0

    has_financial_table = len(monthly_records) >= 3
    strong_finance_text = (
        (total_strong_keyword_hits >= 3 and total_currency_hits >= 1)
        or (total_currency_hits >= 4)
    )
    is_financial = bool(has_financial_table or strong_finance_text)

    expense_df = pd.DataFrame(monthly_records)
    if not expense_df.empty:
        expense_df = (
            expense_df.sort_values(by="Month", key=lambda s: s.map(month_index))
            .drop_duplicates(subset=["Month"], keep="first")
            .reset_index(drop=True)
        )
        expense_df["Total"] = expense_df[
            ["Housing", "Bills & Utilities", "Food & Dining", "Personal", "Auto & Transport"]
        ].sum(axis=1)

    category_totals = pd.DataFrame()
    if not expense_df.empty:
        cat_cols = ["Housing", "Bills & Utilities", "Food & Dining", "Personal", "Auto & Transport"]
        category_totals = pd.DataFrame(
            {
                "Category": cat_cols,
                "Amount": [float(expense_df[c].sum()) for c in cat_cols],
            }
        )

    return {
        "is_financial": is_financial,
        "total_signals": total_signals,
        "page_df": page_df,
        "keyword_df": keyword_df,
        "expense_df": expense_df,
        "category_totals": category_totals,
    }


def process_and_index_pdfs(pdf_paths: List[Path]) -> tuple[int, dict]:
    get_store().clear()
    pages = load_multiple_pdfs(pdf_paths)
    page_dicts = [
        {"file_name": p.file_name, "page_number": p.page_number, "text": p.text}
        for p in pages
    ]
    docs = build_pdf_chunks(page_dicts, chunk_size=600, overlap=100)
    if not docs:
        return 0, {"is_financial": False, "total_signals": 0, "page_df": pd.DataFrame(), "keyword_df": pd.DataFrame()}

    texts = [d["text"] for d in docs]
    vectors = get_embedder().encode_texts(texts)
    get_store().add(vectors, docs)
    return len(docs), _extract_pdf_financial_viz(pages)