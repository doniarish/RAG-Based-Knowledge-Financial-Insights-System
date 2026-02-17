from __future__ import annotations

from pathlib import Path
import re
from typing import List

import pandas as pd
import plotly.express as px
import streamlit as st

from generation.answerer import GroundedAnswerer
from ingestion.pdf_loader import load_multiple_pdfs
from processing.chunker import build_pdf_chunks
from retrieval.embedder import Embedder
from retrieval.faiss_store import FAISSStore
from retrieval.retriever import Retriever


APP_DIR = Path(__file__).resolve().parent
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


def get_answerer() -> GroundedAnswerer:
    embedder = get_embedder()
    store = get_store()
    retriever = Retriever(embedder=embedder, store=store)
    return GroundedAnswerer(retriever=retriever, enable_llm=False)


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


def show_pdf_financial_charts(viz: dict) -> None:
    if not viz.get("is_financial"):
        return

    page_df: pd.DataFrame = viz.get("page_df", pd.DataFrame())
    keyword_df: pd.DataFrame = viz.get("keyword_df", pd.DataFrame())
    expense_df: pd.DataFrame = viz.get("expense_df", pd.DataFrame())
    category_totals: pd.DataFrame = viz.get("category_totals", pd.DataFrame())

    st.subheader("Financial Signals from PDF")
    st.caption(f"Detected financial signals: {viz.get('total_signals', 0)}")

    if not expense_df.empty and not category_totals.empty:
        annual_total = float(expense_df["Total"].sum())
        avg_monthly = float(expense_df["Total"].mean())
        top_month_row = expense_df.sort_values("Total", ascending=False).iloc[0]

        c1, c2, c3 = st.columns(3)
        c1.metric("Annual Spend", f"${annual_total:,.2f}")
        c2.metric("Avg Monthly Spend", f"${avg_monthly:,.2f}")
        c3.metric("Highest Spend Month", f"{top_month_row['Month']} (${top_month_row['Total']:,.2f})")

        st.markdown("### Category Breakdown")
        fig_cat_pie = px.pie(
            category_totals,
            names="Category",
            values="Amount",
            title="Yearly Spend Share by Category",
        )
        st.plotly_chart(fig_cat_pie, use_container_width=True)

        st.markdown("### Monthly Spending Trend")
        expense_long = expense_df.melt(
            id_vars=["Month"],
            value_vars=["Housing", "Bills & Utilities", "Food & Dining", "Personal", "Auto & Transport"],
            var_name="Category",
            value_name="Amount",
        )

        fig_month_bar = px.bar(
            expense_long,
            x="Month",
            y="Amount",
            color="Category",
            barmode="group",
            title="Monthly Spend by Category",
        )
        st.plotly_chart(fig_month_bar, use_container_width=True)

        fig_total_line = px.line(
            expense_df,
            x="Month",
            y="Total",
            markers=True,
            title="Total Monthly Spend",
        )
        st.plotly_chart(fig_total_line, use_container_width=True)

        st.markdown("### Expense Table")
        st.dataframe(expense_df, use_container_width=True, hide_index=True)

    if not page_df.empty:
        total_pages = int(page_df["page"].nunique())
        strongest_page = page_df.sort_values("total_financial_signals", ascending=False).iloc[0]
        strongest_page_label = f"{strongest_page['file']} - page {int(strongest_page['page'])}"
        strongest_page_value = int(strongest_page["total_financial_signals"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Financial Signals", int(viz.get("total_signals", 0)))
        col2.metric("Pages With Signals", int((page_df["total_financial_signals"] > 0).sum()))
        col3.metric("Analyzed Pages", total_pages)

        st.markdown(
            f"**Top page:** {strongest_page_label} with **{strongest_page_value}** financial signals"
        )

        file_agg = (
            page_df.groupby("file", as_index=False)[
                ["keyword_hits", "numeric_hits", "total_financial_signals"]
            ]
            .sum()
        )

        pie_df = file_agg[file_agg["total_financial_signals"] > 0]
        if not pie_df.empty:
            fig_pie = px.pie(
                pie_df,
                values="total_financial_signals",
                names="file",
                title="Signal Share by PDF File",
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        st.markdown("### Easy Summary Tables")
        page_table = (
            page_df[["file", "page", "keyword_hits", "numeric_hits", "total_financial_signals"]]
            .sort_values("total_financial_signals", ascending=False)
            .head(15)
            .reset_index(drop=True)
        )
        st.dataframe(page_table, use_container_width=True, hide_index=True)

        if not keyword_df.empty:
            kw_table = keyword_df.head(15).reset_index(drop=True)
            st.dataframe(kw_table, use_container_width=True, hide_index=True)

        with st.expander("Advanced Charts"):
            fig_page = px.bar(
                page_df,
                x="page",
                y="total_financial_signals",
                color="file",
                title="Financial Signal Density by Page",
            )
            st.plotly_chart(fig_page, use_container_width=True)

            fig_line = px.line(
                page_df,
                x="page",
                y="total_financial_signals",
                color="file",
                markers=True,
                title="Financial Signals Trend Across Pages",
            )
            st.plotly_chart(fig_line, use_container_width=True)

            fig_area = px.area(
                page_df,
                x="page",
                y="total_financial_signals",
                color="file",
                title="Cumulative Financial Signal Coverage by Page",
            )
            st.plotly_chart(fig_area, use_container_width=True)

            fig_scatter = px.scatter(
                page_df,
                x="keyword_hits",
                y="numeric_hits",
                size="total_financial_signals",
                color="file",
                hover_data=["page"],
                title="Keyword vs Numeric Signal Distribution",
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            fig_hist = px.histogram(
                page_df,
                x="total_financial_signals",
                color="file",
                nbins=20,
                title="Distribution of Financial Signals per Page",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

            if len(file_agg) > 1:
                sunburst_df = file_agg.melt(
                    id_vars=["file"],
                    value_vars=["keyword_hits", "numeric_hits"],
                    var_name="signal_type",
                    value_name="count",
                )
                sunburst_df = sunburst_df[sunburst_df["count"] > 0]
                if not sunburst_df.empty:
                    fig_sunburst = px.sunburst(
                        sunburst_df,
                        path=["file", "signal_type"],
                        values="count",
                        title="Signal Type Breakdown by File",
                    )
                    st.plotly_chart(fig_sunburst, use_container_width=True)

            if not keyword_df.empty:
                top_kw = keyword_df.head(10)
                fig_kw = px.bar(
                    top_kw,
                    x="keyword",
                    y="count",
                    title="Top Financial Terms in Uploaded PDFs",
                )
                st.plotly_chart(fig_kw, use_container_width=True)

                fig_kw_pie = px.pie(
                    top_kw,
                    values="count",
                    names="keyword",
                    title="Keyword Composition (Top Terms)",
                )
                st.plotly_chart(fig_kw_pie, use_container_width=True)

                fig_kw_treemap = px.treemap(
                    top_kw,
                    path=["keyword"],
                    values="count",
                    title="Keyword Importance Treemap",
                )
                st.plotly_chart(fig_kw_treemap, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="RAG Knowledge + Financial Insights", layout="wide")
    st.title("RAG-Based Knowledge & Financial Insights System")

    if "processed_pdf_signature" not in st.session_state:
        st.session_state.processed_pdf_signature = ()
    if "pdf_financial_viz" not in st.session_state:
        st.session_state.pdf_financial_viz = None
    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.header("Data Upload")
        pdf_files = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
        )

        index_pdf_clicked = st.button("Index Documents")

        clear_index_clicked = st.button("Clear Index")

    pdf_signature = tuple((f.name, f.size) for f in pdf_files) if pdf_files else ()
    if pdf_signature and pdf_signature != st.session_state.processed_pdf_signature:
        with st.spinner("Processing uploaded PDFs..."):
            paths = save_uploaded_files(pdf_files, suffix="pdf")
            chunks, viz = process_and_index_pdfs(paths)
        st.session_state.processed_pdf_signature = pdf_signature
        st.session_state.pdf_financial_viz = viz
        st.session_state.messages = []
        st.success(f"Processed and indexed {chunks} PDF chunks.")

    if not pdf_signature:
        st.session_state.processed_pdf_signature = ()

    with st.sidebar:
        st.caption(f"Indexed chunks: {get_store().count()}")

    if clear_index_clicked:
        get_store().clear()
        st.session_state.pdf_financial_viz = None
        st.session_state.processed_pdf_signature = ()
        st.session_state.messages = []
        st.success("FAISS index cleared.")

    if index_pdf_clicked:
        if not pdf_files:
            st.warning("Please upload at least one PDF.")
        else:
            paths = save_uploaded_files(pdf_files, suffix="pdf")
            chunks, viz = process_and_index_pdfs(paths)
            st.session_state.pdf_financial_viz = viz
            st.session_state.messages = []
            st.session_state.processed_pdf_signature = pdf_signature
            st.success(f"Indexed {chunks} PDF chunks into FAISS.")

    st.divider()

    st.subheader("Unified Chat")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    question = st.chat_input(
        "Ask about uploaded PDFs, including financial insights found in them..."
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        if get_store().count() == 0:
            warn_text = "No indexed content found. Upload files and wait for processing to complete."
            st.session_state.messages.append({"role": "assistant", "content": warn_text})
            with st.chat_message("assistant"):
                st.write(warn_text)
            return

        answerer = get_answerer()
        result = answerer.answer(question, k=5)
        st.session_state.messages.append({"role": "assistant", "content": result.answer})

        with st.chat_message("assistant"):
            st.markdown("### Answer")
            st.write(result.answer)

            st.markdown("### Citations")
            if result.citations:
                for c in result.citations:
                    st.write(
                        f"- (file={c['file']}, page={c['page']}, chunk_id={c['chunk_id']})"
                    )
            else:
                st.write("No citations available.")

    if st.session_state.pdf_financial_viz is not None:
        show_pdf_financial_charts(st.session_state.pdf_financial_viz)


if __name__ == "__main__":
    main()
