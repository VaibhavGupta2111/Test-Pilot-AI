import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import streamlit as st
import pandas as pd
import torch

from utils import load_issues, export_to_excel
from risk_engine import calculate_scores
from generator import (
    generate_test_case,
    load_outline_gen,
    load_refine_gen,
    build_vector_index
)

try:
    torch.set_num_threads(2)
except RuntimeError:
    pass

st.set_page_config(page_title="Risk & Test Case Generator", layout="wide")
st.title("🛡️ Risk Scoring + Test Case Generator")
st.markdown("Upload your Excel file and generate scored insights + auto test cases.")

uploaded_file = st.file_uploader("📤 Upload your issues.xlsx", type=["xlsx"])


@st.cache_resource(show_spinner=False)
def init_outline_pipeline():
    return load_outline_gen()

@st.cache_resource(show_spinner=False)
def init_refine_pipeline():
    return load_refine_gen()


if uploaded_file:
    try:
        # 1) Load & display raw issues
        df_raw = load_issues(uploaded_file)
        st.success(f"✅ Loaded {len(df_raw)} issues.")
        st.dataframe(df_raw)

        # 2) Build semantic‐search index on the raw issues
        issues_list = df_raw.to_dict(orient="records")
        index, embeddings, indexed_issues = build_vector_index(issues_list)
        st.info("🔎 Semantic index built on all issues.")

        # 3) Calculate Risk Scores
        scores = calculate_scores(df_raw)
        df_raw["Risk Score"] = df_raw["Issue key"].map(scores)
        df = df_raw.sort_values("Risk Score", ascending=False)

        # 4) Sidebar filters
        st.sidebar.header("🔎 Filter Options")
        min_score   = st.sidebar.slider("Minimum Risk Score", 0, 30, 0)
        only_linked = st.sidebar.checkbox("Only include issues with Linked Issues")

        if only_linked:
            df = df[df["Linked Issues"].apply(lambda x: len(x) > 0)]
        df = df[df["Risk Score"] >= min_score]

        st.subheader("📊 Scored & Filtered Issues")
        st.dataframe(df)

        # 5) Download filtered issues
        filtered_excel_bytes = export_to_excel(df)
        st.download_button(
            label="📥 Download Filtered Issues",
            data=filtered_excel_bytes,
            file_name="filtered_issues.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 6) Warm up generators
        st.subheader("🔄 Warming up models…")
        with st.spinner("Loading GPT-2 outline generator…"):
            outline_gen = init_outline_pipeline()
            outline_gen("Hello", max_new_tokens=5)
        with st.spinner("Loading GPT-J refinement generator…"):
            refine_gen = init_refine_pipeline()
            refine_gen("Hello", max_new_tokens=5)
        st.success("✅ Models ready!")

        # 7) Generate test cases with progress bar
        st.subheader("🧪 Generating Test Cases…")
        progress_bar = st.progress(0)
        status_text = st.empty()

        result_rows = []
        total = len(df)
        for i, row in enumerate(df.to_dict(orient="records"), start=1):
            tc = generate_test_case(
                issue=row,
                risk_score=row["Risk Score"],
                index=index,
                issues_list=indexed_issues,
                outline_gen=outline_gen,
                refine_gen=refine_gen
            )
            result_rows.append({
                "Issue key":   row["Issue key"],
                "Summary":     row["Summary"],
                "Priority":    row["Priority"],
                "Status":      row["Status"],
                "Risk Score":  row["Risk Score"],
                "Test Case":   tc
            })

            progress_bar.progress(int(i / total * 100))
            status_text.text(f"Progress: {i}/{total}")

        status_text.text("✅ All test cases generated!")

        # 8) Show & export final results
        result_df = pd.DataFrame(result_rows)
        st.subheader("📋 Generated Test Cases")
        for idx, r in result_df.iterrows():
            with st.expander(f"🔹 Test Case: {r['Issue key']}"):
                st.markdown(f"**Summary:** {r['Summary']}")
                st.markdown(f"**Priority:** {r['Priority']}")
                st.markdown(f"**Status:** {r['Status']}")
                st.markdown(f"**Risk Score:** {r['Risk Score']}")
                st.code(r["Test Case"], language="markdown")
                st.text_area(
                    "📋 Copy this test case",
                    value=r["Test Case"],
                    height=150,
                    key=f"copy_{idx}"
                )

        excel_bytes = export_to_excel(df, result_df)
        st.download_button(
            label="📥 Download Final Results",
            data=excel_bytes,
            file_name="scored_test_cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
