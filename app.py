import os
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import streamlit as st
import pandas as pd
import torch

from utils import load_issues, export_to_excel
from risk_engine import calculate_scores
from generator import (
    load_outline_gen,
    load_refine_gen,
    build_vector_index,
    generate_test_case
)

# limit threads for cpu 
try:
    torch.set_num_threads(2)
except RuntimeError:
    pass

st.set_page_config(page_title="Risk & Test Case Generator", layout="wide")
st.title("🛡️ Risk Scoring + Test Case Generator")
st.markdown("Upload your Excel file to compute risk scores and generate auto test cases.")

uploaded_file = st.file_uploader("📤 Upload your issues.xlsx", type=["xlsx"])


@st.cache_resource(show_spinner=False)
def init_outline_pipeline():
    return load_outline_gen()


@st.cache_resource(show_spinner=False)
def init_refine_pipeline():
    return load_refine_gen()


@st.cache_resource(show_spinner=False)
def init_vector_index(issues_list):
    return build_vector_index(issues_list)


if uploaded_file:
    try:
        # 1) Load & display raw issues
        df = load_issues(uploaded_file)
        st.success(f"✅ Loaded {len(df)} issues.")
        st.dataframe(df)

        # 2) Build semantic‐search index
        issues_list = df.to_dict(orient="records")
        index = init_vector_index(issues_list)
        st.info("🔎 Semantic index ready.")

        # 3) Calculate Risk Scores
        scores = calculate_scores(df)
        df["Risk Score"] = df["Issue key"].map(scores)
        df = df.sort_values("Risk Score", ascending=False)

        # 4) Sidebar filters
        st.sidebar.header("🔎 Filter Options")
        min_score   = st.sidebar.slider("Min Risk Score", 0, 50, 0)
        only_linked = st.sidebar.checkbox("Only include issues with Linked Issues")

        if only_linked:
            df = df[df["Linked Issues"].apply(bool)]
        df = df[df["Risk Score"] >= min_score]

        st.subheader("📊 Scored & Filtered Issues")
        st.dataframe(df)

        # 5) Download filtered issues only
        filtered_bytes = export_to_excel(df, pd.DataFrame())
        st.download_button(
            label="📥 Download Filtered Issues",
            data=filtered_bytes,
            file_name="filtered_issues.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # 6) Warm pipelines
        outline_gen = init_outline_pipeline()
        refine_gen  = init_refine_pipeline()

        # 7) Generate test cases
        st.subheader("🧪 Generating Test Cases")
        progress = st.progress(0)
        status   = st.empty()

        results = []
        total = len(df)
        for i, row in enumerate(df.to_dict(orient="records"), start=1):
            tc_md = generate_test_case(
                issue=row,
                risk_score=row["Risk Score"],
                index=index,
                issues_list=issues_list,
                outline_gen=outline_gen,
                refine_gen=refine_gen
            )
            results.append({
                **row,
                "Test Case": tc_md
            })
            progress.progress(int(i / total * 100))
            status.text(f"Progress: {i}/{total}")

        status.text("✅ All test cases generated!")

        # 8) Display & export final output
        results_df = pd.DataFrame(results)
        st.subheader("📋 Generated Test Cases")
        for idx, rec in results_df.iterrows():
            with st.expander(f"🔹 {rec['Issue key']}"):
                st.markdown(f"**Summary:** {rec['Summary']}")
                st.markdown(f"**Priority:** {rec['Priority']}")
                st.markdown(f"**Status:** {rec['Status']}")
                st.markdown(f"**Risk Score:** {rec['Risk Score']}")
                st.code(rec["Test Case"], language="markdown")
                st.text_area(
                    "📋 Copy test case",
                    value=rec["Test Case"],
                    height=150,
                    key=f"tc_{idx}"
                )

        # final download
        final_bytes = export_to_excel(df, results_df)
        st.download_button(
            label="📥 Download Full Results",
            data=final_bytes,
            file_name="scored_test_cases.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        st.error(f"❌ Error: {e}")
