"""Streamlit UI for JD-based resume screening."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.resume_parser import extract_resume_text_from_bytes
from src.screen import score_resume_texts, scores_to_frame


st.set_page_config(page_title="AI Resume Screening", layout="wide")


def build_download_csv(report: pd.DataFrame) -> bytes:
    return report.to_csv(index=False).encode("utf-8")


st.title("AI Resume Screening")
st.caption("Upload resumes, paste a job description, and get a ranked shortlist.")

with st.sidebar:
    st.header("How it works")
    st.write("1. Paste the job description.")
    st.write("2. Upload one or more PDF/TXT resumes.")
    st.write("3. Click **Rank Resumes**.")
    st.write("4. Review score, keyword match, and project relevance.")

job_description = st.text_area(
    "Job description",
    height=260,
    placeholder=(
        "Paste the complete job description here. Include required skills, tools, "
        "responsibilities, and project expectations."
    ),
)

uploaded_files = st.file_uploader(
    "Upload resumes",
    type=["pdf", "txt"],
    accept_multiple_files=True,
    help="Upload any number of candidate resumes in PDF or TXT format.",
)

top_keywords = st.slider("Number of job-description keywords to compare", 10, 40, 25)
rank_clicked = st.button("Rank Resumes", type="primary", use_container_width=True)

if rank_clicked:
    if not job_description.strip():
        st.error("Please paste a job description first.")
    elif not uploaded_files:
        st.error("Please upload at least one resume.")
    else:
        resumes: list[tuple[str, str, str]] = []
        failed_files: list[str] = []

        with st.spinner("Reading resumes and ranking candidates..."):
            for uploaded_file in uploaded_files:
                try:
                    text = extract_resume_text_from_bytes(uploaded_file.name, uploaded_file.getvalue())
                    if text.strip():
                        resumes.append((uploaded_file.name, uploaded_file.name, text))
                    else:
                        failed_files.append(f"{uploaded_file.name}: no readable text found")
                except Exception as error:  # pragma: no cover - UI feedback path
                    failed_files.append(f"{uploaded_file.name}: {error}")

            if not resumes:
                st.error("No readable resumes were found. Try text-based PDFs or TXT files.")
            else:
                scores = score_resume_texts(job_description, resumes, top_keywords)
                report = scores_to_frame(scores)

                st.success(f"Ranked {len(report)} resume(s).")
                st.dataframe(
                    report[
                        [
                            "rank",
                            "resume",
                            "score",
                            "similarity",
                            "keyword_coverage",
                            "project_relevance",
                            "matched_keywords",
                            "matched_project_keywords",
                            "missing_keywords",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

                best = report.iloc[0]
                st.subheader("Best Match")
                st.write(f"**{best['resume']}** ranked #1 with a score of **{best['score']}**.")
                st.write(f"Project relevance: **{best['project_relevance']}**")

                st.download_button(
                    "Download CSV Report",
                    data=build_download_csv(report),
                    file_name="resume_screening_report.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        if failed_files:
            st.warning("Some files could not be processed:\n\n" + "\n".join(f"- {item}" for item in failed_files))
