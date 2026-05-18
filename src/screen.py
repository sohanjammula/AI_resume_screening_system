"""Rank resumes against a user-provided job description."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocessing import clean_resume_text
from .resume_parser import collect_resume_paths, extract_resume_text


@dataclass(frozen=True)
class ResumeScore:
    resume: str
    source: str
    score: float
    similarity: float
    keyword_coverage: float
    project_relevance: float
    matched_keywords: list[str]
    missing_keywords: list[str]
    matched_project_keywords: list[str]


def load_job_description(text: str | None, file_path: Path | None) -> str:
    """Load a job description from text or a file."""
    if text and file_path:
        raise ValueError("Use either --job-description or --job-file, not both.")
    if file_path:
        return file_path.read_text(encoding="utf-8")
    if text:
        return text
    raise ValueError("Provide a job description with --job-description or --job-file.")


def extract_job_keywords(job_description: str, top_n: int = 25) -> list[str]:
    """Extract important JD keywords and keyphrases from the current job description."""
    snippets = [
        clean_resume_text(snippet)
        for snippet in re.split(r"[\n.;:()]+", job_description)
        if clean_resume_text(snippet)
    ]
    if not snippets:
        return []

    vectorizer = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        ngram_range=(1, 2),
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+#.-]{1,}\b",
        max_features=top_n,
    )
    try:
        matrix = vectorizer.fit_transform(snippets)
    except ValueError:
        return []

    keywords = vectorizer.get_feature_names_out()
    counts = matrix.toarray().sum(axis=0)
    ranked = sorted(zip(keywords, counts, strict=True), key=lambda item: (-item[1], item[0]))
    return [keyword for keyword, count in ranked if count > 0][:top_n]


def keyword_in_text(keyword: str, cleaned_text: str) -> bool:
    """Check a keyword/keyphrase as a token-aware text match."""
    escaped = re.escape(keyword)
    pattern = rf"(?<![a-zA-Z0-9+#.-]){escaped}(?![a-zA-Z0-9+#.-])"
    return re.search(pattern, cleaned_text) is not None


def find_keyword_matches(resume_text: str, keywords: list[str]) -> tuple[list[str], list[str]]:
    """Split JD keywords into matched and missing lists for one resume."""
    cleaned_resume = clean_resume_text(resume_text)
    matched = [keyword for keyword in keywords if keyword_in_text(keyword, cleaned_resume)]
    missing = [keyword for keyword in keywords if not keyword_in_text(keyword, cleaned_resume)]
    return matched, missing


def extract_project_text(resume_text: str) -> str:
    """Return the resume project section when one can be detected."""
    heading_pattern = re.compile(
        r"\b(projects?|project experience|academic projects|personal projects)\b",
        flags=re.IGNORECASE,
    )
    next_heading_pattern = re.compile(
        r"\n\s*(education|skills?|experience|work experience|certifications?|achievements?|"
        r"publications?|languages?|interests?)\b",
        flags=re.IGNORECASE,
    )
    match = heading_pattern.search(resume_text)

    if not match:
        return ""

    remaining_text = resume_text[match.end() :]
    next_heading = next_heading_pattern.search(remaining_text)
    if next_heading:
        return remaining_text[: next_heading.start()].strip()

    return remaining_text.strip()


def calculate_project_relevance(job_description: str, project_text: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Score how closely resume projects match the job description."""
    if not project_text.strip():
        return 0.0, []

    matched_project_keywords, _ = find_keyword_matches(project_text, keywords)
    keyword_score = len(matched_project_keywords) / len(keywords) if keywords else 0.0
    documents = [clean_resume_text(job_description), clean_resume_text(project_text)]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return keyword_score, matched_project_keywords

    similarity_score = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2]).flatten()[0])
    project_score = (0.6 * similarity_score) + (0.4 * keyword_score)
    return project_score, matched_project_keywords


def score_resume_texts(
    job_description: str,
    resumes: list[tuple[str, str, str]],
    top_keywords: int = 25,
) -> list[ResumeScore]:
    """Rank already-extracted resume text by JD fit."""
    resume_texts = [resume_text for _, _, resume_text in resumes]
    cleaned_documents = [clean_resume_text(job_description), *map(clean_resume_text, resume_texts)]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned_documents)
        similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    except ValueError:
        similarities = [0.0 for _ in resume_texts]

    keywords = extract_job_keywords(job_description, top_keywords)

    scores: list[ResumeScore] = []
    for (name, source, text), similarity in zip(resumes, similarities, strict=True):
        matched, missing = find_keyword_matches(text, keywords)
        keyword_coverage = len(matched) / len(keywords) if keywords else 0.0
        project_score, matched_project_keywords = calculate_project_relevance(
            job_description,
            extract_project_text(text),
            keywords,
        )
        final_score = (0.55 * float(similarity)) + (0.25 * keyword_coverage) + (0.20 * project_score)
        scores.append(
            ResumeScore(
                resume=name,
                source=source,
                score=round(final_score * 100, 2),
                similarity=round(float(similarity) * 100, 2),
                keyword_coverage=round(keyword_coverage * 100, 2),
                project_relevance=round(project_score * 100, 2),
                matched_keywords=matched,
                missing_keywords=missing,
                matched_project_keywords=matched_project_keywords,
            )
        )

    return sorted(scores, key=lambda result: result.score, reverse=True)


def score_resumes(job_description: str, resume_paths: list[Path], top_keywords: int = 25) -> list[ResumeScore]:
    """Rank resumes by JD similarity and keyword coverage."""
    resumes = [(path.name, str(path), extract_resume_text(path)) for path in resume_paths]
    return score_resume_texts(job_description, resumes, top_keywords)


def scores_to_frame(scores: list[ResumeScore]) -> pd.DataFrame:
    """Convert ranking results to a report-friendly table."""
    return pd.DataFrame(
        [
            {
                "rank": index,
                "resume": score.resume,
                "source": score.source,
                "score": score.score,
                "similarity": score.similarity,
                "keyword_coverage": score.keyword_coverage,
                "project_relevance": score.project_relevance,
                "matched_keywords": ", ".join(score.matched_keywords),
                "missing_keywords": ", ".join(score.missing_keywords),
                "matched_project_keywords": ", ".join(score.matched_project_keywords),
            }
            for index, score in enumerate(scores, start=1)
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Screen resumes against a job description.")
    parser.add_argument("--job-description", type=str)
    parser.add_argument("--job-file", type=Path)
    parser.add_argument("--resumes", type=Path, nargs="+", required=True)
    parser.add_argument("--top-keywords", type=int, default=25)
    parser.add_argument("--output", type=Path, help="Optional CSV report path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    job_description = load_job_description(args.job_description, args.job_file)
    resume_paths = collect_resume_paths(args.resumes)

    if not resume_paths:
        raise ValueError("No supported resume files found.")

    scores = score_resumes(job_description, resume_paths, args.top_keywords)
    report = scores_to_frame(scores)

    display_columns = [
        "rank",
        "resume",
        "score",
        "similarity",
        "keyword_coverage",
        "project_relevance",
        "matched_keywords",
    ]
    print(report[display_columns].to_string(index=False))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
        print(f"\nSaved screening report to {args.output}")


if __name__ == "__main__":
    main()
