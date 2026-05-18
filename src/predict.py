"""Run predictions with a trained resume classification model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .preprocessing import clean_resume_text
from .resume_parser import extract_resume_text


def load_resume_text(text: str | None, file_path: Path | None) -> str:
    if text and file_path:
        raise ValueError("Use either --text or --file, not both.")
    if file_path:
        return extract_resume_text(file_path)
    if text:
        return text
    raise ValueError("Provide resume content with --text or --file.")


def predict_category(model_path: Path, resume_text: str) -> tuple[str, float | None]:
    model = joblib.load(model_path)
    frame = pd.DataFrame({"resume_text": [clean_resume_text(resume_text)]})
    prediction = str(model.predict(frame)[0])

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(frame)[0]
        confidence = float(probabilities.max())

    return prediction, confidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a resume category.")
    parser.add_argument("--model", type=Path, default=Path("models/resume_classifier.joblib"))
    parser.add_argument("--text", type=str)
    parser.add_argument("--file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resume_text = load_resume_text(args.text, args.file)
    category, confidence = predict_category(args.model, resume_text)

    print(f"Predicted category: {category}")
    if confidence is not None:
        print(f"Confidence: {confidence:.3f}")


if __name__ == "__main__":
    main()
