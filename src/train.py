"""Train and evaluate resume classification models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .preprocessing import clean_resume_text


RANDOM_STATE = 42


def load_dataset(path: Path) -> pd.DataFrame:
    """Load and validate resume training data."""
    data = pd.read_csv(path)
    required_columns = {"resume_text", "category"}
    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    data = data.dropna(subset=["resume_text", "category"]).copy()
    data["resume_text"] = data["resume_text"].astype(str).map(clean_resume_text)
    data["category"] = data["category"].astype(str)

    if data.empty:
        raise ValueError("Dataset has no usable rows after cleaning.")

    return data


def build_pipeline(model: object) -> Pipeline:
    """Create a text classification pipeline."""
    return Pipeline(
        steps=[
            (
                "features",
                ColumnTransformer(
                    transformers=[
                        (
                            "resume_tfidf",
                            TfidfVectorizer(stop_words="english", sublinear_tf=True),
                            "resume_text",
                        )
                    ],
                    remainder="drop",
                ),
            ),
            ("classifier", model),
        ]
    )


def train_best_model(data: pd.DataFrame) -> tuple[GridSearchCV, pd.DataFrame, pd.Series]:
    """Train candidate models and return the best search result plus holdout data."""
    class_counts = data["category"].value_counts()
    test_size = max(len(class_counts), int(round(len(data) * 0.25)))

    x_train, x_test, y_train, y_test = train_test_split(
        data[["resume_text"]],
        data["category"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=data["category"],
    )

    candidates = [
        {
            "classifier": [LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)],
            "features__resume_tfidf__ngram_range": [(1, 1), (1, 2)],
            "features__resume_tfidf__max_features": [1000, 3000],
            "classifier__C": [0.5, 1.0, 2.0],
        },
        {
            "classifier": [MultinomialNB()],
            "features__resume_tfidf__ngram_range": [(1, 1), (1, 2)],
            "features__resume_tfidf__max_features": [1000, 3000],
            "classifier__alpha": [0.25, 0.5, 1.0],
        },
    ]

    min_train_class_count = int(y_train.value_counts().min())
    cv_splits = min(4, min_train_class_count)
    if cv_splits < 2:
        raise ValueError("Each category needs at least two examples for stratified training.")

    search = GridSearchCV(
        estimator=build_pipeline(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        param_grid=candidates,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
        refit=True,
    )
    search.fit(x_train, y_train)

    return search, x_test, y_test


def save_model(model: GridSearchCV, output_path: Path) -> None:
    """Persist the fitted best estimator."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.best_estimator_, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train resume screening classifiers.")
    parser.add_argument("--data", type=Path, default=Path("data/sample_resumes.csv"))
    parser.add_argument("--model-out", type=Path, default=Path("models/resume_classifier.joblib"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = load_dataset(args.data)
    search, x_test, y_test = train_best_model(data)
    predictions = search.predict(x_test)

    print(f"Best model: {search.best_estimator_.named_steps['classifier'].__class__.__name__}")
    print(f"Best CV accuracy: {search.best_score_:.3f}")
    print(f"Holdout accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Best parameters:")
    for key, value in search.best_params_.items():
        print(f"  {key}: {value}")
    print()
    print(classification_report(y_test, predictions, zero_division=0))

    save_model(search, args.model_out)
    print(f"Saved model to {args.model_out}")


if __name__ == "__main__":
    main()
