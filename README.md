# AI Resume Screening System

An NLP-based resume screening project built with Python and Scikit-learn. It supports two practical workflows:

1. Train a resume category classifier with TF-IDF, Logistic Regression, and Naive Bayes.
2. Screen real PDF/TXT resumes against a user-provided job description and rank candidates by fit.
3. Use a simple browser UI to upload many resumes and rank them without command-line input.

## Features

- Parses real resume files in `.pdf` and `.txt` format.
- Accepts a custom job description for each screening run.
- Dynamically extracts job-specific keywords and keyphrases from the job description.
- Converts job descriptions and resumes into TF-IDF vectors.
- Ranks resumes using cosine similarity plus keyword coverage.
- Checks whether the resume projects are relevant to the job description.
- Shows matched and missing job-description keywords for explainable shortlisting.
- Trains Logistic Regression and Naive Bayes classifiers for category prediction.
- Uses hyperparameter search to improve classifier accuracy.

## Project Structure

```text
.
|-- data/
|   |-- sample_job_description.txt
|   `-- sample_resumes.csv
|-- models/
|   |-- .gitkeep
|   `-- resume_classifier.joblib
|-- src/
|   |-- __init__.py
|   |-- predict.py
|   |-- preprocessing.py
|   |-- resume_parser.py
|   |-- screen.py
|   `-- train.py
|-- .gitignore
|-- app.py
|-- README.md
`-- requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Easiest Way: Use the UI

Start the app:

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

In the UI:

1. Paste the full job description in the text box.
2. Upload one or more candidate resumes as PDF or TXT files.
3. Click `Rank Resumes`.
4. Review each candidate's rank, score, keyword match, missing keywords, and project relevance.
5. Download the CSV report if needed.

## Screen Resumes Against a Job Description

Use this when the recruiter or user provides a job description and wants resumes ranked specifically for that role.

```bash
python -m src.screen --job-file data/sample_job_description.txt --resumes "C:\Users\deepu\Downloads\resume_.pdf"
```

You can also pass a whole folder of resumes:

```bash
python -m src.screen --job-file data/sample_job_description.txt --resumes "C:\Users\deepu\Downloads\resumes" --output reports\screening_report.csv
```

Or provide the job description inline:

```bash
python -m src.screen --job-description "Python NLP engineer with scikit-learn, TF-IDF, model evaluation, and resume parsing experience" --resumes "C:\Users\deepu\Downloads\resume_.pdf"
```

The output includes:

- `score`: final candidate fit score.
- `similarity`: TF-IDF cosine similarity between the job description and resume.
- `keyword_coverage`: percentage of extracted JD keywords found in the resume.
- `project_relevance`: how strongly the resume's project section matches the JD.
- `matched_keywords`: JD keywords found in the resume.
- `missing_keywords`: JD keywords not found in the resume, included in CSV reports.

## Train Resume Category Classifier

```bash
python -m src.train --data data/sample_resumes.csv --model-out models/resume_classifier.joblib
```

The training script compares Logistic Regression and Naive Bayes, prints evaluation metrics, and saves the best pipeline.

## Predict a Resume Category

Predict from text:

```bash
python -m src.predict --model models/resume_classifier.joblib --text "Built REST APIs using Python, Flask, SQL, Docker, and AWS."
```

Predict from a PDF or text resume:

```bash
python -m src.predict --model models/resume_classifier.joblib 
```

## Dataset Format

Use a CSV file with these columns:

```csv
resume_text,category
```

`resume_text` should contain the resume content, and `category` should contain the target job role or department label.
