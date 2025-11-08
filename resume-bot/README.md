# Resume Bot (ATS Analyzer)

FastAPI service that:
- Trains a simple TF-IDF + Logistic Regression model to classify "good"/"bad" resumes
- Runs ATS-style heuristics (sections, contact info, bullets, length)
- Suggests keywords from a job description and shows missing ones
- Accepts text or uploaded files (PDF/DOCX/TXT)

## Endpoints

### 1) Health
GET /health

### 2) Train (retrain from bundled dataset)
POST /train

Response:
{
  "train_samples": 20,
  "train_accuracy_on_small_set": 0.95
}

### 3) Analyze (text)
POST /analyze
Body (JSON):
{
  "resume_text": "string",
  "job_description": "optional string",
  "target_role": "optional string"
}

### 4) Analyze File (PDF/DOCX/TXT)
POST /analyze-file
form-data:
- file: (binary)
- job_description: (text, optional)
- target_role: (text, optional)

## Local Dev

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
