# 📄  Resume Intelligence System

A job-agnostic resume screener with semantic matching, skill gap analysis, interactive visualizations, and downloadable ATS-style reports.

---

## 🚀 Features

- **Job-agnostic**: Works for any role or domain
- **Semantic matching**: Uses Sentence Transformers (SBERT) for context-aware similarity
- **Skill extraction**: NLP-based with spaCy and dictionary matching
- **ATS-style scoring**: Weighted semantic + skill match score
- **Interactive visualizations**: Bar charts, pie charts, and progress metrics
- **model feedback**: Suggestions for missing skills and resume improvements
- **PDF report download**
- **Resume upload**: Supports PDF and DOCX files

---

## 🧠 Tech Stack

- **Python 3.10+**
- Streamlit
- Sentence Transformers (`all-MiniLM-L6-v2`)
- PyTorch
- spaCy + NLTK
- Scikit-learn
- pdfplumber + python-docx
- ReportLab
- Matplotlib / Numpy

---

## ⚙️ Installation

1. Clone the repo:

```bash
git clone https://github.com/YOUR_USERNAME/resume-screener.git
cd resume-screener
Install dependencies:

bash
Copy code
pip install -r requirements.txt
python -m spacy download en_core_web_sm
Run locally:

bash
Copy code
streamlit run app.py
