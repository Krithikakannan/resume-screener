from visualization import (
    show_ats_progress,
    show_skill_bar_chart,
    show_skill_coverage_pie,
    show_score_components
)

import streamlit as st
import re
import numpy as np
import nltk
import spacy
import nltk
import spacy
from spacy.cli import download

# Ensure NLTK punkt tokenizer is available
nltk.download("punkt")

# Try to load spaCy English model; download if missing
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from file_parser import parse_file
from report_generator import generate_pdf_report
from llm_feedback import generate_model_feedback

# ------------------ SETUP ------------------

nltk.download("punkt")
nlp = spacy.load("en_core_web_sm")
model = SentenceTransformer("all-MiniLM-L6-v2")

st.set_page_config(page_title="Universal Resume Intelligence", layout="wide")

# ------------------ UTILS ------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_skill_corpus(path="skills.txt"):
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip().lower() for line in f if line.strip())


# ------------------ SKILL EXTRACTION ------------------

def extract_phrases(text):
    doc = nlp(text)
    phrases = set()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip().lower()
        if 2 <= len(phrase.split()) <= 4:
            phrases.add(phrase)

    return phrases


def extract_skills(text, skill_corpus):
    text = clean_text(text)
    phrases = extract_phrases(text)
    found = set()

    for skill in skill_corpus:
        if skill in text:
            found.add(skill)

    for phrase in phrases:
        if phrase in skill_corpus:
            found.add(phrase)

    return sorted(found)


# ------------------ SEMANTIC MATCHING ------------------

def chunk_text(text, min_words=8):
    sentences = sent_tokenize(text)
    return [s for s in sentences if len(s.split()) >= min_words]


def semantic_similarity(resume_text, jd_text):
    resume_chunks = chunk_text(resume_text)
    jd_chunks = chunk_text(jd_text)

    if not resume_chunks or not jd_chunks:
        return 0.0

    resume_emb = model.encode(resume_chunks)
    jd_emb = model.encode(jd_chunks)

    sim_matrix = cosine_similarity(resume_emb, jd_emb)
    best_matches = sim_matrix.max(axis=0)

    return float(np.mean(best_matches))


# ------------------ ATS SCORE ------------------

def skill_match_ratio(resume_skills, jd_skills):
    if not jd_skills:
        return 0.0
    return len(set(resume_skills) & set(jd_skills)) / len(jd_skills)


def compute_ats_score(semantic_score, skill_ratio, semantic_weight=0.6):
    return round(
        semantic_score * semantic_weight +
        (skill_ratio * 100) * (1 - semantic_weight),
        2
    )


# ------------------ FEEDBACK ------------------

def resume_feedback(resume_text, resume_skills, jd_skills):
    feedback = []

    wc = len(resume_text.split())
    if wc < 250:
        feedback.append("Resume is short. Add more details about projects or experience.")
    elif wc > 1500:
        feedback.append("Resume is long. Consider summarizing key achievements.")

    missing = set(jd_skills) - set(resume_skills)
    if missing:
        feedback.append(
            f"Consider adding or strengthening: {', '.join(sorted(missing))}"
        )

    for sec in ["projects", "experience", "skills", "education"]:
        if sec not in resume_text.lower():
            feedback.append(f"Add a '{sec}' section if applicable.")

    return feedback


# ------------------ STREAMLIT UI ------------------

st.title("📄 Universal Resume Intelligence System")
st.write("Job-agnostic | Semantic | Explainable ATS")

# ---------- INPUT SECTION ----------
st.subheader("📥 Input Resume & Job Description")

col1, col2 = st.columns(2)

with col1:
    uploaded_resume = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"]
    )
    resume_text_manual = st.text_area(
        "Or paste resume text",
        height=220
    )

with col2:
    jd_text = st.text_area(
        "Paste Job Description Text",
        height=300
    )

resume_text = ""
if uploaded_resume:
    resume_text = parse_file(uploaded_resume)
elif resume_text_manual.strip():
    resume_text = resume_text_manual.strip()

# ---------- ANALYSIS ----------
if st.button("🚀 Analyze Resume"):
    if resume_text and jd_text:

        skill_corpus = load_skill_corpus()

        resume_skills = extract_skills(resume_text, skill_corpus)
        jd_skills = extract_skills(jd_text, skill_corpus)

        matched_skills = sorted(set(resume_skills) & set(jd_skills))
        missing_skills = sorted(set(jd_skills) - set(resume_skills))

        semantic_score = round(
            semantic_similarity(resume_text, jd_text) * 100, 2
        )

        skill_ratio = skill_match_ratio(resume_skills, jd_skills)
        ats_score = compute_ats_score(semantic_score, skill_ratio)

        feedback = resume_feedback(resume_text, resume_skills, jd_skills)
        model_feedback = generate_model_feedback(resume_text, jd_text)

        # ---------- VISUALS ----------
        show_ats_progress(ats_score)
        show_skill_bar_chart(matched_skills, missing_skills)
        show_skill_coverage_pie(jd_skills, matched_skills)
        show_score_components(semantic_score, skill_ratio)

        # ---------- OUTPUT ----------
        st.success(f"📊 ATS Weighted Score: {ats_score}%")
        st.write(
            f"Semantic Match: {semantic_score}% | "
            f"Skill Match: {round(skill_ratio * 100, 2)}%"
        )

        st.subheader("✅ Matched Skills")
        st.write(", ".join(matched_skills) if matched_skills else "None")

        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "None")

        st.subheader("💡 Resume Feedback")
        if feedback:
            for f in feedback:
                st.write(f"- {f}")
        else:
            st.write("No structural issues detected.")

        st.subheader("🧠 AI-Based Resume Feedback")
        if model_feedback:
            for fb in model_feedback:
                st.write(f"- {fb}")
        else:
            st.write("Your resume addresses most job requirements well.")

        # ---------- PDF REPORT ----------
        report_path = "ats_report.pdf"

        generate_pdf_report(
            report_path,
            ats_score,
            semantic_score,
            skill_ratio,
            matched_skills,
            missing_skills,
            feedback,
            model_feedback
        )

        with open(report_path, "rb") as f:
            st.download_button(
                "📥 Download ATS Report (PDF)",
                f,
                file_name="ATS_Report.pdf",
                mime="application/pdf"
            )

    else:
        st.warning("Please provide both resume and job description.")
