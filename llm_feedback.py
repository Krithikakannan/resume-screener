from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

model = SentenceTransformer("all-MiniLM-L6-v2")


#Extract Resume & JD Sentences
def get_sentences(text, min_words=8):
    sentences = sent_tokenize(text)
    return [s for s in sentences if len(s.split()) >= min_words]

# Detect JD Requirements NOT Covered Well
def find_weak_matches(resume_text, jd_text, threshold=0.55):
    resume_sents = get_sentences(resume_text)
    jd_sents = get_sentences(jd_text)

    if not resume_sents or not jd_sents:
        return []

    resume_emb = model.encode(resume_sents)
    jd_emb = model.encode(jd_sents)

    sim_matrix = cosine_similarity(resume_emb, jd_emb)

    feedback_points = []

    for i, jd_sent in enumerate(jd_sents):
        best_match = sim_matrix[:, i].max()

        if best_match < threshold:
            feedback_points.append(jd_sent)

    return feedback_points


# Convert Weak Matches → Human Feedback
def generate_model_feedback(resume_text, jd_text):
    weak_requirements = find_weak_matches(resume_text, jd_text)

    feedback = []

    for req in weak_requirements[:5]:  # limit feedback
        feedback.append(
            f"Your resume does not clearly address this requirement: '{req}'. "
            "Consider adding a concrete example or achievement related to it."
        )

    return feedback



