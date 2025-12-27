import matplotlib.pyplot as plt
import streamlit as st

def show_ats_progress(ats_score):
    st.subheader("📊 ATS Match Score")
    st.progress(ats_score / 100)
    st.write(f"**ATS Score:** {ats_score}%")


def show_skill_bar_chart(matched_skills, missing_skills):
    matched_count = len(matched_skills)
    missing_count = len(missing_skills)

    fig, ax = plt.subplots()
    ax.bar(
        ["Matched Skills", "Missing Skills"],
        [matched_count, missing_count]
    )

    ax.set_ylabel("Number of Skills")
    ax.set_title("Skill Match Analysis")

    st.pyplot(fig)


def show_skill_coverage_pie(jd_skills, matched_skills):
    covered = len(matched_skills)
    total = len(jd_skills)
    missing = max(total - covered, 0)

    if total == 0:
        st.write("No skills found in job description.")
        return

    fig, ax = plt.subplots()
    ax.pie(
        [covered, missing],
        labels=["Covered Skills", "Missing Skills"],
        autopct="%1.1f%%",
        startangle=90
    )
    ax.set_title("Skill Coverage Ratio")
    ax.axis("equal")

    st.pyplot(fig)


def show_score_components(semantic_score, skill_ratio):
    fig, ax = plt.subplots()
    ax.bar(
        ["Semantic Match", "Skill Match"],
        [semantic_score, skill_ratio * 100]
    )

    ax.set_ylim(0, 100)
    ax.set_ylabel("Score (%)")
    ax.set_title("ATS Score Components")

    st.pyplot(fig)
