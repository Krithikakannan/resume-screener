from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4


def generate_pdf_report(
    filepath,
    ats_score,
    semantic_score,
    skill_ratio,
    matched_skills,
    missing_skills,
    feedback,
    model_feedback
):
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>ATS Resume Analysis Report</b>", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"<b>ATS Score:</b> {ats_score}%", styles["Normal"]))
    story.append(Paragraph(f"<b>Semantic Match:</b> {semantic_score}%", styles["Normal"]))
    story.append(Paragraph(
        f"<b>Skill Match:</b> {round(skill_ratio * 100, 2)}%", styles["Normal"]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Matched Skills</b>", styles["Heading2"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Normal"])) for s in matched_skills]
        or [Paragraph("None", styles["Normal"])]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Missing Skills</b>", styles["Heading2"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Normal"])) for s in missing_skills]
        or [Paragraph("None", styles["Normal"])]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>Resume Feedback</b>", styles["Heading2"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(f, styles["Normal"])) for f in feedback]
        or [Paragraph("No structural issues detected.", styles["Normal"])]
    ))

    story.append(Spacer(1, 12))

    story.append(Paragraph("<b>AI-Based Feedback</b>", styles["Heading2"]))
    story.append(ListFlowable(
        [ListItem(Paragraph(fb, styles["Normal"])) for fb in model_feedback]
        or [Paragraph("Resume aligns well with the job description.", styles["Normal"])]
    ))

    doc = SimpleDocTemplate(filepath, pagesize=A4)
    doc.build(story)
