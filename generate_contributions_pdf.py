import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = "AI_A1_G08_CONTRIBUTIONS.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'TitleStyle',
    parent=styles['Heading1'],
    fontSize=18,
    leading=22,
    textColor=colors.HexColor('#1A365D'),
    alignment=1,
    spaceAfter=10
)

subtitle_style = ParagraphStyle(
    'SubtitleStyle',
    parent=styles['Normal'],
    fontSize=10,
    leading=14,
    textColor=colors.HexColor('#4A5568'),
    alignment=1,
    spaceAfter=15
)

h2_style = ParagraphStyle(
    'H2Style',
    parent=styles['Heading2'],
    fontSize=12,
    leading=16,
    textColor=colors.HexColor('#2B6CB0'),
    spaceBefore=10,
    spaceAfter=5
)

body_style = ParagraphStyle(
    'BodyStyle',
    parent=styles['Normal'],
    fontSize=8.5,
    leading=12,
    textColor=colors.HexColor('#2D3748')
)

elements = []

# Title & Metadata
elements.append(Paragraph("<b>AI ASSIGNMENT 1: GROUP CONTRIBUTION & ROLE MATRIX</b>", title_style))
elements.append(Paragraph("<b>Group Code:</b> AI_A1_G08 | <b>Course:</b> SWE3513 - Artificial Intelligence", subtitle_style))
elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceBefore=5, spaceAfter=15))

elements.append(Paragraph("Individual Member Contributions & Traceability Log", h2_style))
elements.append(Paragraph(
    "This document certifies the individual responsibilities, code contributions, Git traceability, and work deliverables "
    "completed by each group member for AI Assignment 1.",
    body_style
))
elements.append(Spacer(1, 10))

# Table Data
contrib_data = [
    ["Member Name & Reg No.", "Assigned Role", "Requirements Met", "Key Files Modified / Created", "Git Branch & Commits", "Signature"],
    [
        "Aisha\n(22001122)",
        "Data Engineer / Pipeline",
        "Data loading, cleaning, feature scaling, train/test split",
        "src/data_pipeline.py\nsrc/utils.py",
        "feature/data-pipeline\n3+ commits",
        "Signed"
    ],
    [
        "Kassimu\n(22003344)",
        "ML Engineer (Regression & Classification)",
        "Regression model, Classification model, metrics calculation",
        "src/models/regression.py\nsrc/models/classification.py",
        "feature/ml-models\n3+ commits",
        "Signed"
    ],
    [
        "Ineza\n(22005566)",
        "Unsupervised Lead (Clustering)",
        "K-Means clustering, silhouette score, cluster visualization plot",
        "src/models/clustering.py\nartifacts/cluster_plot.png",
        "feature/clustering\n2+ commits",
        "Signed"
    ],
    [
        "Sumaya\n(22007788)",
        "Inference & API Lead",
        "predict.py schema, CLI validation, JSON formatting",
        "predict.py\nrun_all.py",
        "feature/predict-cli\n3+ commits",
        "Signed"
    ],
    [
        "Mahgoub\n(22009900)",
        "QA & Documentation Lead",
        "pytest test suite, test log generation, UI/UX specification",
        "tests/test_pipeline.py\nevidence/TEST_LOG.pdf",
        "feature/testing-qa\n3+ commits",
        "Signed"
    ]
]

t_contrib = Table(contrib_data, colWidths=[95, 85, 120, 110, 80, 50])
t_contrib.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1A365D')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 8),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ('ALIGN', (5,1), (5,-1), 'CENTER'),
    ('FONTNAME', (5,1), (5,-1), 'Helvetica-Oblique'),
    ('TEXTCOLOR', (5,1), (5,-1), colors.HexColor('#2F855A')),
]))

elements.append(t_contrib)
elements.append(Spacer(1, 15))

elements.append(Paragraph("Declaration & Academic Honesty Statement", h2_style))
elements.append(Paragraph(
    "We hereby declare that all work presented in this submission was executed collaboratively by the group members listed above. "
    "Generative AI tools utilized during the development process have been transparently documented inside <code>evidence/AI_USE.md</code>.",
    body_style
))

doc.build(elements)
print("AI_A1_G08_CONTRIBUTIONS.pdf generated successfully!")
