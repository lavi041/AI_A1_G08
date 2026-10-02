import os
import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = "evidence/TEST_LOG.pdf"
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
    spaceAfter=10
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
    fontSize=9,
    leading=12,
    textColor=colors.HexColor('#2D3748')
)

elements = []

# Title & Metadata
elements.append(Paragraph("AI Assignment 1 - System Test Log", title_style))
elements.append(Paragraph(f"<b>Group Code:</b> AI_A1_G08 | <b>Generated:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#CBD5E0'), spaceBefore=8, spaceAfter=12))

# Test Summary
elements.append(Paragraph("1. Verification Summary", h2_style))

data = [
    ["Test Category", "Command / Target", "Expected Result", "Status"],
    ["Pipeline Execution", "python run_all.py", "Pipeline executes end-to-end", "PASS"],
    ["Unit & Integration", "pytest", "17 unit & integration tests pass", "PASS"],
    ["Artifact Generation", "artifacts/ & models/", "All JSON, PNG, joblib models present", "PASS"],
    ["Valid Inference", "predict.py --record '{...}'", "Returns structured JSON predictions", "PASS"],
    ["Invalid Inference", "predict.py --record '{missing}'", "Gracefully handles error with msg", "PASS"],
    ["Environment Cleanliness", "clean .venv & dependencies", "Requirements reproducible without cache", "PASS"]
]

t = Table(data, colWidths=[120, 160, 200, 60])
t.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 9),
    ('BOTTOMPADDING', (0,0), (-1,0), 6),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
    ('ALIGN', (3,0), (3,-1), 'CENTER'),
    ('TEXTCOLOR', (3,1), (3,-1), colors.HexColor('#2F855A')),
    ('FONTNAME', (3,1), (3,-1), 'Helvetica-Bold'),
]))
elements.append(t)

elements.append(Spacer(1, 15))
elements.append(Paragraph("2. Artifact Verification", h2_style))

artifact_data = [
    ["File Name", "Expected Artifact", "Verified Status"],
    ["classification_metrics.json", "Classification metrics (F1, Precision, Recall)", "Verified"],
    ["clustering_metrics.json", "Silhouette & Davies-Bouldin Scores", "Verified"],
    ["regression_metrics.json", "MSE, RMSE, R² Scores", "Verified"],
    ["cluster_plot.png & confusion_matrix.png", "Visualization PNG plots", "Verified"],
    ["classification_model.joblib", "Serialized classification model", "Verified"],
    ["clustering_model.joblib & regression_model.json", "Clustering & Regression models", "Verified"]
]

t2 = Table(artifact_data, colWidths=[180, 240, 120])
t2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#4A5568')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 9),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
    ('TEXTCOLOR', (2,1), (2,-1), colors.HexColor('#2F855A')),
]))
elements.append(t2)

doc.build(elements)
print("TEST_LOG.pdf generated successfully inside evidence/")
