import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

pdf_path = "AI_A1_G08_UIUX.pdf"
doc = SimpleDocTemplate(
    pdf_path,
    pagesize=letter,
    rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'CoverTitle',
    parent=styles['Heading1'],
    fontSize=22,
    leading=26,
    textColor=colors.HexColor('#1A365D'),
    alignment=1,
    spaceAfter=15
)

subtitle_style = ParagraphStyle(
    'CoverSubtitle',
    parent=styles['Normal'],
    fontSize=12,
    leading=16,
    textColor=colors.HexColor('#4A5568'),
    alignment=1,
    spaceAfter=25
)

h1_style = ParagraphStyle(
    'SectionH1',
    parent=styles['Heading1'],
    fontSize=15,
    leading=18,
    textColor=colors.HexColor('#1A365D'),
    spaceBefore=10,
    spaceAfter=10
)

body_style = ParagraphStyle(
    'BodyStyle',
    parent=styles['Normal'],
    fontSize=9.5,
    leading=14,
    textColor=colors.HexColor('#2D3748')
)

elements = []

# --- PAGE 1: COVER & PROBLEM STATEMENT ---
elements.append(Spacer(1, 40))
elements.append(Paragraph("<b>AI ASSIGNMENT 1: UI/UX & SYSTEM SPECIFICATION</b>", title_style))
elements.append(Paragraph("<b>Group Code:</b> AI_A1_G08 | <b>Course:</b> SWE3513 - Artificial Intelligence", subtitle_style))
elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceBefore=10, spaceAfter=20))

elements.append(Paragraph("1. Executive Summary & Problem Statement", h1_style))
elements.append(Paragraph(
    "Smallholder farmers and agricultural managers often face severe yields unpredictability due to erratic weather, "
    "soil acidity variations, and logistical delays. Standard machine learning solutions provide numerical outputs, "
    "but often lack human-centered interpretability and input validation, leading to poor operational decisions.<br/><br/>"
    "<b>Objective:</b> Design an end-to-end Machine Learning interface (CLI and API schema) that predicts farm productivity, "
    "categorizes yields into risk buckets, and clusters operational regions while enforcing strict input boundary checks and "
    "transparent Responsible AI feedback loops.",
    body_style
))

elements.append(Spacer(1, 15))
elements.append(Paragraph("2. Primary User Personas & Target Audience", h1_style))

persona_data = [
    ["User Role", "Primary Goal", "Key Technical Needs", "Pain Points"],
    ["Agronomist Lead", "Analyze historical yield predictors", "Batch inference, JSON output", "Black-box models without metrics"],
    ["Field Coordinator", "Predict yield per plot on real data", "CLI interface, immediate errors", "Cryptic stack traces on invalid data"],
    ["System Auditor", "Verify model metrics and artifacts", "Traceable evaluation logs", "Unverified parameters or missing models"]
]

t_persona = Table(persona_data, colWidths=[110, 140, 150, 140])
t_persona.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 8.5),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
    ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
]))
elements.append(t_persona)
elements.append(PageBreak())

# --- PAGE 2: USER JOURNEY & WORKFLOW DIAGRAM ---
elements.append(Paragraph("3. End-to-End User Journey", h1_style))
elements.append(Paragraph(
    "The diagram below outlines the interaction pipeline for field users executing predictions via the system CLI interface.",
    body_style
))
elements.append(Spacer(1, 10))

journey_data = [
    ["Phase", "User Action", "System Processing", "Output / Feedback"],
    ["1. Initialization", "Runs predict.py with JSON payload", "Loads model JSON/joblib models & scaler", "Models validated in memory"],
    ["2. Input Parsing", "Passes raw JSON record parameters", "Schema validation & missing key audit", "Returns structured error if invalid"],
    ["3. Preprocessing", "Data enters pipeline", "StandardScaling applied to input feature vector", "Features normalized"],
    ["4. Multi-Model Inference", "Inference triggered", "Regression + Classification + Clustering run", "Predictions computed concurrently"],
    ["5. Structuring", "Formatter processes results", "JSON response built with version & group code", "Standardized JSON printed to stdout"]
]

t_journey = Table(journey_data, colWidths=[80, 140, 170, 150])
t_journey.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2D3748')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 8.5),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFFFF')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
elements.append(t_journey)
elements.append(PageBreak())

# --- PAGE 3: ANNOTATED WIREFRAME 1 (INPUT INTERFACE) ---
elements.append(Paragraph("4. Wireframe Layout 1: Input CLI Payload Schema", h1_style))
elements.append(Paragraph(
    "Annotated conceptual wireframe showing the structure and constraints of input payload passing into `predict.py`.",
    body_style
))
elements.append(Spacer(1, 15))

wf1_data = [
    ["[CLI Wireframe Box - Input Interface]", "Annotations & Specifications"],
    ["$ python predict.py --record '{\n  \"plot_area_ha\": 1.2,\n  \"rainfall_mm\": 81,\n  \"soil_ph\": 5.7,\n  \"seed_kg\": 210,\n  \"distance_km\": 14,\n  \"arrival_hour\": 9\n}'",
     "• Requirement: All 6 keys must be present.\n• Type Checking: Numeric floats/ints.\n• Bound Verification:\n  - soil_ph: 0.0 to 14.0\n  - plot_area_ha > 0\n  - arrival_hour: 0 to 23\n• Error Trigger: Any missing key generates explicit JSON error."]
]

t_wf1 = Table(wf1_data, colWidths=[270, 270])
t_wf1.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,0), colors.HexColor('#EDF2F7')),
    ('BACKGROUND', (1,0), (1,0), colors.HexColor('#E2E8F0')),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#4A5568')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('FONTSIZE', (0,1), (0,1), 8.5),
]))
elements.append(t_wf1)
elements.append(PageBreak())

# --- PAGE 4: ANNOTATED WIREFRAME 2 (OUTPUT INTERFACE) ---
elements.append(Paragraph("5. Wireframe Layout 2: Output JSON Response Interface", h1_style))
elements.append(Paragraph(
    "Annotated wireframe illustrating success and error outputs from the prediction module.",
    body_style
))
elements.append(Spacer(1, 15))

wf2_data = [
    ["[CLI Output Wireframe - Response Payload]", "Field Explanations"],
    ["{\n  \"regression_prediction\": 4.12,\n  \"classification_prediction\": 1,\n  \"classification_probability\": 0.89,\n  \"cluster_label\": 2,\n  \"group_code\": \"AI_A1_G08\",\n  \"model_version\": \"1.0.0\"\n}",
     "1. regression_prediction: Yield target in Tons.\n2. classification_prediction: Binary High/Low yield flag.\n3. classification_probability: Model confidence score (0-1).\n4. cluster_label: Assigned agricultural zone index.\n5. Audit Tags: Group identifier and model iteration trace."]
]

t_wf2 = Table(wf2_data, colWidths=[270, 270])
t_wf2.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (0,0), colors.HexColor('#EBF8FF')),
    ('BACKGROUND', (1,0), (1,0), colors.HexColor('#BEE3F8')),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('GRID', (0,0), (-1,-1), 1, colors.HexColor('#2B6CB0')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ('FONTSIZE', (0,1), (0,1), 8.5),
]))
elements.append(t_wf2)
elements.append(PageBreak())

# --- PAGE 5: RESPONSIBLE AI STATES ---
elements.append(Paragraph("6. Responsible AI States & Edge-Case Handling", h1_style))
elements.append(Paragraph(
    "To comply with Responsible AI principles, the user experience must gracefully communicate errors, model limitations, and confidence thresholds.",
    body_style
))
elements.append(Spacer(1, 10))

rai_data = [
    ["System State", "Trigger Condition", "UI / System Response", "Responsible AI Guardrail"],
    ["Missing Key Error", "Payload lacks required feature (e.g. soil_ph)", "HTTP/CLI JSON Error Response", "Prevents model hallucination on zero-imputed data."],
    ["Out-of-Bounds Input", "Extremely unrealistic input (e.g. pH = 25)", "Input Validation Error", "Restricts predictions strictly within physically valid bounds."],
    ["Low Confidence Prediction", "Classification probability between 0.45 - 0.55", "Returns prediction with Low Confidence Flag", "Warns user not to make critical financial decisions without verification."],
    ["Malformed JSON Payload", "Syntax error in input string", "Parses JSON and returns clear JSON syntax error", "Avoids unexpected python crashes or security exposure."]
]

t_rai = Table(rai_data, colWidths=[100, 130, 160, 150])
t_rai.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#C53030')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 8.5),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#FEB2B2')),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFF5F5')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
elements.append(t_rai)
elements.append(PageBreak())

# --- PAGE 6: VISUAL SYSTEM RATIONALE ---
elements.append(Paragraph("7. Visual System Rationale & Design System", h1_style))
elements.append(Paragraph(
    "The visual and structural hierarchy of system outputs adheres to strict design guidelines for readability and integration into production logs.",
    body_style
))
elements.append(Spacer(1, 10))

design_data = [
    ["Design Element", "Choice / Standard", "Rationale"],
    ["Data Format", "Standard JSON", "Ensures language-agnostic integration for frontend dashboards and web services."],
    ["Color Codes (Terminal)", "ANSI Color Traces (Green/Red)", "Improves developer immediate recognition during automated test logging."],
    ["Numerical Precision", "Floats rounded to 2 decimal places", "Avoids false precision perception while maintaining scientific accuracy."],
    ["Error Transparency", "Explicit list of missing keys", "Eliminates developer guesswork during payload debugging."]
]

t_design = Table(design_data, colWidths=[120, 160, 260])
t_design.setStyle(TableStyle([
    ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
    ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
    ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ('FONTSIZE', (0,0), (-1,0), 8.5),
    ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
    ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#F7FAFC')),
    ('VALIGN', (0,0), (-1,-1), 'TOP'),
]))
elements.append(t_design)

doc.build(elements)
print("AI_A1_G08_UIUX.pdf generated successfully with 6 pages!")
