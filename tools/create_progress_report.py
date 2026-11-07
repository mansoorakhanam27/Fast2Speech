from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Report" / "Technical_Report.pdf"

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="CoverTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=24, leading=29, textColor=colors.HexColor("#14263D"), alignment=TA_CENTER, spaceAfter=14))
styles.add(ParagraphStyle(name="Deck", parent=styles["Normal"], fontSize=11, leading=16, textColor=colors.HexColor("#506176"), alignment=TA_CENTER, spaceAfter=18))
styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=14, leading=18, textColor=colors.HexColor("#174A6E"), spaceBefore=12, spaceAfter=6))
styles.add(ParagraphStyle(name="Body2", parent=styles["BodyText"], fontSize=9.5, leading=14, spaceAfter=6))
styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#536273")))
styles.add(ParagraphStyle(name="Cell", parent=styles["BodyText"], fontSize=8.5, leading=11))
styles.add(ParagraphStyle(name="CellHead", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.5, leading=11, textColor=colors.white))


def P(text, style="Body2"):
    return Paragraph(text, styles[style])


def footer(canvas, doc):
    canvas.saveState()
    w, h = letter
    canvas.setStrokeColor(colors.HexColor("#DCE4EA"))
    canvas.line(doc.leftMargin, 0.58 * inch, w - doc.rightMargin, 0.58 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#657587"))
    canvas.drawString(doc.leftMargin, 0.38 * inch, "FastSpeech 2 | Technical progress report | 7 November 2026")
    canvas.drawRightString(w - doc.rightMargin, 0.38 * inch, f"Page {doc.page}")
    canvas.restoreState()


story = [
    Spacer(1, 0.45 * inch),
    P("FastSpeech 2", "CoverTitle"),
    P("English and Telugu Assignment - Technical Progress Report", "Deck"),
    P("Status as of 7 November 2026", "Deck"),
    P("Executive summary", "Section"),
    P("This report documents completed, reproducible work and the current blockers. It is a progress report, not a claim that the assignment is complete. The English Svarah corpus was accessed, exported locally, and audited into duration-targeted train and validation splits. Telugu data have not been obtained or validated. Neither language has completed forced alignment, acoustic feature extraction, model training, or inference; therefore no model quality, loss curves, generated speech, or variance plots are reported.", "Body2"),
    P("Verified English data", "Section"),
]

data = [
    [P("Measure", "CellHead"), P("Observed", "CellHead")],
    [P("Source", "Cell"), P("ai4bharat/Svarah, source split 'test'", "Cell")],
    [P("Pinned revision", "Cell"), P("ebbf7777fe771490696a3f7b007097606fa8c924", "Cell")],
    [P("Export", "Cell"), P("6,656 WAV utterances; 9.6144 measured audio hours", "Cell")],
    [P("Training split", "Cell"), P("3,429 utterances; 5.0007 hours", "Cell")],
    [P("Validation split", "Cell"), P("686 utterances; 1.0001 hours", "Cell")],
    [P("Overlap / speakers", "Cell"), P("0 shared IDs; Female and Male labels in both splits", "Cell")],
    [P("Transcript handling", "Cell"), P("Split audit left transcript strings unchanged. Correctness has not been manually verified.", "Cell")],
]
t = Table(data, colWidths=[1.55 * inch, 4.95 * inch], repeatRows=1, hAlign="LEFT")
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A6E")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE5")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
    ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
]))
story += [t, Spacer(1, 8), P("The audit is recorded in <font name='Courier'>data/Svarah/splits/audit.json</font>; the export metadata is in <font name='Courier'>data/Svarah/export_summary.json</font>. Raw audio and manifests are local and excluded from version control. The Svarah dataset card identifies CC BY 4.0; see References.", "Small"), PageBreak()]

story += [
    P("Implementation and reproducibility", "Section"),
    P("The repository vendors the MIT-licensed ming024/FastSpeech2 reference implementation at commit d4e79eb52e8b01d24703b2dfc0385544092958f3. Language-specific configuration and wrapper scripts, a dataset exporter, and a deterministic duration-based manifest audit are present. The English split uses seed 42. The audit checks audio durations, duplicate IDs, missing files/transcripts, target duration, and train/validation ID overlap.", "Body2"),
    P("Environment checks found PyTorch 2.14.1+cpu with CUDA unavailable. The English preprocessing attempt did not reach audio preparation: initially required runtime dependencies were missing, and a subsequent attempt was stopped after it remained without visible progress. Montreal Forced Aligner (MFA) is not installed. No TextGrid alignments or acoustic features were produced. Some Python dependencies were installed into the project-local .venv; the system-wide Python environment was not intentionally modified.", "Body2"),
    P("Assignment status", "Section"),
]
status = [
    [P("Requirement", "CellHead"), P("Status", "Cell"), P("Evidence / remaining work", "CellHead")],
    [P("English corpus and 5 h / 1 h split", "Cell"), P("Complete", "Cell"), P("Verified Svarah export and disjoint split audit.", "Cell")],
    [P("Telugu corpus and split", "Cell"), P("Not started", "Cell"), P("A public mirror has conflicting language metadata; original IIT Madras access/license process remains unresolved. Do not assume the mirror is Telugu.", "Cell")],
    [P("Text normalization, lexicon, forced alignment", "Cell"), P("Not complete", "Cell"), P("Requires working language-specific frontend and MFA; Telugu pronunciation resources need validation.", "Cell")],
    [P("FastSpeech 2 training and checkpoints", "Cell"), P("Not started", "Cell"), P("Train separate language models; record configuration, hardware, and actual training/validation losses.", "Cell")],
    [P("Inference, five test cases per language, plots", "Cell"), P("Not started", "Cell"), P("Requires trained checkpoints; preserve assignment test strings exactly and render duration, pitch, and energy plots.", "Cell")],
    [P("Final technical report and public repository", "Cell"), P("Not complete", "Cell"), P("This progress report is not the final report. The workspace has no configured Git remote or public repository URL.", "Cell")],
]
st = Table(status, colWidths=[1.55 * inch, 0.92 * inch, 4.03 * inch], repeatRows=1, hAlign="LEFT")
st.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#174A6E")),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#D5DEE5")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
story += [st, PageBreak()]

story += [
    P("Next steps", "Section"),
    P("1. Obtain the original Telugu corpus through an authorized route, review its license, and verify transcript language, audio, speakers, and duration before export. 2. Install and validate MFA and language-specific pronunciation resources. 3. Run preprocessing and forced alignment for each language, manually inspect alignment samples, and extract mel, pitch, energy, and duration targets. 4. Train both models on a CUDA-capable machine, preserving logs and checkpoints. 5. Synthesize the exact assignment test sentences, generate the required plots, and complete the final report from measured results. 6. Create and verify a public GitHub repository containing code and documentation only; do not publish restricted audio or credentials.", "Body2"),
    P("Reproduce the completed split audit", "Section"),
    P("From the project root in PowerShell:", "Body2"),
    P("python tools\\audit_manifests.py --manifest data\\Svarah\\manifest.jsonl --audio-root data\\Svarah --output-dir data\\Svarah\\splits --language English", "Small"),
    P("The workspace is local at <font name='Courier'>C:\\Users\\manso\\OneDrive\\Documents\\ChatGPT\\Fast2Speech</font>. Git currently has no remote configured, so there is no public repository link to provide yet.", "Body2"),
    P("References", "Section"),
    P("1. FastSpeech 2 assignment handout supplied by the student, <i>FastSpeech2_Assignment.pdf</i>.", "Small"),
    P("2. ming024/FastSpeech2 reference implementation, commit d4e79eb52e8b01d24703b2dfc0385544092958f3. The upstream README reports under 10,000 steps (under one hour on a GTX 1080 Ti) to generate acceptable-quality samples; this is not a runtime guarantee for this project or its configurations. <link href='https://github.com/ming024/FastSpeech2' color='#176B96'>github.com/ming024/FastSpeech2</link>.", "Small"),
    P("3. AI4Bharat, Svarah dataset card and terms. <link href='https://huggingface.co/datasets/ai4bharat/Svarah' color='#176B96'>huggingface.co/datasets/ai4bharat/Svarah</link>.", "Small"),
    P("4. SPRINGLab IndicTTS Telugu dataset card. Its description includes conflicting language information; use only after verification. <link href='https://huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu' color='#176B96'>huggingface.co/datasets/SPRINGLab/IndicTTS_Telugu</link>.", "Small"),
    Spacer(1, 10),
    P("Limitations: This report records project status and a dataset split audit only. The audit does not prove transcript accuracy or alignment. No model metrics or generated outputs exist yet.", "Small"),
]

OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=letter, rightMargin=0.72 * inch, leftMargin=0.72 * inch, topMargin=0.65 * inch, bottomMargin=0.78 * inch, title="FastSpeech 2 Technical Progress Report", author="FastSpeech 2 Assignment")
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print(OUT)
