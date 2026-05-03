"""
Manuscript → Corrected PDF  (v2 — new file, existing PDF untouched)
=====================================================================
Output: reports/Manuscript_Draft_Corrected.pdf

Key fix over v1
  • Properly handles SINGLE-LINE display math  $$...$$
    (old parser consumed all lines until the NEXT $$ block)
  • LaTeX macros cleaned to plain readable text before rendering
  • Figures embedded at matching Results sub-headings
"""

import re, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    Image, Table, TableStyle, ListFlowable, ListItem,
)

BASE    = "/Users/apple/Desktop/crc/crc_microbiome_project"
MD_PATH = os.path.join(BASE, "reports", "Manuscript_Draft.md")
PDF_OUT = os.path.join(BASE, "reports", "Manuscript_Draft_Corrected.pdf")   # NEW file
FIG_DIR = os.path.join(BASE, "figures", "publication_ready")

FIGURES = {
    "fig1": (os.path.join(FIG_DIR, "fig1_pca_combat.png"),
             "<b>Figure 1</b> — PCA Before/After ComBat. (A) Raw CLR: tight study-specific clusters. "
             "(B) After ComBat: study clusters dissolved; CRC/CTR disease separation preserved."),
    "fig2": (os.path.join(FIG_DIR, "fig2_roc_curves.png"),
             "<b>Figure 2</b> — Diagnostic Performance. (A) Internal 5-fold CV AUC = 0.900 vs "
             "LODO external mean AUC = 0.785. (B) Per-cohort LODO AUC; US-CRC-2 ensemble = 0.606 "
             "(RF 0.603, XGBoost 0.573, LightGBM 0.588)."),
    "fig3": (os.path.join(FIG_DIR, "fig3_shap_interaction.png"),
             "<b>Figure 3</b> — SHAP Importance and Epistasis. (A) Mean |SHAP|; red = CRC-enriched, "
             "blue = CTR-enriched. (B) P. micra × Anaerotruncus sp. supra-additive interaction "
             "(coeff = +0.147, FDR &lt; 0.001)."),
    "fig4": (os.path.join(FIG_DIR, "fig4_geographic_attenuation.png"),
             "<b>Figure 4</b> — Geographic Attenuation in US-CRC-2. (A) Fold-change attenuation of "
             "top-10 European biomarkers; Firmicutes sp. = 30× drop (p = 2.3 × 10⁻¹¹). "
             "(B) Paired median CLR: European cohorts vs US-CRC-2."),
}

FIG_TRIGGER = {
    "Batch Harmonisation":         "fig1",
    "Diagnostic Performance":      "fig2",
    "Epistatic Microbial Synerg":  "fig3",
    "Geographic Vulnerabilities":  "fig4",
}

# ── Colours ────────────────────────────────────────────────────────────────────
NAVY = colors.HexColor("#1A237E")
DARK = colors.HexColor("#222222")
MID  = colors.HexColor("#666666")
RULE = colors.HexColor("#90CAF9")
TBGH = colors.HexColor("#1A237E")
TBGA = colors.HexColor("#F0F4FA")
TBGG = colors.HexColor("#BBCCDD")

LM, RM, TM, BM = 2.8*cm, 2.8*cm, 2.2*cm, 2.2*cm
PW = A4[0] - LM - RM

S = getSampleStyleSheet()
def ps(name, **kw):
    return ParagraphStyle(name, parent=S["Normal"], **kw)

ST = {
    "title":   ps("T",  fontSize=15, leading=21, textColor=NAVY,
                  fontName="Helvetica-Bold", alignment=TA_CENTER, spaceAfter=5),
    "authors": ps("Au", fontSize=10, leading=14, textColor=DARK,
                  fontName="Helvetica", alignment=TA_CENTER, spaceAfter=3),
    "aff":     ps("Af", fontSize=8.5, leading=12, textColor=MID,
                  fontName="Helvetica-Oblique", alignment=TA_CENTER, spaceAfter=10),
    "h1":      ps("H1", fontSize=13, leading=17, textColor=NAVY,
                  fontName="Helvetica-Bold", spaceBefore=16, spaceAfter=4),
    "h2":      ps("H2", fontSize=11, leading=15, textColor=NAVY,
                  fontName="Helvetica-Bold", spaceBefore=10, spaceAfter=3),
    "h3":      ps("H3", fontSize=10, leading=13, textColor=DARK,
                  fontName="Helvetica-Bold", spaceBefore=7, spaceAfter=2),
    "body":    ps("Bo", fontSize=9.5, leading=14.5, textColor=DARK,
                  fontName="Helvetica", alignment=TA_JUSTIFY, spaceBefore=3, spaceAfter=3),
    "abs":     ps("Ab", fontSize=9, leading=13.5, textColor=DARK,
                  fontName="Helvetica", alignment=TA_JUSTIFY,
                  spaceBefore=2, spaceAfter=2, leftIndent=8, rightIndent=8),
    "abs_h":   ps("AH", fontSize=9, leading=13, textColor=NAVY,
                  fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=1, leftIndent=8),
    "kw":      ps("KW", fontSize=8.5, leading=12, textColor=MID,
                  fontName="Helvetica-Oblique", spaceBefore=5, spaceAfter=10, leftIndent=8),
    "bullet":  ps("BU", fontSize=9.5, leading=14, textColor=DARK,
                  fontName="Helvetica", leftIndent=16, spaceBefore=1, spaceAfter=1),
    "math":    ps("MA", fontSize=9, leading=13, textColor=colors.HexColor("#1A237E"),
                  fontName="Courier", leftIndent=20, spaceBefore=6, spaceAfter=6,
                  backColor=colors.HexColor("#EEF2F7")),
    "caption": ps("CA", fontSize=8, leading=11.5, textColor=MID,
                  fontName="Helvetica-Oblique", alignment=TA_CENTER,
                  spaceBefore=4, spaceAfter=12),
    "tbl_hdr": ps("TH", fontSize=8, leading=11, textColor=colors.white,
                  fontName="Helvetica-Bold", alignment=TA_CENTER),
    "tbl_cel": ps("TC", fontSize=7.5, leading=10.5, textColor=DARK,
                  fontName="Helvetica", alignment=TA_CENTER),
}

# ── LaTeX → readable plain text ────────────────────────────────────────────────
LATEX_MAP = [
    (r'\\text\{([^}]+)\}',   r'\1'),
    (r'\\mathbf\{([^}]+)\}', r'\1'),
    (r'\\mathrm\{([^}]+)\}', r'\1'),
    (r'\\left\(',  '('), (r'\\right\)', ')'),
    (r'\\left\\{', '{'), (r'\\right\\}', '}'),
    (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1/\2'),
    (r'\\prod_\{([^}]+)\}\^\{([^}]+)\}', r'∏(\1, \2)'),
    (r'\\sum_\{([^}]+)\}\^\{([^}]+)\}',  r'∑(\1, \2)'),
    (r'\\log',    'log'), (r'\\ln', 'ln'),
    (r'\\alpha',  'α'),  (r'\\beta', 'β'),
    (r'\\gamma',  'γ'),  (r'\\delta', 'δ'),
    (r'\\epsilon','ε'),  (r'\\phi', 'φ'),
    (r'\\quad',   '  '), (r'\\qquad', '    '),
    (r'\\_',      '_'),  (r'\\,', ' '),
    (r'\^\{([^}]+)\}', r'^\1'),   # superscript
    (r'_\{([^}]+)\}',  r'_\1'),   # subscript
    (r'\\\\',     ''),
    (r'\{|\}',    ''),
]

def latex_to_text(src):
    """Best-effort LaTeX → readable ASCII/Unicode."""
    out = src
    for pat, repl in LATEX_MAP:
        out = re.sub(pat, repl, out)
    # Collapse whitespace
    out = re.sub(r'[ \t]{2,}', '  ', out).strip()
    return out

# ── Inline markdown → ReportLab XML ───────────────────────────────────────────
def mdi(text):
    spans = []
    def _save(m):
        spans.append(m.group(1))
        return f"\x01{len(spans)-1}\x01"
    text = re.sub(r'`([^`]+?)`', _save, text)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.+?)\*\*',     r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',         r'<i>\1</i>', text)
    text = re.sub(r'_([^_\s][^_]*?)_',  r'<i>\1</i>', text)
    text = re.sub(r'\[CITATION\]',
                  '<super><font size="7" color="#C62828">[ref]</font></super>', text)
    def _rest(m):
        raw  = spans[int(m.group(1))]
        safe = raw.replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')
        return f'<font name="Courier" color="#1565C0">{safe}</font>'
    return re.sub(r'\x01(\d+)\x01', _rest, text)

def hr(thick=0.6, col=RULE):
    return HRFlowable(width="100%", thickness=thick, color=col,
                      spaceAfter=4, spaceBefore=4)

def fig_block(key):
    path, caption = FIGURES[key]
    if not os.path.exists(path):
        return []
    img = Image(path, width=PW, height=PW * 0.44, kind="proportional")
    cap = Paragraph(caption, ST["caption"])
    return [Spacer(1, 8), img, cap]

def build_md_table(raw_rows):
    def parse(line):
        return [c.strip() for c in line.strip().strip('|').split('|')]
    if len(raw_rows) < 2:
        return None
    header = parse(raw_rows[0])
    data   = [parse(r) for r in raw_rows[2:]
              if r.strip() and not re.match(r'^[\|\s\-:]+$', r)]
    nc = len(header)
    cw = PW / nc
    rows = [[Paragraph(mdi(h), ST["tbl_hdr"]) for h in header]]
    for row in data:
        row = (row + ['']*nc)[:nc]
        rows.append([Paragraph(mdi(c), ST["tbl_cel"]) for c in row])
    tbl = Table(rows, colWidths=[cw]*nc, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,0), TBGH),
        ('TEXTCOLOR',     (0,0),(-1,0), colors.white),
        ('ALIGN',         (0,0),(-1,-1),'CENTER'),
        ('VALIGN',        (0,0),(-1,-1),'MIDDLE'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white, TBGA]),
        ('GRID',          (0,0),(-1,-1), 0.35, TBGG),
        ('FONTNAME',      (0,0),(-1,0),'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),(-1,0), 8),
        ('FONTNAME',      (0,1),(-1,-1),'Helvetica'),
        ('FONTSIZE',      (0,1),(-1,-1), 7.5),
        ('TOPPADDING',    (0,0),(-1,-1), 3),
        ('BOTTOMPADDING', (0,0),(-1,-1), 3),
        ('LEFTPADDING',   (0,0),(-1,-1), 4),
        ('RIGHTPADDING',  (0,0),(-1,-1), 4),
    ]))
    return tbl

def on_page(canvas, doc):
    canvas.saveState()
    w, h = A4
    canvas.setStrokeColor(NAVY); canvas.setLineWidth(0.9)
    canvas.line(LM, h-TM+7*mm, w-RM, h-TM+7*mm)
    canvas.setFont("Helvetica-Oblique", 7); canvas.setFillColor(MID)
    canvas.drawString(LM, h-TM+9*mm, "Microbiome  |  Research Article")
    canvas.drawRightString(w-RM, h-TM+9*mm, "CRC Multi-Cohort Metagenomics")
    canvas.setStrokeColor(RULE); canvas.setLineWidth(0.4)
    canvas.line(LM, BM-5*mm, w-RM, BM-5*mm)
    canvas.setFont("Helvetica", 7); canvas.setFillColor(MID)
    canvas.drawCentredString(w/2, BM-10*mm, f"Page {doc.page}")
    canvas.restoreState()

# ── The fixed Markdown parser ──────────────────────────────────────────────────
def parse_md(text):
    story       = []
    lines       = text.splitlines()
    i           = 0
    in_abstract = False
    bullets     = []
    tbl_buf     = []

    def flush_bullets():
        if bullets:
            items = [ListItem(Paragraph(mdi(b), ST["bullet"]),
                              leftIndent=22, bulletColor=NAVY,
                              bulletType='bullet', bulletFontSize=9)
                     for b in bullets]
            story.append(ListFlowable(items, bulletType='bullet',
                                      leftIndent=18, spaceBefore=2, spaceAfter=4))
            bullets.clear()

    def flush_table():
        if tbl_buf:
            t = build_md_table(tbl_buf)
            if t:
                story.append(Spacer(1, 6))
                story.append(t)
                story.append(Spacer(1, 8))
            tbl_buf.clear()

    while i < len(lines):
        line = lines[i]

        # ── Markdown table ─────────────────────────────────────────────────────
        if line.strip().startswith('|'):
            flush_bullets()
            tbl_buf.append(line); i += 1; continue
        else:
            flush_table()

        # ── Blank ──────────────────────────────────────────────────────────────
        if not line.strip():
            flush_bullets(); story.append(Spacer(1, 3)); i += 1; continue

        # ── HTML comment ───────────────────────────────────────────────────────
        if line.strip().startswith("<!--"): i += 1; continue

        # ── Horizontal rule ────────────────────────────────────────────────────
        if re.match(r'^---+$', line.strip()):
            flush_bullets(); story.append(hr()); i += 1; continue

        # ── Display math  $$...$$  ─────────────────────────────────────────────
        # Detect BOTH single-line ($$formula$$) and multi-line blocks
        stripped = line.strip()
        if stripped.startswith('$$'):
            flush_bullets()

            # ── Single-line: $$formula$$ ──────────────────────────────────────
            # Count occurrences of $$ on this line
            if stripped.count('$$') >= 2:
                # Extract content between the two $$
                inner = re.sub(r'^\$\$\s*', '', stripped)
                inner = re.sub(r'\s*\$\$$', '', inner)
                formula = latex_to_text(inner)
                story.append(Paragraph(formula, ST["math"]))
                i += 1; continue

            # ── Multi-line: $$ on its own line ────────────────────────────────
            math_lines = []
            i += 1
            while i < len(lines):
                if lines[i].strip().startswith('$$'):
                    i += 1; break     # closing $$, stop here
                math_lines.append(lines[i].strip())
                i += 1
            formula = latex_to_text(' '.join(l for l in math_lines if l))
            story.append(Paragraph(formula, ST["math"]))
            continue

        # ── H1 ────────────────────────────────────────────────────────────────
        if re.match(r'^# [^#]', line):
            flush_bullets()
            story.append(Paragraph(mdi(line[2:].strip()), ST["title"]))
            i += 1; continue

        # ── H2 ────────────────────────────────────────────────────────────────
        if re.match(r'^## [^#]', line):
            flush_bullets()
            txt = line[3:].strip()
            in_abstract = "Abstract" in txt
            story.append(Paragraph(mdi(txt), ST["h1"]))
            story.append(HRFlowable(width="100%", thickness=1.4,
                                    color=NAVY, spaceAfter=4))
            i += 1; continue

        # ── H3 ────────────────────────────────────────────────────────────────
        if re.match(r'^### [^#]', line):
            flush_bullets()
            txt = line[4:].strip()
            sty = ST["abs_h"] if in_abstract else ST["h2"]
            story.append(Paragraph(mdi(txt), sty))
            for keyword, fig_key in FIG_TRIGGER.items():
                if keyword.lower() in txt.lower():
                    story.extend(fig_block(fig_key)); break
            i += 1; continue

        # ── H4 ────────────────────────────────────────────────────────────────
        if re.match(r'^#### [^#]', line):
            flush_bullets()
            story.append(Paragraph(mdi(line[5:].strip()), ST["h3"]))
            i += 1; continue

        # ── Special lines ──────────────────────────────────────────────────────
        if line.startswith("**Authors:**"):
            story.append(Paragraph(mdi(line), ST["authors"])); i += 1; continue
        if re.match(r'^[¹²³]', line) or line.startswith("**Corresponding"):
            story.append(Paragraph(mdi(line), ST["aff"])); i += 1; continue
        if line.startswith("**Keywords:**"):
            flush_bullets()
            story.append(Paragraph(mdi(line), ST["kw"])); i += 1; continue

        # ── Bullet / numbered list ─────────────────────────────────────────────
        if re.match(r'^\s*[-*]\s+', line) or re.match(r'^\s*\d+\.\s+', line):
            txt = re.sub(r'^\s*[-*\d.]+\s+', '', line).strip()
            bullets.append(txt); i += 1; continue
        else:
            flush_bullets()

        # ── Plain paragraph ────────────────────────────────────────────────────
        sty = ST["abs"] if in_abstract else ST["body"]
        story.append(Paragraph(mdi(line.strip()), sty))
        i += 1

    flush_bullets()
    flush_table()
    return story

def build():
    with open(MD_PATH, encoding='utf-8') as f:
        md = f.read()

    doc = SimpleDocTemplate(
        PDF_OUT, pagesize=A4,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM+10*mm, bottomMargin=BM+10*mm,
        title="CRC Metagenomics — Microbiome Manuscript (Corrected)",
        author="Computational Biology Lab",
        subject="Colorectal Cancer Microbiome Meta-Analysis",
    )
    story = parse_md(md)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    sz = os.path.getsize(PDF_OUT) / 1e6
    print(f"\n  ✅  PDF saved → {PDF_OUT}  ({sz:.1f} MB)")
    print("  Figures:")
    for k, (p, _) in FIGURES.items():
        print(f"    {k}: {'✓ embedded' if os.path.exists(p) else '✗ missing'}")

if __name__ == "__main__":
    build()
