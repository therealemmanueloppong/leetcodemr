#!/usr/bin/env python3
"""
LeetMastery — Expose PDFs (21 pattern packs)

One PDF per pattern. Within each pack: Hard → Medium → Easy.
Each question: description (HTML-parsed) + boxed examples + constraints +
interview approach + expose notes (+ description images).

Data:
  public/grind_questions.json   — descriptions, interviewApproach
  public/quick_review_info.json — expose notes (insights + complexity)
  public/description-images/    — local images referenced as /description-images/...
  Optional .lc_content_cache.json editorial HTML (KaTeX-stripped when present)

Usage:
  python3 build_expose_pdf.py                  # all 21 packs
  python3 build_expose_pdf.py --pattern Trees  # substring match
  python3 build_expose_pdf.py --self-test      # KaTeX strip unit check only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag
from PIL import Image as PILImage
from reportlab.lib.colors import HexColor, white
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    Image as RLImage,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Fonts (macOS Lucida/Menlo, else DejaVu on Linux) ──────────────────────────
def _register_fonts() -> tuple[str, str, str, str]:
    candidates = [
        (
            "/System/Library/Fonts/LucidaGrande.ttc",
            "/System/Library/Fonts/Menlo.ttc",
            True,
        ),
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            False,
        ),
        (
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            False,
        ),
    ]
    for body, mono, is_ttc in candidates:
        bp, mp = Path(body), Path(mono)
        if not bp.exists() or not mp.exists():
            continue
        try:
            if is_ttc:
                pdfmetrics.registerFont(TTFont("LG", str(bp), subfontIndex=0))
                pdfmetrics.registerFont(TTFont("LG-Bold", str(bp), subfontIndex=1))
                pdfmetrics.registerFont(TTFont("Menlo", str(mp), subfontIndex=0))
                pdfmetrics.registerFont(TTFont("Menlo-Bold", str(mp), subfontIndex=1))
            else:
                bold_body = bp.with_name(bp.name.replace("Regular", "Bold").replace("Sans.ttf", "Sans-Bold.ttf"))
                if not bold_body.exists():
                    bold_body = Path(str(bp).replace("DejaVuSans.ttf", "DejaVuSans-Bold.ttf"))
                bold_mono = Path(str(mp).replace("Regular", "Bold").replace("Mono.ttf", "Mono-Bold.ttf"))
                if not bold_mono.exists():
                    bold_mono = Path(str(mp).replace("DejaVuSansMono.ttf", "DejaVuSansMono-Bold.ttf"))
                pdfmetrics.registerFont(TTFont("LG", str(bp)))
                pdfmetrics.registerFont(TTFont("LG-Bold", str(bold_body if bold_body.exists() else bp)))
                pdfmetrics.registerFont(TTFont("Menlo", str(mp)))
                pdfmetrics.registerFont(TTFont("Menlo-Bold", str(bold_mono if bold_mono.exists() else mp)))
            return "LG", "LG-Bold", "Menlo", "Menlo-Bold"
        except Exception:
            continue
    return "Helvetica", "Helvetica-Bold", "Courier", "Courier-Bold"


BODY, BODY_B, CODE, CODE_B = _register_fonts()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
GRIND_JSON = SCRIPT_DIR / "public" / "grind_questions.json"
QR_JSON = SCRIPT_DIR / "public" / "quick_review_info.json"
DESC_IMG_DIR = SCRIPT_DIR / "public" / "description-images"
LC_CACHE = SCRIPT_DIR / ".lc_content_cache.json"
OUTPUT_DIR = Path.home() / "Desktop" / "Expose PDFs"

PATTERN_DISPLAY_ORDER = [
    "Arrays & Hashing",
    "String",
    "Two Pointers",
    "Sliding Window",
    "Sorting",
    "Binary Search",
    "Matrix",
    "Trees & BST",
    "DFS",
    "Graphs",
    "BFS",
    "Linked List",
    "Stack",
    "Heap",
    "Trie",
    "Backtracking",
    "Greedy",
    "Dynamic Programming",
    "Bit Manipulation",
    "Math",
    "JavaScript",
]

DIFF_RANK = {"Hard": 0, "Medium": 1, "Easy": 2}
DIFF_COLOR = {"Easy": "#16A34A", "Medium": "#D97706", "Hard": "#DC2626"}
LINK_COLOR = "#1D4ED8"
PAGE_W, PAGE_H = letter
MARGIN = 0.65 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

# ── KaTeX / HTML → text ───────────────────────────────────────────────────────
_MATH_O = re.compile(
    r"(?:\\mathcal\s*\{\s*O\s*\}|\\mathrm\s*\{\s*O\s*\}|\\operatorname\s*\{\s*O\s*\}|O)"
    r"\s*\(([^)]*)\)",
    re.I,
)
_DOUBLED_O = re.compile(r"(O\([^)]*\))\s*\1")
_DOUBLED_IDENT = re.compile(r"\b([A-Za-z])\1\b")  # NN from doubled N (conservative; applied only near O())


def strip_katex_mathml(soup: BeautifulSoup) -> None:
    """Remove screen-reader KaTeX copies so visible .katex-html is kept once."""
    for el in soup.select(".katex-mathml"):
        el.decompose()


def normalize_complexity_math(text: str) -> str:
    """Turn common KaTeX leftovers into plain O(N), O(N log N), O(N²)."""
    if not text:
        return text

    def _repl_o(m: re.Match) -> str:
        inner = m.group(1)
        inner = inner.replace("\\log", "log").replace("\\cdot", "·")
        inner = re.sub(r"\^\{?2\}?", "²", inner)
        inner = re.sub(r"\^\{?3\}?", "³", inner)
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)
        inner = re.sub(r"[{}]", "", inner)
        inner = re.sub(r"\s+", " ", inner).strip()
        return f"O({inner})"

    text = _MATH_O.sub(_repl_o, text)
    # Collapse accidental doubles from residual mathml+html text
    prev = None
    while prev != text:
        prev = text
        text = _DOUBLED_O.sub(r"\1", text)
    # NN → N when adjacent to complexity phrasing
    if re.search(r"O\([^)]*\)", text):
        text = re.sub(r"\b([Nn])\1\b", r"\1", text)
    return text


def html_to_plain_text(html: str | None) -> str:
    """Extract readable text from approach/complexity/editorial HTML.

    Always strips .katex-mathml before get_text to avoid O(N)O(N) / NN.
    """
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    strip_katex_mathml(soup)
    # Prefer visible KaTeX HTML if present; otherwise whole soup
    text = soup.get_text(" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return normalize_complexity_math(text)


def self_test_katex() -> None:
    sample = (
        '<p>Time Complexity: '
        '<span class="katex">'
        '<span class="katex-mathml"><math><semantics><annotation encoding="application/x-tex">'
        r'\mathcal{O}(N)'
        "</annotation></semantics></math>"
        '<span class="katex-mathml">O(N)</span>'
        "</span>"
        '<span class="katex-html"><span class="base">O(N)</span></span>'
        "</span>"
        ", where "
        '<span class="katex"><span class="katex-mathml">N</span>'
        '<span class="katex-html">N</span></span>'
        " is the number of nodes.</p>"
    )
    bad = BeautifulSoup(sample, "lxml").get_text(" ", strip=True)
    good = html_to_plain_text(sample)
    assert "O(N)O(N)" in bad.replace(" ", "") or bad.count("O(N)") >= 2 or "NN" in bad.replace(" ", ""), (
        f"fixture should double without strip; got {bad!r}"
    )
    assert "O(N)O(N)" not in good.replace(" ", ""), f"still doubled: {good!r}"
    assert re.search(r"O\(N\)", good), f"missing O(N): {good!r}"
    # Single N for 'where N is'
    assert "NN" not in good.replace(" ", ""), f"doubled N: {good!r}"
    print("self-test OK:", good)


# ── Helpers ───────────────────────────────────────────────────────────────────
def safe_xml(t: str) -> str:
    return (
        (t or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _para(text: str, style) -> Paragraph:
    """Paragraph with fallback if ReportLab markup is unbalanced."""
    try:
        return Paragraph(text if text.strip() else "&nbsp;", style)
    except Exception:
        return Paragraph(safe_xml(re.sub(r"<[^>]+>", "", text)) or "&nbsp;", style)


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


class NamedDest(Flowable):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def wrap(self, availWidth, availHeight):
        return 0, 0

    def draw(self):
        x, y = self.canv.absolutePosition(0, 0)
        self.canv.bookmarkHorizontal(self.name, x, y)


class FooterCtx(Flowable):
    def __init__(self, label: str):
        super().__init__()
        self.label = label

    def wrap(self, availWidth, availHeight):
        return 0, 0

    def draw(self):
        setattr(self.canv, "_expose_footer", self.label)


def resolve_desc_image(src: str) -> Path | None:
    if not src:
        return None
    if src.startswith("/description-images/"):
        p = DESC_IMG_DIR / src.split("/description-images/", 1)[1]
        return p if p.exists() else None
    if src.startswith("description-images/"):
        p = DESC_IMG_DIR / src.split("description-images/", 1)[1]
        return p if p.exists() else None
    # absolute local path
    p = Path(src)
    return p if p.exists() else None


def rl_local_image(path: Path, max_w: float = CONTENT_W - 12) -> RLImage | None:
    try:
        im = PILImage.open(path)
        w, h = im.size
        if w <= 0 or h <= 0:
            return None
        scale = min(max_w / w, 3.2 * inch / h, 1.0)
        return RLImage(str(path), width=w * scale, height=h * scale)
    except Exception:
        return None


def _inline_html_to_rl(fragment: str) -> str:
    """Convert a small HTML fragment's inline tags to ReportLab markup."""
    if not fragment:
        return ""
    soup = BeautifulSoup(fragment, "lxml")
    strip_katex_mathml(soup)

    def walk(node) -> str:
        if isinstance(node, NavigableString):
            return safe_xml(str(node))
        if not isinstance(node, Tag):
            return ""
        name = node.name.lower()
        if name in ("script", "style"):
            return ""
        if name == "br":
            return "<br/>"
        if name in ("strong", "b"):
            return f"<b>{''.join(walk(c) for c in node.children)}</b>"
        if name in ("em", "i"):
            return f"<i>{''.join(walk(c) for c in node.children)}</i>"
        if name == "code":
            inner = safe_xml(node.get_text())
            return f'<font name="{CODE}" size="9" color="#111827">{inner}</font>'
        if name == "sup":
            return f"<super>{''.join(walk(c) for c in node.children)}</super>"
        if name == "sub":
            return f"<sub>{''.join(walk(c) for c in node.children)}</sub>"
        if name == "a":
            return "".join(walk(c) for c in node.children)
        return "".join(walk(c) for c in node.children)

    body = soup.body or soup
    text = "".join(walk(c) for c in body.children)
    text = text.replace("&nbsp;", " ").replace("\xa0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _is_example_header(tag: Tag) -> bool:
    strong = tag.find("strong", class_=re.compile(r"example", re.I)) if tag.name == "p" else None
    if strong:
        return True
    if tag.name == "strong" and "example" in " ".join(tag.get("class") or []).lower():
        return True
    txt = tag.get_text(" ", strip=True)
    return bool(re.match(r"^Example\s+\d+", txt, re.I))


def _is_constraints_header(tag: Tag) -> bool:
    txt = tag.get_text(" ", strip=True)
    return bool(re.match(r"^Constraints\b", txt, re.I))


# ── Styles ────────────────────────────────────────────────────────────────────
def build_styles() -> dict:
    return {
        "cover_title": ParagraphStyle(
            "cover_title", fontName=BODY_B, fontSize=28, leading=34,
            alignment=TA_CENTER, textColor=HexColor("#111827"), spaceAfter=10,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontName=BODY, fontSize=12, leading=16,
            alignment=TA_CENTER, textColor=HexColor("#4B5563"), spaceAfter=6,
        ),
        "h2": ParagraphStyle(
            "h2", fontName=BODY_B, fontSize=13, leading=17,
            textColor=HexColor("#111827"), spaceBefore=10, spaceAfter=4,
        ),
        "q_title": ParagraphStyle(
            "q_title", fontName=BODY_B, fontSize=16, leading=20,
            textColor=HexColor("#111827"), spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body", fontName=BODY, fontSize=10.5, leading=15,
            textColor=HexColor("#1F2937"), spaceAfter=4,
        ),
        "meta": ParagraphStyle(
            "meta", fontName=BODY, fontSize=9.5, leading=13,
            textColor=HexColor("#6B7280"), spaceAfter=6,
        ),
        "nav": ParagraphStyle(
            "nav", fontName=BODY_B, fontSize=10, leading=13,
            textColor=HexColor(LINK_COLOR), spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontName=BODY, fontSize=10.5, leading=15,
            textColor=HexColor("#1F2937"), leftIndent=14, spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "code", fontName=CODE, fontSize=9, leading=12,
            textColor=HexColor("#111827"), spaceAfter=2,
        ),
        "ex_label": ParagraphStyle(
            "ex_label", fontName=BODY_B, fontSize=10.5, leading=14,
            textColor=HexColor("#111827"), spaceAfter=3,
        ),
        "toc": ParagraphStyle(
            "toc", fontName=BODY, fontSize=10, leading=14,
            textColor=HexColor("#1F2937"), spaceAfter=2,
        ),
    }


# ── Description builder ───────────────────────────────────────────────────────
def build_description(desc_html: str | None, desc_text: str | None, styles: dict) -> list:
    """Parse descHtml via BeautifulSoup; fall back to descText when absent."""
    flow: list = []
    html = (desc_html or "").strip()
    if not html:
        text = (desc_text or "").strip()
        if not text:
            return flow
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if para:
                flow.append(Paragraph(safe_xml(para).replace("\n", "<br/>"), styles["body"]))
        return flow

    soup = BeautifulSoup(html, "lxml")
    strip_katex_mathml(soup)
    root = soup.body or soup

    example_box: list = []
    in_example = False
    example_label = ""

    def flush_example():
        nonlocal example_box, in_example, example_label
        if not example_box and not example_label:
            in_example = False
            return
        inner = []
        if example_label:
            inner.append(Paragraph(safe_xml(example_label), styles["ex_label"]))
        inner.extend(example_box)
        data = [[inner]]
        tbl = Table(data, colWidths=[CONTENT_W - 8])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), HexColor("#F3F4F6")),
                    ("BOX", (0, 0), (-1, -1), 0.6, HexColor("#9CA3AF")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        flow.append(Spacer(1, 4))
        flow.append(tbl)
        flow.append(Spacer(1, 6))
        example_box = []
        example_label = ""
        in_example = False

    def add_para(tag: Tag, target: list):
        markup = _inline_html_to_rl("".join(str(c) for c in tag.children))
        if markup:
            target.append(_para(markup, styles["body"]))

    def add_list(tag: Tag, ordered: bool, target: list):
        for i, li in enumerate(tag.find_all("li", recursive=False), 1):
            mark = f"{i}." if ordered else "•"
            markup = _inline_html_to_rl("".join(str(c) for c in li.children))
            if markup:
                target.append(_para(f"{mark} {markup}", styles["bullet"]))

    def add_pre(tag: Tag, target: list):
        # Keep Input/Output/Explanation labels bold when present.
        # Build per-line so <b> never spans a newline (ReportLab breaks otherwise).
        raw_html = "".join(str(c) for c in tag.children)
        tmp = BeautifulSoup(f"<pre>{raw_html}</pre>", "lxml").pre
        strip_katex_mathml(tmp)

        def _chunks(node) -> list[str]:
            out: list[str] = []
            if isinstance(node, NavigableString):
                out.append(safe_xml(str(node)))
                return out
            if not isinstance(node, Tag):
                return out
            if node.name in ("strong", "b"):
                # Bold each line independently
                txt = safe_xml(node.get_text())
                parts = txt.split("\n")
                for i, part in enumerate(parts):
                    if part:
                        out.append(f"<b>{part}</b>")
                    if i < len(parts) - 1:
                        out.append("\n")
                return out
            if node.name == "br":
                out.append("\n")
                return out
            for child in node.children:
                out.extend(_chunks(child))
            return out

        block = "".join(_chunks(tmp)).replace("\r\n", "\n").replace("\r", "\n")
        block = re.sub(r"\n{3,}", "\n\n", block).strip("\n")
        if not block.strip():
            return
        for line in block.split("\n"):
            # Ensure balanced <b> on each line (defensive)
            if line.count("<b>") != line.count("</b>"):
                line = re.sub(r"</?b>", "", line)
            target.append(_para(line if line.strip() else "&nbsp;", styles["code"]))

    def add_img(tag: Tag, target: list):
        src = tag.get("src") or ""
        path = resolve_desc_image(src)
        if path:
            img = rl_local_image(path)
            if img:
                target.append(Spacer(1, 3))
                target.append(img)
                target.append(Spacer(1, 3))

    for el in list(root.children):
        if isinstance(el, NavigableString):
            t = str(el).strip()
            if t:
                (example_box if in_example else flow).append(
                    Paragraph(safe_xml(t), styles["body"])
                )
            continue
        if not isinstance(el, Tag):
            continue
        name = el.name.lower()
        if name in ("script", "style"):
            continue

        # paragraph that only wraps an image
        if name == "p" and el.find("img") and not el.get_text(strip=True):
            img = el.find("img")
            add_img(img, example_box if in_example else flow)
            continue

        if name == "p" and _is_example_header(el):
            flush_example()
            in_example = True
            example_label = el.get_text(" ", strip=True)
            continue

        if name == "p" and _is_constraints_header(el):
            flush_example()
            flow.append(Paragraph("<b>Constraints</b>", styles["h2"]))
            continue

        if name == "img":
            add_img(el, example_box if in_example else flow)
            continue

        target = example_box if in_example else flow
        if name == "p":
            # image + text?
            for img in el.find_all("img"):
                add_img(img, target)
            # remove imgs from clone for text
            add_para(el, target)
        elif name == "ul":
            add_list(el, False, target)
        elif name == "ol":
            add_list(el, True, target)
        elif name == "pre":
            add_pre(el, target)
        elif name in ("h2", "h3", "h4"):
            if _is_example_header(el):
                flush_example()
                in_example = True
                example_label = el.get_text(" ", strip=True)
            elif _is_constraints_header(el):
                flush_example()
                flow.append(Paragraph("<b>Constraints</b>", styles["h2"]))
            else:
                flush_example()
                flow.append(Paragraph(f"<b>{safe_xml(el.get_text(' ', strip=True))}</b>", styles["h2"]))
        elif name in ("div", "section"):
            # flatten one level
            for child in el.children:
                if isinstance(child, Tag):
                    if child.name == "p" and _is_example_header(child):
                        flush_example()
                        in_example = True
                        example_label = child.get_text(" ", strip=True)
                    elif child.name == "pre":
                        add_pre(child, example_box if in_example else flow)
                    elif child.name == "img":
                        add_img(child, example_box if in_example else flow)
                    elif child.name == "p":
                        add_para(child, example_box if in_example else flow)
                    elif child.name in ("ul", "ol"):
                        add_list(child, child.name == "ol", example_box if in_example else flow)

    flush_example()
    return flow


def interview_flowables(text: str, styles: dict) -> list:
    if not text:
        return []
    out = [Paragraph("<b>Interview Approach</b>", styles["h2"])]
    # Keep comment structure readable
    for line in text.splitlines():
        raw = line.rstrip()
        if not raw:
            out.append(Spacer(1, 3))
            continue
        # Strip leading '# ' comment markers for readability
        clean = re.sub(r"^#\s?", "", raw)
        if clean.startswith("PHASE") or clean.startswith("--"):
            out.append(Paragraph(f"<b>{safe_xml(clean)}</b>", styles["body"]))
        else:
            out.append(Paragraph(safe_xml(clean), styles["body"]))
    return out


def expose_flowables(qr: dict | None, lc_editorial_html: str | None, styles: dict) -> list:
    """Expose notes: prefer editorial HTML (KaTeX-stripped); else quick-review."""
    out: list = [Paragraph("<b>Expose Notes</b>", styles["h2"])]
    used = False

    if lc_editorial_html:
        plain = html_to_plain_text(lc_editorial_html)
        if plain:
            # Split lightly on Complexity Analysis / Approach headers
            parts = re.split(
                r"(?=(?:Complexity Analysis|Approach\s+\d+|Intuition|Algorithm)\b)",
                plain,
                flags=re.I,
            )
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if re.match(r"^(Complexity Analysis|Approach\s+\d+|Intuition|Algorithm)\b", part, re.I):
                    head, _, rest = part.partition(".")
                    # better: first line-ish
                    m = re.match(
                        r"^((?:Complexity Analysis|Approach\s+\d+[^\n]*|Intuition|Algorithm)[^\n]*)\n?(.*)$",
                        part,
                        re.S | re.I,
                    )
                    if m:
                        out.append(Paragraph(f"<b>{safe_xml(m.group(1).strip())}</b>", styles["body"]))
                        body = m.group(2).strip()
                        if body:
                            out.append(Paragraph(safe_xml(body), styles["body"]))
                    else:
                        out.append(Paragraph(safe_xml(part), styles["body"]))
                else:
                    out.append(Paragraph(safe_xml(part), styles["body"]))
            used = True

    if qr:
        insights = (qr.get("key_insights") or "").strip()
        cx = normalize_complexity_math((qr.get("space_and_time_complexity") or "").strip())
        sol = (qr.get("solution") or "").strip()
        if insights:
            out.append(Paragraph("<b>Key insights</b>", styles["body"]))
            for line in insights.splitlines():
                line = line.strip().lstrip("-• ").strip()
                if line:
                    out.append(Paragraph(f"• {safe_xml(line)}", styles["bullet"]))
            used = True
        if cx:
            out.append(Paragraph("<b>Complexity</b>", styles["body"]))
            for line in cx.splitlines():
                line = line.strip()
                if line:
                    out.append(Paragraph(f"• {safe_xml(normalize_complexity_math(line))}", styles["bullet"]))
            used = True
        if sol and not lc_editorial_html:
            out.append(Paragraph("<b>Solution sketch</b>", styles["body"]))
            out.append(Paragraph(safe_xml(sol), styles["body"]))
            used = True

    if not used:
        out.append(Paragraph("<i>No expose notes available for this question.</i>", styles["meta"]))
    return out


# ── Pack build ────────────────────────────────────────────────────────────────
def load_data():
    questions = json.loads(GRIND_JSON.read_text())
    qr_list = json.loads(QR_JSON.read_text()) if QR_JSON.exists() else []
    qr_by_id = {int(x["id"]): x for x in qr_list if "id" in x}
    lc = {}
    if LC_CACHE.exists():
        try:
            lc = json.loads(LC_CACHE.read_text())
        except Exception:
            lc = {}
    return questions, qr_by_id, lc


def group_by_pattern(questions: list) -> dict[str, list]:
    buckets: dict[str, list] = defaultdict(list)
    for q in questions:
        pat = q.get("pattern") or "Other"
        buckets[pat].append(q)
    for pat, qs in buckets.items():
        qs.sort(key=lambda q: (DIFF_RANK.get(q.get("difficulty", "Medium"), 9), q.get("id", 0)))
    return buckets


def nav_row(prev_id: int | None, next_id: int | None, toc_name: str, styles: dict) -> Paragraph:
    parts = [f'<link href="#{toc_name}" color="{LINK_COLOR}"><b>Contents</b></link>']
    if prev_id is not None:
        parts.append(f'<link href="#q_{prev_id}" color="{LINK_COLOR}"><b>← Prev</b></link>')
    else:
        parts.append('<font color="#9CA3AF">← Prev</font>')
    if next_id is not None:
        parts.append(f'<link href="#q_{next_id}" color="{LINK_COLOR}"><b>Next →</b></link>')
    else:
        parts.append('<font color="#9CA3AF">Next →</font>')
    return Paragraph(" &nbsp;|&nbsp; ".join(parts), styles["nav"])


def build_pack_pdf(pattern: str, qs: list, qr_by_id: dict, lc: dict, out_path: Path, pack_idx: int) -> None:
    styles = build_styles()
    toc_name = f"toc_{slugify(pattern)}"
    story: list = []

    # Cover
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("LeetMastery Expose", styles["cover_title"]))
    story.append(Paragraph(safe_xml(pattern), styles["cover_title"]))
    story.append(
        Paragraph(
            f"Pack {pack_idx:02d} of 21 &nbsp;·&nbsp; {len(qs)} questions &nbsp;·&nbsp; Hard → Easy",
            styles["cover_sub"],
        )
    )
    story.append(
        Paragraph(
            "Description · Examples · Constraints · Interview Approach · Expose Notes",
            styles["cover_sub"],
        )
    )
    story.append(PageBreak())

    # TOC
    story.append(NamedDest(toc_name))
    story.append(FooterCtx(f"{pattern} · Contents"))
    story.append(Paragraph("<b>Contents</b>", styles["h2"]))
    story.append(
        Paragraph(
            "Questions ordered <b>Hard → Medium → Easy</b>. Tap a title to jump.",
            styles["meta"],
        )
    )
    for q in qs:
        qid = int(q["id"])
        diff = q.get("difficulty", "")
        col = DIFF_COLOR.get(diff, "#6B7280")
        label = f'#{qid} {safe_xml(q.get("title", ""))} — <font color="{col}">{diff}</font>'
        story.append(
            Paragraph(
                f'<link href="#q_{qid}" color="{LINK_COLOR}">{label}</link>',
                styles["toc"],
            )
        )
    story.append(PageBreak())

    for i, q in enumerate(qs):
        qid = int(q["id"])
        prev_id = int(qs[i - 1]["id"]) if i > 0 else None
        next_id = int(qs[i + 1]["id"]) if i + 1 < len(qs) else None
        diff = q.get("difficulty", "")
        col = DIFF_COLOR.get(diff, "#6B7280")

        story.append(NamedDest(f"q_{qid}"))
        story.append(FooterCtx(f"{pattern} · #{qid}"))
        story.append(nav_row(prev_id, next_id, toc_name, styles))
        story.append(
            Paragraph(
                f'#{qid}. {safe_xml(q.get("title", ""))} '
                f'<font color="{col}" size="11"><b>{diff}</b></font>',
                styles["q_title"],
            )
        )
        story.append(
            Paragraph(
                f'<link href="https://leetcode.com/problems/{safe_xml(q.get("slug", ""))}/" '
                f'color="{LINK_COLOR}">leetcode.com/problems/{safe_xml(q.get("slug", ""))}</link>',
                styles["meta"],
            )
        )
        story.append(HRFlowable(width="100%", thickness=0.6, color=HexColor("#D1D5DB"), spaceAfter=8))

        # Description (HTML-first)
        story.append(Paragraph("<b>Description</b>", styles["h2"]))
        desc_html = q.get("descriptionHtml") or q.get("descHtml") or ""
        desc_text = q.get("description") or q.get("descText") or ""
        story.extend(build_description(desc_html, desc_text, styles))

        # Interview approach
        approach = q.get("interviewApproach") or ""
        # Optional HTML approach fields (editorial) — KaTeX-stripped
        for key in ("approachHtml", "interviewApproachHtml", "solutionHtml"):
            if q.get(key):
                plain = html_to_plain_text(q[key])
                if plain:
                    story.append(Paragraph("<b>Approach (editorial)</b>", styles["h2"]))
                    story.append(Paragraph(safe_xml(plain), styles["body"]))
        story.extend(interview_flowables(approach, styles))

        # Expose notes
        slug = q.get("slug") or ""
        editorial = None
        if slug and isinstance(lc.get(slug), dict):
            editorial = lc[slug].get("editorial") or lc[slug].get("approachHtml")
        # Also accept per-question expose HTML
        editorial = editorial or q.get("exposeHtml") or q.get("editorialHtml") or q.get("complexityHtml")
        story.extend(expose_flowables(qr_by_id.get(qid), editorial, styles))

        if i + 1 < len(qs):
            story.append(PageBreak())

    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _footer(canv, doc):
        canv.saveState()
        label = getattr(canv, "_expose_footer", pattern)
        canv.setFont(BODY, 8)
        canv.setFillColor(HexColor("#6B7280"))
        canv.drawString(MARGIN, 0.4 * inch, label[:80])
        canv.drawRightString(PAGE_W - MARGIN, 0.4 * inch, f"{doc.page}")
        canv.restoreState()

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=0.6 * inch,
        bottomMargin=0.7 * inch,
        title=f"LeetMastery Expose — {pattern}",
        author="LeetMastery",
    )
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build 21 LeetMastery Expose PDFs")
    ap.add_argument("--pattern", default="", help="Substring filter (e.g. Trees)")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR, help="Output directory")
    ap.add_argument("--self-test", action="store_true", help="Run KaTeX strip self-test and exit")
    args = ap.parse_args(argv)

    if args.self_test:
        self_test_katex()
        return 0

    if not GRIND_JSON.exists():
        print(f"Missing data: {GRIND_JSON}", file=sys.stderr)
        return 1

    questions, qr_by_id, lc = load_data()
    buckets = group_by_pattern(questions)

    patterns = [p for p in PATTERN_DISPLAY_ORDER if p in buckets]
    for p in sorted(buckets):
        if p not in patterns:
            patterns.append(p)

    if args.pattern:
        needle = args.pattern.lower()
        patterns = [p for p in patterns if needle in p.lower()]
        if not patterns:
            print(f"No pattern matching {args.pattern!r}", file=sys.stderr)
            return 1

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(patterns)} pack(s) → {args.out}")

    for idx, pattern in enumerate(patterns, 1):
        qs = buckets[pattern]
        fname = f"{idx:02d} {pattern} Expose.pdf"
        # When filtering, keep display-order index from full list
        if args.pattern:
            full_idx = PATTERN_DISPLAY_ORDER.index(pattern) + 1 if pattern in PATTERN_DISPLAY_ORDER else idx
            fname = f"{full_idx:02d} {pattern} Expose.pdf"
            pack_idx = full_idx
        else:
            pack_idx = idx
        out_path = args.out / fname
        print(f"  [{pack_idx:02d}/21] {pattern} ({len(qs)} q) → {out_path.name}")
        build_pack_pdf(pattern, qs, qr_by_id, lc, out_path, pack_idx)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
