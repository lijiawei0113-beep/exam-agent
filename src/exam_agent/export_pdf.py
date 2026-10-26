"""导出学生版和答案版 PDF。

为什么选 ReportLab：纯 Python，`pip install reportlab` 即可，不依赖外部排版程序，部署到服务器也简单。
两份 PDF 都从数据库里的同一个版本快照生成，所以题目、题号、分值必然一致。

中文字体：PDF 需要嵌入字体才能在任何电脑上正常显示中文。程序按顺序查找系统里常见的
TrueType 中文字体（Windows 的宋体/黑体/微软雅黑，macOS 的华文黑体，Linux 的文泉驿等），
也可以用环境变量 EXAM_PDF_FONT / EXAM_PDF_FONT_BOLD 指定字体文件。
"""
from __future__ import annotations

import re

import os
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

from .config import PROJECT_ROOT
from .domain import TYPE_ORDER, PaperSnapshot, QType

OUTPUT_DIR = PROJECT_ROOT / "output"
CN_NUM = "一二三四五六七八九十"
LETTERS = "ABCD"

# (路径, ttc 中的子字体序号)。只列 TrueType 轮廓的字体：ReportLab 不支持 OTF/CFF 轮廓。
REGULAR_CANDIDATES = [
    ("C:/Windows/Fonts/simsun.ttc", 0),       # 宋体：试卷正文最常用
    ("C:/Windows/Fonts/msyh.ttc", 0),         # 微软雅黑
    ("C:/Windows/Fonts/simhei.ttf", 0),
    ("/System/Library/Fonts/STHeiti Light.ttc", 0),
    ("/Library/Fonts/Arial Unicode.ttf", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc", 0),
    ("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 0),
    ("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", 0),
]
BOLD_CANDIDATES = [
    ("C:/Windows/Fonts/simhei.ttf", 0),       # 黑体：标题
    ("C:/Windows/Fonts/msyhbd.ttc", 0),
    ("/System/Library/Fonts/STHeiti Medium.ttc", 0),
]


class ExportError(Exception):
    pass


@dataclass
class Fonts:
    regular: str
    bold: str


_FONTS: Fonts | None = None


def _try_register(name: str, candidates: list[tuple[str, int]]) -> bool:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for path, index in candidates:
        if not Path(path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(name, path, subfontIndex=index))
            return True
        except Exception:
            continue  # 字体文件不兼容，换下一个
    return False


# 中文排版“避头点”：这些标点不能出现在行首。ReportLab 自带的列表漏了全角问号、逗号等，
# 实测会出现“？”单独占一行的情况，这里补全。
NO_LINE_START = "，。、；：？！）》」』】〕…—”’％"


def _patch_line_breaking() -> None:
    import reportlab.lib.textsplit as ts
    import reportlab.platypus.paragraph as para
    extra = "".join(ch for ch in NO_LINE_START if ch not in ts.ALL_CANNOT_START)
    if extra:
        ts.ALL_CANNOT_START += extra
        para.ALL_CANNOT_START = ts.ALL_CANNOT_START


def register_fonts() -> Fonts:
    global _FONTS
    if _FONTS is not None:
        return _FONTS
    _patch_line_breaking()
    env_regular, env_bold = os.getenv("EXAM_PDF_FONT"), os.getenv("EXAM_PDF_FONT_BOLD")
    regular = ([(env_regular, 0)] if env_regular else []) + REGULAR_CANDIDATES
    if not _try_register("CN", regular):
        raise ExportError(
            "没有找到可用的中文 TrueType 字体。请在 .env 中设置 EXAM_PDF_FONT=字体文件路径，"
            "例如 C:/Windows/Fonts/simsun.ttc")
    bold = ([(env_bold, 0)] if env_bold else []) + BOLD_CANDIDATES
    _FONTS = Fonts(regular="CN", bold="CN-Bold" if _try_register("CN-Bold", bold) else "CN")
    return _FONTS


# ---------------- 排版 ----------------

def _styles(fonts: Fonts):
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle

    base = dict(fontName=fonts.regular, wordWrap="CJK", leading=20, fontSize=11)
    return {
        "title": ParagraphStyle("title", **{**base, "fontName": fonts.bold, "fontSize": 17, "leading": 26,
                                            "alignment": TA_CENTER, "spaceAfter": 4}),
        "subtitle": ParagraphStyle("subtitle", **{**base, "fontSize": 10.5, "alignment": TA_CENTER,
                                                  "spaceAfter": 10}),
        "section": ParagraphStyle("section", **{**base, "fontName": fonts.bold, "fontSize": 12,
                                                "spaceBefore": 10, "spaceAfter": 4}),
        "q": ParagraphStyle("q", **{**base, "leftIndent": 18, "firstLineIndent": -18, "spaceBefore": 6}),
        "opt": ParagraphStyle("opt", **{**base, "fontSize": 10.5}),
        "label": ParagraphStyle("label", **{**base, "fontSize": 10.5, "leftIndent": 18}),
        "small": ParagraphStyle("small", **{**base, "fontSize": 9, "leading": 13}),
        "cell": ParagraphStyle("cell", **{**base, "fontSize": 9.5, "leading": 13}),
    }


def _p(text: str, style):
    from reportlab.platypus import Paragraph
    # escape：题干中的 < > & 不能被当成排版标记；换行保留为 <br/>
    return Paragraph(escape(text).replace("\n", "<br/>"), style)


_SEP_ROW = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?$")


def _stem(no: int, stem: str, st, width) -> list:
    """题干 → 排版元素。题干里的 Markdown 表格（统计表题常见）画成真正的表格，而不是一行行竖线。"""
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    out, text, rows = [], [], []

    def flush_text():
        if text:
            body = "\n".join(text).strip("\n")
            if body:
                out.append(_p(body if out else f"{no}. {body}", st["q"]))
            text.clear()

    def flush_table():
        if rows:
            if not out and not text:
                out.append(_p(f"{no}.", st["q"]))
            n = max(len(r) for r in rows)
            data = [[_p(c, st["label"]) for c in r + [""] * (n - len(r))] for r in rows]
            t = Table(data, colWidths=[min(width / n, 80)] * n, hAlign="LEFT")
            t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                                   ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                                   ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                   ("LEFTPADDING", (0, 0), (-1, -1), 6), ("RIGHTPADDING", (0, 0), (-1, -1), 6)]))
            out.append(t)
            rows.clear()

    for line in stem.split("\n"):
        ln = line.strip()
        if ln.startswith("|") and ln.count("|") >= 2:
            flush_text()
            if not _SEP_ROW.match(ln):
                rows.append([c.strip() for c in ln.strip("|").split("|")])
        else:
            flush_table()
            text.append(line)
    flush_table()
    flush_text()
    if not out:
        out.append(_p(f"{no}. {stem}", st["q"]))
    return out


def _sections(paper: PaperSnapshot):
    """按题型分大题，返回 [(题型, [(卷面题号, 题目)])]，题号与蓝图一致。"""
    pos = {s.slot_id: s.position for s in paper.blueprint.slots}
    groups: dict[QType, list] = {}
    for q in paper.ordered_questions():
        groups.setdefault(q.qtype, []).append((pos[q.slot_id], q))
    return [(t, groups[t]) for t in TYPE_ORDER if t in groups]


def _section_title(i: int, qtype: QType, items) -> str:
    scores = {q.score for _, q in items}
    total = sum(q.score for _, q in items)
    each = f"每小题 {scores.pop()} 分，" if len(scores) == 1 else ""
    return f"{CN_NUM[i]}、{qtype.value}（本大题共 {len(items)} 小题，{each}共 {total} 分）"


def _options(q, st, width):
    """选项短就一行四个，长就一行两个，再长就一行一个——和纸质试卷的习惯一致。"""
    from reportlab.platypus import Table, TableStyle

    longest = max(len(o) for o in q.options)
    per_row = 4 if longest <= 8 else 2 if longest <= 20 else 1
    cells = [_p(f"{LETTERS[i]}. {o}", st["opt"]) for i, o in enumerate(q.options)]
    rows = [cells[i:i + per_row] for i in range(0, len(cells), per_row)]
    rows[-1] += [""] * (per_row - len(rows[-1]))
    col = (width - 18) / per_row
    t = Table(rows, colWidths=[col + 18] + [col] * (per_row - 1), hAlign="LEFT")
    t.setStyle(TableStyle([("LEFTPADDING", (0, 0), (0, -1), 18), ("LEFTPADDING", (1, 0), (-1, -1), 0),
                           ("TOPPADDING", (0, 0), (-1, -1), 0), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))
    return t


def _footer(text_fn):
    def draw(canvas, doc):
        canvas.saveState()
        canvas.setFont(register_fonts().regular, 9)
        canvas.drawCentredString(doc.pagesize[0] / 2, 1.2 * 28.35, text_fn(doc.page))
        canvas.restoreState()
    return draw


ANSWER_SPACE_CM = {QType.CALC: 3.2, QType.APPLICATION: 4.5, QType.WRITING: 7.0}


def build_student(paper: PaperSnapshot, path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer, Table, TableStyle

    fonts = register_fonts()
    st = _styles(fonts)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title=paper.title)
    width = A4[0] - 4 * cm
    sections = _sections(paper)
    story = [
        _p(paper.title, st["title"]),
        _p(f"（满分 {paper.total_score} 分）", st["subtitle"]),
        _p("班级 ______________　　姓名 ______________　　学号 ______________", st["subtitle"]),
    ]

    # 得分表：大题 + 总分
    header = ["题号"] + [CN_NUM[i] for i in range(len(sections))] + ["总分"]
    table = Table([header, ["得分"] + [""] * (len(sections) + 1)],
                  colWidths=[1.6 * cm] + [1.6 * cm] * (len(sections) + 1), rowHeights=[0.75 * cm, 0.9 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), fonts.regular), ("FONTSIZE", (0, 0), (-1, -1), 10.5),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story += [table, Spacer(1, 0.3 * cm)]

    for i, (qtype, items) in enumerate(sections):
        story.append(_p(_section_title(i, qtype, items), st["section"]))
        for no, q in items:
            block = _stem(no, q.stem, st, width)
            if q.options:
                block.append(_options(q, st, width))
            if qtype in ANSWER_SPACE_CM:
                block.append(Spacer(1, ANSWER_SPACE_CM[qtype] * cm))
            story.append(KeepTogether(block))  # 一道题不被拆到两页

    doc.build(story, onFirstPage=_footer(lambda n: f"第 {n} 页"),
              onLaterPages=_footer(lambda n: f"第 {n} 页"))
    return path


def _status_summary(paper: PaperSnapshot) -> str:
    from .domain import VStatus

    counts = {s: sum(q.vstatus == s for q in paper.questions) for s in VStatus}
    return "，".join(f"{k.value} {v} 题" for k, v in counts.items() if v)


def build_answers(paper: PaperSnapshot, version: int, path: Path) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.platypus import KeepTogether, SimpleDocTemplate, Spacer, Table, TableStyle

    fonts = register_fonts()
    st = _styles(fonts)
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=2 * cm, title=paper.title + "（答案）")
    width = A4[0] - 4 * cm
    story = [_p(paper.title, st["title"]), _p("参考答案与解析", st["subtitle"]),
             _p("验证情况：" + _status_summary(paper) + "。“待老师确认”表示没有工具能自动核对，请人工审阅。",
                st["label"])]

    for i, (qtype, items) in enumerate(_sections(paper)):
        story.append(_p(_section_title(i, qtype, items), st["section"]))
        for no, q in items:
            block = _stem(no, q.stem, st, width)
            if q.options:
                block.append(_options(q, st, width))
            block.append(_p(f"【答案】{q.answer}", st["label"]))
            block.append(_p(f"【解析】{q.analysis}", st["label"]))
            v = q.verification
            block.append(_p(f"【验证】{q.vstatus.value}" + (f"（{v.method}）" if v and v.method else ""), st["label"]))
            story.append(KeepTogether(block))

    # 双向细目表：给老师看的命题依据
    story += [Spacer(1, 0.4 * cm), _p("附：双向细目表", st["section"])]
    rows = [["题号", "题型", "分值", "知识点", "难度", "认知层次"]]
    for s in sorted(paper.blueprint.slots, key=lambda s: s.position):
        rows.append([str(s.position), s.qtype.value, str(s.score), _p(s.knowledge_point, st["cell"]),
                     s.difficulty.value, s.cognitive_level.value])
    t = Table(rows, colWidths=[1.2 * cm, 1.8 * cm, 1.2 * cm, width - 8.6 * cm, 1.6 * cm, 2.8 * cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), fonts.regular), ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("FONTNAME", (0, 0), (-1, 0), fonts.bold), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (2, -1), "CENTER"), ("ALIGN", (4, 0), (-1, -1), "CENTER"),
    ]))
    story.append(t)
    footer = f"{paper.paper_id} · 版本 v{version} · 教师用卷"
    doc.build(story, onFirstPage=_footer(lambda n: f"第 {n} 页　{footer}"),
              onLaterPages=_footer(lambda n: f"第 {n} 页　{footer}"))
    return path


def export_paper(paper: PaperSnapshot, version: int, out_dir: Path = OUTPUT_DIR) -> tuple[Path, Path]:
    try:
        import reportlab  # noqa: F401
    except ImportError as e:
        raise ExportError("缺少 reportlab，请先运行：pip install reportlab") from e
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{paper.paper_id}_v{version}"
    student = build_student(paper, out_dir / f"{stem}_学生版.pdf")
    answers = build_answers(paper, version, out_dir / f"{stem}_答案版.pdf")
    return student, answers
