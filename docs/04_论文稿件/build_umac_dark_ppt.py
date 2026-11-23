"""Build the advisor briefing deck in the UMAC navy template style.

Base: ppt/深色澳门大学PhD申博面试PPT模板.pptx (master, theme, logo).
Output: docs/04_论文稿件/导师阶段汇报_UMAC_2026-11-23_generated.pptx

NOTE: this always writes a fresh copy under *_generated.pptx so it never
overwrites the hand-edited working deck. To refresh only the architecture
figure inside the working deck, run sync_deck_figure.py instead.

Usage:
    python docs/04_论文稿件/build_umac_dark_ppt.py
"""
from __future__ import annotations

import copy
import zipfile
from pathlib import Path

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT.parent / "ppt" / "深色澳门大学PhD申博面试PPT模板.pptx"
OUT = Path(__file__).resolve().parent / "导师阶段汇报_UMAC_2026-11-23_generated.pptx"
ASSETS = Path(__file__).resolve().parent / "_assets"

NAVY = RGBColor(0x00, 0x2C, 0x54)
NAVY_MID = RGBColor(0x1D, 0x6F, 0xA9)
CYAN = RGBColor(0x36, 0xAF, 0xCE)
GOLD = RGBColor(0xF1, 0x9D, 0x19)
INK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

LATIN = "Times New Roman"
EA = "黑体"

W, H = 13.333, 7.5
SECTIONS = ["1  Research Question", "2  Method", "3  Results", "4  Boundaries"]


# --------------------------------------------------------------------------- helpers
def new_slide(prs, layout):
    s = prs.slides.add_slide(layout)
    for sh in list(s.shapes):
        if sh.is_placeholder:
            sh._element.getparent().remove(sh._element)
    return s


def set_font(run, size: float, bold: bool = False, color: RGBColor = INK,
             latin: str = LATIN, ea: str = EA) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.name = latin
    run.font.color.rgb = color
    rPr = run._r.get_or_add_rPr()
    for tag, face in (("a:latin", latin), ("a:ea", ea)):
        el = rPr.find(qn(tag))
        if el is None:
            el = etree.SubElement(rPr, qn(tag))
        el.set("typeface", face)


def textbox(slide, x, y, w, h, *, anchor=MSO_ANCHOR.TOP, wrap=True):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = Inches(0.02)
    tf.margin_right = Inches(0.02)
    tf.margin_top = Inches(0.01)
    tf.margin_bottom = Inches(0.01)
    return box


def para(box, text, size, *, bold=False, color=INK, first=False, space_after=4,
         align=PP_ALIGN.LEFT, line=1.15):
    tf = box.text_frame
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_after = Pt(space_after)
    p.line_spacing = line
    run = p.add_run()
    run.text = text
    set_font(run, size, bold, color)
    return p


def rect(slide, x, y, w, h, fill: RGBColor, *, line=None, shape=MSO_SHAPE.RECTANGLE, radius=None):
    sh = slide.shapes.add_shape(shape, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    if radius is not None and shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        sh.adjustments[0] = radius
    sh.text_frame.word_wrap = True
    return sh


def slide_number(slide, n: int, total: int) -> None:
    box = textbox(slide, W - 1.6, H - 0.34, 1.2, 0.24)
    para(box, f"{n:02d} / {total:02d}", 10, color=GRAY, first=True, align=PP_ALIGN.RIGHT)


def nav(slide, active: int, page: int, total: int) -> None:
    rect(slide, 0, 0, W, 0.62, NAVY)
    # logo plate
    slide.shapes.add_picture(str(ASSETS / "logo_white.png"), Inches(0.22), Inches(0.11), height=Inches(0.40))
    xs = [2.35, 5.05, 7.75, 10.45]
    for i, label in enumerate(SECTIONS):
        box = textbox(slide, xs[i], 0.12, 2.55, 0.38, anchor=MSO_ANCHOR.MIDDLE)
        para(box, label, 13, bold=(i == active), color=WHITE if i == active else RGBColor(0xC9, 0xD6, 0xE4),
             first=True, align=PP_ALIGN.LEFT)
        if i == active:
            rect(slide, xs[i], 0.55, 1.55, 0.045, CYAN)


def title(slide, text: str, size: float = 26) -> None:
    box = textbox(slide, 0.42, 0.78, W - 0.9, 0.55, anchor=MSO_ANCHOR.MIDDLE)
    para(box, "➢  " + text, size, bold=True, color=NAVY, first=True)
    rect(slide, 0.45, 1.38, W - 0.9, 0.012, RGBColor(0xC9, 0xD6, 0xE4))


def body_bullets(slide, x, y, w, items, size=14, gap=8, color=INK, bullet="▪  "):
    box = textbox(slide, x, y, w, 4.6)
    for i, item in enumerate(items):
        if isinstance(item, tuple):
            head, rest = item
            p = para(box, "", size, first=(i == 0), space_after=gap)
            r1 = p.add_run(); r1.text = bullet; set_font(r1, size, True, NAVY)
            r2 = p.add_run(); r2.text = head; set_font(r2, size, True, color)
            if rest:
                r3 = p.add_run(); r3.text = rest; set_font(r3, size, False, color)
        else:
            p = para(box, "", size, first=(i == 0), space_after=gap)
            r1 = p.add_run(); r1.text = bullet; set_font(r1, size, True, NAVY)
            r2 = p.add_run(); r2.text = item; set_font(r2, size, False, color)
    return box


def table(slide, x, y, w, rows, cols, col_widths, row_h=0.34):
    shape = slide.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(row_h * rows))
    tbl = shape.table
    # no-style, no-grid so fills and fonts are fully controlled
    tblPr = tbl._tbl.tblPr
    for el in tblPr.findall(qn("a:tableStyleId")):
        tblPr.remove(el)
    sid = etree.SubElement(tblPr, qn("a:tableStyleId"))
    sid.text = "{2D5ABB26-0587-4C30-8999-92F81FD0307C}"
    for i, cw in enumerate(col_widths):
        tbl.columns[i].width = Inches(cw)
    for r in range(rows):
        tbl.rows[r].height = Inches(row_h)
    return tbl


def fill_cell(cell, text, size=12, bold=False, color=INK, bg=None, align=PP_ALIGN.CENTER):
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    cell.margin_left = Inches(0.06); cell.margin_right = Inches(0.06)
    cell.margin_top = Inches(0.02); cell.margin_bottom = Inches(0.02)
    if bg is not None:
        cell.fill.solid(); cell.fill.fore_color.rgb = bg
    else:
        cell.fill.background()
    tf = cell.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run(); run.text = text
    set_font(run, size, bold, color)


def card(slide, x, y, w, h, num, head, desc, *, accent=False, size_head=15, size_desc=11.5):
    sh = rect(slide, x, y, w, h, NAVY if accent else LIGHT,
              shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
    color = WHITE if accent else NAVY
    body_color = RGBColor(0xDC, 0xE6, 0xF0) if accent else GRAY
    nb = textbox(slide, x + 0.18, y + 0.14, w - 0.36, 0.3)
    para(nb, num, 10.5, bold=True, color=CYAN if accent else NAVY_MID, first=True)
    hb = textbox(slide, x + 0.18, y + 0.5, w - 0.36, 0.4)
    para(hb, head, size_head, bold=True, color=color, first=True)
    db = textbox(slide, x + 0.18, y + 0.95, w - 0.36, h - 1.05)
    para(db, desc, size_desc, color=body_color, first=True, line=1.25)


def note(slide, text, y=6.55, size=10.5, color=GRAY):
    box = textbox(slide, 0.45, y, W - 0.9, 0.3)
    para(box, text, size, color=color, first=True)


def footer(slide, text="University of Macau  ·  Faculty of Science and Technology"):
    rect(slide, 0, H - 0.36, W, 0.012, RGBColor(0xC9, 0xD6, 0xE4))
    box = textbox(slide, 0.45, H - 0.33, 9.0, 0.24)
    para(box, text, 8.5, color=GRAY, first=True)


# --------------------------------------------------------------------------- build
def build() -> None:
    ASSETS.mkdir(exist_ok=True)
    with zipfile.ZipFile(TEMPLATE) as z:
        (ASSETS / "logo.png").write_bytes(z.read("ppt/media/image1.png"))
        (ASSETS / "logo_white.png").write_bytes(z.read("ppt/media/image2.png"))

    prs = Presentation(TEMPLATE)
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst):
        prs.part.drop_rel(sldId.get(qn("r:id")))
        sldIdLst.remove(sldId)

    blank = prs.slide_layouts[6]
    total = 14

    # 1 cover -----------------------------------------------------------------
    s = new_slide(prs, blank)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = WHITE
    s.shapes.add_picture(str(ASSETS / "logo.png"), Inches((W - 3.4) / 2), Inches(0.55), width=Inches(3.4))
    rect(s, 0, 2.05, W, 2.25, NAVY)
    rect(s, 0, 2.05, W, 0.06, CYAN)
    tb = textbox(s, 0.9, 2.35, W - 1.8, 1.7, anchor=MSO_ANCHOR.MIDDLE)
    para(tb, "对齐之后，边界还在", 40, bold=True, color=WHITE, first=True, align=PP_ALIGN.CENTER)
    para(tb, "LLM 调查模拟的个体、群体与时间边界", 20, color=RGBColor(0xC9, 0xD6, 0xE4),
         space_after=0, align=PP_ALIGN.CENTER)
    info = textbox(s, 3.9, 4.75, 6.0, 1.7)
    for i, line in enumerate([
        "陈昱源 · CHENYU YUAN",
        "导师：王鹏阳 教授",
        "澳门大学 · 科技学院 · 计算机科学系",
        "2026 年 11 月 导师汇报",
    ]):
        para(info, line, 14, bold=(i == 0), color=NAVY if i == 0 else GRAY, first=(i == 0), space_after=8)
    slide_number(s, 1, total)

    # 2 problem ---------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 0, 2, total); title(s, "大模型当\u201c虚拟受访者\u201d，现有评测只测了演技")
    body_bullets(s, 0.55, 1.7, 6.6, [
        ("调查很贵：", "找人、访问、等数据，一轮几个月。"),
        ("现有做法：", "把画像塞进 prompt，问一句，给回答打分（one-shot）。"),
        ("它测的是：", "\u201c模型能不能演得像\u201d，而不是有没有学到调查数据的结构。"),
        ("我这篇问的是：", "用真实调查数据微调之后，到底买到了什么、买在哪儿。"),
    ], size=15, gap=14)
    rect(s, 7.45, 1.75, 5.35, 2.95, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    qb = textbox(s, 7.75, 1.95, 4.8, 3.1)
    para(qb, "本文的问题", 12, bold=True, color=NAVY_MID, first=True, space_after=10)
    para(qb, "不是\u201cLLM 能不能替代受访者\u201d", 17, bold=True, color=NAVY, space_after=6, line=1.2)
    para(qb, "而是\u201c结构化对齐买到了什么，买在哪儿\u201d", 17, bold=True, color=NAVY, space_after=14, line=1.2)
    para(qb, "三层答案：多少 / 哪里 / 没买到", 13, color=GRAY, line=1.3)
    footer(s); slide_number(s, 2, total)

    # 3 map -------------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 0, 3, total); title(s, "论文地图：六件事，一句话一件")
    items = [
        ("01", "数据", "132,948 → 98,159 行；T / R 两套协议。", True),
        ("02", "阶梯", "微调吃掉 82% 差距（77–95%），但不反超。", False),
        ("03", "剖面", "两端赢、中间输；12/12 格同向。", False),
        ("04", "群体", "先验更准：0.0076 对 0.0215（EDC）。", False),
        ("05", "时间", "表观塌缩来自输入缺失，不是时间结论。", False),
        ("06", "边界", "814 行（29.9%）渲染矛盾；无跨年 person key。", False),
    ]
    x0, y0, cw, ch, gx, gy = 0.55, 1.65, 4.02, 2.3, 0.14, 0.22
    for i, (num, head, desc, accent) in enumerate(items):
        c, r = i % 3, i // 3
        card(s, x0 + c * (cw + gx), y0 + r * (ch + gy), cw, ch, num, head, desc, accent=accent)
    footer(s); slide_number(s, 3, total)

    # 4 architecture ----------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 1, 4, total); title(s, "方法总览：两条 pipeline，三层评测")
    fig = ASSETS / "fig1_pipeline.png"
    if not fig.exists():
        fig = ROOT / "thesis" / "figures" / "fig1_pipeline.png"
    s.shapes.add_picture(str(fig), Inches(1.93), Inches(1.52), height=Inches(5.25))
    note(s, "左：个体级 PAS（画像 → prompt → LoRA → 读出 → 留出评测）。右：群体级 FAS（外推 → IPF → 合成人口，与 prior 抽样对照）。底部：个体 / 群体 / 时间三层评测。", y=6.86, size=9.5)
    footer(s); slide_number(s, 4, total)

    # 4 data ------------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 1, 5, total); title(s, "数据：先把它做对，再谈效果")
    body_bullets(s, 0.55, 1.7, 6.1, [
        ("ANES 六个波次：", "132,948 行清洗到 98,159 行。"),
        ("三类容易漏的问题：", "8,165 行尾点候选人名（其中 8,141 行重复）、23,060 行社会群体温度计混入候选人列、3,588 行缺失或越界评分。"),
        ("两套协议：", "T 按年份切；R 在同一年内按编号留出 20%。"),
        ("一句实话：", "R 是编号级留出，不是人物级；发布文件没有跨年 person key。"),
    ], size=13.5, gap=12)
    tbl = table(s, 6.95, 1.78, 5.9, 7, 4, [1.7, 1.55, 1.2, 1.45], row_h=0.42)
    data = [
        ("Split", "年份", "行数", "编号数"),
        ("train_t", "2016–2020", "6,516", "2,720"),
        ("val_t", "2022", "2,501", "1,251"),
        ("test_t", "2024", "3,421", "1,712"),
        ("train_r", "2016–2020", "5,094", "2,131"),
        ("test_r", "2016–2022", "6,014", "192"),
        ("test_tr", "2024", "226", "113"),
    ]
    for r, row in enumerate(data):
        for c, val in enumerate(row):
            fill_cell(tbl.cell(r, c), val, size=11.5, bold=(r == 0),
                      color=WHITE if r == 0 else INK,
                      bg=NAVY if r == 0 else (LIGHT if r % 2 == 0 else WHITE))
    note(s, "主表用 2016 / 2018 / 2019 的 test_r：2,726 行。", y=5.05, size=10.5)
    footer(s); slide_number(s, 5, total)

    # 5 PAS task --------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 1, 6, total); title(s, "个体级任务：从画像到 0–100 分")
    steps = [
        ("选民画像", "年龄 / 性别 / 种族 / 教育 / 党派 / 意识形态"),
        ("候选人属性", "党派 / 意识形态 / 年龄 / 职位"),
        ("Qwen3-8B + LoRA", "r=8, α=32, 3 epochs, 3 seeds"),
        ("评分与选择", "0–100 分 → 组内评分诱导选择"),
    ]
    bw, bh, gap = 2.85, 1.75, 0.3
    for i, (head, desc) in enumerate(steps):
        x = 0.6 + i * (bw + gap)
        rect(s, x, 1.8, bw, bh, NAVY if i == 2 else LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06)
        hb = textbox(s, x + 0.15, 1.95, bw - 0.3, 0.5)
        para(hb, head, 15, bold=True, color=WHITE if i == 2 else NAVY, first=True)
        db = textbox(s, x + 0.15, 2.5, bw - 0.3, 0.95)
        para(db, desc, 11.5, color=RGBColor(0xDC, 0xE6, 0xF0) if i == 2 else GRAY, first=True, line=1.3)
        if i < 3:
            ab = textbox(s, x + bw - 0.02, 2.5, gap + 0.05, 0.4)
            para(ab, "→", 18, bold=True, color=NAVY_MID, first=True, align=PP_ALIGN.CENTER)
    body_bullets(s, 0.6, 4.1, 12.1, [
        ("基线：", "Ridge 与 HistGB，用同一套特征；另有未微调的同 prompt 对照。"),
        ("读出：", "generation（贪心解码取整数）为主；anchor likelihood 作校准诊断。"),
        ("指标：", "MAE、Pearson、rating-induced choice（含 tie-aware）。"),
    ], size=13.5, gap=10)
    footer(s); slide_number(s, 6, total)

    # 6 two inputs ------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 1, 7, total); title(s, "两种输入，并不相同")
    rect(s, 0.55, 1.7, 5.95, 3.9, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    lb = textbox(s, 0.85, 1.9, 5.35, 3.5)
    para(lb, "LLM · core prompt", 14, bold=True, color=NAVY_MID, first=True, space_after=8)
    for line in ["文本里写的是人口学、党派、意识形态、年龄差", "空值会被填成默认值（Independent / Moderate）",
                 "2024 这类整列缺失时，所有人看起来都一样"]:
        para(lb, "· " + line, 12.5, color=INK, space_after=8, line=1.25)
    rect(s, 6.75, 1.7, 6.0, 3.9, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    rb = textbox(s, 7.05, 1.9, 5.4, 3.5)
    para(rb, "表格 · HistGB / Ridge", 14, bold=True, color=NAVY_MID, first=True, space_after=8)
    for line in ["数值 pid7、party_match，加上相似度等派生列", "同一个 pid7 越界时，表格看到数字、文本看到\u201cIndependent\u201d",
                 "结果：814 行（29.9%）上，两边信息不同"]:
        para(rb, "· " + line, 12.5, color=INK, space_after=8, line=1.25)
    note(s, "这个差异对 LLM 不利，所以 82% 是下限；去掉 2018 后是 77%。", y=5.78, size=12, color=NAVY)
    footer(s); slide_number(s, 7, total)

    # 7 ladder ----------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 2, 8, total); title(s, "效果阶梯：吃掉 82%，但没有反超")
    tbl = table(s, 0.55, 1.72, 6.55, 7, 3, [2.6, 1.95, 2.0], row_h=0.46)
    rows = [("模型", "MAE", "说明"), ("无信息（常数）", "32.65", "训练均值"),
            ("候选人均值", "32.35", "corr 0.155"), ("base Qwen", "24.15", "同一批测试行"),
            ("Ridge", "22.86", "线性对照"), ("微调 Qwen ×3", "21.35", "± 0.19"),
            ("HistGB", "20.73", "强基线")]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            fill_cell(tbl.cell(r, c), val, size=12, bold=(r == 0 or r == 5),
                      color=WHITE if r == 0 else (NAVY if r == 5 else INK),
                      bg=NAVY if r == 0 else (RGBColor(0xDC, 0xE6, 0xF0) if r == 5 else (LIGHT if r % 2 == 0 else WHITE)))
    rect(s, 7.45, 1.75, 5.35, 1.55, NAVY, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    nb = textbox(s, 7.75, 1.92, 4.8, 1.2)
    para(nb, "82%", 40, bold=True, color=WHITE, first=True, space_after=2)
    para(nb, "吃掉 base → HistGB 的 MAE 差距", 12, color=RGBColor(0xC9, 0xD6, 0xE4), space_after=0)
    body_bullets(s, 7.45, 3.55, 5.4, [
        ("同一张表：", "两个模型都读这张处理表，所以是相对距离，不是绝对上限。"),
        ("换分母：", "从无信息起算 95%；去掉 2018 是 77%。"),
        ("没吃掉的 18%：", "corr 0.666 对 0.703；tie-aware choice 0.481 对 0.555。"),
    ], size=12, gap=10)
    footer(s); slide_number(s, 8, total)

    # 8 profile ---------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 2, 9, total); title(s, "误差剖面：两端赢，中间输")
    tbl = table(s, 0.55, 1.72, 6.4, 5, 5, [1.35, 1.3, 1.3, 1.3, 1.15], row_h=0.58)
    rows = [("年份", "Q1 最低", "Q2", "Q3", "Q4 最高"),
            ("2016", "19.4 / 25.7", "27.1 / 18.4", "19.7 / 16.5", "24.6 / 30.0"),
            ("2018", "17.1 / 21.1", "34.4 / 19.1", "29.8 / 19.2", "11.2 / 28.1"),
            ("2019", "13.6 / 19.5", "26.4 / 16.9", "22.0 / 15.5", "17.8 / 24.2"),
            ("谁更好", "LLM", "HistGB", "HistGB", "LLM")]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            bold = r == 0 or r == 4
            bg = NAVY if r == 0 else (RGBColor(0xDC, 0xE6, 0xF0) if r == 4 else (LIGHT if r % 2 == 0 else WHITE))
            fill_cell(tbl.cell(r, c), val, size=11.5, bold=bold, color=WHITE if r == 0 else INK, bg=bg)
    body_bullets(s, 7.3, 1.72, 5.5, [
        ("机制：", "HistGB 预测标准差 24、真值 36，所以两端必错；LLM 是 40，两端跟得上但中间更吵。"),
        ("检验：", "三年 × 三 seed，12/12 格同向，置换检验 p=0.0003。"),
        ("主检验：", "中间减两端 +16.45 MAE [14.78, 18.09]。"),
        ("误差-极值相关：", "LLM −0.184，HistGB +0.251。"),
    ], size=12.5, gap=12)
    note(s, "每格是 LLM / HistGB 的 MAE；Q1 最低、Q4 最高真值分段。", y=6.3, size=10.5)
    footer(s); slide_number(s, 9, total)

    # 9 population ------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 2, 10, total); title(s, "群体层面：接近先验，但没有超过")
    tbl = table(s, 0.55, 1.8, 7.1, 4, 4, [1.5, 1.55, 1.55, 2.5], row_h=0.55)
    rows = [("配置", "prior-only", "LLM 最佳（v3）", "结论"),
            ("EDC 边际 / 联合", "0.0004 / 0.0076", "0.0040 / 0.0215", "prior 更准"),
            ("EHC 边际 / 联合", "0.0011 / 0.0225", "0.0077 / 0.0363", "prior 更准"),
            ("差值 CI（EDC 联合）", "−0.0138 [−0.021, −0.0067]", "", "区间不含 0")]
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            fill_cell(tbl.cell(r, c), val, size=11.5, bold=(r == 0),
                      color=WHITE if r == 0 else INK,
                      bg=NAVY if r == 0 else (LIGHT if r % 2 == 0 else WHITE),
                      align=PP_ALIGN.CENTER if c else PP_ALIGN.LEFT)
    body_bullets(s, 8.0, 1.8, 4.8, [
        ("SFT 有用：", "把 LLM 从 0.0552 拉到 0.0048（−91%）。"),
        ("但没超过：", "直接从不含语言模型的重建先验抽样更准。"),
        ("价值在哪：", "条件生成与自然语言控制，当前评测还没隔离出来。"),
    ], size=12, gap=12)
    note(s, "400 行合成样本 vs 真实 2024 参照（EDC 1,601 / EHC 1,552）；JS 越低越好。", y=5.9, size=10.5)
    footer(s); slide_number(s, 10, total)

    # 10 temporal -------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 3, 11, total); title(s, "时间层面：没有干净的外推证据")
    body_bullets(s, 0.55, 1.7, 7.0, [
        ("表观下降：", "2019 的 0.722 → 2020 的 0.478 → 2022 的 0.408 → 2024 的 −0.056；HistGB 同样下降。"),
        ("三个波次不能用：", "2020 没有 5 点意识形态、党派刻度反向；2022 只有 194 行；2024 的政治变量在表里全丢。"),
        ("探针：", "把 2019 同一批行的政治块抹掉，corr 从 0.714 掉到 0.166，比真实的 2024 还接近零。"),
        ("所以结论是：", "输入缺失本身就能复现大部分损失；不能说\u201cLLM 四年后失效\u201d。"),
    ], size=13.5, gap=12)
    rect(s, 7.9, 1.75, 4.9, 3.9, LIGHT, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.05)
    kb = textbox(s, 8.2, 1.95, 4.3, 3.5)
    para(kb, "2024 的实际输入", 12, bold=True, color=NAVY_MID, first=True, space_after=10)
    para(kb, "3,421", 34, bold=True, color=NAVY, space_after=2)
    para(kb, "条 prompt，只有 1 种政治档案", 12, color=GRAY, space_after=14, line=1.25)
    para(kb, "0.714 → 0.166", 24, bold=True, color=NAVY, space_after=2)
    para(kb, "抹掉政治块后的 2019 相关性", 12, color=GRAY, line=1.25)
    footer(s); slide_number(s, 11, total)

    # 11 boundaries -----------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 3, 12, total); title(s, "三项边界，写在 Limitations 里")
    card(s, 0.55, 1.7, 4.02, 2.9, "01 · 身份", "编号 ≠ 人物", "发布文件 caseid 是波内编号，跨年出生年一致率仅 1.2–2.6%。R 是编号留出，不是人物留出。", accent=False)
    card(s, 4.71, 1.7, 4.02, 2.9, "02 · 输入", "同一张表，两种渲染", "814 行（29.9%）上，文本说\u201cIndependent\u201d而表格说\u201c同党\u201d；这个差异对 LLM 不利。", accent=True)
    card(s, 8.87, 1.7, 4.02, 2.9, "03 · 修复", "已修表，未重训", "2024 恢复 pid7 94%、ideo5 90%、educ 100%（档案 1 → 575），但旧特征管线不可复现，重训列为 future work。", accent=False)
    note(s, "三项都写进了 Limitations：前两项决定主张的边界，第三项是 future work。", y=5.05, size=12, color=NAVY)
    footer(s); slide_number(s, 12, total)

    # 12 conclusions ----------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 3, 13, total); title(s, "结论：买到\u201c能用\u201d，没买到\u201c更好\u201d")
    lines = [
        ("多少", "微调吃掉 82%（77–95%）的差距，但没有反超强基线。"),
        ("哪里", "增益在量表两端，损失在中间；两个模型的误差方向相反。"),
        ("没买到", "群体不如直接抽样；跨期没有干净证据——表观塌缩主要来自输入退化。"),
    ]
    for i, (tag, text) in enumerate(lines):
        y = 1.85 + i * 1.45
        rect(s, 0.6, y, 1.5, 0.95, NAVY)
        tb = textbox(s, 0.6, y, 1.5, 0.95, anchor=MSO_ANCHOR.MIDDLE)
        para(tb, tag, 15, bold=True, color=WHITE, first=True, align=PP_ALIGN.CENTER)
        b = textbox(s, 2.35, y, 10.2, 0.95, anchor=MSO_ANCHOR.MIDDLE)
        para(b, text, 17, color=INK, first=True, line=1.2)
    note(s, "一句话：结构化对齐是必要的，但不是充分的；它的收益是局部的、有形状的。", y=6.2, size=12.5, color=NAVY)
    footer(s); slide_number(s, 13, total)

    # 13 decisions ------------------------------------------------------------
    s = new_slide(prs, blank); nav(s, 3, 14, total); title(s, "请老师定三件事")
    items = [
        ("01", "页数", "正文 18 页（共约 41 页）是否符合学院要求？如需要，我按附录压缩。"),
        ("02", "是否补修复表重训", "2–3 天：先做单年 parity test，再重训；会把时间章从\u201c没有证据\u201d变成\u201c残余为 X\u201d。"),
        ("03", "投稿定位", "workshop / 期刊 / 只作为毕业论文。"),
    ]
    for i, (num, head, desc) in enumerate(items):
        y = 1.75 + i * 1.55
        rect(s, 0.6, y, 0.9, 0.9, NAVY)
        nb = textbox(s, 0.6, y, 0.9, 0.9, anchor=MSO_ANCHOR.MIDDLE)
        para(nb, num, 16, bold=True, color=WHITE, first=True, align=PP_ALIGN.CENTER)
        hb = textbox(s, 1.75, y - 0.02, 10.8, 0.45)
        para(hb, head, 18, bold=True, color=NAVY, first=True)
        db = textbox(s, 1.75, y + 0.42, 10.8, 0.6)
        para(db, desc, 13, color=GRAY, first=True, line=1.2)
    note(s, "谢谢老师。", y=6.35, size=16, color=NAVY)
    footer(s); slide_number(s, 14, total)

    prs.save(OUT)
    print("saved", OUT, "slides:", len(prs.slides))


if __name__ == "__main__":
    build()
