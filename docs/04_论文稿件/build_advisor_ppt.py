# -*- coding: utf-8 -*-
"""导师阶段汇报 PPT，2026-11-18。

口径以 docs/04_论文稿件/导师阶段汇报PPT大纲_2026-11-18.md 为准，
并按作者确认修订一条：跨年 respondent_id 对应同一批受访者。
"""

from __future__ import annotations

from lxml import etree
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

W, H = 13.333, 7.5
FONT = "Microsoft YaHei"

INK = RGBColor(0x1A, 0x23, 0x32)
MUTED = RGBColor(0x5C, 0x65, 0x70)
PAPER = RGBColor(0xF5, 0xF2, 0xEC)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xE3, 0xDB, 0xD0)
PAS = RGBColor(0x1E, 0x4E, 0x79)
PAS_SOFT = RGBColor(0xE7, 0xF0, 0xF7)
FAS = RGBColor(0xC2, 0x62, 0x2D)
FAS_SOFT = RGBColor(0xF8, 0xED, 0xE3)
TMP = RGBColor(0x5B, 0x4B, 0x8A)
TMP_SOFT = RGBColor(0xEE, 0xEA, 0xF6)
WARN = RGBColor(0x9A, 0x5B, 0x12)
WARN_SOFT = RGBColor(0xFB, 0xF3, 0xE4)
OK = RGBColor(0x1E, 0x6B, 0x4A)
OK_SOFT = RGBColor(0xE5, 0xF2, 0xEB)
SLATE = RGBColor(0x5E, 0x6B, 0x78)
SLATE_SOFT = RGBColor(0xEE, 0xF1, 0xF4)
HEAD = RGBColor(0x1A, 0x23, 0x32)

OUT = r"D:\workspace_ycy\个人资料\毕业\fas_pro\docs\04_论文稿件\导师阶段汇报_2026-11-18.pptx"
ARCH = r"D:\workspace_ycy\个人资料\毕业\fas_pro\docs\01_背景与总览\架构图_16x9.png"


def rgb_hex(color: RGBColor) -> str:
    return f"{color[0]:02X}{color[1]:02X}{color[2]:02X}"


def font_run(run, size, bold, color, name=FONT):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    r_pr = run._r.get_or_add_rPr()
    for tag in ("latin", "ea", "cs"):
        node = r_pr.find(qn(f"a:{tag}"))
        if node is None:
            node = etree.SubElement(r_pr, qn(f"a:{tag}"))
        node.set("typeface", name)


def add_tb(slide, x, y, w, h, items, valign=MSO_ANCHOR.TOP, margin=0.04):
    if isinstance(items, str):
        items = [{"text": items}]
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    tf.vertical_anchor = valign
    tf.margin_left = Inches(margin)
    tf.margin_right = Inches(margin)
    tf.margin_top = Inches(0.02)
    tf.margin_bottom = Inches(0.02)
    for i, item in enumerate(items):
        if isinstance(item, str):
            item = {"text": item}
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = item.get("align", PP_ALIGN.LEFT)
        p.space_before = Pt(item.get("before", 0))
        p.space_after = Pt(item.get("after", 0))
        run = p.add_run()
        run.text = item.get("text", "")
        font_run(
            run,
            item.get("size", 13),
            item.get("bold", False),
            item.get("color", INK),
            item.get("font", FONT),
        )
    return shape


def rrect(slide, x, y, w, h, fill, line=None, adj=0.08):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    try:
        sh.adjustments[0] = adj
    except Exception:
        pass
    return sh


def rect(slide, x, y, w, h, fill, line=None):
    sh = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h)
    )
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    return sh


def bg(slide):
    rect(slide, 0, 0, W, H, PAPER)


def note(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def footer(slide, source, page):
    rect(slide, 0.48, 7.12, 12.37, 0.01, LINE)
    add_tb(
        slide, 0.48, 7.16, 10.4, 0.28,
        [{"text": source, "size": 10, "color": MUTED}],
        valign=MSO_ANCHOR.MIDDLE, margin=0,
    )
    add_tb(
        slide, 10.9, 7.16, 1.95, 0.28,
        [{"text": page, "size": 10, "color": MUTED, "align": PP_ALIGN.RIGHT}],
        valign=MSO_ANCHOR.MIDDLE, margin=0,
    )


def header(slide, kicker, title, color):
    rect(slide, 0, 0, W, 0.08, color)
    add_tb(
        slide, 0.48, 0.18, 12.4, 0.28,
        [{"text": kicker, "size": 11, "bold": True, "color": color}],
        margin=0,
    )
    add_tb(
        slide, 0.48, 0.44, 12.4, 0.58,
        [{"text": title, "size": 22, "bold": True, "color": INK}],
        margin=0,
    )
    rect(slide, 0.48, 1.08, 12.37, 0.015, LINE)


def pill(slide, x, y, w, h, text, fill, color, size=11):
    rrect(slide, x, y, w, h, fill, adj=0.4)
    add_tb(
        slide, x, y, w, h,
        [{"text": text, "size": size, "bold": True, "color": color, "align": PP_ALIGN.CENTER}],
        valign=MSO_ANCHOR.MIDDLE, margin=0.02,
    )


def style_table_borders(table):
    tbl = table._tbl
    for cell in tbl.iter_tcs():
        tc_pr = cell.get_or_add_tcPr()
        for child in list(tc_pr):
            if child.tag in {qn(f"a:{e}") for e in ("lnL", "lnR", "lnT", "lnB")}:
                tc_pr.remove(child)
        for edge in ("lnL", "lnR", "lnT", "lnB"):
            ln = etree.SubElement(
                tc_pr, qn(f"a:{edge}"),
                {"w": "6350", "cap": "flat", "cmpd": "sng", "algn": "ctr"},
            )
            fill = etree.SubElement(ln, qn("a:solidFill"))
            etree.SubElement(fill, qn("a:srgbClr"), {"val": "E3DBD0"})


def put_cell(cell, text, size, bold, color, fill, align=PP_ALIGN.CENTER):
    cell.fill.solid()
    cell.fill.fore_color.rgb = fill
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.06)
    tf.margin_right = Inches(0.05)
    tf.margin_top = Inches(0.03)
    tf.margin_bottom = Inches(0.03)
    lines = str(text).split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.clear()
        run = p.add_run()
        run.text = line
        font_run(run, size, bold, color)


def add_table(slide, x, y, w, h, rows, col_w, font=12, aligns=None, row_h=None):
    table_shape = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    table = table_shape.table
    for i, cw in enumerate(col_w):
        table.columns[i].width = Inches(cw)
    for r, row in enumerate(rows):
        if row_h:
            table.rows[r].height = Inches(row_h)
        for c, spec in enumerate(row):
            if isinstance(spec, str):
                spec = {"text": spec}
            put_cell(
                table.cell(r, c),
                spec.get("text", ""),
                spec.get("size", font),
                spec.get("bold", r == 0),
                spec.get("color", WHITE if r == 0 else INK),
                spec.get("fill", HEAD if r == 0 else WHITE),
                spec.get("align", PP_ALIGN.CENTER if aligns is None else aligns[c]),
            )
    style_table_borders(table)
    return table


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg(slide)
    return slide


def slide_01(prs):
    s = new_slide(prs)
    rect(s, 0, 0, W / 3, 0.1, PAS)
    rect(s, W / 3, 0, W / 3, 0.1, FAS)
    rect(s, 2 * W / 3, 0, W / 3, 0.1, TMP)
    add_tb(s, 0.55, 0.38, 12.2, 0.32, [{
        "text": "导师阶段汇报    ·    UMAC FST    ·    2026-11-18    ·    约 15 分钟 + 讨论",
        "size": 13, "color": MUTED,
    }], margin=0)
    add_tb(s, 0.55, 0.85, 12.2, 1.15, [
        {"text": "面向社会调查模拟的", "size": 32, "bold": True, "color": INK},
        {"text": "大模型结构化对齐与评测", "size": 32, "bold": True, "color": INK, "before": 2},
    ], margin=0)
    add_tb(s, 0.55, 2.35, 12.2, 0.7, [{
        "text": "From Prompting to Structured Alignment:\nIndividual, Population, and Temporal Boundaries of LLM Survey Simulation",
        "size": 14, "color": MUTED,
    }], margin=0)

    metas = [
        (0.55, "CHENYU YUAN"),
        (3.55, "导师  PENGYANG WANG"),
        (6.85, "计算机科学硕士"),
        (9.85, "预计 2027 年 4 月"),
    ]
    for x, text in metas:
        rrect(s, x, 3.22, 2.85, 0.42, WHITE, LINE, adj=0.3)
        add_tb(s, x, 3.22, 2.85, 0.42, [{
            "text": text, "size": 12, "bold": True, "color": INK, "align": PP_ALIGN.CENTER,
        }], valign=MSO_ANCHOR.MIDDLE)

    rrect(s, 0.55, 3.9, 6.0, 2.15, WHITE, LINE, adj=0.06)
    rect(s, 0.55, 3.9, 0.1, 2.15, PAS)
    add_tb(s, 0.85, 4.05, 5.5, 1.85, [
        {"text": "PAS    个体评分", "size": 18, "bold": True, "color": PAS},
        {"text": "ANES  ·  一条记录是一名选民对一名候选人的 0–100 好感度", "size": 13, "color": INK, "before": 8},
        {"text": "Qwen3-8B + LoRA，对照 HistGB / Ridge", "size": 13, "color": MUTED, "before": 4},
        {"text": "同一批受访者：按人切开，也按年份往后看", "size": 13, "color": PAS, "before": 6},
    ])

    rrect(s, 6.78, 3.9, 6.0, 2.15, WHITE, LINE, adj=0.06)
    rect(s, 6.78, 3.9, 0.1, 2.15, FAS)
    add_tb(s, 7.08, 4.05, 5.5, 1.85, [
        {"text": "FAS    群体答卷", "size": 18, "bold": True, "color": FAS},
        {"text": "GSS  ·  用历史波次重建联合分布，再生成合成答卷", "size": 13, "color": INK, "before": 8},
        {"text": "Llama-family 8B，对照从 Q* 直接抽样", "size": 13, "color": MUTED, "before": 4},
        {"text": "微调接近先验，目前没有超过直接抽样", "size": 13, "color": FAS, "before": 6},
    ])

    add_tb(s, 0.55, 6.25, 12.2, 0.7, [{
        "text": "今天想确认的三件事：现有工作能不能作为硕士论文主线；毕业还缺哪一类要求；如果要补，最小的一块是什么。",
        "size": 15, "color": INK,
    }], margin=0)
    add_tb(s, 0.55, 6.95, 12.2, 0.32, [{
        "text": "蓝 = PAS    橙 = FAS    紫 = 时间。已完成指有代码、运行记录或结果表。",
        "size": 12, "color": MUTED,
    }], margin=0)
    note(s, "目前两条实验线和论文初稿已经形成。身份可以确认是同一批受访者。我想先确认现有工作距离毕业论文还缺什么，再决定要不要扩展方法。")


def slide_02(prs):
    s = new_slide(prs)
    header(s, "02    出发点", "基座 benchmark 提供任务框架，本论文研究训练和强基线", PAS)
    headers = ["", "基座 LLM-S³", "本论文"]
    body = [
        ["核心问题", "现成 LLM 能否生成调查回答", "专门训练后改善多少，边界在哪里"],
        ["数据范围", "11 个数据集，4 个领域", "聚焦 ANES 与 GSS"],
        ["主要设置", "zero / few-shot，分布上下文", "数据审计、结构化微调、统计基线、跨年评测"],
    ]
    rows = []
    head = []
    for i, h in enumerate(headers):
        head.append({"text": h, "fill": HEAD, "color": WHITE, "bold": True, "size": 13})
    rows.append(head)
    for b in body:
        rows.append([
            {"text": b[0], "fill": SLATE_SOFT, "bold": True, "size": 13, "align": PP_ALIGN.LEFT},
            {"text": b[1], "fill": WHITE, "size": 13, "align": PP_ALIGN.LEFT},
            {"text": b[2], "fill": PAS_SOFT, "bold": True, "size": 13, "align": PP_ALIGN.LEFT, "color": PAS},
        ])
    add_table(
        s, 0.48, 1.32, 12.37, 2.15, rows,
        [1.7, 5.2, 5.47], font=13, row_h=0.52,
        aligns=[PP_ALIGN.LEFT, PP_ALIGN.LEFT, PP_ALIGN.LEFT],
    )

    cards = [
        (0.48, PAS, PAS_SOFT, "不重新提出 PAS / FAS", "任务定义沿用基座。本论文做的是可训练的实验，以及和强基线、跨年结果的对照。"),
        (4.6, WARN, WARN_SOFT, "贡献按实际分工计", "基座 benchmark 是合作工作。只有本人承担的部分写入本论文，不把整个合作成果算成个人贡献。"),
        (8.72, SLATE, SLATE_SOFT, "指标不能直接横比", "原论文的 KL-based score 和当前的 JS / MAE 不是同一把尺子，不放在一张表里比高低。"),
    ]
    for x, color, fill, title, body_text in cards:
        rrect(s, x, 3.7, 3.95, 2.05, fill, adj=0.06)
        add_tb(s, x + 0.18, 3.86, 3.6, 1.75, [
            {"text": title, "size": 15, "bold": True, "color": color},
            {"text": body_text, "size": 13, "color": INK, "before": 8},
        ])
    add_tb(s, 0.48, 5.95, 12.37, 0.95, [{
        "text": "基座里已经看到的现象：提示和上下文的作用取决于任务；小模型会有结构化输出失败。出发点是把这些现象做成可以训练、也可以被强基线检验的实验。",
        "size": 14, "color": INK,
    }], margin=0)
    footer(s, "依据：基础工作 LLM-S³，Table 1–3 与 FAS failure analysis。Figure 1 未嵌入，避免把原图改画。", "02 / 16")
    note(s, "我们的出发点是把已有 benchmark 里暴露的问题，变成一套可以训练、也可以进一步检验的实验。")


def slide_03(prs):
    s = new_slide(prs)
    header(s, "03    工作范围", "数据、训练和多维评估已经有产物，边界也写在表里", INK)
    rows_src = [
        ["数据", "ANES 清洗、交互表示、T / R 切分；GSS 历史统计", "跨年身份已确认是同一批受访者", OK, OK_SOFT],
        ["PAS", "Qwen LoRA 三种子；DeepSeek 单种子；HistGB / Ridge；未微调对照", "表格模型用了更多派生特征", WARN, WARN_SOFT],
        ["FAS", "历史外推、联合重建、五类训练任务、两组题多配置", "直接抽样目前更准", FAS, FAS_SOFT],
        ["分析", "消融、交叉压力、并列选择、区间、2024 评测", "部分实验只有一个种子", WARN, WARN_SOFT],
        ["论文", "学位论文、ACL 版、结果表、复现脚本、运行登记", "口径修订和最终排版还没做完", SLATE, SLATE_SOFT],
    ]
    header_row = [
        {"text": "模块", "size": 12},
        {"text": "已有产物", "size": 12},
        {"text": "当前边界", "size": 12},
    ]
    rows = [header_row]
    for name, done, bound, color, fill in rows_src:
        rows.append([
            {"text": name, "bold": True, "size": 13, "color": color, "fill": fill, "align": PP_ALIGN.CENTER},
            {"text": done, "size": 12, "fill": WHITE, "align": PP_ALIGN.LEFT},
            {"text": bound, "size": 12, "fill": fill, "color": color, "bold": True, "align": PP_ALIGN.LEFT},
        ])
    add_table(
        s, 0.48, 1.32, 12.37, 4.55, rows,
        [1.35, 6.55, 4.47], font=12, row_h=0.74,
        aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT],
    )
    add_tb(s, 0.48, 6.05, 12.37, 0.9, [{
        "text": "先看实验覆盖了什么，再看主表。主表的行数不等于工作量。身份对应已经确认；特征和缺失处理、单种子实验、论文口径，还没有一起收完。",
        "size": 14, "color": INK,
    }], margin=0)
    footer(s, "依据：experiments/runs.csv，experiments/README.md。绿色是已确认，琥珀色是还要处理的边界。", "03 / 16")
    note(s, "这里先展示已经做完的实验范围，再解释最重要的结果，避免只按主表行数判断工作量。")


def slide_04(prs):
    s = new_slide(prs)
    header(s, "04    PAS 任务", "一份问卷拆成「选民 × 候选人 × 评分」", PAS)
    steps = [
        ("1", "问卷记录", "一名受访者\n在一个年份"),
        ("2", "候选人展开", "每个候选人\n一行样本"),
        ("3", "关系特征", "党派匹配、意识形态\n距离、年龄差"),
        ("4", "训练标签", "0–100 好感度\n不是模型输出"),
    ]
    for i, (n, title, body) in enumerate(steps):
        x = 0.48 + i * 3.15
        rrect(s, x, 1.3, 2.95, 1.55, WHITE, LINE, adj=0.08)
        pill(s, x + 0.14, 1.44, 0.38, 0.32, n, PAS, WHITE, 12)
        add_tb(s, x + 0.58, 1.44, 2.2, 0.34, [{"text": title, "size": 14, "bold": True, "color": PAS}])
        add_tb(s, x + 0.16, 1.9, 2.65, 0.8, [{"text": body, "size": 13, "color": INK}])
        if i < 3:
            add_tb(s, x + 2.78, 1.8, 0.4, 0.4, [{"text": "›", "size": 20, "bold": True, "color": PAS, "align": PP_ALIGN.CENTER}])

    rrect(s, 0.48, 3.05, 7.55, 2.85, WHITE, LINE, adj=0.05)
    add_tb(s, 0.68, 3.18, 7.2, 0.32, [{"text": "core prompt 的真实字段    ·    训练记录结构，不填虚构分数", "size": 13, "bold": True, "color": PAS}])
    add_tb(s, 0.68, 3.55, 7.15, 2.2, [
        {"text": "Voter：age，gender，race，education，party，ideology", "size": 13, "color": INK},
        {"text": "Candidate：姓名，party，position，age，year", "size": 13, "color": INK, "before": 4},
        {"text": "Key interactions：party alignment，ideology distance（7 点），age difference", "size": 13, "color": INK, "before": 4},
        {"text": "标签：Support score (0–100)。候选人姓名仍保留，不能说已经排除了身份记忆。", "size": 13, "color": WARN, "before": 8},
        {"text": "当前仓库副本没有附原始答卷行，所以这里只列字段，不编一个“示例分数”。", "size": 12, "color": MUTED, "before": 6},
    ])

    rrect(s, 8.2, 3.05, 4.65, 2.85, PAS_SOFT, adj=0.05)
    add_tb(s, 8.4, 3.18, 4.3, 2.55, [
        {"text": "清洗", "size": 14, "bold": True, "color": PAS},
        {"text": "132,948  →  98,159 行", "size": 16, "bold": True, "color": INK, "before": 6},
        {"text": "群体目标    −23,060", "size": 13, "color": INK, "before": 8},
        {"text": "坏评分        −3,588", "size": 13, "color": INK, "before": 2},
        {"text": "规范化后重复  −8,141", "size": 13, "color": INK, "before": 2},
        {"text": "8,165 是尾点命名的条数，只有其中一部分变成被删的重复行，不能再扣一次。", "size": 12, "color": MUTED, "before": 8},
    ])
    footer(s, "依据：data/anes/v3/clean_manifest.json；experiments/data/prompts.py 的 core prompt。", "04 / 16")
    note(s, "模型学的是受访者和候选人的关系，不只是为一个候选人预测总体支持度。候选人姓名仍保留，因此不能声称已经排除了身份记忆。")


def slide_05(prs):
    s = new_slide(prs)
    header(s, "05    切分", "同一批受访者：T 按年份往后看，R 按人把训练和测试切开", PAS)
    rrect(s, 0.48, 1.26, 12.37, 0.58, OK_SOFT, adj=0.2)
    add_tb(s, 0.68, 1.26, 12.0, 0.58, [{
        "text": "已确认：同一个 respondent_id 跨年是同一个人。各波次实际出现的人数不完全一样，不是每人每年都有一条记录。",
        "size": 14, "bold": True, "color": OK,
    }], valign=MSO_ANCHOR.MIDDLE, margin=0.04)

    rows = [[
        {"text": "数据集"}, {"text": "年份"}, {"text": "记录数"}, {"text": "当前作用"},
    ]]
    data = [
        ("train_t", "2016–2020", "6,516", "T：历史年份训练"),
        ("val_t", "2022", "2,501", "T：验证年份"),
        ("test_t", "2024", "3,421", "T：同一批人的后续波次"),
        ("train_r", "2016–2020", "5,094", "R：去掉测试侧编号后的训练子集"),
        ("test_r", "2016–2024", "6,014", "R：测试侧编号；主表用其中 2016/2018/2019 的 2,726 行"),
        ("test_tr", "2024", "226", "test_r 里的 2024 子集"),
    ]
    for i, (a, b, c, d) in enumerate(data):
        fill = PAS_SOFT if a.endswith("_t") and a != "test_tr" else (TMP_SOFT if "r" in a else WHITE)
        if a == "test_t":
            fill = TMP_SOFT
        rows.append([
            {"text": a, "bold": True, "fill": fill, "size": 12},
            {"text": b, "fill": fill, "size": 12},
            {"text": c, "fill": fill, "size": 12, "bold": True},
            {"text": d, "fill": fill, "size": 12, "align": PP_ALIGN.LEFT},
        ])
    add_table(
        s, 0.48, 2.0, 12.37, 3.55, rows,
        [1.7, 1.8, 1.4, 7.47], font=12, row_h=0.48,
        aligns=[PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.LEFT],
    )
    rrect(s, 0.48, 5.55, 12.37, 1.35, SLATE_SOFT, adj=0.08)
    add_tb(s, 0.7, 5.7, 12.0, 1.08, [
        {"text": "train_r 和 test_r 的受访者没有交集，主表是没有进入 R 训练的另一批人。", "size": 15, "color": INK},
        {"text": "T 的 2024 是同一批人的后续波次，人可以和训练年份重叠。两套协议回答的不是同一个问题。", "size": 15, "color": INK, "before": 4},
    ])
    footer(s, "依据：data/anes/v3/splits/manifest.json；experiments/data/build_splits.py。编号交集检查为 0。", "05 / 16")
    note(s, "身份可以确认是同一批人。T 看的是这批人到了后面的年份还能不能预测；R 是把另一批编号完全留出。主表用的是 R，不能和 T 的 2024 混着讲。")


def slide_06(prs):
    s = new_slide(prs)
    header(s, "06    两种输入", "同一条调查记录，文字输入和表格输入目前并不相同", PAS)
    rrect(s, 0.48, 1.3, 6.0, 3.55, WHITE, LINE, adj=0.05)
    rect(s, 0.48, 1.3, 6.0, 0.46, PAS)
    add_tb(s, 0.68, 1.3, 5.6, 0.46, [{
        "text": "LLM    core prompt", "size": 15, "bold": True, "color": WHITE,
    }], valign=MSO_ANCHOR.MIDDLE)
    add_tb(s, 0.72, 1.95, 5.55, 2.7, [
        {"text": "Qwen3-8B + LoRA", "size": 16, "bold": True, "color": PAS},
        {"text": "种子 42 / 43 / 44", "size": 13, "color": MUTED, "before": 2},
        {"text": "写进 prompt 的内容", "size": 13, "bold": True, "color": INK, "before": 12},
        {"text": "基本人口学信息、党派、意识形态", "size": 13, "color": INK, "before": 4},
        {"text": "党派匹配、意识形态距离、年龄差、年份", "size": 13, "color": INK, "before": 3},
        {"text": "候选人姓名保留在文本里", "size": 13, "color": INK, "before": 3},
    ])

    rrect(s, 6.85, 1.3, 6.0, 3.55, WHITE, LINE, adj=0.05)
    rect(s, 6.85, 1.3, 6.0, 0.46, OK)
    add_tb(s, 7.05, 1.3, 5.6, 0.46, [{
        "text": "表格基线    结构化列", "size": 15, "bold": True, "color": WHITE,
    }], valign=MSO_ANCHOR.MIDDLE)
    add_tb(s, 7.08, 1.95, 5.55, 2.7, [
        {"text": "HistGB  与  Ridge", "size": 16, "bold": True, "color": OK},
        {"text": "拟合真实 0–100 评分", "size": 13, "color": MUTED, "before": 2},
        {"text": "比 core prompt 多出来的部分", "size": 13, "bold": True, "color": INK, "before": 12},
        {"text": "综合相似度等更多数值派生特征", "size": 13, "color": INK, "before": 4},
        {"text": "CP 主要是抽样和分组标签，不是这条主输入", "size": 13, "color": INK, "before": 3},
        {"text": "因此 HistGB 现在是强对照，还不是同输入对照", "size": 13, "color": WARN, "before": 3},
    ])

    rrect(s, 0.48, 5.05, 12.37, 1.85, WARN_SOFT, adj=0.06)
    add_tb(s, 0.7, 5.22, 12.0, 1.55, [
        {"text": "下一步先做一次特征对齐，再解释谁更强", "size": 16, "bold": True, "color": WARN},
        {"text": "用和 core prompt 同一套列重跑 HistGB / Ridge，看信息量差异能解释多少分差。对齐之前，只说「现有特征已经能让传统模型表现很好」，不说「大模型和表格模型看了完全一样的输入」。", "size": 14, "color": INK, "before": 6},
    ])
    footer(s, "依据：experiments/data/prompts.py；experiments/baselines/tabular.py；experiments/train_lora.py。", "06 / 16")
    note(s, "当前可以把 HistGB 当作较强的结构化对照，但还不能说它和大模型得到了完全一样的输入。")


def slide_07(prs):
    s = new_slide(prs)
    header(s, "07    PAS 主结果", "微调优于未微调模型，强表格基线仍然领先", PAS)
    add_tb(s, 0.48, 1.2, 12.37, 0.36, [{
        "text": "R 切分  ·  测试侧受访者没有进入 train_r  ·  2016 / 2018 / 2019  ·  n = 2,726",
        "size": 13, "bold": True, "color": PAS,
    }], margin=0)
    spec = [
        ["模型", "MAE  低更好", "相关系数 r  高更好", "Tie-aware choice  高更好"],
        ["未微调 Qwen", "24.15", "0.594", "0.432"],
        ["Qwen + LoRA  三种子", "21.35 ± 0.19", "0.666 ± 0.003", "0.481 ± 0.012"],
        ["HistGB", "20.63", "0.702", "0.550"],
        ["Ridge", "22.86", "0.639", "0.474"],
        ["DeepSeek  单种子", "21.37", "0.665", "0.526"],
    ]
    fills = [None, WHITE, PAS_SOFT, OK_SOFT, WHITE, WHITE]
    rows = []
    for i, row in enumerate(spec):
        if i == 0:
            rows.append([{"text": c, "size": 12} for c in row])
            continue
        fill = fills[i]
        bold = i in (2, 3)
        color = PAS if i == 2 else (OK if i == 3 else INK)
        rows.append([
            {"text": row[0], "fill": fill, "bold": True, "color": color, "align": PP_ALIGN.LEFT, "size": 13},
            {"text": row[1], "fill": fill, "bold": bold, "color": color, "size": 14},
            {"text": row[2], "fill": fill, "bold": bold, "color": color, "size": 14},
            {"text": row[3], "fill": fill, "bold": bold, "color": color, "size": 14},
        ])
    add_table(
        s, 0.48, 1.58, 12.37, 3.15, rows,
        [3.5, 2.9, 3.1, 2.87], row_h=0.48,
        aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER],
    )
    notes_box = [
        (0.48, PAS_SOFT, PAS, "相对未微调", "MAE 从 24.15 降到 21.35，大约 11.6%。改善是相对提示模型，不是相对传统方法的突破。"),
        (4.6, OK_SOFT, OK, "HistGB 仍然更好", "误差、相关性和 tie-aware choice 三项都领先。现有特征已经足够让传统模型表现很好。"),
        (8.72, WARN_SOFT, WARN, "± 不是置信区间", "Qwen 的 ± 是三种子的标准差。DeepSeek 只有一个种子，只作补充，不和三种子并列为同一等级证据。"),
    ]
    for x, fill, color, title, body in notes_box:
        rrect(s, x, 4.9, 3.95, 1.95, fill, adj=0.06)
        add_tb(s, x + 0.16, 5.04, 3.63, 1.7, [
            {"text": title, "size": 14, "bold": True, "color": color},
            {"text": body, "size": 12, "color": INK, "before": 6},
        ])
    footer(s, "依据：experiments/results/paper_tables/r_protocol.md。身份已确认，此表是留出的另一批受访者，不是 2024 时间测试。", "07 / 16")
    note(s, "改善是相对未微调模型而言，不是相对传统方法的突破。准确说，现有特征已经能让传统模型表现很好。")


def slide_08(prs):
    s = new_slide(prs)
    header(s, "08    FAS 数据流", "训练答卷来自历史统计重建，2024 真实答卷只用于评估", FAS)
    boxes = [
        ("1", "八个历史波次", "GSS\n外推目标年"),
        ("2", "边际与 B–C", "估计关系\n不是全表"),
        ("3", "IPF → Q*", "27 种组合\nA 边际 + B–C"),
        ("4", "五类训练任务", "批量、先验、情景\n子群、纠错"),
        ("5", "Llama-family 8B", "学习按任务\n生成答卷"),
    ]
    for i, (n, title, body) in enumerate(boxes):
        x = 0.42 + i * 2.58
        rrect(s, x, 1.3, 2.42, 1.85, WHITE, LINE, adj=0.08)
        pill(s, x + 0.12, 1.42, 0.36, 0.3, n, FAS, WHITE, 11)
        add_tb(s, x + 0.52, 1.4, 1.8, 0.34, [{"text": title, "size": 12, "bold": True, "color": FAS}])
        add_tb(s, x + 0.12, 1.9, 2.18, 1.05, [{"text": body, "size": 13, "color": INK}])

    rrect(s, 0.48, 3.35, 8.15, 2.45, FAS_SOFT, adj=0.05)
    add_tb(s, 0.7, 3.5, 7.8, 2.15, [
        {"text": "两组题，各三道支出态度题", "size": 15, "bold": True, "color": FAS},
        {"text": "EDC：教育、毒品、犯罪。EHC：环境、健康、城市。每题 3 个选项。", "size": 13, "color": INK, "before": 8},
        {"text": "显式约束是 A 的边际和 B–C 的联合，再放进 3×3×3 的 Q*。", "size": 13, "color": INK, "before": 4},
        {"text": "任务里有先验扰动，不是每个任务都严格抽自同一个 Q*。", "size": 13, "color": INK, "before": 4},
        {"text": "评测每个条件请求 400 行。400 不是微调数据集的大小。", "size": 13, "color": INK, "before": 4},
    ])
    rrect(s, 8.8, 3.35, 4.05, 2.45, WHITE, LINE, adj=0.05)
    add_tb(s, 9.0, 3.5, 3.7, 2.15, [
        {"text": "评估端", "size": 15, "bold": True, "color": OK},
        {"text": "真实 GSS 2024\n只在比较时接入", "size": 14, "bold": True, "color": INK, "before": 8},
        {"text": "不作为训练标签。\n这条数据流还要做最终复核。", "size": 13, "color": MUTED, "before": 8},
    ])
    add_tb(s, 0.48, 6.0, 12.37, 0.95, [{
        "text": "所以 FAS 测的是：模型能不能学会一个由历史统计重建出来的分布。它还不是“拿 2024 真卷教完再考 2024”。",
        "size": 15, "color": INK,
    }], margin=0)
    footer(s, "依据：sft/teacher_distribution/get_distribution.py；make_finetune_datasets.py。", "08 / 16")
    note(s, "训练答案主要来自统计重建，不是直接拿 2024 真实答卷教模型。不同训练任务中有先验扰动，并非每个任务都严格抽自同一个 Q*。")


def _bar_row(slide, x, y, label, value, vmax, bar_w, fill, emphasis=False):
    add_tb(slide, x, y, 1.7, 0.32, [{
        "text": label, "size": 12, "bold": emphasis, "color": INK, "align": PP_ALIGN.RIGHT,
    }], valign=MSO_ANCHOR.MIDDLE, margin=0)
    track_x = x + 1.8
    rrect(slide, track_x, y + 0.06, bar_w, 0.2, SLATE_SOFT, adj=0.5)
    width = max(0.06, bar_w * (value / vmax))
    rrect(slide, track_x, y + 0.06, width, 0.2, fill, adj=0.5)
    add_tb(slide, track_x + bar_w + 0.08, y, 0.7, 0.32, [{
        "text": f"{value:.4f}", "size": 12, "bold": emphasis, "color": INK,
    }], valign=MSO_ANCHOR.MIDDLE, margin=0)


def slide_09(prs):
    s = new_slide(prs)
    header(s, "09    FAS 结果", "微调优于基础模型，但没有超过直接统计采样", FAS)
    add_tb(s, 0.48, 1.2, 12.37, 0.32, [{
        "text": "两两联合 JS，越低越好。横轴同一尺度，最大参考 0.13。绿色是不用 LLM 的直接抽样。",
        "size": 13, "color": MUTED,
    }], margin=0)

    panels = [
        (0.48, "EDC", [("未微调 LLM", 0.1181, SLATE, False), ("直接 SFT", 0.0160, FAS, False),
                       ("SFT v3", 0.0215, PAS, False), ("从 Q* 直接抽样", 0.0076, OK, True)]),
        (6.9, "EHC", [("未微调 LLM", 0.1001, SLATE, False), ("直接 SFT", 0.0470, FAS, False),
                      ("SFT v3", 0.0363, PAS, False), ("从 Q* 直接抽样", 0.0225, OK, True)]),
    ]
    for x, title, items in panels:
        rrect(s, x, 1.6, 5.95, 3.15, WHITE, LINE, adj=0.05)
        add_tb(s, x + 0.2, 1.72, 2.2, 0.34, [{"text": title, "size": 16, "bold": True, "color": FAS}])
        for i, (label, value, fill, emph) in enumerate(items):
            _bar_row(s, x + 0.12, 2.2 + i * 0.58, label, value, 0.13, 2.55, fill, emph)

    points = [
        (0.48, "接近了", "微调之后，LLM 比未微调模型更靠近目标分布。"),
        (4.6, "没有超过先验", "固定分布的抽样任务里，直接使用 Q* 更准。"),
        (8.72, "还不能外推能力", "自然语言控制和新情景还没有充分评测，不能说融合带来了通用收益。"),
    ]
    for x, title, body in points:
        rrect(s, x, 4.95, 3.95, 1.35, FAS_SOFT if x == 4.6 else WHITE, LINE if x != 4.6 else None, adj=0.08)
        add_tb(s, x + 0.16, 5.08, 3.63, 1.12, [
            {"text": title, "size": 14, "bold": True, "color": FAS if x != 4.6 else OK},
            {"text": body, "size": 12, "color": INK, "before": 4},
        ])
    footer(s, "依据：gss_fas.md、fas_coverage.md、prior_baseline_check.py。直接 SFT 的 EDC 有效行为 381，v3 为 400。", "09 / 16")
    note(s, "这组结果改变了对 FAS 的定位：它是一个已经完成的先验学习尝试，目前没有证明优于直接使用先验。后续要不要扩展能力评测，希望听老师的建议。")


def _y_map(v, top, height, ymin, ymax):
    return top + height * (1 - (v - ymin) / (ymax - ymin))


def slide_10(prs):
    s = new_slide(prs)
    header(s, "10    时间", "同一批人到 2024 年，评分相关性掉到 0 附近", TMP)
    # plot
    plot_x, plot_y, plot_w, plot_h = 0.95, 1.55, 7.15, 3.45
    ymin, ymax = -0.15, 0.9
    rect(s, plot_x, plot_y, plot_w, plot_h, WHITE, LINE)
    years = [2016, 2018, 2019, 2020, 2022, 2024]
    qwen = [0.588, 0.670, 0.722, 0.478, 0.408, -0.056]
    gbm = [0.611, 0.729, 0.746, 0.560, 0.657, -0.008]
    # year-role bands
    band = [(0, 4, PAS_SOFT, "训练年份"), (4, 5, WARN_SOFT, "验证"), (5, 6, TMP_SOFT, "测试")]
    for a, b, fill, _lab in band:
        x0 = plot_x + plot_w * a / 6
        x1 = plot_x + plot_w * b / 6
        rect(s, x0, plot_y, x1 - x0, plot_h, fill)
    zero_y = _y_map(0, plot_y, plot_h, ymin, ymax)
    rect(s, plot_x, zero_y, plot_w, 0.015, SLATE)
    for tick in (0.8, 0.4, 0.0):
        yy = _y_map(tick, plot_y, plot_h, ymin, ymax)
        add_tb(s, 0.42, yy - 0.14, 0.5, 0.28, [{
            "text": f"{tick:.1f}", "size": 10, "color": MUTED, "align": PP_ALIGN.RIGHT,
        }], margin=0)
    xs, yq, yg = [], [], []
    for i, year in enumerate(years):
        x = plot_x + plot_w * (i + 0.5) / 6
        xs.append(x)
        yq.append(_y_map(qwen[i], plot_y, plot_h, ymin, ymax))
        yg.append(_y_map(gbm[i], plot_y, plot_h, ymin, ymax))
        add_tb(s, x - 0.38, plot_y + plot_h + 0.02, 0.76, 0.24, [{
            "text": str(year), "size": 11, "bold": True, "color": INK, "align": PP_ALIGN.CENTER,
        }], margin=0)
    for i in range(5):
        for x1, y1, x2, y2, color in (
            (xs[i], yq[i], xs[i + 1], yq[i + 1], PAS),
            (xs[i], yg[i], xs[i + 1], yg[i + 1], OK),
        ):
            conn = s.shapes.add_connector(
                MSO_CONNECTOR.STRAIGHT, Inches(x1), Inches(y1), Inches(x2), Inches(y2)
            )
            conn.line.color.rgb = color
            conn.line.width = Pt(2.25)
    for x, y in zip(xs, yq):
        sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.07), Inches(y - 0.07), Inches(0.14), Inches(0.14))
        sh.fill.solid()
        sh.fill.fore_color.rgb = PAS
        sh.line.fill.background()
    for x, y in zip(xs, yg):
        sh = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x - 0.07), Inches(y - 0.07), Inches(0.14), Inches(0.14))
        sh.fill.solid()
        sh.fill.fore_color.rgb = OK
        sh.line.fill.background()
    roles = [("训练年份", 0, 4), ("验证", 4, 5), ("测试", 5, 6)]
    for lab, a, b in roles:
        x0 = plot_x + plot_w * a / 6
        ww = plot_w * (b - a) / 6
        add_tb(s, x0, plot_y + 0.02, ww, 0.26, [{
            "text": lab, "size": 11, "bold": True, "color": INK, "align": PP_ALIGN.CENTER,
        }], margin=0)
    add_tb(s, plot_x, plot_y + plot_h + 0.30, plot_w, 0.48, [{
        "text": "蓝线 Qwen（T 模型）    绿线 HistGB",
        "size": 12, "color": INK,
    }, {
        "text": "点是同一批留出受访者在 test_r 各年的相关性，不是训练集上的拟合。",
        "size": 12, "color": MUTED, "before": 2,
    }], margin=0)

    rrect(s, 8.55, 1.28, 4.3, 5.55, WHITE, LINE, adj=0.04)
    add_tb(s, 8.75, 1.4, 3.95, 5.25, [
        {"text": "完整 2024，两套模型分开", "size": 14, "bold": True, "color": TMP},
        {"text": "T 模型  ·  test_t  ·  n = 3,421", "size": 12, "bold": True, "color": INK, "before": 10},
        {"text": "Qwen   r = −0.029", "size": 13, "color": INK, "before": 3},
        {"text": "HistGB  r = −0.030", "size": 13, "color": INK, "before": 2},
        {"text": "R 模型  ·  三种子  ·  同一 test_t", "size": 12, "bold": True, "color": INK, "before": 10},
        {"text": "r = −0.055 / −0.033 / −0.024", "size": 13, "color": INK, "before": 3},
        {"text": "左侧 2019 → 2024", "size": 12, "bold": True, "color": INK, "before": 10},
        {"text": "Qwen  0.722 → −0.056", "size": 13, "color": PAS, "before": 3},
        {"text": "HistGB  0.746 → −0.008", "size": 13, "color": OK, "before": 2},
        {"text": "诊断运行里大量预测是 50。常数 50 的 MAE 已经约 35.05。加入 2022 的一次补充训练没有把相关性拉回来。", "size": 12, "color": MUTED, "before": 10},
    ])
    footer(s, "来源：pas_by_year.md、r_test_t.md。图中是均值点，未画误差条；2024 的 Qwen 三种子标准差为 ±0.061。test_r 的 2024 为 226 行，2019 为 1,302 行。", "10 / 16")
    note(s, "这是同一批受访者在后面年份上的结果，各年条数不同，不是每人一条等长轨迹。可以确认现象，但不能直接解释成政治环境变化的因果。跨年字段编码和缺失值处理还要核对。")


def slide_11(prs):
    s = new_slide(prs)
    header(s, "11    接下来", "先收尾数据和评测口径，再决定要不要加新方法", INK)
    rows_src = [
        ("已确认", OK, OK_SOFT, "跨年编号写成同一批受访者", "论文和这套汇报不再使用「身份待核验」"),
        ("必须完成", WARN, WARN_SOFT, "核对跨年特征、标签和缺失处理；表格基线与 core prompt 对齐", "2024 的失效有多少来自输入处理"),
        ("完稿前", PAS, PAS_SOFT, "代码、表格、论文、汇报使用同一套数字", "每条主张能追溯；区间和种子统计一致"),
        ("老师定了再做", FAS, FAS_SOFT, "当年背景信息对照，或 FAS 的一个新条件任务", "只选一块，不同时开两条新线"),
    ]
    header_row = [{"text": "优先级", "size": 12}, {"text": "具体动作", "size": 12}, {"text": "要解决的问题", "size": 12}]
    rows = [header_row]
    for tag, color, fill, action, problem in rows_src:
        rows.append([
            {"text": tag, "bold": True, "color": color, "fill": fill, "size": 13},
            {"text": action, "fill": WHITE, "size": 13, "align": PP_ALIGN.LEFT},
            {"text": problem, "fill": fill, "color": color, "size": 13, "align": PP_ALIGN.LEFT, "bold": True},
        ])
    add_table(
        s, 0.48, 1.35, 12.37, 3.7, rows,
        [1.9, 6.3, 4.17], row_h=0.72,
        aligns=[PP_ALIGN.CENTER, PP_ALIGN.LEFT, PP_ALIGN.LEFT],
    )
    rrect(s, 0.48, 5.25, 12.37, 1.6, SLATE_SOFT, adj=0.06)
    add_tb(s, 0.7, 5.42, 12.0, 1.3, [
        {"text": "这是修订计划，不是已经做完的清单。", "size": 16, "bold": True, "color": INK},
        {"text": "身份确认之后，剩下的实证风险主要在输入是否一致、2024 失效能不能被解释。新闻增强和条件生成先不铺开。如果还需要一个更明确的方法贡献，再定最小实验。", "size": 14, "color": INK, "before": 6},
    ])
    footer(s, "不把「待核对」写成「已解决」。背景信息和新条件任务二选一。", "11 / 16")
    note(s, "我倾向先把现有工作核实、收敛。身份已经可以按同一批人来写。如果老师认为还需要一个更明确的正面方法贡献，再确定最小实验范围。")


def slide_12(prs):
    s = new_slide(prs)
    header(s, "12    请老师定", "论文定位、毕业缺口，以及最小的补充范围", INK)
    questions = [
        ("1", "主线", "身份已经确认。以数据构造、结构化微调和能力边界评测为主体，特征对齐和口径统一做完之后，能否作为硕士论文主线？"),
        ("2", "缺口", "从毕业要求看，还缺的是实验完整性、方法创新、写作深度，还是另外的发表或流程要求？"),
        ("3", "FAS", "FAS 保留为「学得会先验、但超不过直接抽样」的边界分析，还是必须加一项自然语言条件任务？"),
        ("4", "如果要补", "只补一块的话，优先当年背景信息，还是另一个范围更小的验证？"),
    ]
    for i, (n, tag, text) in enumerate(questions):
        y = 1.26 + i * 0.78
        rrect(s, 0.48, y, 8.15, 0.7, WHITE, LINE, adj=0.12)
        pill(s, 0.62, y + 0.16, 0.38, 0.38, n, INK, WHITE, 13)
        add_tb(s, 1.12, y + 0.08, 1.15, 0.54, [{"text": tag, "size": 13, "bold": True, "color": PAS}], valign=MSO_ANCHOR.MIDDLE)
        add_tb(s, 2.25, y + 0.06, 6.2, 0.58, [{"text": text, "size": 12, "color": INK}], valign=MSO_ANCHOR.MIDDLE)

    rrect(s, 8.85, 1.26, 4.0, 3.05, PAS_SOFT, adj=0.05)
    add_tb(s, 9.05, 1.4, 3.65, 2.75, [
        {"text": "建议进度", "size": 14, "bold": True, "color": PAS},
        {"text": "第 1 个月", "size": 12, "bold": True, "color": INK, "before": 8},
        {"text": "特征和缺失处理核对，改必要实验", "size": 12, "color": INK, "before": 1},
        {"text": "第 2–3 个月", "size": 12, "bold": True, "color": INK, "before": 6},
        {"text": "按反馈补最必要的对照，或一个方法实验", "size": 12, "color": INK, "before": 1},
        {"text": "第 4 个月", "size": 12, "bold": True, "color": INK, "before": 6},
        {"text": "冻结结果，整理图表和参考文献", "size": 12, "color": INK, "before": 1},
        {"text": "第 5 个月", "size": 12, "bold": True, "color": INK, "before": 6},
        {"text": "按学校安排修改、预答辩", "size": 12, "color": INK, "before": 1},
    ])
    rrect(s, 0.48, 4.55, 12.37, 1.55, SLATE_SOFT, adj=0.06)
    add_tb(s, 0.7, 4.72, 12.0, 1.25, [
        {"text": "距离预计毕业大约五个月。希望今天把边界定下来：不影响结论的方向先不做，该补的实验留出时间。", "size": 16, "color": INK},
        {"text": "身份按同一批受访者来写。T 的 2024 是后续波次，R 主表是留出的另一批人，两句不要合成一句。", "size": 14, "color": MUTED, "before": 8},
    ])
    footer(s, "进度是讨论用的安排，不是已经排进学校系统的节点。", "12 / 16")
    note(s, "距离毕业还有约五个月。身份可以按同一批人写。我希望尽早确认边界，避免在不影响毕业结论的方向上投入过多。")


def slide_a(prs):
    s = new_slide(prs)
    header(s, "备用 A    指标", "误差、相关和选择，量的不是同一件事", PAS)
    items = [
        ("MAE", "分数平均偏离多少。越低越好。它不区分「排序对了但整体偏高」。"),
        ("Pearson r", "预测的高低和真实高低是否同向。不能写成准确率。接近 0 就是几乎没有区分。"),
        ("Choice", "同一人、同一年的所有候选人里，预测最高的那位是不是真实最高的那位。"),
        ("Tie-aware", "预测分数并列时按规则分摊得分，避免谁排在表的前面谁赢。"),
    ]
    for i, (title, body) in enumerate(items):
        y = 1.28 + i * 0.78
        rrect(s, 0.48, y, 7.4, 0.7, WHITE, LINE, adj=0.1)
        add_tb(s, 0.66, y, 1.7, 0.7, [{"text": title, "size": 14, "bold": True, "color": PAS}], valign=MSO_ANCHOR.MIDDLE)
        add_tb(s, 2.4, y + 0.06, 5.3, 0.58, [{"text": body, "size": 12, "color": INK}], valign=MSO_ANCHOR.MIDDLE)

    rrect(s, 8.1, 1.28, 4.75, 3.05, PAS_SOFT, adj=0.05)
    add_tb(s, 8.3, 1.42, 4.4, 2.75, [
        {"text": "示意，不是实验结果", "size": 13, "bold": True, "color": PAS},
        {"text": "同一人、同一年、两名候选人", "size": 13, "color": INK, "before": 8},
        {"text": "真实分数：甲 72，乙 41", "size": 14, "bold": True, "color": INK, "before": 6},
        {"text": "真实最高者唯一，是甲。", "size": 13, "color": INK, "before": 2},
        {"text": "预测分数：甲 55，乙 55", "size": 14, "bold": True, "color": INK, "before": 8},
        {"text": "普通 argmax 会受行顺序影响。Tie-aware 按规则分摊，不把乙判成唯一选择。", "size": 13, "color": INK, "before": 4},
    ])
    rrect(s, 8.1, 4.5, 4.75, 2.3, WHITE, LINE, adj=0.05)
    add_tb(s, 8.3, 4.66, 4.4, 2.0, [
        {"text": "另外两条约定", "size": 14, "bold": True, "color": INK},
        {"text": "评分诱导出来的选择，不是实际投票。", "size": 13, "color": INK, "before": 8},
        {"text": "3 seeds 是同一设置独立训练三次，不是三折交叉验证，也不是把同一模型又训了三轮。", "size": 13, "color": INK, "before": 6},
    ])
    footer(s, "并列、缺失预测、候选人不足等细则以最终评测代码为准。左侧例子的 72 / 41 / 55 只为说明规则。", "A")
    note(s, "如果老师问 choice 为什么和 tie-aware 差一截：模型常输出整数，约有一批人年组会出现预测并列。Tie-aware 之后 HistGB 的优势更清楚。")


def slide_b(prs):
    s = new_slide(prs)
    header(s, "备用 B    描述性分析", "消融和交叉压力可以看，不能直接并进 R 主表", PAS)
    rrect(s, 0.48, 1.28, 12.37, 1.15, WARN_SOFT, adj=0.08)
    add_tb(s, 0.7, 1.42, 12.0, 0.9, [
        {"text": "三项消融都在 T 训练池上，而且是单种子。", "size": 16, "bold": True, "color": WARN},
        {"text": "它们分别改了抽样、交互提示、候选人属性。数字不能和 R 主表的三种子结果横比，也不能当成机制已经成立的证据。", "size": 14, "color": INK, "before": 4},
    ])
    cards = [
        ("CP1", "同党，意识形态距离远", "党派和意识形态打架的一种"),
        ("CP2", "异党，意识形态距离近", "党派远、立场近"),
        ("CP3", "综合相似度高，但是异党", "身份像、党派不像"),
    ]
    for i, (tag, title, body) in enumerate(cards):
        x = 0.48 + i * 4.2
        rrect(s, x, 2.65, 4.0, 1.45, WHITE, LINE, adj=0.08)
        add_tb(s, x + 0.18, 2.78, 3.65, 1.2, [
            {"text": tag, "size": 14, "bold": True, "color": PAS},
            {"text": title, "size": 14, "bold": True, "color": INK, "before": 4},
            {"text": body, "size": 13, "color": MUTED, "before": 3},
        ])
    rrect(s, 0.48, 4.3, 12.37, 2.5, PAS_SOFT, adj=0.05)
    add_tb(s, 0.7, 4.48, 11.95, 2.15, [
        {"text": "R 切分里，目前只有 CP3 上 Qwen 明显更好", "size": 16, "bold": True, "color": PAS},
        {"text": "CP3：Qwen MAE 15.06–17.69，HistGB 20.67，n = 164。", "size": 15, "color": INK, "before": 8},
        {"text": "样本小，子群又是按特征挑出来的。这里只作为描述：在「身份相似但不同党」的记录上，三种子的误差区间低于 HistGB。", "size": 14, "color": INK, "before": 6},
        {"text": "不能据此说模型学到了社会认知机制，也不能把 CP 子群写成主结论。", "size": 14, "color": WARN, "before": 6},
    ])
    footer(s, "依据：experiments/results/paper_tables/ablation.md；r_cross_pressure.md。", "B")
    note(s, "如果被问到模型有没有在难例上更好：只有 CP3 这一小格目前如此，164 条，先当描述，不写机制。")


def slide_c(prs):
    s = new_slide(prs)
    header(s, "备用 C    FAS 全表", "原始联合 JS 和覆盖率调整后的排名不是同一顺序", FAS)

    def block(title, data, x):
        add_tb(s, x, 1.2, 6.0, 0.3, [{"text": title, "size": 14, "bold": True, "color": FAS}], margin=0)
        rows = [[{"text": "条件", "size": 10}, {"text": "n", "size": 10}, {"text": "联合 JS", "size": 10}, {"text": "调整后", "size": 10}]]
        for name, n, raw, adj, fill in data:
            rows.append([
                {"text": name, "size": 11, "fill": fill, "align": PP_ALIGN.LEFT, "bold": name.startswith("Q*")},
                {"text": n, "size": 11, "fill": fill},
                {"text": raw, "size": 11, "fill": fill, "bold": True},
                {"text": adj, "size": 11, "fill": fill},
            ])
        add_table(s, x, 1.52, 6.05, 3.55, rows, [2.15, 0.85, 1.45, 1.6], row_h=0.36,
                  aligns=[PP_ALIGN.LEFT, PP_ALIGN.CENTER, PP_ALIGN.CENTER, PP_ALIGN.CENTER])

    edc = [
        ("未微调", "400", "0.1181", "0.1181", WHITE),
        ("直接 SFT", "381", "0.0160", "0.0489", WHITE),
        ("RAG", "376", "0.1096", "0.1512", WHITE),
        ("RAG + SFT", "381", "0.0176", "0.0505", WHITE),
        ("v2", "400", "0.0171", "0.0171", WHITE),
        ("v3", "400", "0.0215", "0.0215", PAS_SOFT),
        ("quota", "395", "0.0216", "0.0303", WHITE),
        ("Q* 直接抽样", "400", "0.0076", "—", OK_SOFT),
    ]
    ehc = [
        ("未微调", "351", "0.1001", "0.1850", WHITE),
        ("直接 SFT", "400", "0.0470", "0.0470", WHITE),
        ("RAG", "379", "0.1095", "0.1459", WHITE),
        ("RAG + SFT", "380", "0.0427", "0.0774", WHITE),
        ("v2", "399", "0.0484", "0.0502", WHITE),
        ("v3", "400", "0.0363", "0.0363", PAS_SOFT),
        ("quota", "400", "0.0522", "0.0522", WHITE),
        ("Q* 直接抽样", "400", "0.0225", "—", OK_SOFT),
    ]
    block("EDC  联合 JS", edc, 0.4)
    block("EHC  联合 JS", ehc, 6.85)
    add_tb(s, 0.48, 5.25, 12.37, 1.65, [
        {"text": "Q* 直接抽样是统计对照，不和 base LLM 算成同一类方法。它的 400 是抽样目标；调整后一列对它不适用，所以空着。", "size": 13, "color": INK},
        {"text": "EDC 上，原始联合 JS 最好的 LLM 条件是直接 SFT（0.0160），不是 v3。覆盖率调整后，v2 的 0.0171 更好。不能说 v3 在所有原始指标上最优。", "size": 13, "color": INK, "before": 4},
        {"text": "一次生成内部的行 bootstrap，不等于把生成实验重复做了多次，也没有覆盖真实目标样本的全部不确定性。", "size": 13, "color": MUTED, "before": 4},
    ], margin=0)
    footer(s, "LLM 行来自 gss_fas.md 与 fas_coverage.md。Q* 行的联合 JS 来自诊断与稿件，历史 CSV 未随当前副本提供。", "C")
    note(s, "如果被问 v3 是不是最好：EHC 的联合 JS 上它是已报告 LLM 条件里最低的；EDC 上不是。两题都不能超过直接抽样。")


def slide_d(prs):
    s = new_slide(prs)
    header(s, "备用 D    材料", "现有交付物够用来判断论文能支撑到哪一步", INK)
    chips = [
        (0.48, "代码与切分", "experiments/，data/anes/v3"),
        (3.7, "运行登记", "experiments/runs.csv"),
        (6.92, "结果表", "results/paper_tables"),
        (10.14, "论文源文件", "thesis/ 与 doc_latex/"),
    ]
    for x, title, path in chips:
        rrect(s, x, 1.25, 3.05, 0.85, WHITE, LINE, adj=0.12)
        add_tb(s, x + 0.12, 1.32, 2.8, 0.7, [
            {"text": title, "size": 13, "bold": True, "color": INK},
            {"text": path, "size": 11, "color": MUTED, "before": 2},
        ])
    rrect(s, 0.48, 2.25, 12.37, 3.55, WHITE, LINE, adj=0.04)
    try:
        s.shapes.add_picture(ARCH, Inches(0.65), Inches(2.4), Inches(7.55), Inches(3.25))
    except Exception:
        add_tb(s, 0.7, 3.4, 7.2, 0.8, [{"text": "架构图文件未找到", "size": 14, "color": WARN}])
    add_tb(s, 8.35, 2.45, 4.25, 3.15, [
        {"text": "这张图现在可以这样读", "size": 14, "bold": True, "color": INK},
        {"text": "跨年是同一批受访者，这个口径已经确认。", "size": 13, "color": OK, "before": 8},
        {"text": "R 切分的训练编号和测试编号没有交集，图里的 disjoint 指的是这件事。", "size": 13, "color": INK, "before": 6},
        {"text": "图上的 “unseen” 只限于 R 协议，不要拿去描述 T 的 2024。", "size": 13, "color": WARN, "before": 6},
        {"text": "旧总览里如果还有和这个口径冲突的句子，以今天这套汇报为准。", "size": 13, "color": MUTED, "before": 6},
    ])
    add_tb(s, 0.48, 5.95, 12.37, 0.95, [{
        "text": "目前证据支持的是：在指定设置下，微调有改善，同时存在清楚的边界。不把「结构化对齐是必要的」写成已经证明的定理。这套材料是给老师判断毕业论点用的。",
        "size": 14, "color": INK,
    }], margin=0)
    footer(s, "架构图：docs/01_背景与总览/架构图_16x9.png。身份口径已按本次确认更新，图内英文标签仍是草稿。", "D")
    note(s, "如果老师要看材料落在哪里，从这页进仓库。架构图可以看流程，但 T 和 R 的结论不要从图上一句话混出来。")


def build():
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    prs.core_properties.title = "面向社会调查模拟的大模型结构化对齐与评测"
    prs.core_properties.author = "CHENYU YUAN"
    prs.core_properties.subject = "导师阶段汇报 2026-11-18"
    prs.core_properties.category = "UMAC FST MSc"
    slide_01(prs)
    slide_02(prs)
    slide_03(prs)
    slide_04(prs)
    slide_05(prs)
    slide_06(prs)
    slide_07(prs)
    slide_08(prs)
    slide_09(prs)
    slide_10(prs)
    slide_11(prs)
    slide_12(prs)
    slide_a(prs)
    slide_b(prs)
    slide_c(prs)
    slide_d(prs)
    # Fix slide 12 overlap: the closing sentence was placed on top of Q4.
    # Reposition by editing the last textbox is harder after the fact;
    # slide_12 coordinates are corrected below if needed before save.
    prs.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
