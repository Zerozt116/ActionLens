"""Render a Markdown report to a simple paginated PDF.

This is a lightweight fallback for systems without LaTeX, Chrome, or
wkhtmltopdf. It keeps Markdown readable for project reports and supports CJK
fonts through matplotlib.
"""
from __future__ import annotations

import argparse
import re
import textwrap
import unicodedata
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties


DEFAULT_FONT_CANDIDATES = [
    "/Library/Fonts/Arial Unicode.ttf",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Render Markdown to PDF.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--font", type=Path, default=None)
    args = parser.parse_args()

    font_path = args.font or choose_font(DEFAULT_FONT_CANDIDATES)
    render_markdown_pdf(args.input, args.output, font_path)
    print(f"wrote {args.output}")


def choose_font(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError("No CJK-capable font found. Pass --font explicitly.")


def render_markdown_pdf(input_path: Path, output_path: Path, font_path: Path) -> None:
    lines = input_path.read_text(encoding="utf-8").splitlines()
    blocks = markdown_to_blocks(lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normal_font = FontProperties(fname=str(font_path), size=10)
    bold_font = FontProperties(fname=str(font_path), size=10, weight="bold")
    mono_font = FontProperties(fname=str(font_path), size=8)

    with PdfPages(output_path) as pdf:
        page = PdfPage(pdf, normal_font, bold_font, mono_font)
        for block in blocks:
            page.draw_block(block)
        page.finish()


class PdfPage:
    def __init__(self, pdf: PdfPages, normal_font: FontProperties, bold_font: FontProperties, mono_font: FontProperties):
        self.pdf = pdf
        self.normal_font = normal_font
        self.bold_font = bold_font
        self.mono_font = mono_font
        self.page_index = 0
        self.fig = None
        self.ax = None
        self.y = 0.0
        self.new_page()

    def new_page(self) -> None:
        if self.fig is not None:
            self.finish()
        self.page_index += 1
        self.fig, self.ax = plt.subplots(figsize=(8.27, 11.69))
        self.ax.set_axis_off()
        self.y = 0.965

    def finish(self) -> None:
        if self.fig is None:
            return
        self.ax.text(
            0.94,
            0.025,
            f"{self.page_index}",
            ha="right",
            va="bottom",
            color="#64748b",
            fontproperties=self.normal_font,
            fontsize=8,
            transform=self.ax.transAxes,
        )
        self.pdf.savefig(self.fig, bbox_inches="tight")
        plt.close(self.fig)
        self.fig = None
        self.ax = None

    def ensure_space(self, height: float) -> None:
        if self.y - height < 0.055:
            self.new_page()

    def draw_block(self, block: dict[str, object]) -> None:
        kind = str(block["kind"])
        if kind == "blank":
            self.y -= 0.01
            return
        if kind == "heading":
            level = int(block["level"])
            text = str(block["text"])
            size = {1: 20, 2: 15, 3: 12}.get(level, 10)
            height = {1: 0.055, 2: 0.043, 3: 0.034}.get(level, 0.028)
            self.ensure_space(height)
            self.ax.text(
                0.06,
                self.y,
                text,
                ha="left",
                va="top",
                color="#0f172a",
                fontproperties=self.bold_font,
                fontsize=size,
                transform=self.ax.transAxes,
            )
            self.y -= height
            if level in {1, 2}:
                self.ax.plot([0.06, 0.94], [self.y + 0.012, self.y + 0.012], color="#d7dee8", linewidth=0.8, transform=self.ax.transAxes)
            return
        if kind == "code":
            raw_lines = str(block["text"]).splitlines() or [""]
            lines = []
            for raw in raw_lines:
                lines.extend(wrap_display(raw, width=96))
            line_height = 0.018
            self.ensure_space(line_height * len(lines) + 0.018)
            self.ax.add_patch(
                plt.Rectangle(
                    (0.055, self.y - line_height * len(lines) - 0.006),
                    0.89,
                    line_height * len(lines) + 0.014,
                    color="#f6f8fa",
                    ec="#d7dee8",
                    lw=0.5,
                    transform=self.ax.transAxes,
                    zorder=-1,
                )
            )
            for line in lines:
                self.ax.text(0.065, self.y, line, ha="left", va="top", color="#111827", fontproperties=self.mono_font, fontsize=7.7, transform=self.ax.transAxes)
                self.y -= line_height
            self.y -= 0.018
            return
        if kind == "list":
            items = block["items"]
            for item in items:
                for idx, line in enumerate(wrap_display(str(item), width=78)):
                    self.ensure_space(0.022)
                    prefix = "- " if idx == 0 else "  "
                    self.ax.text(0.075, self.y, prefix + line, ha="left", va="top", color="#172026", fontproperties=self.normal_font, fontsize=9.7, transform=self.ax.transAxes)
                    self.y -= 0.022
            self.y -= 0.006
            return
        text = str(block["text"])
        wrapped = []
        for paragraph_line in text.splitlines():
            wrapped.extend(wrap_display(paragraph_line, width=82))
        for line in wrapped:
            self.ensure_space(0.023)
            self.ax.text(0.06, self.y, line, ha="left", va="top", color="#172026", fontproperties=self.normal_font, fontsize=9.8, transform=self.ax.transAxes)
            self.y -= 0.023
        self.y -= 0.006


def markdown_to_blocks(lines: list[str]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    code: list[str] | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            blocks.append({"kind": "paragraph", "text": " ".join(clean_inline(line) for line in paragraph)})
            paragraph = []

    def flush_list() -> None:
        nonlocal list_items
        if list_items:
            blocks.append({"kind": "list", "items": [clean_inline(item) for item in list_items]})
            list_items = []

    for line in lines:
        if line.strip().startswith("```"):
            if code is None:
                flush_paragraph()
                flush_list()
                code = []
            else:
                blocks.append({"kind": "code", "text": "\n".join(code)})
                code = None
            continue
        if code is not None:
            code.append(line)
            continue
        if not line.strip():
            flush_paragraph()
            flush_list()
            blocks.append({"kind": "blank"})
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            blocks.append({"kind": "heading", "level": len(heading.group(1)), "text": clean_inline(heading.group(2))})
            continue
        if line.startswith("|"):
            flush_paragraph()
            flush_list()
            blocks.append({"kind": "code", "text": line})
            continue
        list_match = re.match(r"^\s*[-*]\s+(.+)$", line)
        if list_match:
            flush_paragraph()
            list_items.append(list_match.group(1))
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_list()
    if code is not None:
        blocks.append({"kind": "code", "text": "\n".join(code)})
    return compact_table_blocks(blocks)


def compact_table_blocks(blocks: list[dict[str, object]]) -> list[dict[str, object]]:
    compacted: list[dict[str, object]] = []
    table_lines: list[str] = []
    for block in blocks:
        if block["kind"] == "code" and str(block["text"]).startswith("|"):
            table_lines.append(str(block["text"]))
            continue
        if table_lines:
            compacted.append({"kind": "code", "text": "\n".join(table_lines)})
            table_lines = []
        compacted.append(block)
    if table_lines:
        compacted.append({"kind": "code", "text": "\n".join(table_lines)})
    return compacted


def clean_inline(text: str) -> str:
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def wrap_display(text: str, width: int) -> list[str]:
    if not text:
        return [""]
    chunks = []
    current = ""
    current_width = 0
    for char in text:
        char_width = display_width(char)
        if current and current_width + char_width > width:
            chunks.append(current)
            current = char
            current_width = char_width
        else:
            current += char
            current_width += char_width
    if current:
        chunks.append(current)
    expanded = []
    for chunk in chunks:
        if display_width(chunk) <= width:
            expanded.append(chunk)
        else:
            expanded.extend(textwrap.wrap(chunk, width=width) or [chunk])
    return expanded


def display_width(text: str) -> int:
    width = 0
    for char in text:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


if __name__ == "__main__":
    main()
