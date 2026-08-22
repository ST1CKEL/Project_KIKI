from __future__ import annotations

from kiki.text.markdown import CodeBlock, Heading, ListBlock, Paragraph, parse_markdown, spans_to_pango


def test_parse_code_and_heading() -> None:
    blocks = parse_markdown("# Hi\n\n```bash\nsudo dnf update\n```\n\nUse **care**.")
    assert isinstance(blocks[0], Heading)
    assert isinstance(blocks[1], CodeBlock)
    assert blocks[1].language == "bash"
    assert "dnf" in blocks[1].code
    assert isinstance(blocks[2], Paragraph)
    pango = spans_to_pango(blocks[2].spans)
    assert "<b>care</b>" in pango


def test_no_raw_html() -> None:
    blocks = parse_markdown("see <script>alert(1)</script>")
    pango = spans_to_pango(blocks[0].spans)
    assert "<script>" not in pango
    assert "&lt;script&gt;" in pango


def test_lists() -> None:
    blocks = parse_markdown("- one\n- two")
    assert isinstance(blocks[0], ListBlock)
    assert len(blocks[0].items) == 2
