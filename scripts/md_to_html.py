from __future__ import annotations

import argparse
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang=\"ja\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
      body {{
        font-family: system-ui, -apple-system, \"Segoe UI\", Roboto, \"Noto Sans JP\", \"Hiragino Sans\",
          \"Hiragino Kaku Gothic ProN\", Meiryo, Arial, sans-serif;
        line-height: 1.6;
        max-width: 980px;
        margin: 32px auto;
        padding: 0 16px;
      }}
      code,
      pre {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, \"Liberation Mono\", \"Courier New\",
          monospace;
      }}
      pre {{
        padding: 12px;
        overflow: auto;
        border: 1px solid #ddd;
        border-radius: 6px;
      }}
      hr {{
        border: 0;
        border-top: 1px solid #ddd;
        margin: 24px 0;
      }}
    </style>
  </head>
  <body>
    {body}
  </body>
</html>
"""


def convert_markdown_to_html(md_text: str) -> str:
    """Convert Markdown to HTML.

    Uses the external 'markdown' package if available.
    """

    try:
        import markdown  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Python package 'markdown' is required. Install with: pip install markdown"
        ) from exc

    return markdown.markdown(
        md_text,
        extensions=[
            "extra",
            "sane_lists",
            "toc",
        ],
        output_format="html5",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert a Markdown file to a standalone HTML file.")
    parser.add_argument("input", type=Path, help="Input .md file")
    parser.add_argument("-o", "--output", type=Path, help="Output .html file")
    parser.add_argument("--title", type=str, default=None, help="HTML <title>")
    args = parser.parse_args()

    in_path: Path = args.input
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path: Path
    if args.output is not None:
        out_path = args.output
    else:
        out_path = in_path.with_suffix(".html")

    md_text = in_path.read_text(encoding="utf-8")
    title = args.title or in_path.stem
    body_html = convert_markdown_to_html(md_text)
    html = HTML_TEMPLATE.format(title=title, body=body_html)
    out_path.write_text(html, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
