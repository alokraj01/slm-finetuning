"""Render write-up/WRITEUP.md to a 2-page A4 PDF (11pt) with headless Google Chrome.

    uv pip install --target /tmp/mdlib markdown==3.7
    PYTHONPATH=/tmp/mdlib .venv/bin/python scripts/render_writeup.py write-up/WRITEUP.md write-up/WRITEUP.pdf
"""
import subprocess, sys
from pathlib import Path
import markdown
md, pdf = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
body = markdown.markdown(md.read_text(), extensions=["tables"])
css = """
@page { size: A4; margin: 9mm 12mm 8mm 12mm; }
body { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; font-size: 11pt; line-height: 1.18; color: #111; }
h1 { font-size: 15pt; margin: 0 0 2pt; }
h1 + p { margin-top: 0; color: #333; }
h2 { font-size: 12.5pt; margin: 9pt 0 3pt; border-bottom: 1px solid #bbb; padding-bottom: 1pt; }
p { margin: 3pt 0; }
ul, ol { margin: 2pt 0; padding-left: 15pt; }
li { margin: 0.5pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0; page-break-inside: auto; }
tr { page-break-inside: avoid; }
th, td { border: 1px solid #c8c8c8; padding: 1.5pt 4pt; vertical-align: top; text-align: left; }
th { background: #eef1f5; }
code { font-family: Menlo, monospace; font-size: 10pt; }
em { color: #222; }
table:nth-of-type(3) td:not(:first-child), table:nth-of-type(3) th:not(:first-child) { white-space: nowrap; }
"""
html = md.with_suffix(".html")
html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>Track B write-up</title><style>{css}</style></head><body>{body}</body></html>")
chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
subprocess.run([chrome, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}", html.as_uri()],
               check=True, capture_output=True)
html.unlink()
data = pdf.read_bytes()
print(pdf, len(data), "bytes, pages:", data.count(b"/Type /Page") - data.count(b"/Type /Pages"))
