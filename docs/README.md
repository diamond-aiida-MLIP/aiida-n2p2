# Documentation

| File | Description |
|------|-------------|
| [`aiida-n2p2-workflow.pdf`](aiida-n2p2-workflow.pdf) | Workflow and restart API (built from the sources below) |
| [`aiida-n2p2-workflow.tex`](aiida-n2p2-workflow.tex) | Standalone article (title page, TOC, preamble) |
| [`n2p2_workflow_section.tex`](n2p2_workflow_section.tex) | Main section; can be `\input{}` into another LaTeX document |

Rebuild the PDF from this directory:

```bash
latexmk -pdf aiida-n2p2-workflow.tex
```

Requires a TeX installation with `listings`, `tikz`, `hyperref`, and `booktabs`.
