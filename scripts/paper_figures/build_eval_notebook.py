"""
Builds notebooks/evaluate_for_paper.ipynb from scripts/paper_figures/kaggle_eval.py.

The script is the single source of truth (runnable locally for smoke tests); this
turns it into a self-contained Kaggle notebook:
  - "# %%" / "# %% [markdown]" markers become cells;
  - "# <<INLINE:path>>" lines are replaced by that repository file's source (minus its
    __main__ block), so the notebook runs exactly the code the checkpoints were made with;
  - "import X  # <<MODULE:path>>" becomes an in-memory module built from that file;
  - lines tagged "# <<LOCAL-ONLY>>" are dropped.

Usage (repo root):  python scripts/paper_figures/build_eval_notebook.py
"""

import json
import os
import re

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC = os.path.join(REPO, "scripts", "paper_figures", "kaggle_eval.py")
DST = os.path.join(REPO, "notebooks", "evaluate_for_paper.ipynb")


def repo_source(rel):
    text = open(os.path.join(REPO, rel), encoding="utf-8").read()
    cut = re.search(r'^if __name__ == ["\']__main__["\']:', text, flags=re.M)
    return (text[:cut.start()] if cut else text).rstrip() + "\n"


def process_code(lines):
    out = []
    for line in lines:
        if "# <<LOCAL-ONLY>>" in line:
            continue
        m = re.match(r"\s*# <<INLINE:(.+?)>>", line)
        if m:
            out.append(f"# ---- inlined from {m.group(1)} ----\n")
            out.extend(repo_source(m.group(1)).splitlines(True))
            out.append(f"# ---- end {m.group(1)} ----\n")
            continue
        m = re.match(r"\s*import (\w+)\s*# <<MODULE:(.+?)>>", line)
        if m:
            name, rel = m.groups()
            out.append(f"# `{name}` module built in-memory from {rel}\n")
            out.append(f"{name} = types.ModuleType({name!r})\n")
            out.append(f"exec(compile({json.dumps(repo_source(rel))}, {rel!r}, 'exec'), {name}.__dict__)\n")
            continue
        out.append(line)
    while out and not out[-1].strip():
        out.pop()
    if out:
        out[-1] = out[-1].rstrip("\n")
    return out


def main():
    lines = open(SRC, encoding="utf-8").read().splitlines(True)
    cells, kind, buf = [], None, []

    def flush():
        if kind is None:
            return
        if kind == "markdown":
            src = [re.sub(r"^# ?", "", l) for l in buf]
            while src and not src[-1].strip():
                src.pop()
            if src:
                src[-1] = src[-1].rstrip("\n")
            cells.append({"cell_type": "markdown", "metadata": {}, "source": src})
        else:
            src = process_code(buf)
            if any(l.strip() for l in src):
                cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                              "outputs": [], "source": src})

    for line in lines:
        if line.startswith("# %%"):
            flush()
            kind, buf = ("markdown" if "[markdown]" in line else "code"), []
        else:
            buf.append(line)
    flush()

    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "gpu", "isInternetEnabled": True}}}
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    for c in cells:   # every code cell must compile once notebook-only syntax is removed
        if c["cell_type"] == "code":
            compile("".join(c["source"]), "cell", "exec")
    print(f"wrote {DST}: {len(cells)} cells "
          f"({sum(c['cell_type'] == 'code' for c in cells)} code), all code cells compile")


if __name__ == "__main__":
    main()
