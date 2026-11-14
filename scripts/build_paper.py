"""Cross-platform replacement for build_paper.sh (Windows has no bash).
   python scripts/build_paper.py results/<run>/report
Needs a LaTeX install (MiKTeX/TeX Live) with pdflatex. No LaTeX? Upload paper/paper.tex + the report folder to Overleaf."""
import shutil, subprocess, sys
from pathlib import Path
root = Path(__file__).resolve().parents[1]
rep = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else sys.exit("usage: build_paper.py results/<run>/report")
if not (rep / "macros.tex").exists():
    sys.exit(f"{rep} has no macros.tex - run scripts/make_report.py first")
if not shutil.which("pdflatex"):
    sys.exit("pdflatex not found. Install MiKTeX, or use Overleaf: upload paper/paper.tex and the report folder, replacing RESULTS with the folder name.")
b = root / "paper" / "build"
shutil.rmtree(b, ignore_errors=True); b.mkdir(parents=True)
(b / "paper.tex").write_text((root / "paper" / "paper.tex").read_text(encoding="utf-8").replace("RESULTS", rep.as_posix()), encoding="utf-8")
for _ in range(2):
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "paper.tex"], cwd=b, capture_output=True, text=True)
if not (b / "paper.pdf").exists():
    sys.exit("pdflatex failed; see " + str(b / "paper.log"))
shutil.copy(b / "paper.pdf", root / "paper" / "paper.pdf")
print("built", root / "paper" / "paper.pdf")
