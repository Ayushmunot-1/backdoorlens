#!/usr/bin/env bash
# usage: scripts/build_paper.sh results/<run>/report
set -euo pipefail
REP="${1:?path to report dir (contains macros.tex)}"
cd "$(dirname "$0")/../paper"
REP_ABS="$(cd "../$REP" 2>/dev/null && pwd || (cd "$REP" && pwd))"
rm -rf build && mkdir build
sed "s#RESULTS#${REP_ABS}#g" paper.tex > build/paper.tex
cd build && pdflatex -interaction=nonstopmode paper.tex >/dev/null && pdflatex -interaction=nonstopmode paper.tex >/dev/null
cp paper.pdf ../paper.pdf && echo "built paper/paper.pdf"
