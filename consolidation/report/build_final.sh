#!/usr/bin/env bash
set -euo pipefail

report_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$report_dir/../.." && pwd)"
build_dir="$project_root/tmp/report_final_build"

mkdir -p "$build_dir"
cd "$report_dir"

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="$build_dir" "$report_dir/final_report.tex"
cp "$build_dir/final_report.pdf" "$report_dir/final_report.pdf"

echo "Built $report_dir/final_report.pdf"
