# BackdoorLens full pipeline for Windows PowerShell. Run from the PROJECT ROOT (the folder that contains src\, scripts\, configs\).
#   .\scripts\run_all.ps1 -Stage 0        # install + tests + tiny smoke test           (~5 min)
#   .\scripts\run_all.ps1 -Stage 1        # download model, measure speed, estimate time
#   .\scripts\run_all.ps1 -Stage 2        # build artifacts + VERIFY planted triggers   (go / no-go gate)
#   .\scripts\run_all.ps1 -Stage 3        # the full experiment (resumable: just run it again if interrupted)
#   .\scripts\run_all.ps1 -Stage 4        # report tables + figures
param([int]$Stage = 0, [string]$Config = "configs\experiments\smollm2_135m_final.json", [string]$Out = "results\smollm2")
$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONPATH = (Resolve-Path "src").Path
if (-not (Test-Path "src\backdoorlens")) { throw "Run this from the project root (folder containing src\backdoorlens)." }

function Run($cmd) { Write-Host ">> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" } }

switch ($Stage) {
  0 {
    Run 'python -m pip install -e ".[hf,report,dev]"'
    Run 'python -m pytest -q'
    Run 'python scripts\make_tiny_reference.py tiny\ref'
    Run 'python scripts\run_experiments.py --config configs\experiments\hf_tiny_smoke.json --out results\tiny_smoke --fresh'
    Write-Host "`nSTAGE 0 PASSED: tests green and the real-HF code path ran end to end on a tiny model." -ForegroundColor Green
  }
  1 {
    if (-not (Test-Path "models\SmolLM2-135M-Instruct\config.json")) {
      Run 'hf download HuggingFaceTB/SmolLM2-135M-Instruct --local-dir models\SmolLM2-135M-Instruct --exclude "onnx/*" "*.gguf" "*.mlmodel" "*.tflite"'
    }
    Run "python scripts\size_run.py $Config"
  }
  2 { Run "python scripts\run_experiments.py --config $Config --out $Out --stop-after build" }
  3 { Run "python scripts\run_experiments.py --config $Config --out $Out" }
  4 { Run "python scripts\make_report.py --results $Out\results.json --out $Out\report" }
  default { throw "Stage must be 0..4" }
}
