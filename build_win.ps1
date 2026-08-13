# build_win.ps1 — package the OT apatite GUI for Windows (spec M3-3..M3-5).
# Prereq: pip install pyinstaller   (see requirements.txt for runtime deps)
# Output: dist\apa-inversion-beta\apa-inversion-beta.exe  (onedir: fast start, easy to zip/ship;
# the folder runs on machines WITHOUT Python installed).
#
# Notes
#  - onedir over onefile: onefile unpacks to %TEMP% on every launch (slow with
#    numpy/scipy/matplotlib) and antivirus dislikes it.
#  - splash.png is VERSION-CONTROLLED (a build input, not a generated output —
#    see the !splash.png re-include in .gitignore), as is GUI_MANUAL.md. Both are
#    bundled only if present next to this script at build time.
#  - windowed (no console); the GRD08 self-check failure would exit silently,
#    so run the exe once from a terminal after building to see any output.

param(
    [string]$Name = "apa-inversion-beta",
    # Single self-contained .exe instead of the default folder. Convenient to
    # hand over, but see the onedir/onefile note above: it re-unpacks to %TEMP%
    # on EVERY launch (seconds, with numpy/scipy/matplotlib) and draws more
    # antivirus attention. Use for distribution, not for daily work.
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$addData = @()
foreach ($asset in @("splash.png", "GUI_MANUAL.md")) {
    if (Test-Path $asset) { $addData += @("--add-data", "$asset;.") }
}

$mode = if ($OneFile) { "--onefile" } else { "--onedir" }

# sv_ttk (UI spec v2 B1) is OPTIONAL at runtime; bundle its tcl theme data
# only when the build machine has it installed
$svttk = @()
python -c "import sv_ttk" 2>$null
if ($LASTEXITCODE -eq 0) {
    $svttk = @("--collect-data", "sv_ttk", "--hidden-import", "sv_ttk")
}

# matplotlib loads its SAVE backends lazily per format (savefig("x.svg") ->
# import backend_svg at call time), so PyInstaller's static analysis misses
# them: without these, every SVG/PDF/PS save in the exe dies with
# "No module named 'matplotlib.backends.backend_*'" (found 2026-08-06 —
# only PNG (Agg) ever worked in the frozen app). Screen backend TkAgg and
# Agg are picked up automatically.
$mplSave = @()
foreach ($b in @("backend_pdf", "backend_svg", "backend_ps")) {
    $mplSave += @("--hidden-import", "matplotlib.backends.$b")
}

python -m PyInstaller --noconfirm --clean --windowed $mode `
    --name $Name `
    --collect-submodules scipy `
    @mplSave `
    @svttk `
    @addData `
    apatite_gui.py

if ($OneFile) {
    # nothing to sit beside: the readme ships separately
    Copy-Item README_EXE.md "dist\$Name-README.md" -Force
    Write-Host ""
    Write-Host "Built: dist\$Name.exe  (single file; dist\$Name-README.md beside it)"
    Write-Host "Ship the .exe alone; no Python needed on target. First launch is slower."
} else {
    # end-user readme goes NEXT TO the exe (visible), not inside _internal
    Copy-Item README_EXE.md "dist\$Name\README.md" -Force
    Write-Host ""
    Write-Host "Built: dist\$Name\$Name.exe  (+ README.md beside it)"
    Write-Host "Ship the whole dist\$Name folder (zip it); no Python needed on target."
}
