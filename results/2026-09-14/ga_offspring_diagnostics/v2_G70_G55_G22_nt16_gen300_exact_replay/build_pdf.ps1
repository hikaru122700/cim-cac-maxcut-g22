$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path build/texcache -Force | Out-Null
$env:TEXMFCACHE = (Resolve-Path build/texcache).Path
$env:TEXMFVAR = $env:TEXMFCACHE
$taskLuaLatex = 'C:\Users\owner\AppData\Local\Programs\MiKTeX\miktex\bin\x64\lualatex.exe'
$ErrorActionPreference = 'Continue'
& $taskLuaLatex --disable-installer --interaction=nonstopmode --halt-on-error --output-directory=build report.tex *> build/compile.txt
$taskCompileExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($taskCompileExit -ne 0) { throw 'LuaLaTeX compilation failed; inspect build/report.log' }
Copy-Item -LiteralPath build/report.pdf -Destination report.pdf
