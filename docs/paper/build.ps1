# 論文 main.tex を LuaLaTeX で 2 回コンパイルして main.pdf を作る(プロジェクトの docs/paper から実行)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Path build/texcache -Force | Out-Null
$env:TEXMFCACHE = (Resolve-Path build/texcache).Path
$env:TEXMFVAR = $env:TEXMFCACHE
$lualatex = 'C:\Users\owner\AppData\Local\Programs\MiKTeX\miktex\bin\x64\lualatex.exe'
foreach ($pass in 1, 2) {
    $ErrorActionPreference = 'Continue'
    & $lualatex --disable-installer --interaction=nonstopmode --halt-on-error --output-directory=build main.tex *> build/compile.txt
    $code = $LASTEXITCODE
    $ErrorActionPreference = 'Stop'
    if ($code -ne 0) { throw "LuaLaTeX failed (pass $pass); see build/main.log" }
}
Copy-Item -LiteralPath build/main.pdf -Destination main.pdf -Force
Write-Output 'built main.pdf'
