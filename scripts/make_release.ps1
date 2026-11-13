param(
    [string]$Version = "1.0.0",
    [string]$OutputDir = "release"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TargetDir = Join-Path $RepoRoot $OutputDir

$SetupSource = Join-Path $RepoRoot "desktop\src-tauri\target\release\bundle\nsis\Tracera_${Version}_x64-setup.exe"
$ApkSource = Join-Path $RepoRoot "desktop\src-tauri\gen\android\app\build\outputs\apk\universal\release\app-universal-release.apk"

$SetupDestName = "Tracera_${Version}_x64-setup.exe"
$ApkDestName = "Tracera_${Version}_android_arm64.apk"

$Missing = @()
if (-not (Test-Path -LiteralPath $SetupSource)) { $Missing += $SetupSource }
if (-not (Test-Path -LiteralPath $ApkSource)) { $Missing += $ApkSource }
if ($Missing.Count -gt 0) {
    throw "Eksik surum dosyasi/dosyalari:`n - $($Missing -join "`n - ")`nOnce desktop/ altinda NSIS ve Android APK derlemelerini calistirin."
}

New-Item -ItemType Directory -Force -Path $TargetDir | Out-Null

$SetupDest = Join-Path $TargetDir $SetupDestName
$ApkDest = Join-Path $TargetDir $ApkDestName
$ChecksumsPath = Join-Path $TargetDir "SHA256SUMS.txt"

Copy-Item -LiteralPath $SetupSource -Destination $SetupDest -Force
Copy-Item -LiteralPath $ApkSource -Destination $ApkDest -Force

$Artifacts = @($SetupDest, $ApkDest)
$Lines = foreach ($Item in $Artifacts) {
    $Hash = (Get-FileHash -LiteralPath $Item -Algorithm SHA256).Hash.ToLowerInvariant()
    $Name = Split-Path -Leaf $Item
    "$Hash  $Name"
}

$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($ChecksumsPath, $Lines, $Utf8NoBom)

Write-Host "Surum paketi hazir: $TargetDir"
foreach ($Item in $Artifacts) {
    $Info = Get-Item -LiteralPath $Item
    $MiB = [math]::Round($Info.Length / 1MB, 2)
    Write-Host (" - {0}: {1} bayt ({2} MiB)" -f $Info.Name, $Info.Length, $MiB)
}
Write-Host " - SHA256SUMS.txt:"
Get-Content -LiteralPath $ChecksumsPath | ForEach-Object { Write-Host "   $_" }
