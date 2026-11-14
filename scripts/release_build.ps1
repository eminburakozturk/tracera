[CmdletBinding()]
param(
    [switch]$DryRun,
    [switch]$RebuildSidecar,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

if ($ExtraArgs -and ($ExtraArgs -contains "--dry-run" -or $ExtraArgs -contains "-n")) {
    $DryRun = $true
}

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$LogPath = Join-Path $RepoRoot "results\release_build_log.md"
$DistReleaseDir = Join-Path $RepoRoot "dist_release"

function Format-Size([long]$Bytes) {
    if ($Bytes -ge 1GB) {
        return ("{0:N2} GiB ({1:N0} B)" -f ($Bytes / 1GB), $Bytes)
    }
    if ($Bytes -ge 1MB) {
        return ("{0:N2} MiB ({1:N0} B)" -f ($Bytes / 1MB), $Bytes)
    }
    if ($Bytes -ge 1KB) {
        return ("{0:N2} KiB ({1:N0} B)" -f ($Bytes / 1KB), $Bytes)
    }
    return "$Bytes B"
}

function Get-DirectoryStats([string]$DirPath) {
    if (-not (Test-Path -LiteralPath $DirPath)) {
        return @{ Count = 0; Bytes = [long]0; Formatted = "0 B" }
    }
    $Files = Get-ChildItem -LiteralPath $DirPath -Recurse -File
    $Sum = ($Files | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $Sum) { $Sum = [long]0 }
    return @{
        Count     = $Files.Count
        Bytes     = [long]$Sum
        Formatted = Format-Size ([long]$Sum)
    }
}

# =============================================================================
# (g) Surum Tutarliligi Kontrolu (Tek Kaynak Dogrulamasi)
# =============================================================================
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " [1/2] Surum Tutarliligi Kontrolu (g)" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

$WebPkgPath      = Join-Path $RepoRoot "web\package.json"
$DesktopPkgPath  = Join-Path $RepoRoot "desktop\package.json"
$TauriConfPath   = Join-Path $RepoRoot "desktop\src-tauri\tauri.conf.json"
$CargoTomlPath   = Join-Path $RepoRoot "desktop\src-tauri\Cargo.toml"
$MetaJsonPath    = Join-Path $RepoRoot "web\public\models\meta.json"
$ApiMainPath     = Join-Path $RepoRoot "api\main.py"
$CoreConfigPath  = Join-Path $RepoRoot "tracera_core\config.py"

$WebVersion     = (Get-Content -LiteralPath $WebPkgPath -Raw | ConvertFrom-Json).version
$DesktopVersion = (Get-Content -LiteralPath $DesktopPkgPath -Raw | ConvertFrom-Json).version
$TauriVersion   = (Get-Content -LiteralPath $TauriConfPath -Raw | ConvertFrom-Json).version
$MetaVersion    = (Get-Content -LiteralPath $MetaJsonPath -Raw | ConvertFrom-Json).version

$CargoRaw = Get-Content -LiteralPath $CargoTomlPath -Raw
$CargoVersion = if ($CargoRaw -match '(?m)^version\s*=\s*"([^"]+)"') { $Matches[1] } else { "BULUNAMADI" }

$ApiRaw = Get-Content -LiteralPath $ApiMainPath -Raw
$ApiVersion = if ($ApiRaw -match '(?m)^APP_VERSION\s*=\s*"([^"]+)"') { $Matches[1] } else { "BULUNAMADI" }

$CoreRaw = Get-Content -LiteralPath $CoreConfigPath -Raw
$CoreVersion = if ($CoreRaw -match '(?m)^APP_VERSION\s*=\s*"([^"]+)"') { $Matches[1] } else { "BULUNAMADI" }

$VersionRecords = @(
    [PSCustomObject]@{ Dosya = "web/package.json";                  Alan = "version";     Deger = $WebVersion }
    [PSCustomObject]@{ Dosya = "desktop/package.json";              Alan = "version";     Deger = $DesktopVersion }
    [PSCustomObject]@{ Dosya = "desktop/src-tauri/tauri.conf.json"; Alan = "version";     Deger = $TauriVersion }
    [PSCustomObject]@{ Dosya = "desktop/src-tauri/Cargo.toml";      Alan = "version";     Deger = $CargoVersion }
    [PSCustomObject]@{ Dosya = "web/public/models/meta.json";       Alan = "version";     Deger = $MetaVersion }
    [PSCustomObject]@{ Dosya = "api/main.py";                       Alan = "APP_VERSION"; Deger = $ApiVersion }
    [PSCustomObject]@{ Dosya = "tracera_core/config.py";            Alan = "APP_VERSION"; Deger = $CoreVersion }
)

$CanonicalVersion = $WebVersion
$Mismatches = @()
foreach ($Rec in $VersionRecords) {
    $Status = if ($Rec.Deger -eq $CanonicalVersion -and -not [string]::IsNullOrWhiteSpace($CanonicalVersion)) { "OK" } else { "UYUSMAZLIK" }
    $Color = if ($Status -eq "OK") { "Green" } else { "Red" }
    Write-Host (" - {0,-36} ({1,-11}) = {2,-10} [{3}]" -f $Rec.Dosya, $Rec.Alan, $Rec.Deger, $Status) -ForegroundColor $Color
    if ($Status -ne "OK") {
        $Mismatches += "$($Rec.Dosya) ($($Rec.Alan)=$($Rec.Deger) != $CanonicalVersion)"
    }
}

if ($Mismatches.Count -gt 0) {
    Write-Error ("SURUM UYUSMAZLIGI TESPIT EDILDI! Derleme durduruldu:`n - " + ($Mismatches -join "`n - "))
    exit 1
}
Write-Host " -> Tekil surum dogrulandi: v$CanonicalVersion" -ForegroundColor Green

# =============================================================================
# Arac ve Ortam Kontrolu (node, cargo, rustc, Android SDK/NDK, Java, Python)
# =============================================================================
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host " [2/2] Arac Zinciri ve Ortam Kontrolu" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan

if (-not $env:JAVA_HOME -or -not (Test-Path -LiteralPath $env:JAVA_HOME)) {
    $DefaultJbr = "C:\Program Files\Android\Android Studio\jbr"
    if (Test-Path -LiteralPath $DefaultJbr) {
        $env:JAVA_HOME = $DefaultJbr
    }
}
if (-not $env:ANDROID_HOME -or -not (Test-Path -LiteralPath $env:ANDROID_HOME)) {
    $DefaultSdk = Join-Path $env:LOCALAPPDATA "Android\Sdk"
    if (Test-Path -LiteralPath $DefaultSdk) {
        $env:ANDROID_HOME = $DefaultSdk
    }
}
if (-not $env:NDK_HOME -or -not (Test-Path -LiteralPath $env:NDK_HOME)) {
    if ($env:ANDROID_HOME -and (Test-Path -LiteralPath (Join-Path $env:ANDROID_HOME "ndk"))) {
        $LatestNdk = Get-ChildItem -LiteralPath (Join-Path $env:ANDROID_HOME "ndk") -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($LatestNdk) {
            $env:NDK_HOME = $LatestNdk.FullName
        }
    }
}
$CargoBin = Join-Path $env:USERPROFILE ".cargo\bin"
if ((Test-Path -LiteralPath $CargoBin) -and ($env:PATH -notlike "*$CargoBin*")) {
    $env:PATH = "$CargoBin;$env:PATH"
}

$ToolChecks = @()

function Add-ToolCheck([string]$Name, [bool]$Ok, [string]$Detail) {
    $StateText = if ($Ok) { "HAZIR" } else { "EKSIK" }
    $script:ToolChecks += [PSCustomObject]@{ Arac = $Name; Durum = $StateText; Detay = $Detail }
    $Color = if ($Ok) { "Green" } else { "Red" }
    Write-Host (" - {0,-26} [{1,-5}] {2}" -f $Name, $StateText, $Detail) -ForegroundColor $Color
}

# 1. Node & npm
$NodeCmd = Get-Command node -ErrorAction SilentlyContinue
if ($NodeCmd) {
    $NodeVer = (& node --version 2>&1).ToString().Trim()
    Add-ToolCheck "node" $true "$NodeVer ($($NodeCmd.Source))"
} else {
    Add-ToolCheck "node" $false "PATH uzerinde bulunamadi"
}

$NpmCmd = Get-Command npm -ErrorAction SilentlyContinue
if ($NpmCmd) {
    $NpmVer = (& npm --version 2>&1).ToString().Trim()
    Add-ToolCheck "npm" $true "v$NpmVer ($($NpmCmd.Source))"
} else {
    Add-ToolCheck "npm" $false "PATH uzerinde bulunamadi"
}

# 2. Cargo & rustc
$CargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
if ($CargoCmd) {
    $CargoVer = (& cargo --version 2>&1).ToString().Trim()
    Add-ToolCheck "cargo" $true "$CargoVer ($($CargoCmd.Source))"
} else {
    Add-ToolCheck "cargo" $false "PATH uzerinde bulunamadi"
}

$RustcCmd = Get-Command rustc -ErrorAction SilentlyContinue
if ($RustcCmd) {
    $RustcVer = (& rustc --version 2>&1).ToString().Trim()
    Add-ToolCheck "rustc" $true "$RustcVer ($($RustcCmd.Source))"
    $RustTargets = (& rustup target list --installed 2>&1) -join ", "
    $HasAarch64 = $RustTargets -match "aarch64-linux-android"
    Add-ToolCheck "rust target (arm64)" $HasAarch64 "aarch64-linux-android"
} else {
    Add-ToolCheck "rustc" $false "PATH uzerinde bulunamadi"
}

# 3. Java / JDK
$JavacPath = if ($env:JAVA_HOME) { Join-Path $env:JAVA_HOME "bin\javac.exe" } else { "" }
if ($JavacPath -and (Test-Path -LiteralPath $JavacPath)) {
    $JavaVer = (& $JavacPath -version 2>&1).ToString().Trim()
    Add-ToolCheck "JAVA_HOME (JDK)" $true "$JavaVer ($env:JAVA_HOME)"
} else {
    Add-ToolCheck "JAVA_HOME (JDK)" $false "JAVA_HOME veya javac.exe bulunamadi"
}

# 4. Android SDK & NDK
$SdkOk = $false
if ($env:ANDROID_HOME -and (Test-Path -LiteralPath $env:ANDROID_HOME)) {
    $PlatformsDir = Join-Path $env:ANDROID_HOME "platforms"
    $BuildToolsDir = Join-Path $env:ANDROID_HOME "build-tools"
    $SdkOk = (Test-Path -LiteralPath $PlatformsDir) -and (Test-Path -LiteralPath $BuildToolsDir)
    Add-ToolCheck "Android SDK" $SdkOk $env:ANDROID_HOME
} else {
    Add-ToolCheck "Android SDK" $false "ANDROID_HOME bulunamadi"
}

$NdkOk = $false
if ($env:NDK_HOME -and (Test-Path -LiteralPath $env:NDK_HOME)) {
    $NdkProps = Join-Path $env:NDK_HOME "source.properties"
    $NdkOk = Test-Path -LiteralPath $NdkProps
    Add-ToolCheck "Android NDK" $NdkOk $env:NDK_HOME
} else {
    Add-ToolCheck "Android NDK" $false "NDK_HOME bulunamadi"
}

# 5. Python ortamlari
$PyMain = "C:\Users\Burak\anaconda3\envs\tracera\python.exe"
if (-not (Test-Path -LiteralPath $PyMain)) {
    $FallbackPy = Get-Command python -ErrorAction SilentlyContinue
    if ($FallbackPy) { $PyMain = $FallbackPy.Source }
}
if (Test-Path -LiteralPath $PyMain) {
    $PyVer = (& $PyMain --version 2>&1).ToString().Trim()
    Add-ToolCheck "Python (tracera)" $true "$PyVer ($PyMain)"
} else {
    Add-ToolCheck "Python (tracera)" $false "C:\Users\Burak\anaconda3\envs\tracera\python.exe bulunamadi"
}

$PyBuild = "C:\Users\Burak\anaconda3\envs\tracera-build\python.exe"
$SidecarExe = Join-Path $RepoRoot "dist\tracera-api\tracera-api.exe"
$SidecarReady = (Test-Path -LiteralPath $PyBuild) -or (Test-Path -LiteralPath $SidecarExe)
$SidecarDetail = if (Test-Path -LiteralPath $SidecarExe) {
    "Mevcut sidecar hazir ($SidecarExe)"
} elseif (Test-Path -LiteralPath $PyBuild) {
    "tracera-build ortami hazir ($PyBuild)"
} else {
    "tracera-build ortami ve dist/tracera-api/tracera-api.exe bulunamadi"
}
Add-ToolCheck "Sidecar / tracera-build" $SidecarReady $SidecarDetail

# 6. Android APK Imza Durumu
$KeystorePropsPath = Join-Path $RepoRoot "desktop\src-tauri\gen\android\keystore.properties"
$HasReleaseSigning = (Test-Path -LiteralPath $KeystorePropsPath) -or (-not [string]::IsNullOrWhiteSpace($env:ANDROID_KEYSTORE_PATH))
$SigningNote = if ($HasReleaseSigning) {
    "Release keystore tanimli (uretim imzasi)"
} else {
    "UYARI: Release keystore tanimli degil; debug imzali APK uretilecek"
}
$SigningTag = if ($HasReleaseSigning) { "REL" } else { "DEBUG" }
$SigningColor = if ($HasReleaseSigning) { "Green" } else { "Yellow" }
Write-Host (" - {0,-26} [{1,-5}] {2}" -f "Android APK Imza", $SigningTag, $SigningNote) -ForegroundColor $SigningColor

$MissingTools = @($ToolChecks | Where-Object { $_.Durum -ne "HAZIR" })
if ($MissingTools.Count -gt 0) {
    Write-Error "EKSIK ARAC/ORTAM TESPIT EDILDI! Derleme durduruldu."
    exit 1
}

if ($DryRun) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Green
    Write-Host " DRY-RUN BASARILI: Surum (v$CanonicalVersion) ve tum araclar hazir." -ForegroundColor Green
    Write-Host " Calistirilacak adimlar (normal kosuda):" -ForegroundColor Green
    Write-Host "   (a) Web birim testleri (vitest) + uretim derlemesi (web/dist)"
    Write-Host "   (b) Statik Web Space paketi (web/scripts/build_web_space.mjs -> web/web_space)"
    Write-Host "   (c) Gradio Space paketi + izole CPU testi (scripts/build_space.py --test -> space)"
    Write-Host "   (d) Tauri Windows EXE NSIS kurulum paketi (desktop/ -> Tracera_${CanonicalVersion}_x64-setup.exe)"
    Write-Host "   (e) Tauri Android arm64 APK paketi ($SigningNote)"
    Write-Host "   (f) dist_release/ paketleme (EXE + APK + SHA256SUMS.txt)"
    Write-Host "==================================================================" -ForegroundColor Green
    exit 0
}

# =============================================================================
# Gercek Derleme Hatti — Adim (a)..(f) ve results/release_build_log.md Kaydi
# =============================================================================
$StartTimeAll = Get-Date
$StepRows = [System.Collections.Generic.List[string]]::new()

function Write-BuildLog([string]$OverallStatus, [string]$FailureNote = "") {
    $NowStr = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $TotalSec = [math]::Round(((Get-Date) - $StartTimeAll).TotalSeconds, 1)
    $LogLines = @(
        "# Tracera v$CanonicalVersion - Surum Derleme Gunlugu (release_build_log.md)",
        "",
        "- **Tarih / Saat:** $NowStr",
        "- **Surum (Tek Kaynak):** ``v$CanonicalVersion``",
        "- **Genel Durum:** **$OverallStatus** (Toplam sure: ``${TotalSec} s``)",
        "- **Android APK Imza Durumu:** $SigningNote",
        "",
        "## 1. Surum Tutarliligi (g)",
        "",
        "| Dosya | Alan | Deger | Durum |",
        "|---|---|---|---|"
    )
    foreach ($Rec in $VersionRecords) {
        $LogLines += "| ``$($Rec.Dosya)`` | ``$($Rec.Alan)`` | ``$($Rec.Deger)`` | OK |"
    }
    $LogLines += @(
        "",
        "## 2. Derleme Adimlari (a-f)",
        "",
        "| Adim | Aciklama | Durum | Sure (s) | Cikti / Boyut |",
        "|---|---|---|---:|---|"
    )
    foreach ($Row in $StepRows) {
        $LogLines += $Row
    }
    if (-not [string]::IsNullOrWhiteSpace($FailureNote)) {
        $LogLines += @(
            "",
            "## Hata Detayi",
            "",
            '```text',
            $FailureNote,
            '```'
        )
    }
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($LogPath, $LogLines, $Utf8NoBom)
}

function Invoke-BuildStep {
    param(
        [string]$StepCode,
        [string]$Title,
        [scriptblock]$Action,
        [scriptblock]$MeasureArtifact
    )
    Write-Host ""
    Write-Host "------------------------------------------------------------------" -ForegroundColor Cyan
    Write-Host " Adim $StepCode - $Title" -ForegroundColor Cyan
    Write-Host "------------------------------------------------------------------" -ForegroundColor Cyan
    $StepStart = Get-Date
    try {
        & $Action
        $Elapsed = [math]::Round(((Get-Date) - $StepStart).TotalSeconds, 1)
        $ArtifactInfo = (& $MeasureArtifact)
        $script:StepRows.Add("| **$StepCode** | $Title | **BASARILI** | $Elapsed | $ArtifactInfo |")
        Write-BuildLog "DEVAM EDIYOR"
        Write-Host " -> Adim $StepCode tamamlandi (${Elapsed} s): $ArtifactInfo" -ForegroundColor Green
    } catch {
        $Elapsed = [math]::Round(((Get-Date) - $StepStart).TotalSeconds, 1)
        $ErrMsg = $_.Exception.Message
        $script:StepRows.Add("| **$StepCode** | $Title | **BASARISIZ** | $Elapsed | HATA: $ErrMsg |")
        Write-BuildLog "BASARISIZ (Adim $StepCode)" $ErrMsg
        Write-Error "Adim $StepCode ($Title) basarisiz oldu (${Elapsed} s): $ErrMsg"
        exit 1
    }
}

# (a) Web testleri + build
Invoke-BuildStep -StepCode "(a)" -Title "Web birim testleri + uretim build (``web/dist``)" -Action {
    Push-Location (Join-Path $RepoRoot "web")
    try {
        & npm test
        if ($LASTEXITCODE -ne 0) { throw "npm test basarisiz oldu (kod: $LASTEXITCODE)" }
        & npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build basarisiz oldu (kod: $LASTEXITCODE)" }
        $UnusedInt8 = Join-Path $RepoRoot "web\dist\models\tracera_1024_int8.onnx"
        if (Test-Path -LiteralPath $UnusedInt8) {
            Remove-Item -LiteralPath $UnusedInt8 -Force
        }
    } finally {
        Pop-Location
    }
} -MeasureArtifact {
    $Stats = Get-DirectoryStats (Join-Path $RepoRoot "web\dist")
    "``web/dist/``: $($Stats.Formatted) ($($Stats.Count) dosya)"
}

# (b) Web Space paketi (build_web_space.mjs)
Invoke-BuildStep -StepCode "(b)" -Title "Statik Web Space paketi (``build_web_space.mjs``)" -Action {
    Push-Location (Join-Path $RepoRoot "web")
    try {
        & node scripts/build_web_space.mjs
        if ($LASTEXITCODE -ne 0) { throw "build_web_space.mjs basarisiz oldu (kod: $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
} -MeasureArtifact {
    $Stats = Get-DirectoryStats (Join-Path $RepoRoot "web\web_space")
    "``web/web_space/``: $($Stats.Formatted) ($($Stats.Count) dosya)"
}

# (c) Gradio Space paketi (build_space.py --test)
Invoke-BuildStep -StepCode "(c)" -Title "Gradio Space paketi ve izole testi (``build_space.py --test``)" -Action {
    Push-Location $RepoRoot
    try {
        & $PyMain scripts/build_space.py --test
        if ($LASTEXITCODE -ne 0) { throw "build_space.py --test basarisiz oldu (kod: $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
} -MeasureArtifact {
    $Stats = Get-DirectoryStats (Join-Path $RepoRoot "space")
    "``space/``: $($Stats.Formatted) ($($Stats.Count) dosya)"
}

# (d) Tauri EXE (NSIS)
$SetupExePath = Join-Path $RepoRoot "desktop\src-tauri\target\release\bundle\nsis\Tracera_${CanonicalVersion}_x64-setup.exe"
Invoke-BuildStep -StepCode "(d)" -Title "Tauri Windows Masaustu Kurulumu (NSIS EXE)" -Action {
    if ($RebuildSidecar -or -not (Test-Path -LiteralPath $SidecarExe)) {
        if (-not (Test-Path -LiteralPath $PyBuild)) {
            throw "Sidecar derlemesi icin $PyBuild bulunamadi."
        }
        Push-Location $RepoRoot
        try {
            & $PyBuild api/build_exe.py
            if ($LASTEXITCODE -ne 0) { throw "api/build_exe.py basarisiz oldu (kod: $LASTEXITCODE)" }
        } finally {
            Pop-Location
        }
    }
    Push-Location (Join-Path $RepoRoot "desktop")
    try {
        & npx tauri build --config src-tauri/tauri.windows.conf.json
        if ($LASTEXITCODE -ne 0) { throw "npx tauri build basarisiz oldu (kod: $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    if (-not (Test-Path -LiteralPath $SetupExePath)) {
        throw "Beklenen NSIS kurulum dosyasi bulunamadi: $SetupExePath"
    }
} -MeasureArtifact {
    $Item = Get-Item -LiteralPath $SetupExePath
    "``$($Item.Name)``: $(Format-Size $Item.Length)"
}

# (e) Tauri Android APK (arm64)
$ApkOutDir = Join-Path $RepoRoot "desktop\src-tauri\gen\android\app\build\outputs\apk\universal\release"
$SignedApkCandidate = Join-Path $ApkOutDir "app-universal-release.apk"
$UnsignedApkCandidate = Join-Path $ApkOutDir "app-universal-release-unsigned.apk"

Invoke-BuildStep -StepCode "(e)" -Title "Tauri Android APK (arm64)" -Action {
    if (-not $HasReleaseSigning) {
        Write-Warning "Android release keystore ayarlanmamis; APK debug anahtari ile imzalanacak."
    }
    Push-Location (Join-Path $RepoRoot "desktop")
    try {
        & npx tauri android build --apk --target aarch64
        if ($LASTEXITCODE -ne 0) { throw "npx tauri android build basarisiz oldu (kod: $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
    if (-not $HasReleaseSigning -and (Test-Path -LiteralPath $UnsignedApkCandidate)) {
        $BuildTools = Get-ChildItem -LiteralPath (Join-Path $env:ANDROID_HOME "build-tools") -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $BuildTools) { throw "Android build-tools bulunamadi." }
        $ZipAlign = Join-Path $BuildTools.FullName "zipalign.exe"
        $ApkSigner = Join-Path $BuildTools.FullName "apksigner.bat"
        $DebugKeystore = Join-Path $env:USERPROFILE ".android\debug.keystore"
        if (-not (Test-Path -LiteralPath $DebugKeystore)) {
            $Keytool = Join-Path $env:JAVA_HOME "bin\keytool.exe"
            New-Item -ItemType Directory -Force -Path (Split-Path $DebugKeystore) | Out-Null
            & $Keytool -genkeypair -v -keystore $DebugKeystore -storepass android -alias androiddebugkey -keypass android -keyalg RSA -keysize 2048 -validity 10000 -dname "CN=Android Debug,O=Android,C=US"
        }
        $AlignedApk = Join-Path $ApkOutDir "app-universal-release-aligned.apk"
        if (Test-Path -LiteralPath $AlignedApk) { Remove-Item -LiteralPath $AlignedApk -Force }
        & $ZipAlign -p -f -v 4 $UnsignedApkCandidate $AlignedApk | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "zipalign basarisiz oldu." }
        & $ApkSigner sign --ks $DebugKeystore --ks-key-alias androiddebugkey --ks-pass pass:android --key-pass pass:android --out $SignedApkCandidate $AlignedApk
        if ($LASTEXITCODE -ne 0) { throw "apksigner basarisiz oldu." }
        Remove-Item -LiteralPath $AlignedApk -Force -ErrorAction SilentlyContinue
    }
    if (-not (Test-Path -LiteralPath $SignedApkCandidate)) {
        throw "Beklenen Android APK bulunamadi: $SignedApkCandidate"
    }
} -MeasureArtifact {
    $Item = Get-Item -LiteralPath $SignedApkCandidate
    $Tag = if ($HasReleaseSigning) { "release imzali" } else { "debug imzali (UYARI)" }
    "``$($Item.Name)``: $(Format-Size $Item.Length) [$Tag]"
}

# (f) dist_release/ altina EXE, APK ve SHA256SUMS.txt
$ReleaseExeDest = Join-Path $DistReleaseDir "Tracera_${CanonicalVersion}_x64-setup.exe"
$ReleaseApkDest = Join-Path $DistReleaseDir "Tracera_${CanonicalVersion}_android_arm64.apk"
$SumsDest       = Join-Path $DistReleaseDir "SHA256SUMS.txt"

Invoke-BuildStep -StepCode "(f)" -Title "``dist_release/`` paketleme ve SHA256SUMS.txt" -Action {
    New-Item -ItemType Directory -Force -Path $DistReleaseDir | Out-Null
    Copy-Item -LiteralPath $SetupExePath -Destination $ReleaseExeDest -Force
    Copy-Item -LiteralPath $SignedApkCandidate -Destination $ReleaseApkDest -Force

    $Artifacts = @($ReleaseExeDest, $ReleaseApkDest)
    $Lines = foreach ($Item in $Artifacts) {
        $Hash = (Get-FileHash -LiteralPath $Item -Algorithm SHA256).Hash.ToLowerInvariant()
        $Name = Split-Path -Leaf $Item
        "$Hash  $Name"
    }
    $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines($SumsDest, $Lines, $Utf8NoBom)
} -MeasureArtifact {
    $ExeInfo = Get-Item -LiteralPath $ReleaseExeDest
    $ApkInfo = Get-Item -LiteralPath $ReleaseApkDest
    "EXE: $(Format-Size $ExeInfo.Length), APK: $(Format-Size $ApkInfo.Length), ``SHA256SUMS.txt`` hazir"
}

Write-BuildLog "BASARILI"
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host " TUM ADIMLAR TAMAMLANDI (v$CanonicalVersion)" -ForegroundColor Green
Write-Host " Gunluk: $LogPath" -ForegroundColor Green
Write-Host " Paketler: $DistReleaseDir" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
